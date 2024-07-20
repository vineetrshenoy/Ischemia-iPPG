import os
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import mlflow
from hand_ischemia.data import Hand_Ischemia_Dataset, Hand_Ischemia_Dataset_Test
from hand_ischemia.engine import Hand_Ischemia_Trainer

from .simple_trainer import SimpleTrainer
#from hand_ischemia.models import build_model
from hand_ischemia.optimizers import build_optimizer, build_lr_scheduler

__all__ = ['Hand_Ischemia_Tester']

logger = logging.getLogger(__name__)


class Hand_Ischemia_Tester(SimpleTrainer):

    def __init__(self, cfg):

        super(Hand_Ischemia_Tester, self).__init__(cfg)
        self.cfg = cfg
        
        self.MIN_WINDOW_SEC = cfg.TIME_SCALE_PPG.MIN_WINDOW_SEC
        self.TIME_WINDOW_SEC = cfg.TIME_SCALE_PPG.TIME_WINDOW_SEC
        self.FPS = cfg.TIME_SCALE_PPG.FPS
        self.SLIDING_WINDOW_LENGTH = self.FPS * self.TIME_WINDOW_SEC
        self.L = 10*self.SLIDING_WINDOW_LENGTH + 1
        self.batch_size = cfg.DENOISER.BATCH_SIZE
        self.epochs = cfg.DENOISER.EPOCHS
        self.eps = 1e-6

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        logger.info('Inside Hand_Ischemia_Tester')
    



    def test(self, experiment_id, curr_exp_id):
        """The main training loop for the partition trainer

        Args:
            experiment_id (_type_): The MLFlow experiment ID under which to list training runs
        """
        HR_nn_full, HR_gt_full = [], []
        sub_experiments = mlflow.search_runs([experiment_id], output_format='list')
        # Generates a partition of the data
        for sub_exp in sub_experiments:
            
            test_subject = sub_exp.data.tags['mlflow.runName']
            #if test_subject != 'F018':
            #    continue
            artifact_loc = sub_exp.info.artifact_uri.replace('file://', '')

            # Build model, optimizer, lr_scheduler
            model = build_model(self.cfg)
            optimizer = build_optimizer(self.cfg, model)
            lr_scheduler = build_lr_scheduler(self.cfg, optimizer)

            # Load checkpoint if it exists
            checkpoint_loc = os.path.join(artifact_loc, 'model' + test_subject + '_.pth')
            try:
                checkpoint = torch.load(checkpoint_loc, map_location=self.device)
            except:
                continue
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            logger.info('Testing subject {}'.format(test_subject))

            # Build dataset
            train_dataset = MMSE_HR_Dataset(self.cfg, test_subject)
            test_dataset = MMSE_HR_Dataset_Test(
                self.cfg, train_dataset.gt_test_subject, train_dataset.ts_test_subject)

            # Build dataloader
            test_dataloader = DataLoader(
                test_dataset, batch_size=1, shuffle=False)
           

            # Create experiment and log training parameters
            run_name = '{}'.format(test_subject)
            mlflow.start_run(experiment_id=curr_exp_id,
                             run_name=run_name, nested=True)
            self.log_config_dict(self.cfg)

            
            # Test the model
            HR_nn, HR_gt, mSNR = MMSE_Denoiser_Trainer.test_partition(self, 
                model, optimizer, lr_scheduler, test_dataloader, self.cfg.DENOISER.EPOCHS)
            HR_nn_full = HR_nn_full + HR_nn
            HR_gt_full = HR_gt_full + HR_gt
            max_memory = torch.cuda.max_memory_allocated(device=self.device)
            metrics = self._compute_rmse_and_pte6(HR_gt, HR_nn)
            rmse, mae, pte6, mape, rho = metrics['rmse'], metrics['mae'], metrics['pte6'], metrics['mape'], metrics['rho']
            logger.warning('MMSE-HR results Subject {}: MAE =  {}; RMSE = {}; PTE6 = {}; MAPE: {}; Pearson: {}; mSNR: {}'.format(test_subject,  mae, rmse, pte6, mape, rho, mSNR))
            metrics['mSNR'], metrics['max_memory'] = mSNR, max_memory
            mlflow.log_metrics(metrics, step=self.epochs)



            # End the run
            mlflow.end_run()

        metrics = self._compute_rmse_and_pte6(HR_gt_full, HR_nn_full)
        rmse, mae, pte6, mape, rho = metrics['rmse'], metrics['mae'], metrics['pte6'], metrics['mape'], metrics['rho']
        logger.warning('MMSE-HR Results: MAE =  {}; RMSE = {}; PTE6 = {}; MAPE: {}; Pearson: {}; mSNR: {}'.format( mae, rmse, pte6, mape, rho, mSNR))
        mlflow.log_metrics(metrics, step=self.epochs)

