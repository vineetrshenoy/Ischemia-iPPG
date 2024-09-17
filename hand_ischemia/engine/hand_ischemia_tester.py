import os
import logging
import numpy as np
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import mlflow
from hand_ischemia.data import H5Dataset, H5DatasetTest
from hand_ischemia.engine import Hand_Ischemia_Trainer

from sklearn.model_selection import KFold
from .simple_trainer import SimpleTrainer
from hand_ischemia.models import build_model
from hand_ischemia.optimizers import build_optimizer, build_lr_scheduler

__all__ = ['Hand_Ischemia_Tester']

logger = logging.getLogger(__name__)


class Hand_Ischemia_Tester(SimpleTrainer):

    def __init__(self, cfg):

        super(Hand_Ischemia_Tester, self).__init__(cfg)
        self.cfg = cfg
        self.train_json_path = cfg.INPUT.TRAIN_JSON_PATH
        self.test_json_path = cfg.INPUT.TEST_JSON_PATH
        self.MIN_WINDOW_SEC = cfg.TIME_SCALE_PPG.MIN_WINDOW_SEC
        self.TIME_WINDOW_SEC = cfg.TIME_SCALE_PPG.TIME_WINDOW_SEC
        
        self.USE_DENOISER = cfg.TIME_SCALE_PPG.USE_DENOISER
        self.CLS_MODEL_TYPE = cfg.TIME_SCALE_PPG.CLS_MODEL_TYPE
        self.FPS = cfg.TIME_SCALE_PPG.FPS
        self.SLIDING_WINDOW_LENGTH = self.FPS * self.TIME_WINDOW_SEC
        self.batch_size = cfg.DENOISER.BATCH_SIZE
        self.epochs = cfg.DENOISER.EPOCHS
        self.eval_period = cfg.TEST.EVAL_PERIOD
        self.PLOT_INPUT_OUTPUT = cfg.TEST.PLOT_INPUT_OUTPUT
        self.PLOT_LAST = cfg.TEST.PLOT_LAST
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
        with open(self.train_json_path, 'r') as f:
            train_list = json.load(f)

        HR_nn_full, HR_gt_full = [], []
        sub_experiments = mlflow.search_runs([experiment_id], output_format='list')
        # Generates a partition of the data
        for sub_exp in sub_experiments:
            
            test_subject = sub_exp.data.tags['mlflow.runName']
            if test_subject.split('-')[0] != 'hand':
                continue            
            
            val_subdict = {test_subject: train_list[test_subject]}
            
            #if test_subject != 'F018':
            #    continue
            artifact_loc = sub_exp.info.artifact_uri.replace('file://', '')

            # Build model, optimizer, lr_scheduler
            model, cls_model = build_model(self.cfg)
            optimizer = build_optimizer(self.cfg, model)
            lr_scheduler = build_lr_scheduler(self.cfg, optimizer)

            # Load checkpoint if it exists
            checkpoint_loc = os.path.join(artifact_loc, 'model_{}.pth'.format(test_subject))
            try:
                checkpoint = torch.load(checkpoint_loc, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
            except:
                raise Exception
            
            #model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            logger.info('Testing subject {}'.format(test_subject))

            # Build dataset
            val_dataset = H5DatasetTest(self.cfg, val_subdict)
            logger.info('Test dataset size: {}'.format(len(val_dataset)))
            
            #Update CFG
            self.cfg.INPUT.TEST_ISCHEMIC = val_dataset.num_ischemic
            self.cfg.INPUT.TEST_PERFUSE = val_dataset.num_perfuse
            #self.PLOT_INPUT_OUTPUT = False

            ##Build dataloader
            val_dataloader = DataLoader(
                val_dataset, batch_size=1, shuffle=False)

    

            # Create experiment and log training parameters
            run_name = '{}'.format(test_subject)
            mlflow.start_run(experiment_id=curr_exp_id,
                             run_name=run_name, nested=True)
            self.log_config_dict(self.cfg)

            
            # Test the model
            HR_nn, HR_gt = Hand_Ischemia_Trainer.test_partition(self, None, model, None, optimizer, lr_scheduler, val_dataloader, self.cfg.DENOISER.EPOCHS)
            
            HR_nn_full = HR_nn_full + HR_nn
            HR_gt_full = HR_gt_full + HR_gt
            
            metrics = self._compute_rmse_and_pte6(HR_gt, HR_nn)
            rmse, mae, pte6 = metrics['rmse'], metrics['mae'], metrics['pte6']
            logger.warning('Hand Ischemia results Subject {}: MAE =  {}; RMSE = {}; PTE6 = {}'.format(test_subject,  mae, rmse, pte6))
    
            mlflow.log_metrics(metrics, step=self.epochs)



            # End the run
            mlflow.end_run()

        metrics = self._compute_rmse_and_pte6(HR_gt_full, HR_nn_full)
        rmse, mae, pte6 = metrics['rmse'], metrics['mae'], metrics['pte6']
        logger.warning('Hand Ischemia Results: MAE =  {}; RMSE = {}; PTE6 = {}'.format( mae, rmse, pte6))
        mlflow.log_metrics(metrics, step=self.epochs)

