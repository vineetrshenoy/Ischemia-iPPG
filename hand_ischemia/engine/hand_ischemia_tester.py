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

    def __init__(self, cfg, args):

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
        self.TEST_CV = args.test_CV
        self.eps = 1e-6
        self.rank = 0

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        logger.info('Inside Hand_Ischemia_Tester')
    



    def test(self, args, experiment_id, curr_exp_id):
        """The main training loop for the partition trainer

        Args:
            experiment_id (_type_): The MLFlow experiment ID under which to list training runs
        """
        with open(self.train_json_path, 'r') as f:
            train_list = json.load(f)
        with open('/cis/home/vshenoy/durr_hand/Physnet_Ischemia/hand_ischemia/data/tourniquet_ischemia.json', 'r') as f:
            tourniquet_list = json.load(f)
        with open('/cis/net/r22a/data/vshenoy/durr_hand/model_code/physnet_ischemia/hand_ischemia/data/ubfc_only.json', 'r') as f:
            ubfc_dict = json.load(f)
        keys = np.array([*train_list])
        tourniquet_keys = np.array([*tourniquet_list])

        HR_nn_full, HR_gt_full = [], []

        
        kf = KFold(5, shuffle=False)
        HR_nn_full, HR_gt_full = [], []
        # Generates a partition of the data
        cls_out_all, cls_label_all, pred_class_all, gt_class_all = [], [], [], []
        for idx, (perf_keys, tourn_keys) in enumerate(zip(kf.split(keys), kf.split(tourniquet_keys))):
            #if idx != 1:
            #    continue
            # Generating the one-versus-all partition of subjects for Hand Surgeon
            train_per, val_per = perf_keys
            
            
            # Generating the one-versus-all partition of subjects for Hand Surgeon
            train_subjects = keys[train_per]
            val_subjects = keys[val_per]
            val_subject = val_subjects[0]
                      
        
            query = "tag.mlflow.runName = 'split{}'".format(idx)
            sub_exp = mlflow.search_runs([args.experiment_id], filter_string=query, output_format='list')[0]
            
            if args.test_CV:
                            
                # Generating the one-versus-all partition of subjects for Hand Surgeon
                train_subjects = keys[train_per]
                val_subjects = keys[val_per]
                
                val_subdict = dict((k, train_list[k]) for k in val_subjects if k in train_list)
                val_dataset = H5DatasetTest(self.cfg, val_subdict)
            else: 
                with open(self.test_json_path, 'r') as f:
                    val_subdict = json.load(f)
                val_dataset = H5DatasetTest(self.cfg, val_subdict)
            # Build dataset
            self.cfg.INPUT.TEST_ISCHEMIC = val_dataset.num_ischemic
            self.cfg.INPUT.TEST_PERFUSE = val_dataset.num_perfuse
            
            logger.info('Test dataset size: {}'.format(len(val_dataset)))

            
            ## Build dataloader
            val_dataloader = DataLoader(
                val_dataset, batch_size=1, shuffle=False)
            #if test_subject != 'F018':
            #    continue
            artifact_loc = sub_exp.info.artifact_uri.replace('file://', '')

            # Build model, optimizer, lr_scheduler
            model, cls_model = build_model(self.cfg)
            optimizer = build_optimizer(self.cfg, model)
            lr_scheduler = build_lr_scheduler(self.cfg, optimizer)

            # Load checkpoint if it exists
            checkpoint_loc = os.path.join(artifact_loc, 'model_final.pth'.format(val_subject))
            try:
                checkpoint = torch.load(checkpoint_loc, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                
            except:
                raise Exception
            
            #model.load_state_dict(checkpoint['model_state_dict'])
            model, cls_model = model.to(self.device), cls_model.to(self.device)
            logger.info('Testing split{}'.format(idx))

    

            # Create experiment and log training parameters
            run_name = 'split{}'.format(idx)
            mlflow.start_run(experiment_id=curr_exp_id,
                             run_name=run_name, nested=True)
            self.log_config_dict(self.cfg)

            
            # Test the model
            with torch.no_grad():
                HR_nn, HR_gt = Hand_Ischemia_Trainer.test_partition(self, None, model, None, optimizer, lr_scheduler, val_dataloader, self.cfg.DENOISER.EPOCHS)
            
            if idx == 0:
                np.savez('all_hrs.npz', emp=HR_nn, gt=HR_gt)
            HR_nn_full = HR_nn_full + HR_nn
            HR_gt_full = HR_gt_full + HR_gt
            
            metrics = self._compute_rmse_and_pte6(HR_gt, HR_nn)
            rmse, mae, pte6 = metrics['rmse'], metrics['mae'], metrics['pte6']
            logger.warning('Hand Ischemia results Split {}: MAE =  {}; RMSE = {}; PTE6 = {}'.format(idx,  mae, rmse, pte6))
    
            mlflow.log_metrics(metrics, step=self.epochs)



            # End the run
            mlflow.end_run()

        metrics = self._compute_rmse_and_pte6(HR_gt_full, HR_nn_full)
        rmse, mae, pte6 = metrics['rmse'], metrics['mae'], metrics['pte6']
        logger.warning('Hand Ischemia Results: MAE =  {}; RMSE = {}; PTE6 = {}'.format( mae, rmse, pte6))
        mlflow.log_metrics(metrics, step=self.epochs)

        np.savez('all_hrs.npz', emp=HR_nn_full, gt=HR_gt_full)
        
    
    