import os
import logging
import numpy as np
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import mlflow
from hand_ischemia.data import H5Dataset, H5DatasetTest
from hand_ischemia.engine import Ischemia_Classifier_Trainer

from sklearn.model_selection import KFold
from .simple_trainer import SimpleTrainer
from hand_ischemia.models import build_model
from hand_ischemia.optimizers import build_optimizer, build_lr_scheduler

__all__ = ['Ischemia_Classifier_Tester']

logger = logging.getLogger(__name__)


class Ischemia_Classifier_Tester(SimpleTrainer):

    def __init__(self, cfg, args):

        super(Ischemia_Classifier_Tester, self).__init__(cfg)
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
        self.cls_loss = torch.nn.BCELoss()
        self.eps = 1e-6
        self.rank = 0
        

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        self.hann_window = torch.hann_window(300).to(self.device)

        logger.info('Inside Ischemia_Classifier_Tester')
    



    def test(self, args, curr_exp_id):
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
        for idx, (train, val) in enumerate(zip(kf.split(keys), kf.split(tourniquet_keys))):
            
            # Generating the one-versus-all partition of subjects for Hand Surgeon
            train_per, train_tourn = train
            val_per, val_tourn = val
            
            # Generating the one-versus-all partition of subjects for Hand Surgeon
            train_subjects = keys[train_per]
            val_subjects = keys[val_per]
            val_subject = val_subjects[0]
            
            #if val_subject != 'hand-subject6':
            #    continue
            
            train_tourniquet = tourniquet_keys[train_tourn]
            val_tourniquet = tourniquet_keys[val_tourn]
            
            query = "tag.mlflow.runName = 'split{}'".format(idx)
            sub_exp = mlflow.search_runs([args.experiment_id], filter_string=query, output_format='list')[0]
            cls_sub_exp = mlflow.search_runs([args.cls_experiment_id], filter_string=query, output_format='list')[0]
            
            if args.test_CV:
                            
                # Generating the one-versus-all partition of subjects for Hand Surgeon
                train_subjects = keys[train_per]
                val_subjects = keys[val_per]
                
                train_tourniquet = tourniquet_keys[train_tourn]
                val_tourniquet = tourniquet_keys[val_tourn]
                
                train_subdict = dict((k, train_list[k]) for k in train_subjects if k in train_list)
                tourniquet_train_subdict = dict((k, tourniquet_list[k]) for k in train_tourniquet if k in tourniquet_list)
                train_subdict.update(tourniquet_train_subdict)
                
                val_subdict = dict((k, train_list[k]) for k in val_subjects if k in train_list)
                tourniquet_val_subdict = dict((k, tourniquet_list[k]) for k in val_tourniquet if k in tourniquet_list)
                val_subdict.update(tourniquet_val_subdict)
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
            cls_artifact_loc = cls_sub_exp.info.artifact_uri.replace('file://', '')

            # Build model, optimizer, lr_scheduler
            model, cls_model = build_model(self.cfg)
            optimizer = build_optimizer(self.cfg, model)
            lr_scheduler = build_lr_scheduler(self.cfg, optimizer)

            # Load checkpoint if it exists
            checkpoint_loc = os.path.join(artifact_loc, 'model_final.pth'.format(val_subject))
            cls_checkpoint_loc = os.path.join(cls_artifact_loc, 'clsmodel_final.pth')
            try:
                checkpoint = torch.load(checkpoint_loc, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                
                checkpoint = torch.load(cls_checkpoint_loc, map_location=self.device)
                cls_model.load_state_dict(checkpoint['model_state_dict'])
            except:
                raise Exception
            
            #model.load_state_dict(checkpoint['model_state_dict'])
            model, cls_model = model.to(self.device), cls_model.to(self.device)
            logger.info('Testing subject {}'.format(val_subject))

            
            # Create experiment and log training parameters
            run_name = '{}'.format(val_subject)
            mlflow.start_run(experiment_id=curr_exp_id,
                             run_name=run_name, nested=True)
            self.log_config_dict(self.cfg)

            
            # Test the model
            met, full_mets = Ischemia_Classifier_Trainer.test_partition(self, None, model, cls_model, optimizer, lr_scheduler, val_dataloader, self.epochs)
            acc, auroc, prec =  met['test_acc'], met['test_auroc'], met['test_precision'],
            recall, f1, conf = met['test_recall'], met['test_f1score'], met['test_confusion']
            logger.warning('RESULTS: acc={}; auroc={}; prec={}; recall={}; f1={};'.format(acc, auroc, recall, prec, f1))
            mlflow.log_metrics(met, step=self.epochs)

            cls_out_all += full_mets[0]
            cls_label_all += full_mets[1]
            pred_class_all += full_mets[2]
            gt_class_all += full_mets[3]

            # End the run
            mlflow.end_run()


        
        metric_caulator = SimpleTrainer(self.cfg)
        cls_out_all, cls_label_all = torch.stack(cls_out_all), torch.stack(cls_label_all) 
        pred_class_all, gt_class_all = torch.stack(pred_class_all), torch.stack(gt_class_all) 
        metric_caulator.update_torchmetrics(cls_out_all, cls_label_all, pred_class_all, gt_class_all, mode='last')
        met = metric_caulator.compute_torchmetrics(self.epochs)
        
        acc, auroc, prec =  met['test_acc'], met['test_auroc'], met['test_precision'],
        recall, f1, conf = met['test_recall'], met['test_f1score'], met['test_confusion']
        logger.warning('OVERALL RESULTS: acc={}; auroc={}; prec={}; recall={}; f1={};'.format(acc, auroc, recall, prec, f1))
        mlflow.log_metrics(met, step=self.epochs)