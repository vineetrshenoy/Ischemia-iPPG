import os
import json
import logging
import matplotlib.pyplot as plt
import scipy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import wandb
import mlflow
from sklearn.model_selection import KFold
from hand_ischemia.data import Hand_Ischemia_Dataset, Hand_Ischemia_Dataset_Test, H5Dataset, H5DatasetTest
from .evaluation_helpers import separate_by_task, _frequency_plot_grid, _evaluate_hr, _evaluate_prediction
from .plotting_functions import plot_window_ts, plot_30sec, plot_test_results, plot_window_post_algo, plot_window_physnet

from .simple_trainer import SimpleTrainer

from hand_ischemia.models import build_model, CorrelationLoss
from hand_ischemia.optimizers import build_optimizer, build_lr_scheduler
from hand_ischemia.config import get_cfg_defaults

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os


__all__ = ['Hand_Ischemia_Trainer']

logger = logging.getLogger(__name__)
wandb.require("core")

class Hand_Ischemia_Trainer(SimpleTrainer):

    def __init__(self, cfg, gpu_id):

        super(Hand_Ischemia_Trainer, self).__init__(cfg)
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
        self.cls_loss = torch.nn.BCELoss()
        self.regression_loss = CorrelationLoss()
        self.eps = 1e-6

        self.device = gpu_id

        logger.info('Inside Hand_Ischemia_Trainer')
    
    
    @staticmethod
    def test_partition(self, run, model, cls_model, optimizer, scheduler, dataloader, epoch):
        """Evaluating the algorithm on the held-out test subject

        Args:
            model (torch.nn.Module): The denoiser as a torch.nn.Module
            optimizer (torch.optim.Optimizer): The optimizer
            scheduler (torch.optim.lr_scheduler): The learning rate scheduler
            dataloader (torch.util.data.DataLoader): The data loader


        Returns:
            torch.nn.Module, torch.nn.optim, torch.: The neural network modules
        """   
        #cls_model.eval()
        pred_labels, pred_vector, gt_labels, gt_vector, hr_nn, hr_gt = [], [], [], [], [], []
        for iter, (time_series, ground_truth, cls_label, window_label) in enumerate(dataloader):

            #
            time_series = time_series.to(self.device)
            ground_truth = ground_truth.unsqueeze(1).to(self.device)

            denoised_ts = model(time_series.float())[:, -1:]
            #logger.info('Processed test sample {}'.format(window_label))
            '''
            if self.USE_DENOISER:
                #Denoiser
                with torch.no_grad():
                    time_series = denoiser_model(time_series.float()).T
                    time_series = time_series.unsqueeze(1)

            if self.CLS_MODEL_TYPE == 'SPEC':
                if time_series.shape[1] > 1: #Because the denoiser didn't collapse to one dimension
                    time_series = time_series[:, 0:1, :]
                ################################################## Pre-processing for complex model
                L = 10*time_series.shape[2] + 1
                X = self._adjoint_model(time_series, L)
                ##################################################
                #denoised_ts = denoised_ts.unsqueeze(0)
                #denoised_ts = torch.permute(denoised_ts, [0, 2, 1])
                #outloc = '/cis/net/r22a/data/vshenoy/durr_hand/pre_denoising/{}.jpg'.format(window_label[0])
                #plot_window_ts(self.FPS, time_series, denoised_ts, outloc, ground_truth)

                # Running the algorithm
                out = cls_model(X)
            
            elif self.CLS_MODEL_TYPE == 'TiSc':
                if time_series.shape[1] > 1: #Because the denoiser didn't collapse to one dimension
                    time_series = time_series[:, 0:1, :]
                time_series = time_series.squeeze().float()
                out = cls_model(time_series)
            '''
            '''
            pred_class, gt_class = torch.argmax(out), torch.argmax(ground_truth)
            pred_labels.append(pred_class), gt_labels.append(gt_class)
            pred_labels.append(pred_class), gt_labels.append(gt_class)
            pred_vector.append(out), gt_vector.append(ground_truth)
            pred_class = 'ischemic' if pred_class == 1 else 'perfuse'
            gt_class = 'ischemic' if gt_class == 1 else 'perfuse'
            '''
            pred_hr = _evaluate_hr(denoised_ts.detach(), self.FPS)
            hr_nn.append(pred_hr)
            gt_hr =  _evaluate_hr(ground_truth, self.FPS)
            hr_gt.append(gt_hr)
            
            if self.PLOT_INPUT_OUTPUT and epoch == self.epochs:
                #plot_test_results(self.FPS, time_series, window_label, epoch, gt_class, pred_class)
                #if iter % 10 == 0: #Plot only every tenth
                denoised_ts = denoised_ts.detach().cpu().numpy()
                denoised_ts = H5Dataset.normalize_filter_gt(self, denoised_ts[0, 0, :], self.FPS)
                denoised_ts = np.expand_dims(np.expand_dims(denoised_ts, axis=0), axis=0)
                if self.device == 0:
                    plot_window_physnet(run, self.FPS, ground_truth, denoised_ts, window_label, epoch, 0, 0)
                    x = 5
            #metrics = {'denoiser_loss': loss.detach().cpu().item()}
            #mlflow.log_metrics(metrics, step=step)
            #step += 1
            ###
        
        #pred_labels, gt_labels = torch.stack(pred_labels), torch.stack(gt_labels)
        #pred_vector, gt_vector = torch.squeeze(torch.stack(pred_vector)), torch.squeeze(torch.stack(gt_vector))
        #metrics = self.compute_torchmetrics(pred_vector, gt_vector, epoch)
        #
        #return metrics        
        
        
        #model = sparsePPGnn.model
        #optimizer = sparsePPGnn.optimizer
        #scheduler = sparsePPGnn.scheduler

        #mSNR = np.mean(snr_arr)
        return hr_nn, hr_gt


    def _adjoint_model(self, Y, L):
        """Applies the adjoint model. Calculates the gradients

        Args:
            Y (torch.Tensor): The matrix upon which to apply the adjoint
            L (int): Length fo the FFT

        Returns:
            torch.Tensor: Tensor representing the application of the adjoint model
        """
        X = torch.fft.rfft(Y, n=L, axis=2) * \
            (1 / torch.sqrt(torch.Tensor([L])).to(self.device))
        X = X[:, :, 0: (L//2) + 1].to(torch.cfloat)


        return X
        
    def train_partition(self, run, model, cls_model, optimizer, scheduler, dataloader, test_dataloader):
        """Training the denoiser on all subjects except one held-out test subjection

        Args:
            model (torch.nn.Module): The denoiser as a torch.nn.Module
            optimizer (torch.optim.Optimizer): The optimizer
            scheduler (torch.optim.lr_scheduler): The learning rate scheduler
            dataloader (torch.util.data.DataLoader): The data loader


        Returns:
            torch.nn.Module, torch.nn.optim, torch.: The neural network modules
        """
        model.train()
        #cls_model.train()
        step = 0
        
        for i in range(0, self.epochs):

            logger.info('Training on Epoch {}'.format(i))
            pred_labels, pred_vector, gt_labels, gt_vector = [], [], [], []

            for iter, (time_series, ground_truth, cls_label, window_label) in enumerate(dataloader):
            
                optimizer.zero_grad()
                time_series = time_series.to(self.device)
                ground_truth = ground_truth.unsqueeze(1).to(self.device)
                cls_label = cls_label.to(self.device)
                
                out = model(time_series.float())[:, -1:]
                zero_mean_out = (out - torch.mean(out, axis=2, keepdim=True)) / (torch.abs(torch.mean(out, axis=2, keepdim=True)) + 1e-6) #AC-DC Normalization
                '''
                if self.CLS_MODEL_TYPE == 'SPEC':
                    
                    ################################################## Pre-processing for complex model
                    L = 10*zero_mean_out.shape[2] + 1
                    X = self._adjoint_model(zero_mean_out, L)
                    ##################################################
                    #denoised_ts = denoised_ts.unsqueeze(0)
                    #denoised_ts = torch.permute(denoised_ts, [0, 2, 1])
                    #outloc = '/cis/net/r22a/data/vshenoy/durr_hand/pre_denoising/{}.jpg'.format(window_label[0])
                    #plot_window_ts(self.FPS, zero_mean_out, denoised_ts, outloc, ground_truth)

                    # Running the algorithm
                    cls_out = cls_model(X)
                elif self.CLS_MODEL_TYPE == 'TiSc':
                    if zero_mean_out.shape[1] > 1: #Because the denoiser didn't collapse to one dimension
                        zero_mean_out = zero_mean_out[:, 0:1, :]
                    zero_mean_out = zero_mean_out.squeeze().float()
                    cls_out = cls_model(zero_mean_out)
                '''
                
                loss = self.regression_loss(zero_mean_out, ground_truth.float()) #+ self.cls_loss(cls_out, cls_label)
                loss.backward()
                optimizer.step()
                
                
                #pred_vector.append(out), gt_vector.append(ground_truth)
                lr = scheduler.optimizer.param_groups[0]['lr']
                metrics = {'loss': loss.detach().cpu().item(),
                           'lr': lr}
                run.log(metrics, step=step) if run != None else False
                if self.device == 0:
                    mlflow.log_metrics(metrics, step=step)
                step += 1
                ####
            
            scheduler.step()
            '''
            #Getting test metrics
            pred_vector, gt_vector = torch.squeeze(torch.cat(pred_vector)), torch.squeeze(torch.cat(gt_vector))
            metrics = self.compute_torchmetrics(pred_vector, gt_vector, i, mode='train')
            mlflow.log_metrics(metrics, step=i)
            '''
            if i % self.eval_period == 0:
                hr_nn, hr_gt = self.test_partition(self, run, model, None, optimizer, scheduler, test_dataloader, i)
                #acc, auroc, prec =  met['test_acc'], met['test_auroc'], met['test_precision'],
                #recall, f1, conf = met['test_recall'], met['test_f1score'], met['test_confusion']
                #logger.warning('RESULTS: acc={}; auroc={}; prec={}; recall={}; f1={};'.format(acc, auroc, recall, prec, f1))
                met = self._compute_rmse_and_pte6(hr_gt, hr_nn)
                mae, rmse, pte6 =  met['mae'], met['rmse'], met['pte6']
                logger.warning('RESULTS: MAE={}; RMSE={}; PTE6={}'.format(mae, rmse, pte6))
                mlflow.log_metrics(met, step=i)
            

        
        return cls_model, optimizer, scheduler

    def train(self, experiment_id):
        """The main training loop for the partition trainer

        Args:
            experiment_id (_type_): The MLFlow experiment ID under which to list training runs
        """
        
        with open(self.train_json_path, 'r') as f:
            train_list = json.load(f)
        with open('/cis/net/r22a/data/vshenoy/durr_hand/model_code/physnet_ischemia/hand_ischemia/data/ubfc_only.json', 'r') as f:
            ubfc_dict = json.load(f)
        keys = np.array([*train_list])
        kf = KFold(6, shuffle=False)
        # Generates a partition of the data
        for idx, (train, val) in enumerate(kf.split(keys)):
            
            
            # Generating the one-versus-all partition of subjects for Hand Surgeon
            train_subjects = keys[train]
            val_subjects = keys[val]
            
            train_subdict = dict((k, train_list[k]) for k in train_subjects if k in train_list)
            val_subdict = dict((k, train_list[k]) for k in val_subjects if k in train_list)

            # Update training set with UBFC data
            train_subdict.update(ubfc_dict) 
            test_subject = keys[val][0]
            logger.info('Training Fold {}'.format(test_subject))

            
            # Build dataset
            train_dataset = H5Dataset(self.cfg, train_subdict)
            val_dataset = H5DatasetTest(self.cfg, val_subdict)
            logger.info('Train dataset size: {}'.format(len(train_dataset)))
            logger.info('Test dataset size: {}'.format(len(val_dataset)))
            
            #Update CFG
            self.cfg.INPUT.TRAIN_ISCHEMIC = train_dataset.num_ischemic
            self.cfg.INPUT.TRAIN_PERFUSE = train_dataset.num_perfuse
            self.cfg.INPUT.TEST_ISCHEMIC = val_dataset.num_ischemic
            self.cfg.INPUT.TEST_PERFUSE = val_dataset.num_perfuse
            #self.PLOT_INPUT_OUTPUT = False

            ##Build dataloader
            train_dataloader = DataLoader(
                train_dataset, batch_size=self.batch_size, sampler=DistributedSampler(train_dataset))
            val_dataloader = DataLoader(
                val_dataset, batch_size=1, shuffle=False)

            #Build the model, optimizer, and scheduler
            model, cls_model = build_model(self.cfg)
            model = model.to(self.device)
            model = DDP(model, device_ids=[self.device])
            optimizer = build_optimizer(self.cfg, model)
            lr_scheduler = build_lr_scheduler(self.cfg, optimizer)

            # Create experiment and log training parameters
            run_name = '{}'.format(test_subject)
            config_dictionary = dict(yaml=self.cfg)
            run = None
            if self.device == 0:
                run = wandb.init(
                    entity='vshenoy',
                    project='hand_surgeon',
                    group=experiment_id,
                    name=run_name,
                    config=config_dictionary
                )
                mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True)
                self.log_config_dict(self.cfg)

            # Train the model
            cls_model, optimizer, lr_scheduler = self.train_partition(run, model,
                None, optimizer, lr_scheduler, train_dataloader, val_dataloader)
            
            logger.warning('Finished training ')

            # Test the model
            hr_nn, hr_gt = self.test_partition(self, run, model, None, optimizer, lr_scheduler, val_dataloader, self.cfg.DENOISER.EPOCHS)
            
            
            #Compute and log the metrics; finish the run
            met = self._compute_rmse_and_pte6(hr_gt, hr_nn)
            if self.device == 0:
                mlflow.log_metrics(met, step=self.epochs)
                mlflow.end_run()
                run.finish() if run != None else False
            
            mae, rmse, pte6 =  met['mae'], met['rmse'], met['pte6']
            logger.warning('RESULTS: MAE={}; RMSE={}; PTE6={}'.format(mae, rmse, pte6))
            
                        
            ### Save the Model
            #out_dir = os.path.join(self.cfg.OUTPUT.OUTPUT_DIR, test_subject)
            #os.makedirs(out_dir, exist_ok=True)
            #model_name = 'model{}_.pth'.format(test_subject)
            #
            #out_path = os.path.join(out_dir, model_name)
            #torch.save({'model_state_dict': model.state_dict(),
            #            'optimizer_state_dict': optimizer.state_dict()}, out_path)
            #mlflow.log_artifacts(out_dir)


    def train_no_val(self, experiment_id):
        """The main training loop for the partition trainer

        Args:
            experiment_id (_type_): The MLFlow experiment ID under which to list training runs
        """
        
        with open(self.train_json_path, 'r') as f:
            train_list = json.load(f)
        with open(self.test_json_path, 'r') as f:
            test_list = json.load(f)
              
        
        # Build dataset
        train_dataset = H5Dataset(self.cfg, train_list)
        test_dataset = H5DatasetTest(self.cfg, test_list)
        logger.info('Train dataset size: {}'.format(len(train_dataset)))
        logger.info('Test dataset size: {}'.format(len(test_dataset)))        
        

        #Update CFG
        self.cfg.INPUT.TRAIN_ISCHEMIC = train_dataset.num_ischemic
        self.cfg.INPUT.TRAIN_PERFUSE = train_dataset.num_perfuse
        self.cfg.INPUT.TEST_ISCHEMIC = test_dataset.num_ischemic
        self.cfg.INPUT.TEST_PERFUSE = test_dataset.num_perfuse


        ## Build dataloader
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=DistributedSampler(train_dataset))
        test_dataloader = DataLoader(
            test_dataset, batch_size=1, shuffle=False)
        
        
        #Build the model, optimizer, and scheduler
        model, cls_model = build_model(self.cfg)
        model = model.to(self.device)
        model = DDP(model, device_ids=[self.device])
        optimizer = build_optimizer(self.cfg, model)
        lr_scheduler = build_lr_scheduler(self.cfg, optimizer)

        # Create experiment and log training parameters
        config_dictionary = dict(
            yaml=self.cfg,
        )
        run = None
        if self.device == 0:
            run = wandb.init(
                entity='vshenoy',
                project='hand_surgeon',
                config=config_dictionary
            )
            mlflow.start_run(experiment_id=experiment_id,nested=True)
            self.log_config_dict(self.cfg)

        
        # Train the model
        cls_model, optimizer, lr_scheduler = self.train_partition(run, model,
                None, optimizer, lr_scheduler, train_dataloader, test_dataloader)
        
        logger.warning('Finished training ')

        
        # Test the model
        hr_nn, hr_gt = self.test_partition(self, run, model, None, optimizer, lr_scheduler, test_dataloader, self.cfg.DENOISER.EPOCHS)
        
        
        #Comput eand log the metrics; end the run.
        met = self._compute_rmse_and_pte6(hr_gt, hr_nn)
        if self.device == 0:
            mlflow.log_metrics(met, step=self.epochs)
            mlflow.end_run()
            run.finish() if run != None else False
        
        mae, rmse, pte6 =  met['mae'], met['rmse'], met['pte6']
        logger.warning('RESULTS: MAE={}; RMSE={}; PTE6={}'.format(mae, rmse, pte6))
        
        ''' 
        acc, auroc, prec =  met['test_acc'], met['test_auroc'], met['test_precision']
        recall, f1, conf = met['test_recall'], met['test_f1score'], met['test_confusion']
        logger.warning('RESULTS: acc={}; auroc={}; prec={}; recall={}; f1={};'.format(acc, auroc, recall, prec, f1))
        mlflow.log_metrics(met, step=self.epochs)
        # Save the Model
        #out_dir = os.path.join(self.cfg.OUTPUT.OUTPUT_DIR, test_subject)
        #os.makedirs(out_dir, exist_ok=True)
        #model_name = 'model{}_.pth'.format(test_subject)
        #
        #out_path = os.path.join(out_dir, model_name)
        #torch.save({'model_state_dict': model.state_dict(),
        #            'optimizer_state_dict': optimizer.state_dict()}, out_path)
        #mlflow.log_artifacts(out_dir)
        
        # End the run
        mlflow.end_run()
        '''