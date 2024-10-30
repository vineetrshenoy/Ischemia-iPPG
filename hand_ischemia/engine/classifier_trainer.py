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
import torchaudio
from hand_ischemia.models import build_model, CorrelationLoss
from hand_ischemia.optimizers import build_optimizer, build_lr_scheduler
from hand_ischemia.config import get_cfg_defaults

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os


__all__ = ['Ischemia_Classifier_Trainer']

logger = logging.getLogger(__name__)
#wandb.require("core")

class Ischemia_Classifier_Trainer(SimpleTrainer):

    def __init__(self, cfg, gpu_id):

        super(Ischemia_Classifier_Trainer, self).__init__(cfg)
        self.cfg = cfg
        self.train_json_path = cfg.INPUT.TRAIN_JSON_PATH
        self.test_json_path = cfg.INPUT.TEST_JSON_PATH
        self.MIN_WINDOW_SEC = cfg.TIME_SCALE_PPG.MIN_WINDOW_SEC
        self.TIME_WINDOW_SEC = cfg.TIME_SCALE_PPG.TIME_WINDOW_SEC
        
        self.rank = gpu_id
        #if torch.cuda.is_available():
        #    self.device = torch.device('cuda')
        #else:
        #    self.device = torch.device('cpu')
        
        self.USE_DENOISER = cfg.TIME_SCALE_PPG.USE_DENOISER
        self.CLS_MODEL_TYPE = cfg.TIME_SCALE_PPG.CLS_MODEL_TYPE
        self.FPS = cfg.TIME_SCALE_PPG.FPS
        self.SLIDING_WINDOW_LENGTH = self.FPS * self.TIME_WINDOW_SEC
        self.batch_size = cfg.DENOISER.BATCH_SIZE
        self.epochs = cfg.DENOISER.EPOCHS
        self.eval_period = cfg.TEST.EVAL_PERIOD
        self.PLOT_INPUT_OUTPUT = cfg.TEST.PLOT_INPUT_OUTPUT
        self.PLOT_LAST = cfg.TEST.PLOT_LAST
        self.cls_loss = torch.nn.BCEWithLogitsLoss()
        self.hann_window = torch.hann_window(300).to(self.rank)
        #self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=300, power=1,
        #                          normalized=True, window_fn=self.hann_window)
        self.eps = 1e-6

        

        logger.info('Inside Ischemia_Classifier_Trainer')
    
    
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
        cls_model.eval()
        pred_labels, pred_vector, gt_labels, gt_vector, hr_nn, hr_gt, test_loss = [], [], [], [], [], [], []
        cls_out_all, cls_label_all, pred_class_all, gt_class_all = [], [], [], []
        for iter, (time_series, ground_truth, cls_label, window_label) in enumerate(dataloader):
                
                
            
            time_series = time_series.to(self.rank)
            if ground_truth is not None:
                ground_truth = ground_truth.unsqueeze(1).to(self.rank)
            cls_label = cls_label.to(self.rank)
            
            with torch.no_grad():
                out = model(time_series.float())[:, -1:]
                zero_mean_out = (out - torch.mean(out, axis=2, keepdim=True)) / (torch.abs(torch.mean(out, axis=2, keepdim=True)) + 1e-6) #AC-DC Normalization
            
                if self.CLS_MODEL_TYPE == 'SPEC':
                    '''
                    ################################################## Pre-processing for complex model
                    L = 10*zero_mean_out.shape[2] + 1
                    X = Ischemia_Classifier_Trainer._adjoint_model(self, zero_mean_out, L)
                    ##################################################
                    #denoised_ts = denoised_ts.unsqueeze(0)
                    #denoised_ts = torch.permute(denoised_ts, [0, 2, 1])
                    #outloc = '/cis/net/r22a/data/vshenoy/durr_hand/pre_denoising/{}.jpg'.format(window_label[0])
                    #plot_window_ts(self.FPS, zero_mean_out, denoised_ts, outloc, ground_truth)

                    # Running the algorithm
                    cls_out = cls_model(torch.abs(X)).unsqueeze(0)
                    
                    
                    '''
                    spec = torchaudio.functional.spectrogram(zero_mean_out, pad=0, window=self.hann_window, n_fft=300,
                                                win_length=300, hop_length=300//2, power=1, normalized=True)
                    spec = spec.repeat(1, 3, 1, 1)
                    cls_out = cls_model(spec)
                
                elif self.CLS_MODEL_TYPE == 'TiSc':
                    if zero_mean_out.shape[1] > 1: #Because the denoiser didn't collapse to one dimension
                        zero_mean_out = zero_mean_out[:, 0:1, :]
                    zero_mean_out = zero_mean_out.squeeze().float()
                    cls_out = cls_model(zero_mean_out)
            
            #logger.info('out shape{} ; label shape {}'.format(cls_out.shape, cls_label.shape))
            #logger.info(cls_out)
            #loss = self.cls_loss(cls_out, cls_label)
            #test_loss.append(loss.detach().cpu().numpy().item())
            
            cls_out = torch.nn.functional.sigmoid(cls_out)
            pred_class, gt_class = torch.round(cls_out), cls_label
            pred_class, gt_class = pred_class[0].to(torch.int32), gt_class[0].to(torch.int32)
            #logger.info('cls_out {}; cls_label {}; pred_class {}; gt_class {}'.format(cls_out, cls_label, pred_class, gt_class))
            self.update_torchmetrics(cls_out, cls_label, pred_class, gt_class)
            cls_out_all.append(cls_out), cls_label_all.append(cls_label)
            pred_class_all.append(pred_class), gt_class_all.append(gt_class)
            #pred_labels.append(pred_class), gt_labels.append(gt_class)
            #pred_vector.append(cls_out), gt_vector.append(cls_label)
            pred_class = 'ischemic' if pred_class.item() == 1 else 'perfuse'
            gt_class = 'ischemic' if gt_class.item() == 1 else 'perfuse'
            
            if self.PLOT_INPUT_OUTPUT and epoch == self.epochs:
                #plot_test_results(self.FPS, time_series, window_label, epoch, gt_class, pred_class)
                #if iter % 10 == 0: #Plot only every tenth
                denoised_ts = zero_mean_out.detach().cpu().numpy()
                denoised_ts = H5Dataset.normalize_filter_gt(self, denoised_ts[0, 0, :], self.FPS)
                denoised_ts = np.expand_dims(np.expand_dims(denoised_ts, axis=0), axis=0)
                if self.rank == 0:
                    plot_window_physnet(run, self.FPS, ground_truth, denoised_ts, window_label, epoch, gt_class, pred_class, cls_out)
                    x = 5
            
            ###
        metrics = self.compute_torchmetrics(epoch)
        #pred_labels, gt_labels = torch.stack(pred_labels), torch.stack(gt_labels)
        #pred_vector, gt_vector = torch.squeeze(torch.stack(pred_vector)), torch.squeeze(torch.stack(gt_vector))
        #metrics = self.update_and_compute_torchmetrics(pred_vector, gt_vector, epoch)
        #metrics['test_loss'] = np.mean(test_loss)
        #
        return metrics, [cls_out_all, cls_label_all, pred_class_all, gt_class_all]        
    
    @staticmethod
    def _adjoint_model(self, Y, L):
        """Applies the adjoint model. Calculates the gradients

        Args:
            Y (torch.Tensor): The matrix upon which to apply the adjoint
            L (int): Length fo the FFT

        Returns:
            torch.Tensor: Tensor representing the application of the adjoint model
        """
        X = torch.fft.rfft(Y, n=L, axis=2) * \
            (1 / torch.sqrt(torch.Tensor([L])).to(self.rank))
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
        model.eval()
        cls_model.train()
        step = 0
        
        for i in range(0, self.epochs):

            logger.info('Training on Epoch {}'.format(i))
            pred_labels, pred_vector, gt_labels, gt_vector = [], [], [], []
            training_loss = []
            for iter, (time_series, ground_truth, cls_label, window_label) in enumerate(dataloader):
                
                
                optimizer.zero_grad()
                time_series = time_series.to(self.rank)
                ground_truth = ground_truth.unsqueeze(1).to(self.rank)
                cls_label = cls_label.to(self.rank)
                
                with torch.no_grad():
                    out = model(time_series.float())[:, -1:]
                    zero_mean_out = (out - torch.mean(out, axis=2, keepdim=True)) / (torch.abs(torch.mean(out, axis=2, keepdim=True)) + 1e-6) #AC-DC Normalization
                
                if self.CLS_MODEL_TYPE == 'SPEC':
                    '''
                    ################################################## Pre-processing for complex model
                    L = 10*zero_mean_out.shape[2] + 1
                    X = Ischemia_Classifier_Trainer._adjoint_model(self, zero_mean_out, L)
                    ##################################################
                    #denoised_ts = denoised_ts.unsqueeze(0)
                    #denoised_ts = torch.permute(denoised_ts, [0, 2, 1])
                    #outloc = '/cis/net/r22a/data/vshenoy/durr_hand/pre_denoising/{}.jpg'.format(window_label[0])
                    #plot_window_ts(self.FPS, zero_mean_out, denoised_ts, outloc, ground_truth)

                    # Running the algorithm
                    #X_abs = torch.abs(X)
                    #max = torch.max(X_abs, dim=2, keepdim=True)[0]
                    #X_abs = X_abs / max
                    #cls_out = cls_model(X_abs)
                    logger.info('Shape: {}'.format(torch.abs(X).shape))
                    cls_out = cls_model(torch.abs(X))
                    '''
                    spec = torchaudio.functional.spectrogram(zero_mean_out, pad=0, window=self.hann_window, n_fft=300,
                                                win_length=300, hop_length=300//2, power=1, normalized=True)
                    spec = spec.repeat(1, 3, 1, 1)
                    cls_out = cls_model(spec)
                
                elif self.CLS_MODEL_TYPE == 'TiSc':
                    if zero_mean_out.shape[1] > 1: #Because the denoiser didn't collapse to one dimension
                        zero_mean_out = zero_mean_out[:, 0:1, :]
                    zero_mean_out = zero_mean_out.squeeze().float()
                    cls_out = cls_model(zero_mean_out)
                
                
                loss = self.cls_loss(cls_out, cls_label)
                loss.backward()
                optimizer.step()
                
                
                
                #pred_vector.append(out), gt_vector.append(ground_truth)
                training_loss.append(loss.detach().cpu().numpy().item())
                lr = scheduler.optimizer.param_groups[0]['lr']
                metrics = {'loss': loss.detach().cpu().item(),
                           'lr': lr}
                #run.log(metrics, step=step) if run != None else False
                if self.rank == 0:
                    mlflow.log_metrics(metrics, step=step)
                step += 1
                ####
            
            
            if self.rank == 0:
                mean_training_loss = np.mean(training_loss)
                metrics = {'epoch_training_loss': mean_training_loss.item()}
                mlflow.log_metrics(metrics, step=step)
            scheduler.step()
            '''
            #Getting test metrics
            pred_vector, gt_vector = torch.squeeze(torch.cat(pred_vector)), torch.squeeze(torch.cat(gt_vector))
            metrics = self.compute_torchmetrics(pred_vector, gt_vector, i, mode='train')
            mlflow.log_metrics(metrics, step=i)
            '''
            if i % self.eval_period == 0:
                logger.warning('Evaluating at epoch {}'.format(i))
                met, _ = self.test_partition(self, run, model, cls_model, optimizer, scheduler, test_dataloader, i)
                
                acc, auroc, prec =  met['test_acc'], met['test_auroc'], met['test_precision'],
                recall, f1, conf = met['test_recall'], met['test_f1score'], met['test_confusion']
                logger.warning('RESULTS: acc={}; auroc={}; prec={}; recall={}; f1={};'.format(acc, auroc, recall, prec, f1))
                
                
                if self.rank == 0:
                    mlflow.log_metrics(metrics, step=step)
                
                
                cls_model.train()

        
        return cls_model, optimizer, scheduler

    
    
    def train_classifier(self, experiment_id, curr_exp_id):
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
        for idx, (perf_keys, tourn_keys) in enumerate(zip(kf.split(keys), kf.split(tourniquet_keys))):

            if idx != 1:
                continue
            
            train_per, val_per = perf_keys
            train_tourn, val_tourn = tourn_keys
            
            # Generating the one-versus-all partition of subjects for Hand Surgeon
            train_subjects = keys[train_per]
            val_subjects = keys[val_per]
            val_subject = val_subjects[0]
            
            #if val_subject != 'hand-subject6':
            #    continue
            
            train_tourniquet = tourniquet_keys[train_tourn]
            val_tourniquet = tourniquet_keys[val_tourn]
            
            
            query = "tag.mlflow.runName = 'split{}'".format(idx)
            sub_exp = mlflow.search_runs([experiment_id], filter_string=query, output_format='list')[0]
        
            train_subdict = dict((k, train_list[k]) for k in train_subjects if k in train_list)
            tourniquet_train_subdict = dict((k, tourniquet_list[k]) for k in train_tourniquet if k in tourniquet_list)
            train_subdict.update(tourniquet_train_subdict)
            #train_subdict.update(ubfc_dict)
            
            val_subdict = dict((k, train_list[k]) for k in val_subjects if k in train_list)
            tourniquet_val_subdict = dict((k, tourniquet_list[k]) for k in val_tourniquet if k in tourniquet_list)
            val_subdict.update(tourniquet_val_subdict)
        
            # Build dataset
            train_dataset = H5Dataset(self.cfg, train_subdict)
            val_dataset = H5DatasetTest(self.cfg, val_subdict)
            #val_dataset = H5Dataset(self.cfg, train_subdict) #Debug only
            logger.info('Train dataset size: {}'.format(len(train_dataset)))
            logger.info('Test dataset size: {}'.format(len(val_dataset)))
            
            
            self.cfg.INPUT.TRAIN_ISCHEMIC = train_dataset.num_ischemic
            self.cfg.INPUT.TRAIN_PERFUSE = train_dataset.num_perfuse
            self.cfg.INPUT.TEST_ISCHEMIC = val_dataset.num_ischemic
            self.cfg.INPUT.TEST_PERFUSE = val_dataset.num_perfuse
            
            pos_weight = torch.tensor([train_dataset.num_perfuse/train_dataset.num_ischemic]).to(self.rank)
            self.cls_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            ## Build dataloader
            train_dataloader = DataLoader(
                train_dataset, batch_size=self.batch_size, drop_last=True, sampler=DistributedSampler(train_dataset))
            #train_dataloader = DataLoader(
            #    train_dataset, batch_size=self.batch_size, drop_last=True, shuffle=True)
            test_dataloader = DataLoader(
                val_dataset, batch_size=1, shuffle=False)
            
        
            # Build model
            model, cls_model = build_model(self.cfg)
            model, cls_model  = model.to(self.rank), cls_model.to(self.rank)
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            cls_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(cls_model)
            model, cls_model = DDP(model, device_ids=[self.rank]), DDP(cls_model, device_ids=[self.rank])
            
            # Load checkpoint if it exists
            artifact_loc = sub_exp.info.artifact_uri.replace('file://', '')
            checkpoint_loc = os.path.join(artifact_loc, 'model_final.pth')
            try:
                checkpoint = torch.load(checkpoint_loc, map_location=self.device)
                model.module.load_state_dict(checkpoint['model_state_dict'])
            except:
                raise Exception
            
            #Send the model to DDP
            #model, cls_model = model.to(self.rank), cls_model.to(self.rank)
            #model, cls_model = DDP(model, device_ids=[self.rank]), DDP(cls_model, device_ids=[self.rank])
            
            #Build the optimizer, lr_scheduler
            optimizer = build_optimizer(self.cfg, cls_model)
            lr_scheduler = build_lr_scheduler(self.cfg, optimizer)
            logger.info('Training split {}'.format(idx))

            
            # Create experiment and log training parameters
            run_name = 'split{}'.format(idx)
            # Create experiment and log training parameters
            config_dictionary = dict(
                yaml=self.cfg,
            )
            run = None
            if self.rank == 0:
                run = wandb.init(
                    entity='vshenoy',
                    project='hand_surgeon',
                    config=config_dictionary
                )
                mlflow.start_run(experiment_id=curr_exp_id, run_name=run_name, nested=True)
                self.log_config_dict(self.cfg)

            
            
            # Train the model
            cls_model, optimizer, lr_scheduler = self.train_partition(run, model,
                    cls_model, optimizer, lr_scheduler, train_dataloader, test_dataloader)
            
            logger.warning('Finished Training; now testing')
            #Test the model
            met, _ = self.test_partition(self, run, model, cls_model, optimizer, lr_scheduler, test_dataloader, self.epochs)    
            acc, auroc, prec =  met['test_acc'], met['test_auroc'], met['test_precision'],
            recall, f1, conf = met['test_recall'], met['test_f1score'], met['test_confusion']
            logger.warning('RESULTS: acc={}; auroc={}; prec={}; recall={}; f1={};'.format(acc, auroc, recall, prec, f1))
            mlflow.log_metrics(met, step=self.epochs)

            
            ## Save the Model
            out_dir = os.path.join(self.cfg.OUTPUT.OUTPUT_DIR, val_subject)
            os.makedirs(out_dir, exist_ok=True)
            model_name = 'clsmodel_final.pth'
            
            out_path = os.path.join(out_dir, model_name)
            torch.save({'model_state_dict': cls_model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, out_path)
            mlflow.log_artifacts(out_dir)


            # End the run
            mlflow.end_run()
            

        #metrics = self._compute_rmse_and_pte6(HR_gt_full, HR_nn_full)
        #rmse, mae, pte6 = metrics['rmse'], metrics['mae'], metrics['pte6']
        #logger.warning('Hand Ischemia Results: MAE =  {}; RMSE = {}; PTE6 = {}'.format( mae, rmse, pte6))
        #mlflow.log_metrics(metrics, step=self.epochs)