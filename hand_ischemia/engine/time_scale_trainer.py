import os
import json
import logging
import matplotlib.pyplot as plt
import scipy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import mlflow
from sklearn.model_selection import KFold
from hand_ischemia.data import MMSE_HR_Dataset, MMSE_HR_Dataset_Test
from .evaluation_helpers import separate_by_task, _frequency_plot_grid, _process_ground_truth_window, _evaluate_prediction
from .plotting_functions import plot_window_ts, plot_30sec

from .simple_trainer import SimpleTrainer
from hand_ischemia.models import build_model
from hand_ischemia.optimizers import build_optimizer, build_lr_scheduler
from hand_ischemia.config import get_cfg_defaults


__all__ = ['MMSE_Denoiser_Trainer']

logger = logging.getLogger(__name__)


class MMSE_Denoiser_Trainer(SimpleTrainer):

    def __init__(self, cfg):

        super(MMSE_Denoiser_Trainer, self).__init__(cfg)
        self.cfg = cfg
        
        self.MIN_WINDOW_SEC = cfg.SPARSE_PPG.MIN_WINDOW_SEC
        self.TIME_WINDOW_SEC = cfg.SPARSE_PPG.TIME_WINDOW_SEC
        self.FPS = cfg.SPARSE_PPG.FPS
        self.SLIDING_WINDOW_LENGTH = self.FPS * self.TIME_WINDOW_SEC
        self.batch_size = cfg.DENOISER.BATCH_SIZE
        self.epochs = cfg.DENOISER.EPOCHS
        self.eval_period = cfg.TEST.EVAL_PERIOD
        self.eps = 1e-6

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        logger.info('Inside MMSE_Denoiser_Trainer')
    
        
    
    @staticmethod
    def test_partition(self, model, optimizer, scheduler, dataloader, epoch):
        """Evaluating the algorithm on the held-out test subject

        Args:
            model (torch.nn.Module): The denoiser as a torch.nn.Module
            optimizer (torch.optim.Optimizer): The optimizer
            scheduler (torch.optim.lr_scheduler): The learning rate scheduler
            dataloader (torch.util.data.DataLoader): The data loader


        Returns:
            torch.nn.Module, torch.nn.optim, torch.: The neural network modules
        """

        sparsePPGnn = SparsePPGnn(self.cfg, model, optimizer, scheduler)
        label_window, nn_waveform, gt_waveform = [], [], []
        for iter, (time_series, ground_truth, window_label) in enumerate(dataloader):

            logger.info('Evaluating window{}; shape {}'.format(window_label[0], time_series.shape))
            batch_size, num_regions, time_length = time_series.shape

            time_series = time_series.to(self.device)
            ground_truth = ground_truth.to(self.device)

            # Running the algorithm
            freq_bins, X, Z_est, = sparsePPGnn.run(
                time_series, ground_truth, window_label, mode='test')
            ###
            Z_est = Z_est[:, :, 0:self.SLIDING_WINDOW_LENGTH]
            # Waveform MAE and MSE
            repated_gt = torch.repeat_interleave(
                ground_truth, num_regions, dim=1)
    
            #Append the waveforms for 30-second concatenation
            nn_waveform.append(Z_est), gt_waveform.append(repated_gt)
            label_window.append(window_label[0]) #Append window label for 30-second concatenation
            
            # Plotting functions
            Z_est = Z_est.cpu().detach().numpy()
            ground_truth = ground_truth.cpu()
            #self.plot_window_gt(self.FPS, ground_truth.numpy(), window_label[0])
            plot_window_ts(self.FPS, time_series.cpu().numpy(),
                Z_est, ground_truth.numpy(), window_label[0], iter, epoch)

            HR_estimate = _evaluate_prediction(freq_bins, X, window_label[0])
            
            
            HR_ground_truth = _process_ground_truth_window(
                ground_truth, self.FPS)
            
            metrics = {
                'HR_gt': HR_ground_truth,
                'HR_estimate': HR_estimate,
            }

            
        #Separate the windows by tasks. Before, dataloader loads windows sequentially independent of task
        nnwave, gtwave, label = separate_by_task(nn_waveform, gt_waveform, label_window) 
        
        num_tasks = len(nnwave)
        subject_hr_nn, subject_hr_gt, snr_arr = [], [], []
        
        step = 0 
        for i in range(0, num_tasks): #Process 30-second windows and get results
            nn_hr, gt_hr, snr = MMSE_Denoiser_Trainer.process_30_sec_windows(self, nnwave[i], gtwave[i], label[i]+'epoch'+str(epoch), epoch)
            subject_hr_nn += nn_hr
            subject_hr_gt += gt_hr
            snr_arr += snr
            #Log the metrics to mlflow
            for k in range(0, len(subject_hr_nn)):
                metrics = {
                    'HR_gt': subject_hr_gt[k],
                    'HR_estimate': subject_hr_nn[k],
                    'SNR': snr_arr[k]
                }
                mlflow.log_metrics(metrics, step=k)
                step += 1
        
        
        
        model = sparsePPGnn.model
        optimizer = sparsePPGnn.optimizer
        scheduler = sparsePPGnn.scheduler

        mSNR = np.mean(snr_arr)
        return subject_hr_nn, subject_hr_gt, mSNR

    def process_30_sec_windows(self, nnwave, gtwave, label, epoch):
        """This function performs evaluation on the 30 second windows 
        that were concatenated from the previous function; then does evaluation

        Args:
            nnwave (torch.Tensor): The entire time series for a certain subject and task
            gtwave (torch.Tensor): The entire time series for a certain subject and task
            label (str): The label of thewindow (i.e. F017_T10)

        Returns:
            list: The heart rates predicted and from ground-truth
        """
        
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        bs, reg, siglen = nnwave.shape
        sliding_window_length = self.FPS * 30 #For 30 second windows
        sliding_window_start = 0
        sliding_window_end = sliding_window_start + sliding_window_length
        count = 0
        L = 7501
        freq_bins = self.FPS * torch.arange(0, L/2) / L
        nn_hr, gt_hr, snr_arr = [], [], []
        
        while sliding_window_end <= siglen: #While we haven't exceeded the end of the signal
            
            #Get the 30-second windows
            ppg_mat_window = nnwave[0:1, :,  sliding_window_start:sliding_window_end]
            gt_wave_window = gtwave[0:1, 0:1,  sliding_window_start:sliding_window_end].cpu()
            
            X = torch.fft.rfft(ppg_mat_window, n=L, axis=2) * \
                (1 / torch.sqrt(torch.Tensor([L])).to(device))
            X = X[:, :, 0: (L//2) + 1].to(torch.cfloat)
            window_label = label + 'win{}'.format(count)
            nn_win_hr = _evaluate_prediction(freq_bins, X, window_label)
            nn_hr.append(nn_win_hr)
            
            gt_win_hr = _process_ground_truth_window(gt_wave_window, 25)
            gt_hr.append(gt_win_hr)
            signal = ppg_mat_window.clone().detach().cpu().numpy()
            snr = self._calculate_SNR(signal, gt_win_hr, fs=self.FPS)
            snr_arr.append(snr)
            #Update the signal pointers to the correct locations
            sliding_window_start = sliding_window_start + sliding_window_length
            sliding_window_end = sliding_window_start + sliding_window_length
            count += 1
            
            plot_30sec(self.FPS, ppg_mat_window, gt_wave_window, window_label, epoch)
            
        min_window_sec = self.MIN_WINDOW_SEC #Need atleast MIN_WINDOW_SEC to evaluate
        partial_window_end = sliding_window_start + (self.FPS * min_window_sec)
        if partial_window_end < siglen: #Get the last window longer than 10 sec
            
            sliding_window_end = siglen
            ppg_mat_window = nnwave[0:1, :,  sliding_window_start:sliding_window_end]
            gt_wave_window = gtwave[0:1, 0:1,  sliding_window_start:sliding_window_end].cpu()
            
            X = torch.fft.rfft(ppg_mat_window, n=L, axis=2) * \
                (1 / torch.sqrt(torch.Tensor([L])).to(device))
            X = X[:, :, 0: (L//2) + 1].to(torch.cfloat)
            window_label = label + 'win{}'.format(count)
            nn_win_hr = _evaluate_prediction(freq_bins, X, window_label)
            nn_hr.append(nn_win_hr)
            
            gt_win_hr = _process_ground_truth_window(gt_wave_window, 25)
            gt_hr.append(gt_win_hr)
            signal = ppg_mat_window.clone().detach().cpu().numpy()
            snr = self._calculate_SNR(signal, gt_win_hr, fs=self.FPS)
            snr_arr.append(snr)
            
            plot_30sec(self.FPS, ppg_mat_window, gt_wave_window, window_label, epoch)
        
        return nn_hr, gt_hr, snr_arr
        
    def train_partition(self, model, optimizer, scheduler, dataloader, test_dataloader):
        """Training the denoiser on all subjects except one held-out test subjection

        Args:
            model (torch.nn.Module): The denoiser as a torch.nn.Module
            optimizer (torch.optim.Optimizer): The optimizer
            scheduler (torch.optim.lr_scheduler): The learning rate scheduler
            dataloader (torch.util.data.DataLoader): The data loader


        Returns:
            torch.nn.Module, torch.nn.optim, torch.: The neural network modules
        """

        sparsePPGnn = SparsePPGnn(self.cfg, model, optimizer, scheduler)

        for i in range(0, self.epochs):

            logger.info('Training on Epoch {}'.format(i))

            for iter, (time_series, ground_truth, window_label) in enumerate(dataloader):

                #

                time_series = time_series.to(self.device)
                ground_truth = ground_truth.to(self.device)

                # Running the algorithm
                freq_bins, X, Z_est, = sparsePPGnn.run(
                    time_series, ground_truth, window_label, mode='train')
                ###
            scheduler.step()

            if i % self.eval_period == 0:
                HR_nn, HR_gt, mSNR = MMSE_Denoiser_Trainer.test_partition(self, model, optimizer, scheduler, test_dataloader, i)
                metrics = self._compute_rmse_and_pte6(HR_gt, HR_nn)
                rmse, mae, pte6, mape, rho = metrics['rmse'], metrics['mae'], metrics['pte6'], metrics['mape'], metrics['rho']
                logger.warning('MMSE-HR results Epoch {}: MAE =  {}; RMSE = {}; PTE6 = {}; MAPE: {}; Pearson: {}; mSNR: {}'.format( i,mae, rmse, pte6, mape, rho, mSNR))
                #mlflow.log_metrics({'mae': mae, 'rmse': rmse, 'pte6': pte6}, step=i)


        model = sparsePPGnn.model
        optimizer = sparsePPGnn.optimizer
        scheduler = sparsePPGnn.scheduler

        return model, optimizer, scheduler

    def train(self, experiment_id):
        """The main training loop for the partition trainer

        Args:
            experiment_id (_type_): The MLFlow experiment ID under which to list training runs
        """
        with open('torch_SparsePPG/data/vid_102.json', 'r') as f:
            data = json.load(f)

        subject_list = list(data.keys())
        subject_list.reverse()
        HR_nn_full, HR_gt_full = [], []
        kf = KFold(len(subject_list))
        # Generates a partition of the data
        for idx, (train, test) in enumerate(kf.split(subject_list)):
            
            
            # Generating the one-versus-all partition of subjects for MMSE-HR
            test_subject = subject_list[test[0]]
            train_subject = [subject_list[i] for i in train]

            #if test_subject != 'F017':# or test_subject != 'F023' or test_subject != 'F009':
            #    continue
            logger.info(
                'Partition {} --- Train {} ; Test {}'.format(idx, train_subject, test_subject))

            #self.subject_cfg = self.get_subject_cfg(test_subject) #Get subject specific config for LR, etc.
            
            # Build dataset
            train_dataset = MMSE_HR_Dataset(self.cfg, test_subject)
            test_dataset = MMSE_HR_Dataset_Test(
                self.cfg, train_dataset.gt_test_subject, train_dataset.ts_test_subject)

            # Build dataloader
            train_dataloader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True)
            test_dataloader = DataLoader(
                test_dataset, batch_size=1, shuffle=False)

            # Build model, optimizer, lr_scheduler
            model = build_model(self.cfg)
            model = model.to(self.device)
            optimizer = build_optimizer(self.cfg, model)
            lr_scheduler = build_lr_scheduler(self.cfg, optimizer)

            # Create experiment and log training parameters
            run_name = '{}'.format(test_subject)
            mlflow.start_run(experiment_id=experiment_id,
                            run_name=run_name, nested=True)
            torch.cuda.reset_peak_memory_stats(device=self.device)
            self.log_config_dict(self.cfg)

            start_training = torch.cuda.Event(enable_timing=True)
            stop_training = torch.cuda.Event(enable_timing=True)
            # Train the model
            start_training.record()
            model, optimizer, lr_scheduler = self.train_partition(
                model, optimizer, lr_scheduler, train_dataloader, test_dataloader)
            stop_training.record()
            torch.cuda.synchronize()
            logger.warning(
                'Finished training denoiser for held-out test subject {}; now testing on subject {}'.format(test_subject, test_subject))

            # Test the model
            start_inference = torch.cuda.Event(enable_timing=True)
            stop_inference = torch.cuda.Event(enable_timing=True)
            start_inference.record()
            HR_nn, HR_gt, mSNR = self.test_partition(self, 
                model, optimizer, lr_scheduler, test_dataloader, self.cfg.DENOISER.EPOCHS)
            stop_inference.record()
            torch.cuda.synchronize()
            max_memory = torch.cuda.max_memory_allocated(device=self.device)
            HR_nn_full = HR_nn_full + HR_nn
            HR_gt_full = HR_gt_full + HR_gt
            metrics = self._compute_rmse_and_pte6(HR_gt, HR_nn)
            rmse, mae, pte6, mape, rho = metrics['rmse'], metrics['mae'], metrics['pte6'], metrics['mape'], metrics['rho']
            logger.warning('MMSE-HR Results: MAE =  {}; RMSE = {}; PTE6 = {}; MAPE: {}; Pearson: {}; mSNR: {}'.format( mae, rmse, pte6, mape, rho, mSNR))
            metrics['inference_time'] = start_inference.elapsed_time(stop_inference)
            metrics['mSNR'], metrics['max_memory'] = mSNR, max_memory
            mlflow.log_metrics(metrics, step=self.epochs)

            # Save the Model
            out_dir = os.path.join(self.cfg.OUTPUT.OUTPUT_DIR, test_subject)
            os.makedirs(out_dir, exist_ok=True)
            model_name = 'model{}_.pth'.format(test_subject)
            
            out_path = os.path.join(out_dir, model_name)
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, out_path)
            mlflow.log_artifacts(out_dir)
            
            # End the run
            mlflow.end_run()

        metrics = self._compute_rmse_and_pte6(HR_gt_full, HR_nn_full)
        rmse, mae, pte6, mape, rho = metrics['rmse'], metrics['mae'], metrics['pte6'], metrics['mape'], metrics['rho']
        logger.warning('MMSE-HR Results: MAE =  {}; RMSE = {}; PTE6 = {}; MAPE: {}; Pearson: {}; mSNR: {}'.format( mae, rmse, pte6, mape, rho, mSNR))
        mlflow.log_metrics(metrics, step=self.epochs)

    def get_subject_cfg(self, test_subject):
        """Gets the subject specific configuration in case something needs to be trained individually

        Args:
            test_subject (str): The name of the subject to obtain

        Returns:
            CfgNode: Configuration node in the YACS format
        """
        
        cfg = get_cfg_defaults()  # Get the defaults from torch_SparsePPG/config/config.py
        # overwrite default configs args with those from file
        config_file = os.path.join('config/{}_cfg.yaml'.format(test_subject))
        cfg.merge_from_file(config_file)
        # overwrite config args with those from command line
        
        cfg.freeze()

        return cfg
