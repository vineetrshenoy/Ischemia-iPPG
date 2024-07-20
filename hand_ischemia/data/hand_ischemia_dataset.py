import sys
import os
import json
import logging
from pathlib import Path
import numpy as np
import scipy.fft
import scipy.io as sio
from scipy.signal import firwin, filtfilt
import torch
from torch.utils.data import Dataset
from torchvision.transforms import RandomApply
from .augmentations import SpeedUp, SlowDown

__all__ = ['Hand_Ischemia_Dataset', 'Hand_Ischemia_Dataset_Test']
logger = logging.getLogger(__name__)


class Hand_Ischemia_Dataset(Dataset):

    def __init__(self, cfg, data_dict):
        self.gt_datapath = cfg.INPUT.GT_FILEPATH
        self.ts_filepath = cfg.INPUT.TIME_SERIES_FILEPATH
        self.train_json_path = cfg.INPUT.TRAIN_JSON_PATH

        self.cfg = cfg
        self.PASSBAND_FREQ = cfg.TIME_SCALE_PPG.PASSBAND_FREQ
        self.CUTOFF_FREQ = cfg.TIME_SCALE_PPG.CUTOFF_FREQ
        self.NUM_TAPS = cfg.TIME_SCALE_PPG.NUM_TAPS
        self.TIME_WINDOW_SEC = cfg.TIME_SCALE_PPG.TIME_WINDOW_SEC
        self.FPS = cfg.TIME_SCALE_PPG.FPS
        self.FRAME_STRIDE = int(cfg.TIME_SCALE_PPG.FRAME_STRIDE * self.FPS)
        self.SLIDING_WINDOW_LENGTH = int(self.FPS * self.TIME_WINDOW_SEC)
        cutoff = [(self.PASSBAND_FREQ / 60), (self.CUTOFF_FREQ/60)]
        self.bp_filt = firwin(numtaps=self.NUM_TAPS,
                              cutoff=cutoff, pass_zero='bandpass', fs=self.FPS)
        self.L = 10*self.SLIDING_WINDOW_LENGTH + 1
        self.ch = cfg.INPUT.CHANNEL
        self.b, self.a = scipy.signal.butter(
            self.NUM_TAPS, cutoff, btype='bandpass', fs=self.FPS)
        #with open(self.train_json_path, 'r') as f:
        #    self.ts_list = json.load(f)
        self.ts_time_windows, self.time_window_label = self._get_timeseries(self,
            data_dict)
        self.num_perfuse, self.num_ischemic = Hand_Ischemia_Dataset._count_class_numbers(self.ts_time_windows)
        

        # Transformations
        #self.speed_up = SpeedUp(cfg)
        #self.slow_down = SlowDown(cfg)
        #self.generator = torch.Generator()
        #self.transforms = transforms
        
        ##### DEBUG Only
        #self.ts_time_windows, self.time_window_label = self.ts_time_windows[0:1], self.time_window_label[0:1]
        #self.gt_test_subject, self.ts_test_subject = self._gt_file_list, self._ts_file_list
        x = 5
        ##### DEBUG Only
        
    
    @staticmethod
    def _count_class_numbers(ts_time_windows):
        N = len(ts_time_windows)
        perfuse, ischemic = 0, 0
        
        for i in range(0, N):
            class_labels = ts_time_windows[i][1]
            if class_labels[0] == 1:
                perfuse += 1
            elif class_labels[1] == 1:
                ischemic += 1
        
        return perfuse, ischemic
                
        
        
        
    @staticmethod
    def _get_timeseries(self, ts_list):
        """Get the filepaths for the ground-truth and input time series and store
        in separate arrays. The gt_file_list[i] corresponds to ts_file_list[i]

        Args:
            gt_datapath (str): Data path to ground-truth files
            ts_filepath (str): Data path to time-series files
            test_subject (str): Subject to exclude from training set
        Returns:
            gt_file_list (list[str]): Fullpaths of ground-truth files
            ts_file_list (list[str]): Fullpaths of time-series files
        """
        
        ts_time_window, time_window_label = [], []
        # Load the json file describing subjects and task

        for ts_filename, task_list in ts_list.items():  # Load the gt files
            subject = ts_filename.split('/')[-2]
            mat = sio.loadmat(ts_filename)
            for key, value in task_list.items():

                ts, label = Hand_Ischemia_Dataset.load_time_windows(self, mat[key], subject, value, key)

                ts_time_window += ts
                time_window_label += label

        return ts_time_window, time_window_label

    @staticmethod
    def _process_ppg_mat_window(numerator, denominator, ppg_mat_window):
        """Filter the ppg mat window region-wise

        Args:
            numerator (np.ndarray): The numerator coefficients of the filter
            denominator (np.ndarray): The denominator coefficients of the filter
            ppg_mat_window (np.ndarray): The signal to filter. Of shape (self.SLIDING_WINDOW_LENGTH, 5)

        Returns:
            torch.Tensor: The filtered signal of shape (5, self.SLIDING_WINDOW_LENGTH)
        """
        eps = 1e-6
        m, n = ppg_mat_window.shape
        full_idx = torch.arange(n)

        filt_ppg_win = filtfilt(
            numerator, denominator, ppg_mat_window, axis=0)
        filt_ppg_win = torch.from_numpy(
            filt_ppg_win.copy())
        filt_ppg_win = filt_ppg_win.T

        filt_ppg_win = filt_ppg_win / torch.linalg.vector_norm(filt_ppg_win, dim=1, keepdim=True)

        return torch.squeeze(filt_ppg_win)

    @staticmethod
    def load_time_windows(self, time_series, subject, label, key):
        """Loads the time windows into an array. The time windows are filtered
        and converted to torch format

        Args:
            gt_file_list (list(str)): A list of filepaths
            ts_file_list (list(str)): A list of filepaths

        Returns:
            list: The list of file ts time windows, gt time windows, and window_labels
        """
        ts_time_windows, time_window_label = [], []
        time_steps = time_series.shape[0]
        sliding_window_start, window_num = 0, 0
        sliding_window_end = sliding_window_start + self.SLIDING_WINDOW_LENGTH

        while sliding_window_end <= time_steps:

            ppg_mat_window = time_series[sliding_window_start:sliding_window_end, :]

            # Debugging only
            '''
            Hand_Ischemia_Dataset.plot_window_gt(gt_wave_window.T, 'F006_T10_win0_BEFORE')
            Z_gt = Hand_Ischemia_Dataset._process_ppg_mat_window(self.bp_filt, gt_wave_window)
            Hand_Ischemia_Dataset.plot_window_gt(Z_gt.numpy(), 'F006_T10_win0_AFTER')

            Hand_Ischemia_Dataset.plot_window_ts(ppg_mat_window.T, 'F006_T10_win0_BEFORE')
            Z = Hand_Ischemia_Dataset._process_ppg_mat_window(self.bp_filt, ppg_mat_window)
            Hand_Ischemia_Dataset.plot_window_ts(Z.numpy(), 'F006_T10_win0_AFTER')
            '''
            ##########################
            Z = Hand_Ischemia_Dataset._process_ppg_mat_window(
                self.b, self.a, ppg_mat_window)

            cls_value = torch.zeros((2,))
            #cls_value[0,1] = 0 if label == 0 else 0
            if label == 0:
                cls_value[0] = 1
            else:
                cls_value[1] = 1
            
            signal = Z.repeat(5, 1)
            assert signal.shape[0] == 5
            ts_time_windows.append((signal, cls_value))

            window_label = '{}_{}_win{}'.format(
                subject, key, window_num)
            time_window_label.append(window_label)

            sliding_window_start = sliding_window_start + self.FRAME_STRIDE
            sliding_window_end = sliding_window_start + self.SLIDING_WINDOW_LENGTH
            window_num += 1

        return ts_time_windows, time_window_label

    @staticmethod
    def plot_window_gt(signal, filename):
        """Plots the ground-truth signal. For debugging purposes only

        Args:
            signal (np.ndarray): The ground-truth signal. Shape is (1, self.SLIDING_WINDOW_LENGTH)
            filename (str): Name of output file
        """
        num_regions, length = signal.shape
        fig, ax = plt.subplots(2, 1)
        plt.subplots_adjust(hspace=0.5)
        #signal = signal.numpy()
        signal = (signal - np.mean(signal))  # / (np.mean(signal) + 1e-6)
        idx = np.arange(length)

        L = 100*len(signal[0, :])
        Fs = 25  # samples per second
        fft_gt = (1/L)*scipy.fft.fft(signal[0, :], n=L)
        fft_gt = np.abs(fft_gt)

        P1 = np.square(2*fft_gt[0:(L//2)])
        freq_bins = (Fs/L)*(np.arange(0, L//2)) * 60
        freq_of_interest = np.where(np.logical_and(
            freq_bins >= 60, freq_bins <= 150))
        freq_bins, P1 = freq_bins[freq_of_interest], P1[freq_of_interest]

        val, freqidx = torch.topk(torch.from_numpy(P1), k=2)
        peak_freq_idx = freqidx[0]
        ratio = val[0] / val[1]
        if ratio > 0.95 and freq_bins[peak_freq_idx] < 65:
            peak_freq_idx = freqidx[1]

        #peak_freq_idx = np.argmax(P1)
        peak_freq = freq_bins[peak_freq_idx]

        lim = np.where(freq_bins < 200)[0]  # only get beats under 200
        lim_idx = lim[-1]

        ax[0].plot(idx, signal[0, :])
        ax[0].set_title('Time Domain')
        ax[0].set_xlabel('time')
        ax[1].plot(freq_bins[0:lim_idx], P1[0:lim_idx])
        ax[1].set_title('Freq. Dom. Peak @ {} bpm'.format(peak_freq))
        ax[1].set_xlabel('Frequency (beats/min)')
        fig.suptitle(filename + ' GroundTruth')
        fig.savefig('{}.jpg'.format(filename))
        plt.close()

    @staticmethod
    def plot_window_ts(signal, filename):
        """Plotting the time series. For debugging purposes only

        Args:
            signal (np.ndarray): The time-series signal. Of shape (5, self.SLIDING_WINDOW_LENGTH)
            filename (str): The name of the output file
        """
        num_regions, length = signal.shape
        idx = np.arange(length)
        #signal = signal.numpy()
        for i in range(0, num_regions):
            sig_reg = signal[i, :]
            # / (np.mean(signal) + 1e-6)
            sig_reg = (sig_reg - np.mean(sig_reg))

            fig, ax = plt.subplots(2, 1)
            plt.subplots_adjust(hspace=0.5)
            L = 100*len(sig_reg)
            Fs = 25  # samples per second
            fft_gt = (1/L)*scipy.fft.fft(sig_reg, n=L)
            fft_gt = np.abs(fft_gt)

            P1 = 2*fft_gt[0:(L//2)]
            freq_bins = (Fs/L)*(np.arange(0, L//2)) * 60
            peak_freq_idx = np.argmax(P1)
            peak_freq = freq_bins[peak_freq_idx]

            lim = np.where(freq_bins < 200)[0]  # only get beats under 200
            lim_idx = lim[-1]

            ax[0].plot(idx, sig_reg)
            ax[0].set_title('Time Domain')
            ax[0].set_xlabel('time')
            ax[1].plot(freq_bins[0:lim_idx], P1[0:lim_idx])
            ax[1].set_title('Freq. Dom. Peak @ {} bpm'.format(peak_freq))
            ax[1].set_xlabel('Frequency (beats/min)')
            ax[1].set_xlabel('Frequency (beats/min)')
            fig.suptitle(filename)
            name = filename + '_region{}.jpg'.format(i)
            fig.savefig(name)
            plt.close()

    def __len__(self):
        return len(self.ts_time_windows)

    def __getitem__(self, idx):

        ts, label = self.ts_time_windows[idx]
        ts_shape = ts.shape
        window_label = self.time_window_label[idx]
        
        #Transforms
        #p = torch.randn(1, generator=self.generator)
        #if p <= 0.5:
        #    return ts, gt, window_label
        #elif p > 0.5 and p <= 0.75:
        #    ts, gt = self.speed_up(ts, gt, self.generator)
        #else:
        #    #pass
        #    if idx+1 == self.__len__():
        #        return ts, gt, window_label
        #    
        #    ts_next = self.ts_time_windows[idx+1]
        #    gt_next = self.gt_time_windows[idx+1]
        #    window_label_next = self.time_window_label[idx+1]
        #    ts, gt, window_label_all = [ts, ts_next], [gt, gt_next], [window_label, window_label_next]
        #    ts, gt = self.slow_down(ts, gt, window_label_all, self.generator)
        #    
        #
        #assert ts.shape == ts_shape
        
        
        return ts, label, window_label


class Hand_Ischemia_Dataset_Test(Dataset):

    def __init__(self, cfg, data_dict):
        self.gt_datapath = cfg.INPUT.GT_FILEPATH
        self.ts_filepath = cfg.INPUT.TIME_SERIES_FILEPATH
        self.test_json_path = cfg.INPUT.TEST_JSON_PATH

        self.cfg = cfg
        self.PASSBAND_FREQ = cfg.TIME_SCALE_PPG.PASSBAND_FREQ
        self.CUTOFF_FREQ = cfg.TIME_SCALE_PPG.CUTOFF_FREQ
        self.NUM_TAPS = cfg.TIME_SCALE_PPG.NUM_TAPS
        self.TIME_WINDOW_SEC = cfg.TIME_SCALE_PPG.TIME_WINDOW_SEC
        self.FPS = cfg.TIME_SCALE_PPG.FPS
        self.MIN_WINDOW_SEC = 10
        self.SLIDING_WINDOW_LENGTH = int(self.FPS * self.TIME_WINDOW_SEC)
        self.FRAME_STRIDE = int(cfg.TIME_SCALE_PPG.FRAME_STRIDE * self.FPS)
        cutoff = [(self.PASSBAND_FREQ / 60), (self.CUTOFF_FREQ/60)]
        self.bp_filt = firwin(numtaps=self.NUM_TAPS,
                              cutoff=cutoff, pass_zero='bandpass', fs=self.FPS)
        self.L = 10*self.SLIDING_WINDOW_LENGTH + 1
        self.ch = cfg.INPUT.CHANNEL
        self.b, self.a = scipy.signal.butter(
            self.NUM_TAPS, cutoff, btype='bandpass', fs=self.FPS)
        #with open(self.test_json_path, 'r') as f:
        #    self.ts_list = json.load(f)
        self.ts_time_windows, self.time_window_label = Hand_Ischemia_Dataset._get_timeseries(self,
            data_dict)
        x = 5
        #self.ts_time_windows, self.time_window_label = self.ts_time_windows[0:1], self.time_window_label[0:1]
        self.num_perfuse, self.num_ischemic = Hand_Ischemia_Dataset._count_class_numbers(self.ts_time_windows)
        
        
    
    
    def __len__(self):
        return len(self.ts_time_windows)

    def __getitem__(self, idx):

        ts, label = self.ts_time_windows[idx]
        ts_shape = ts.shape
        window_label = self.time_window_label[idx]
        
        return ts, label, window_label


if __name__ == '__main__':

    import sys
    sys.path.append(
        '/projects/CV2/Biosignals/FY_2022/nonlinear_SparsePPG/torch_SparsePPG/config')
    from config import get_cfg_defaults
    import matplotlib.pyplot as plt

    test_subject = 'F006'
    cfg = get_cfg_defaults()

    transform = SpeedUp()
    train_dataset = Hand_Ischemia_Dataset(cfg, transform, test_subject)
    test_dataset = Hand_Ischemia_Dataset_Test(
        cfg, train_dataset.gt_test_subject, train_dataset.ts_test_subject)

    #data = test_dataset[0]
    #ts, gt, label = data

    #Hand_Ischemia_Dataset.plot_window_gt(gt.numpy(), label)
    #Hand_Ischemia_Dataset.plot_window_ts(ts.numpy(), label)

    for i in range(0, 5):
        ts, gt, label = train_dataset[i]
        Hand_Ischemia_Dataset.plot_window_gt(gt.numpy(), label)
        #Hand_Ischemia_Dataset.plot_window_ts(ts.numpy(), label)
