import numpy as np
import os
import h5py
import torch
from torch.utils.data import Dataset
from scipy.fft import fft
import scipy.signal
#from scipy.signal import butter, filtfilt
import glob
from numpy.random import default_rng
from hand_ischemia.engine import plot_window_gt


class H5Dataset(Dataset):

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
        #self.bp_filt = firwin(numtaps=self.NUM_TAPS,
        #                      cutoff=cutoff, pass_zero='bandpass', fs=self.FPS)
        self.L = 10*self.SLIDING_WINDOW_LENGTH + 1
        self.ch = cfg.INPUT.CHANNEL
        self.b, self.a = scipy.signal.butter(self.NUM_TAPS, cutoff, btype='bandpass', fs=self.FPS)
        #with open(self.train_json_path, 'r') as f:
        #    self.ts_list = json.load(f)
        self.ts_time_windows, self.time_window_label = self._get_timeseries(self, data_dict)
        self.num_perfuse, self.num_ischemic = H5Dataset._count_class_numbers(self.ts_time_windows)
        
        #Debug only
        self.ts_time_windows = self.ts_time_windows[0:1]
        x = 5
    
    @staticmethod
    def _get_timeseries(self, ts_list):
        
        
        ts_time_window, time_window_label = [], []
        # Load the json file describing subjects and task

        for ts_filename, task_list in ts_list.items():  # Load the gt files
            subject = ts_filename
            #mat = sio.loadmat(ts_filename)
            
            for sub_video, data_files in task_list.items():
                vname = sub_video.split('/')[-2]
                subject = '{}-{}'.format(subject, vname)
            
                for key, value in data_files.items():
                    
                    h5_filepath = os.path.join(sub_video, key)
                    ts, label = H5Dataset.load_time_windows(self, h5_filepath, subject, value, key)

                    ts_time_window += ts
                    time_window_label += label

        return ts_time_window, time_window_label
    
    @staticmethod
    def load_time_windows(self, h5_filepath, subject, label, key):
        """Loads the time windows into an array. The time windows are filtered
        and converted to torch format

        Args:
            gt_file_list (list(str)): A list of filepaths
            ts_file_list (list(str)): A list of filepaths

        Returns:
            list: The list of file ts time windows, gt time windows, and window_labels
        """
        ts_time_windows, time_window_label = [], []
        #time_steps = time_series.shape[0]
        sliding_window_start, window_num = 0, 0
        sliding_window_end = sliding_window_start + self.SLIDING_WINDOW_LENGTH
        

        with h5py.File(h5_filepath, 'r') as f:
            
            data_length = np.min([f['imgs'].shape[0], f['bvp'].shape[0]])
            time_series = np.arange(0, data_length)
            
            while sliding_window_end <= data_length:

                #ppg_mat_window = time_series[sliding_window_start:sliding_window_end, :]

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
                #Z = Hand_Ischemia_Dataset._process_ppg_mat_window(
                #    self.b, self.a, ppg_mat_window)

                cls_value = torch.zeros((2,))
                #cls_value[0,1] = 0 if label == 0 else 0
                if label == 0:
                    cls_value[0] = 1
                else:
                    cls_value[1] = 1
                
                #signal = Z.repeat(5, 1)
                #assert signal.shape[0] == 5
                #ts_time_windows.append((signal, cls_value))

                window_label = '{}_{}_win{}'.format(
                    subject, key, window_num)
                
                time_window_tuple = (h5_filepath, sliding_window_start, sliding_window_end, cls_value, window_label)
                ts_time_windows.append(time_window_tuple)
                time_window_label.append(window_label)

                sliding_window_start = sliding_window_start + self.FRAME_STRIDE
                sliding_window_end = sliding_window_start + self.SLIDING_WINDOW_LENGTH
                window_num += 1

            return ts_time_windows, time_window_label     
    
    @staticmethod
    def normalize_filter_gt(self, signal, samp):
    
        sig = (signal - np.mean(signal, axis=0)) / (np.abs(np.mean(signal, axis=0)) + 1e-6) #AC-DC Normalization
        #plot_signal_window(signal, sig)
        normsig = sig / np.linalg.norm(sig, axis=0)
        #plot_signal_window(sig, normsig)


        N = signal.shape[0]
        win = scipy.signal.windows.hann(N)
        winsig = normsig * win
        #plot_signal_window(normsig, winsig)

        b_low, a_low   = scipy.signal.butter(5, 2.5, 'low', fs=samp) # was 2.5
        b_high, a_high = scipy.signal.butter(5, 0.7, 'high', fs=samp) # was 0.7
        filtsig = scipy.signal.filtfilt(b_low, a_low, normsig, axis=0, padtype='odd', padlen=3*(max(len(b_low),1)-1)) # len(a_low) = 1
        sig = scipy.signal.filtfilt(b_high, a_high, filtsig, axis=0, padtype='odd', padlen=3*(max(len(b_high),len(a_high))-1))

        #B, A = butter(5, [0.7, 2.5], 'bandpass', fs=samp)
        #sig = filtfilt(B, A, sig, axis=0)
        

        return sig
    
    @staticmethod
    def _count_class_numbers(ts_time_windows):
        N = len(ts_time_windows)
        perfuse, ischemic = 0, 0
        
        for i in range(0, N):
            class_labels = ts_time_windows[i][3]
            if class_labels[0] == 1:
                perfuse += 1
            elif class_labels[1] == 1:
                ischemic += 1
        
        return perfuse, ischemic
    
    def __len__(self):
        return len(self.ts_time_windows)

    def __getitem__(self, idx):
        ts_tuple = self.ts_time_windows[idx]
        window_label = self.time_window_label[idx]
        
        filename, idx_start, idx_end = ts_tuple[0], ts_tuple[1], ts_tuple[2]
        cls_label, window_label = ts_tuple[3], ts_tuple[4]

        with h5py.File(filename, 'r') as f:
            bvp = f['bvp'][idx_start:idx_end].astype('float32')
            bvp = H5Dataset.normalize_filter_gt(self, bvp, self.FPS)
            bvp = torch.from_numpy(bvp.copy())
            #plot_window_gt(self.FPS, bvp, 'temp')
            img_seq = f['imgs'][idx_start:idx_end]
            
        img_seq = np.transpose(img_seq, (3, 0, 1, 2)).astype('float32')
        img_seq = torch.from_numpy(img_seq.copy())
        return img_seq, bvp, cls_label, window_label


class H5DatasetTest(Dataset):

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
        self.FRAME_STRIDE = int(cfg.TIME_SCALE_PPG.FRAME_STRIDE_TEST * self.FPS)
        self.SLIDING_WINDOW_LENGTH = int(self.FPS * self.TIME_WINDOW_SEC)
        cutoff = [(self.PASSBAND_FREQ / 60), (self.CUTOFF_FREQ/60)]
        #self.bp_filt = firwin(numtaps=self.NUM_TAPS,
        #                      cutoff=cutoff, pass_zero='bandpass', fs=self.FPS)
        self.L = 10*self.SLIDING_WINDOW_LENGTH + 1
        self.ch = cfg.INPUT.CHANNEL
        self.b, self.a = scipy.signal.butter(self.NUM_TAPS, cutoff, btype='bandpass', fs=self.FPS)
        #with open(self.train_json_path, 'r') as f:
        #    self.ts_list = json.load(f)
        self.ts_time_windows, self.time_window_label = H5Dataset._get_timeseries(self, data_dict)
        self.num_perfuse, self.num_ischemic = H5Dataset._count_class_numbers(self.ts_time_windows)
        
        #Debug only
        #self.ts_time_windows = self.ts_time_windows[0:1]
        x = 5

    def __len__(self):
        return len(self.ts_time_windows)

    def __getitem__(self, idx):
        ts_tuple = self.ts_time_windows[idx]
        window_label = self.time_window_label[idx]
        
        filename, idx_start, idx_end = ts_tuple[0], ts_tuple[1], ts_tuple[2]
        cls_label, window_label = ts_tuple[3], ts_tuple[4]

        with h5py.File(filename, 'r') as f:
            bvp = f['bvp'][idx_start:idx_end].astype('float32')
            bvp = H5Dataset.normalize_filter_gt(self, bvp, self.FPS)
            bvp = torch.from_numpy(bvp.copy())
            #plot_window_gt(self.FPS, bvp, 'temp')
            img_seq = f['imgs'][idx_start:idx_end]
            
        img_seq = np.transpose(img_seq, (3, 0, 1, 2)).astype('float32')
        img_seq = torch.from_numpy(img_seq.copy())
        return img_seq, bvp, cls_label, window_label
if __name__ == "__main__":
    print('Hello World')
    train_list = ['/cis/net/io72a/data/vshenoy/durr_hand/contrast-w-gt-08-01/chi_034/finger3-distal.h5', '/cis/net/io72a/data/vshenoy/durr_hand/contrast-w-gt-08-01/chi_034/finger3-intermediate.h5']
    window_length = 30 * 10
    H5Dataset(train_list, window_length, 1)
    
    x = 5