import sys
import os
import torch
import torch.nn.functional as F


__all__ = ['SpeedUp', 'SlowDown']

class SpeedUp(torch.nn.Module):
    
    def __init__(self, cfg):
        
        super(SpeedUp, self).__init__()
        self.generator = torch.Generator()
        self.FPS = cfg.SPARSE_PPG.FPS
        self.TIME_WINDOW_SEC = cfg.SPARSE_PPG.TIME_WINDOW_SEC
        self.SLIDING_WINDOW_LENGTH = self.FPS * self.TIME_WINDOW_SEC
        self.MARGIN_AUG = cfg.SPARSE_PPG.MARGIN_AUG
    
    def forward(self, ts, gt, generator):
        """Applies the augmentation

        Args:
            ts (torch.Tensor): The empirical time-series waveform
            gt (torch.Tensor): The ground-truth waveform
            generator (_type_): _description_

        Returns:
            torch.Tensor: The time-series after augmentation
        """
        
        margin = torch.randn(1, generator=generator) * self.MARGIN_AUG
        window_length = int((0.8 - margin) * self.SLIDING_WINDOW_LENGTH)
        
        ts, gt = ts[:, 0:window_length], gt[:, 0:window_length]
        
        ts, gt = ts.unsqueeze(0), gt.unsqueeze(0) # Needed for interpolation
        
        ts = F.interpolate(ts, size=self.SLIDING_WINDOW_LENGTH).squeeze()
        gt = F.interpolate(gt, size=self.SLIDING_WINDOW_LENGTH).squeeze(dim=0)
        
          
        return ts, gt

class SlowDown(torch.nn.Module):
    
    def __init__(self, cfg):
            
        super(SlowDown, self).__init__()
        self.generator = torch.Generator()
        self.FPS = cfg.SPARSE_PPG.FPS
        self.TIME_WINDOW_SEC = cfg.SPARSE_PPG.TIME_WINDOW_SEC
        self.SLIDING_WINDOW_LENGTH = self.FPS * self.TIME_WINDOW_SEC
        self.MARGIN_AUG = cfg.SPARSE_PPG.MARGIN_AUG
    
    def get_window_parts(self, label):
        """Get the window subject, task, and number

        Args:
            label (str): The full window string

        Returns:
            str: The subject, task, and window number
        """
        parts = label.split('_')
        win_num = int(parts[2].strip('win'))
        
        subject, task = parts[0], parts[1]
        
        return  subject, task, win_num    
    
    def check_window(self, window_label, window_label_next):
        """Checks if the following window belongs to the same task;
        returns True if the following window belongs to the same task

        Args:
            window_label (str): The name of the window
            window_label_next (str): The name of the next window

        Returns:
           bool: Whether the window belongs to the same task
        """
        sub, task, num = self.get_window_parts(window_label)
        sub_next, task_next, num_next = self.get_window_parts(window_label_next)
        
        if sub != sub_next:
            return False
        if task != task_next:
            return False
        if (num + 1) != num_next:
            return False
        
        return True
    
    def forward(self, ts, gt, window_label, generator):
        """Applies the data augmentation

        Args:
            ts (torch.Tensor): The empirical time series
            gt (torch.Tensor): The ground-truth time series
            window_label (str): The label of the window
            generator (torch.Generator): A random number generator

        Returns:
            torch.Tensor: The time-series after augmentation
        """
        ts, ts_next = ts
        gt, gt_next = gt
        window_label, window_label_next = window_label
        
        check = self.check_window(window_label, window_label_next)
        if check == False:
            return ts, gt
        
        
        margin = torch.randn(1, generator=self.generator) * self.MARGIN_AUG
        window_length = int((1.2 + margin) * self.SLIDING_WINDOW_LENGTH)
        window_extra = window_length - self.SLIDING_WINDOW_LENGTH
        
        ts_extra, gt_extra = ts_next[:, 0:window_extra], gt_next[:, 0:window_extra]
        ts, gt = torch.cat((ts, ts_extra), dim=1), torch.cat((gt, gt_extra), dim=1)
         
        ts, gt = ts.unsqueeze(0), gt.unsqueeze(0) # Needed for interpolation
        
        ts = F.interpolate(ts, size=self.SLIDING_WINDOW_LENGTH).squeeze()
        gt = F.interpolate(gt, size=self.SLIDING_WINDOW_LENGTH).squeeze(dim=0)
        
        
        return ts, gt
    



if __name__ == '__main__':
    
    win_name = 'F006_T10_win0'
    
    parts = win_name.split('_')
    win_num = int(parts[2].strip('win'))
    
    print(win_num)