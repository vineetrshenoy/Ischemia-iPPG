import numpy as np
import torch
import matplotlib.pyplot as plt



def append_waveforms(waveform_list):
    """Appends 10-second waveforms together

    Args:
        waveform_list (list[torch.Tensor]): The list of 10-second waveforms

    Returns:
        torch.Tensor: The complete waveform for a certain subject and task
    """
    
    
    waveform_cat = torch.cat(waveform_list, dim=2)

    return waveform_cat

def separate_by_task(nn_waveform, gt_waveform, window_label):
    """Accepts a list of 10-second waveforms, and concatenates them based
    on subject and task 

    Args:
        nn_waveform (list[torch.Tensor]): The 10-second waveform for a subject; not arranged by task
        gt_waveform (list[torch.Tensor]): The 10-second waveform for a subject; not arranged by task
        window_label (list[str]): The labels for each window

    Returns:
        lists: The waveforms organized by task. Label corresponds to the task
    """
    nn_append, gt_append = [], []
    N = len(window_label)
    prev_task = ''
    nn_taskwaveforms, gt_taskwaveforms, label = [], [], []
    
    for i in range(0, N): #Iterate over the windows
    
        lab = window_label[i]
        subject, task, win_num = get_window_parts(lab) #Need to separate str to get subject, task, window number
        
        if prev_task != '': #If not the first window

            if task != prev_task: #If we have switched to a different task
                
                assert len(nn_append) > 0
                #Append all waveforms for this task
                nn_full = append_waveforms(nn_append)
                gt_full = append_waveforms(gt_append)
                
                #Add task waveform to list of tasks
                nn_taskwaveforms.append(nn_full)
                gt_taskwaveforms.append(gt_full)
                label.append(subject + '_' + prev_task)
                
                nn_append, gt_append = [], []
                
        #
        prev_task = task
        nn_append.append(nn_waveform[i]), gt_append.append(gt_waveform[i]) #append 10-second window to task
      
    #Loop ended before concatenating the last tasks; do so now  
    nn_full = append_waveforms(nn_append)
    gt_full = append_waveforms(gt_append)
    nn_taskwaveforms.append(nn_full)
    gt_taskwaveforms.append(gt_full)
    label.append(subject + '_' + task)

    return nn_taskwaveforms, gt_taskwaveforms, label

def separate_by_task_ubfc_rppg(nn_waveform, gt_waveform, window_label):
    """Accepts a list of 10-second waveforms, and concatenates them based
    on subject and task 

    Args:
        nn_waveform (list[torch.Tensor]): The 10-second waveform for a subject; not arranged by task
        gt_waveform (list[torch.Tensor]): The 10-second waveform for a subject; not arranged by task
        window_label (list[str]): The labels for each window

    Returns:
        lists: The waveforms organized by task. Label corresponds to the task
    """
    nn_append, gt_append = [], []
    N = len(window_label)
    prev_subject= ''
    nn_taskwaveforms, gt_taskwaveforms, label = [], [], []
    
    for i in range(0, N): #Iterate over the windows
    
        lab = window_label[i]
        subject, task, win_num = get_window_parts(lab) #Need to separate str to get subject, task, window number
        
        if prev_subject!= '': #If not the first window

            if subject!= prev_subject: #If we have switched to a different subject
                
                assert len(nn_append) > 0
                #Append all waveforms for this task
                nn_full = append_waveforms(nn_append)
                gt_full = append_waveforms(gt_append)
                
                #Add task waveform to list of tasks
                nn_taskwaveforms.append(nn_full)
                gt_taskwaveforms.append(gt_full)
                label.append(prev_subject + '_' + 'T1')
                
                nn_append, gt_append = [], []
                
        #
        prev_subject= subject
        nn_append.append(nn_waveform[i]), gt_append.append(gt_waveform[i]) #append 10-second window to task
      
    #Loop ended before concatenating the last tasks; do so now  
    nn_append.append(nn_waveform[i]), gt_append.append(gt_waveform[i]), label.append(subject + '_' + 'T1')
    nn_full = append_waveforms(nn_append)
    gt_full = append_waveforms(gt_append)
    nn_taskwaveforms.append(nn_full)
    gt_taskwaveforms.append(gt_full)
    

    assert len(nn_taskwaveforms) == len(gt_taskwaveforms) == len(label)
    
    return nn_taskwaveforms, gt_taskwaveforms, label

def separate_by_task_pure(nn_waveform, gt_waveform, window_label):
    """Accepts a list of 10-second waveforms, and concatenates them based
    on subject and task 

    Args:
        nn_waveform (list[torch.Tensor]): The 10-second waveform for a subject; not arranged by task
        gt_waveform (list[torch.Tensor]): The 10-second waveform for a subject; not arranged by task
        window_label (list[str]): The labels for each window

    Returns:
        lists: The waveforms organized by task. Label corresponds to the task
    """
    nn_append, gt_append = [], []
    N = len(window_label)
    prev_task = ''
    nn_taskwaveforms, gt_taskwaveforms, label = [], [], []
    
    for i in range(0, N): #Iterate over the windows
    
        lab = window_label[i]
        subject, task, win_num = get_window_parts(lab) #Need to separate str to get subject, task, window number
        
        if prev_task != '': #If not the first window

            if prev_task!= task: #If we have switched to a different subject
                
                assert len(nn_append) > 0
                #Append all waveforms for this task
                nn_full = append_waveforms(nn_append)
                gt_full = append_waveforms(gt_append)
                
                #Add task waveform to list of tasks
                nn_taskwaveforms.append(nn_full)
                gt_taskwaveforms.append(gt_full)
                label.append(subject + '_' + task)
                
                nn_append, gt_append = [], []
                
        #
        prev_task= task
        nn_append.append(nn_waveform[i]), gt_append.append(gt_waveform[i]) #append 10-second window to task
      
    #Loop ended before concatenating the last tasks; do so now  
    nn_append.append(nn_waveform[i]), gt_append.append(gt_waveform[i]), label.append(subject + '_' + task)
    nn_full = append_waveforms(nn_append)
    gt_full = append_waveforms(gt_append)
    nn_taskwaveforms.append(nn_full)
    gt_taskwaveforms.append(gt_full)

    assert len(nn_taskwaveforms) == len(gt_taskwaveforms) == len(label)

    return nn_taskwaveforms, gt_taskwaveforms, label

def get_window_parts(window_label):
    """Separate a window of style 'F017_T10_win5' into 
    its separate parts

    Args:
        window_label (str): Str such as 'F017_T10_win5'

    Returns:
        str: The subject, task, and window number
    """
    
    parts = window_label.split('_')
    subject = parts[0]
    task = parts[1]
    num = parts[2]

    win_num = int(num.split('win')[-1])

    return subject, task, win_num

def _frequency_plot_grid(fps, ppg):
    """Gets the frequency buckets for the ground-truth 

    Args:
        fps (int): The frame rate
        ppg (numpy.ndarray): The GT-window in the time domain

    Returns:
        numpy.ndarray: The Fourier buckets and Power spectrum
    """
    ppg = ppg.numpy().squeeze()
    time_length = ppg.shape[0]
    L = time_length * 100
    freq = fps * np.arange(L/2) * (1 / L)
    hann_win = np.hanning(time_length)
    Y = np.fft.fft(ppg, n=L)
    P = np.abs(Y / L)**2

    P_one_sided = P[0:L//2]
    P_one_sided[1:-1] = 2*P_one_sided[1:-1]

    return freq, P_one_sided


def _evaluate_hr(wave, fps):
    """Generates the ground-truth heart-rate in the window

    Args:
        gt_wave (torch.Tensor): GT wave of shape [batch_size, 1, window_length]
        fps (int): The frame rate

    Returns:
        float: The heart-rate in the window
    """

    # AC/DC normalization
    
    windowed_pulse = wave / torch.linalg.norm(wave, dim=2, keepdim=True)

    freq_GT, Pk_GT = _frequency_plot_grid(
        fps, windowed_pulse.cpu())
    
    lower_bound = np.greater_equal(freq_GT, 0.6)
    upper_bound = np.less_equal(freq_GT, 2.5)
    freq_idx = np.where(np.logical_and(lower_bound, upper_bound))[0]
    
    Pk_GT = Pk_GT[freq_idx]
    limit_freq_GT = freq_GT[freq_idx]
    #freq_bins, P1 = freq_bins[freq_of_interest], P1[freq_of_interest]
    max_idx = np.argmax(np.abs(Pk_GT))

    HR_GT_in_window = limit_freq_GT[max_idx]*60

    return HR_GT_in_window.item()


def _evaluate_prediction(freq_bins, X, window_label):
    """Determines heart-rate from result

    Args:
        freq_bins (torch.Tensor): The equally-spaced bins
        X (torch.Tensor): The Fourier coefficients


    Returns:
        float: The HR estimate in the time window
    """
    lower_limit = torch.where(freq_bins < 0.6)[0][-1].item()
    upper_limit = torch.where(freq_bins > 2.5)[0][0].item()
    freq_bins = freq_bins[lower_limit:upper_limit]
    # remove batch dimension and transpose
    X = torch.squeeze(X).T
    X = X[lower_limit:upper_limit, :]
    X_sum = torch.sum(torch.pow(torch.abs(X), 2), 1)
    max_idx = torch.argmax(X_sum)
    HR_estimated = freq_bins[max_idx] * 60

    
    return HR_estimated.item()