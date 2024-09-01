import mlflow
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
import torch.nn.functional as F
import os
def plot_window_gt(fps, gt_sig, filename):
    """Plots the ground-truth signal. For debugging purposes only

    Args:
        signal (np.ndarray): The ground-truth signal. Shape is (1, self.SLIDING_WINDOW_LENGTH)
        filename (str): Name of output file
    """
    Fs = 60
    L = 100*len(gt_sig)
    f_gt, pxx_gt= scipy.signal.periodogram(gt_sig, fs=Fs, nfft=L, detrend=False)
    f_gt, f_gt = f_gt * 60, f_gt * 60
    lim = np.where(f_gt < 200)[0]  # only get beats under 200
    lim_idx = lim[-1]
    
    abs_pxx_gt = np.abs(pxx_gt)
    peak_freq_idx = np.argmax(abs_pxx_gt)
    peak_freq = f_gt[peak_freq_idx]
    N = len(gt_sig)
    idx = np.arange(0, N)
    
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(idx, gt_sig)
    ax[0].set_xlabel('samples')
    ax[0].set_title('Time Domain')
    
    
    ax[1].plot(f_gt[0:lim_idx], abs_pxx_gt[0:lim_idx])
    ax[1].set_xlabel('beats per minute')
    ax[1].set_title('Frequency Domain')
    fig.suptitle('{}: Peak Freq {}'.format(filename, peak_freq))
    fig.tight_layout()
    fig.savefig('temp.jpg')

def plot_30sec(fps, signal, gt_wave_window, filename, epoch):
    """Plots the ground-truth signal. For debugging purposes only

    Args:
        signal (np.ndarray): The ground-truth signal. Shape is (1, self.SLIDING_WINDOW_LENGTH)
        filename (str): Name of output file
    """
    signal, gt_wave_window = signal.cpu().detach().numpy(), gt_wave_window.cpu().detach().numpy()
    # Plotting the ground-truth
    batch_size, num_regions, length = signal.shape
    fig, ax = plt.subplots(2, 1)
    plt.subplots_adjust(hspace=0.5)
    signal = signal[0, 0, :]
    gt_wave_window = gt_wave_window[0, 0, :]
    idx = np.arange(length)

    L = 100*len(signal)
    Fs = fps  # samples per second
    fft_signal = (1/L)*scipy.fft.fft(signal, n=L)
    fft_signal = np.abs(fft_signal)
    fft_gt = (1/L)*scipy.fft.fft(gt_wave_window, n=L)
    fft_gt = np.abs(fft_gt)

    P1_signal = 2*fft_signal[0:(L//2)]
    P1_gt = 2*fft_gt[0:(L//2)]
    freq_bins = (Fs/L)*(np.arange(0, L//2)) * 60
    
    peak_freq_idx = np.argmax(P1_signal)
    peak_freq_idx_gt = np.argmax(P1_gt)
    peak_freq = freq_bins[peak_freq_idx]
    peak_freq_gt = freq_bins[peak_freq_idx_gt]

    lim = np.where(freq_bins < 200)[0]  # only get beats under 200
    lim_idx = lim[-1]

    ax[0].plot(idx, signal)
    ax[0].plot(idx, gt_wave_window)
    ax[0].set_title('Time Domain')
    ax[0].set_xlabel('time')
    ax[0].legend(['signal', 'gt'], prop={'size': 4})
    
    ax[1].plot(freq_bins[0:lim_idx], P1_signal[0:lim_idx])
    ax[1].plot(freq_bins[0:lim_idx], P1_gt[0:lim_idx])
    ax[1].set_title('Freq. Dom. Peak @ {} bpm'.format(peak_freq))
    ax[1].set_xlabel('Frequency (beats/min)')
    ax[1].legend(['signal', 'gt'], prop={'size': 4})
    
    fig.suptitle(filename + 'GT Peak @ {} bpm'.format(peak_freq_gt))
    filename = '{}_epoch{}.jpg'.format(filename, epoch)
    mlflow.log_figure(fig, filename)
    plt.close()


@staticmethod
def plot_test_results(fps, org_sig, win_num, epoch, gt_class, pred_class):
    """Plots the signal and spectrums. Used during test time


    Args:
        org_sig (np.ndarray): The original time-series signal
        proj_sig (np.ndarray): The signal after running the algorithm
        Z_gt (np.ndarray): The ground-truth signal
        filename (str): The output file name
    """
    org_sig = org_sig.detach().cpu().numpy()
    # Plotting the time series
    batch_size, length = org_sig.shape
    idx = np.arange(length)
    org_sig = org_sig[0, :]
    #out_dir = os.path.join('output', filename)
    #os.makedirs(out_dir, exist_ok=True)
    
    fig, ax = plt.subplots(2, 1)
    plt.subplots_adjust(hspace=0.75, wspace=0.5)

    #Get spectrum
    L = 100*len(org_sig)
    Fs = fps  # samples per second
    fft_org = (1/L)*scipy.fft.fft(org_sig, n=L)    
    fft_org = np.abs(fft_org)
    P1_org = 2 * fft_org[0:(L//2)]
    freq_bins = (Fs/L)*(np.arange(0, L//2)) * 60
    peak_freq_idx_org= np.argmax(P1_org)
    peak_freq_org = np.round_(freq_bins[peak_freq_idx_org], 3)
    

    lim = np.where(freq_bins < 200)[0]  # only get beats under 200
    lim_idx = lim[-1]

    ax[0].plot(idx, org_sig)
    ax[0].set_title('Time Domain: Original')
    ax[0].set_xlabel('samples')
    #ax[0].legend(['proj', 'gt'], prop={'size': 4})

    ax[1].plot(freq_bins[0:lim_idx], P1_org[0:lim_idx])
    ax[1].set_title(
        'Freq. Dom. Peak @ {} bpm'.format(peak_freq_org))
    ax[1].set_xlabel('Frequency (beats/min)')
    #ax[1].legend(['proj', 'gt'], prop={'size': 4})

        

    fig.suptitle('Name: {}; GT {}; Pred: {}'.format(win_num[0], gt_class, pred_class))
    fig.tight_layout()
    name = '_win{}_epoch{}: GT: {} Pred: {}.jpg'.format(win_num[0], epoch, gt_class, pred_class)
    mlflow.log_figure(fig, name)
    filename = '{}_gt_{}_pred_{}'.format(win_num[0], gt_class,pred_class)
    #fig.savefig(filename)
    plt.close()

    #logging the above artifacts for later plotting
    '''
    name = os.path.join(out_dir, filename + '_Region{}'.format(i))
    np.savez(name, org_sig_reg=org_sig_reg,
            proj_sig_reg=proj_sig_reg,
            Z_gt=Z_gt,
            freq_bins=freq_bins,
            P1_org=P1_org,
            P1_proj=P1_proj,
            P1_Z_gt=P1_Z_gt)
    mlflow.log_artifact(name + '.npz')
    '''

@staticmethod
def plot_window_ts(fps, org_sig, proj_sig, Z_gt, outloc, gt_label):
    """Plots the signal and spectrums. Used during test time


    Args:
        org_sig (np.ndarray): The original time-series signal
        proj_sig (np.ndarray): The signal after running the algorithm
        Z_gt (np.ndarray): The ground-truth signal
        filename (str): The output file name
    """
    gt = 'perfused'
    if gt_label[0,0] == 0:
        gt = 'ischemic'
    win_label = outloc.split('/')[-1]
    # Plotting the time series
    org_sig = org_sig.detach().cpu().numpy()
    proj_sig =  proj_sig.detach().cpu().numpy()
    Z_gt = Z_gt.detach().cpu().numpy()
    batch_size, num_regions, length = org_sig.shape
    idx = np.arange(length)
    org_sig, proj_sig, Z_gt = org_sig[0, :,
                                        :], proj_sig[0, :, :], Z_gt[0, 0, :]
    #out_dir = os.path.join('output', filename)
    #os.makedirs(out_dir, exist_ok=True)
    for i in range(0, 1):
        org_sig_reg, proj_sig_reg = org_sig[i, :], proj_sig[i, :]
        #mse_loss = F.mse_loss(torch.from_numpy(proj_sig_reg), torch.from_numpy(Z_gt))

        fig, ax = plt.subplots(2, 2)
        plt.subplots_adjust(hspace=0.75, wspace=0.5)

        L = 100*len(org_sig_reg)
        Fs = fps  # samples per second
        fft_org = (1/L)*scipy.fft.fft(org_sig_reg, n=L)
        fft_proj = (1/L)*scipy.fft.fft(proj_sig_reg, n=L)
        fft_Z_gt = (1/L)*scipy.fft.fft(Z_gt, n=L)
        
        fft_org, fft_proj, fft_Z_gt = np.abs(
            fft_org), np.abs(fft_proj), np.abs(fft_Z_gt)

        P1_org, P1_proj, P1_Z_gt = 2 * \
            fft_org[0:(L//2)], 2*fft_proj[0:(L//2)], 2*fft_Z_gt[0:(L//2)]
        freq_bins = (Fs/L)*(np.arange(0, L//2)) * 60

        peak_freq_idx_org, peak_freq_idx_proj, peak_fft_Z_gt = np.argmax(
            P1_org), np.argmax(P1_proj), np.argmax(P1_Z_gt)
        
        peak_freq_org = np.round_(freq_bins[peak_freq_idx_org], 3)
        peak_freq_proj = np.round_(freq_bins[peak_freq_idx_proj], 3)
        peak_freq_Z_gt = np.round_(freq_bins[peak_fft_Z_gt], 3)

        lim = np.where(freq_bins < 200)[0]  # only get beats under 200
        lim_idx = lim[-1]

        ax[0, 0].plot(idx, org_sig_reg)
        #ax[0, 0].plot(idx, Z_gt)
        ax[0, 0].set_title('Time Domain: Original')
        ax[0, 0].set_xlabel('time')
        ax[0, 0].legend(['pre'], prop={'size': 4})

        ax[0, 1].plot(freq_bins[0:lim_idx], P1_org[0:lim_idx])
        #ax[0, 1].plot(freq_bins[0:lim_idx], P1_Z_gt[0:lim_idx])
        ax[0, 1].set_title(
            'Freq. Dom. Peak @ {} bpm'.format(peak_freq_org))
        ax[0, 1].set_xlabel('Frequency (beats/min)')
        ax[0, 1].legend(['pre'], prop={'size': 4})

        ax[1, 0].plot(idx, proj_sig_reg)
        #ax[1, 0].plot(idx, Z_gt)
        ax[1, 0].set_title('Time Domain: Post-Algorithm')
        ax[1, 0].set_xlabel('time')
        ax[1, 0].legend(['post'], prop={'size': 4})

        ax[1, 1].plot(freq_bins[0:lim_idx], P1_proj[0:lim_idx])
        #ax[1, 1].plot(freq_bins[0:lim_idx], P1_Z_gt[0:lim_idx])
        ax[1, 1].set_title(
            'Freq. Dom. Peak @ {} bpm'.format(peak_freq_proj))
        ax[1, 1].set_xlabel('Frequency (beats/min)')
        ax[1, 1].legend(['post'], prop={'size': 4})

        fig.suptitle(' {}:{}'.format(win_label, gt))
        fig.tight_layout()
        #name = filename + '_region{}win{}_epoch{}.jpg'.format(i, win_num, epoch)
        #name = filename + '.jpg'.format(i, win_num, epoch)
        #mlflow.log_figure(fig, name)
        fig.savefig(outloc)
        plt.close()

        #logging the above artifacts for later plotting
        '''
        name = os.path.join(out_dir, filename + '_Region{}'.format(i))
        np.savez(name, org_sig_reg=org_sig_reg,
                proj_sig_reg=proj_sig_reg,
                Z_gt=Z_gt,
                freq_bins=freq_bins,
                P1_org=P1_org,
                P1_proj=P1_proj,
                P1_Z_gt=P1_Z_gt)
        mlflow.log_artifact(name + '.npz')
        '''

@staticmethod
def plot_window_post_algo(fps, org_sig, proj_sig, win_num, epoch, gt_class, pred_class):
    """Plots the signal and spectrums. Used during test time


    Args:
        org_sig (np.ndarray): The original time-series signal
        proj_sig (np.ndarray): The signal after running the algorithm
        Z_gt (np.ndarray): The ground-truth signal
        filename (str): The output file name
    """
    # Plotting the time series
    org_sig = org_sig.detach().cpu().numpy()
    proj_sig =  proj_sig.detach().cpu().numpy()
    batch_size, num_regions, length = org_sig.shape
    idx = np.arange(length)
    org_sig = org_sig[0, 0,:]
    #out_dir = os.path.join('output', filename)
    #os.makedirs(out_dir, exist_ok=True)
    ####Actually plotting
    org_sig_reg, proj_sig_reg = org_sig, proj_sig[0, 0,:]
    #mse_loss = F.mse_loss(torch.from_numpy(proj_sig_reg), torch.from_numpy(Z_gt))

    fig, ax = plt.subplots(2, 2)
    plt.subplots_adjust(hspace=0.75, wspace=0.5)

    L = 100*len(org_sig_reg)
    Fs = fps  # samples per second
    fft_org = (1/L)*scipy.fft.fft(org_sig_reg, n=L)
    fft_proj = (1/L)*scipy.fft.fft(proj_sig_reg, n=L)
    
    
    fft_org, fft_proj = np.abs(fft_org), np.abs(fft_proj)

    P1_org, P1_proj = 2 * \
        fft_org[0:(L//2)], 2*fft_proj[0:(L//2)]
    freq_bins = (Fs/L)*(np.arange(0, L//2)) * 60

    peak_freq_idx_org, peak_freq_idx_proj= np.argmax(
        P1_org), np.argmax(P1_proj)
    
    peak_freq_org = np.round_(freq_bins[peak_freq_idx_org], 3)
    peak_freq_proj = np.round_(freq_bins[peak_freq_idx_proj], 3)
    

    lim = np.where(freq_bins < 200)[0]  # only get beats under 200
    lim_idx = lim[-1]

    ax[0, 0].plot(idx, org_sig_reg)        
    ax[0, 0].set_title('Time Domain: Original')
    ax[0, 0].set_xlabel('time')
    ax[0, 0].legend(['pre'], prop={'size': 4})

    ax[0, 1].plot(freq_bins[0:lim_idx], P1_org[0:lim_idx])        
    ax[0, 1].set_title(
        'Freq. Dom. Peak @ {} bpm'.format(peak_freq_org))
    ax[0, 1].set_xlabel('Frequency (beats/min)')
    ax[0, 1].legend(['pre'], prop={'size': 4})

    ax[1, 0].plot(idx, proj_sig_reg)
    ax[1, 0].set_title('Time Domain: Post-Algorithm')
    ax[1, 0].set_xlabel('time')
    ax[1, 0].legend(['post'], prop={'size': 4})

    ax[1, 1].plot(freq_bins[0:lim_idx], P1_proj[0:lim_idx])
    ax[1, 1].set_title(
        'Freq. Dom. Peak @ {} bpm'.format(peak_freq_proj))
    ax[1, 1].set_xlabel('Frequency (beats/min)')
    ax[1, 1].legend(['post'], prop={'size': 4})

    
    
    fig.suptitle('Name: {}; GT {}; Pred: {}'.format(win_num[0], gt_class, pred_class))
    fig.tight_layout()
    #name = '_win{}_epoch{}: GT: {} Pred: {}.jpg'.format(win_num[0], epoch, gt_class, pred_class)
    #mlflow.log_figure(fig, name)
    filename = '{}_gt_{}_pred_{}.jpg'.format(win_num[0], gt_class,pred_class)
    mlflow.log_figure(fig, filename)
    #fig.savefig('temp.jpg')
    plt.close()

    #logging the above artifacts for later plotting
    '''
    name = os.path.join(out_dir, filename + '_Region{}'.format(i))
    np.savez(name, org_sig_reg=org_sig_reg,
            proj_sig_reg=proj_sig_reg,
            Z_gt=Z_gt,
            freq_bins=freq_bins,
            P1_org=P1_org,
            P1_proj=P1_proj,
            P1_Z_gt=P1_Z_gt)
    mlflow.log_artifact(name + '.npz')
    '''
    
@staticmethod
def plot_window_physnet(fps, gt_sig, proj_sig, win_num, epoch, gt_class, pred_class):
    """Plots the signal and spectrums. Used during test time


    Args:
        org_sig (np.ndarray): The original time-series signal
        proj_sig (np.ndarray): The signal after running the algorithm
        Z_gt (np.ndarray): The ground-truth signal
        filename (str): The output file name
    """
    # Plotting the time series
    gt_sig, proj_sig =  gt_sig.detach().cpu().numpy(), proj_sig
    batch_size, num_regions, length = proj_sig.shape
    idx = np.arange(length)
    
    #out_dir = os.path.join('output', filename)
    #os.makedirs(out_dir, exist_ok=True)
    ####Actually plotting
    gt_sig, proj_sig_reg = gt_sig[0, 0,:], proj_sig[0, 0,:]
    #mse_loss = F.mse_loss(torch.from_numpy(proj_sig_reg), torch.from_numpy(Z_gt))  
    gt_sig = gt_sig / np.linalg.norm(gt_sig, keepdims=True)
    proj_sig_reg = proj_sig_reg / np.linalg.norm(proj_sig_reg, keepdims=True)

    fig, ax = plt.subplots(2, 2)
    """Plots the ground-truth signal. For debugging purposes only

    Args:
        signal (np.ndarray): The ground-truth signal. Shape is (1, self.SLIDING_WINDOW_LENGTH)
        filename (str): Name of output file
    """
    Fs = fps
    L = 100*length
    f_pred, pxx_pred = scipy.signal.periodogram(proj_sig_reg, fs=Fs, nfft=L, detrend=False)
    f_gt, pxx_gt= scipy.signal.periodogram(gt_sig, fs=Fs, nfft=L, detrend=False)
    f_pred, f_gt = f_pred * 60, f_gt * 60
    
    lim = np.where(f_gt < 200)[0]  # only get beats under 200
    lim_idx = lim[-1]
    
    abs_pxx_pred, abs_pxx_gt = np.abs(pxx_pred), np.abs(pxx_gt)
    peak_freq_idx_pred, peak_freq_idx_gt = np.argmax(abs_pxx_pred), np.argmax(abs_pxx_gt)
    peak_freq_pred, peak_freq_gt = f_gt[peak_freq_idx_pred], f_gt[peak_freq_idx_gt]
    peak_freq_pred, peak_freq_gt = np.round(peak_freq_pred, 3), np.round(peak_freq_gt, 3)
   
    N = len(proj_sig_reg)
    idx = np.arange(0, N)
    
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(idx, proj_sig_reg)
    ax[0].plot(idx, gt_sig)
    ax[0].set_xlabel('samples')
    ax[0].set_title('Time Domain')
    ax[0].legend(['pred, gt'], prop={'size': 4})
    
    
    ax[1].plot(f_pred[0:lim_idx], abs_pxx_pred[0:lim_idx])
    ax[1].plot(f_gt[0:lim_idx], abs_pxx_gt[0:lim_idx])
    ax[1].set_xlabel('beats per minute')
    ax[1].set_title('Frequency Domain: Pred HR {}; GT HR {}'.format(peak_freq_pred, peak_freq_gt))
    ax[1].legend(['pred, gt'], prop={'size': 4})
        
    fig.suptitle('Name: {}; GT {}; Pred: {}'.format(win_num[0], gt_class, pred_class))
    fig.tight_layout()
    #name = '_win{}_epoch{}: GT: {} Pred: {}.jpg'.format(win_num[0], epoch, gt_class, pred_class)
    #mlflow.log_figure(fig, name)
    filename = '{}_gt_{}_pred_{}.jpg'.format(win_num[0], gt_class,pred_class)
    mlflow.log_figure(fig, filename)
    #fig.savefig('temp.jpg')
    plt.close()

    #logging the above artifacts for later plotting
    '''
    name = os.path.join(out_dir, filename + '_Region{}'.format(i))
    np.savez(name, org_sig_reg=org_sig_reg,
            proj_sig_reg=proj_sig_reg,
            Z_gt=Z_gt,
            freq_bins=freq_bins,
            P1_org=P1_org,
            P1_proj=P1_proj,
            P1_Z_gt=P1_Z_gt)
    mlflow.log_artifact(name + '.npz')
    '''