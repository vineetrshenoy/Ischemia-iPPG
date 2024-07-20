#(c) MERL 2024
import torch
import torch.nn as nn


class CorrelationLoss(nn.Module):
    
    def __init__(self):
        super(CorrelationLoss, self).__init__()

    
    def forward(self, Z_est, Z_gt):
        """Computes the correlation loss between the signal and ground-truth

        Args:
            Z_est (torch.Tensor): The reconstructed pulse wave
            Z_gt (torch.Tensor): The ground-truth signal

        Returns:
            torch.Tensor: The correlation loss
    """
        bs, region, sig_length = Z_est.shape
        numerator = (sig_length * torch.sum(Z_est * Z_gt)) - (torch.sum(Z_est, dim=2) * torch.sum(Z_gt, dim=2))
        denominator1 = (sig_length * torch.sum(Z_est * Z_est)) - (torch.sum(Z_est, dim=2) * torch.sum(Z_est, dim=2))
        denominator2 = (sig_length * torch.sum(Z_gt * Z_gt)) - (torch.sum(Z_gt, dim=2) * torch.sum(Z_gt, dim=2))
        denominator = denominator1 * denominator2
        
        corr_loss = 1 - numerator / torch.sqrt(denominator)
        corr_loss = torch.sum(corr_loss)
        
        return corr_loss
    
if __name__ == '__main__':
    
    Z_est = torch.randn(100, 5, 250)
    Z_gt = torch.randn(100, 5, 250)
    Z_est = Z_gt
    
    bs, region, sig_length = Z_est.shape
    numerator = (sig_length * torch.sum(Z_est * Z_gt)) - (torch.sum(Z_est, dim=2) * torch.sum(Z_gt, dim=2))
    denominator1 = (sig_length * torch.sum(Z_est * Z_est)) - (torch.sum(Z_est, dim=2) * torch.sum(Z_est, dim=2))
    denominator2 = (sig_length * torch.sum(Z_gt * Z_gt)) - (torch.sum(Z_gt, dim=2) * torch.sum(Z_gt, dim=2))
    denominator = denominator1 * denominator2
    
    corr_loss = 1 - numerator / torch.sqrt(denominator)
    corr_loss = torch.sum(corr_loss)
    x = 5