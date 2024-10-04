import torch
import torch
from jaxtyping import Float
from torch import Tensor
from nerfstudio.model_components.losses import ScaleAndShiftInvariantLoss

def silog_loss(predicted_depth, depth_gt, depth_mask, epsilon=1e-6):
    """
    Computes the Scale-Invariant Logarithmic (SILog) Loss between predicted and ground truth depth.
    
    Args:
        predicted_depth (torch.Tensor): The predicted depth map (B, H, W).
        depth_gt (torch.Tensor): The ground truth depth map (B, H, W).
        depth_mask (torch.Tensor): The binary mask indicating valid depth pixels (B, H, W).
        epsilon (float): A small constant to avoid log(0).
        
    Returns:
        torch.Tensor: The computed SILog loss.
    """
    # Apply the depth mask to select valid pixels
    valid_predicted_depth = predicted_depth[depth_mask]
    valid_depth_gt = depth_gt[depth_mask]

    # Ensure valid depth values (positive) by clamping small values
    valid_predicted_depth = torch.clamp(valid_predicted_depth, min=epsilon)
    valid_depth_gt = torch.clamp(valid_depth_gt, min=epsilon)

    # Compute the logarithmic difference
    log_diff = torch.log(valid_predicted_depth) - torch.log(valid_depth_gt)

    # Scale-Invariant Logarithmic Loss
    silog_loss_value = torch.mean(log_diff ** 2) - (torch.mean(log_diff) ** 2)

    return silog_loss_value

def berhu_loss(predicted_depth, depth_gt, depth_mask, c=0.2):
    """
    Computes the BerHu (Reverse Huber) Loss between predicted and ground truth depth.
    
    Args:
        predicted_depth (torch.Tensor): The predicted depth map (B, H, W).
        depth_gt (torch.Tensor): The ground truth depth map (B, H, W).
        depth_mask (torch.Tensor): The binary mask indicating valid depth pixels (B, H, W).
        c (float): Threshold parameter that controls the transition from L1 to L2 loss.
        
    Returns:
        torch.Tensor: The computed BerHu loss.
    """
    # Apply the depth mask
    valid_predicted_depth = predicted_depth[depth_mask]
    valid_depth_gt = depth_gt[depth_mask]

    # Calculate absolute difference between predicted and ground truth depth
    diff = torch.abs(valid_predicted_depth - valid_depth_gt)

    # Find the threshold for BerHu (c * max absolute difference)
    threshold = c * torch.max(diff)

    # Apply BerHu loss formula
    l1_part = diff[diff <= threshold]
    l2_part = diff[diff > threshold]

    # L1 loss for small differences, L2 loss for large differences
    loss = torch.cat([l1_part, (l2_part ** 2 + threshold ** 2) / (2 * threshold)])

    return loss.mean()



def monosci_depth_loss(predicted_depth, depth_gt, depth_mask):
    scale_shift_invariant_loss_fn = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1, reduction_type="batch")
    ssi_loss = scale_shift_invariant_loss_fn(predicted_depth, depth_gt, depth_mask)
    return ssi_loss