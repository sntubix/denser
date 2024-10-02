import torch
import torch.nn.functional as F
from torch.nn.functional import interpolate
from rich.console import Console

CONSOLE = Console()


def depth_loss(pred_depth, gt_depth, mask=None, lambda_smooth=0.1):
    """
    Compute depth loss for Gaussian splatting to ensure Gaussians are placed correctly.
    
    Args:
    pred_depth (torch.Tensor): Predicted depth map from the model.
    gt_depth (torch.Tensor): Ground-truth depth map.
    mask (torch.Tensor, optional): Mask indicating valid depth regions. Default is None.
    lambda_smooth (float, optional): Weight for the smoothness regularization term. Default is 0.1.
    
    Returns:
    torch.Tensor: Calculated depth loss.
    # """
    # CONSOLE.log(f"pred_depth.shape = {pred_depth.shape}")
    # CONSOLE.log(f"gt_depth.shape = {gt_depth.shape}")
    # Ensure pred_depth has 3 dimensions and gt_depth has 2 dimensions
    if pred_depth.dim() == 3:
        pred_depth = pred_depth.squeeze(-1)  # Remove the last dimension if it is 1

    if gt_depth.dim() == 3:
        gt_depth = gt_depth.squeeze(-1)  # Remove the last dimension if it is 1

    # Ensure pred_depth and gt_depth have the same shape
    if pred_depth.shape != gt_depth.shape:
        pred_depth = interpolate(pred_depth.unsqueeze(0).unsqueeze(0), size=gt_depth.shape[-2:], mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

    if pred_depth is None:
        # Return a large loss value if pred_depth is None
        return torch.tensor(1e6, device=gt_depth.device)
    
    
    assert pred_depth.shape == gt_depth.shape, "Predicted and ground-truth depth maps must have the same shape."

    # If a mask is provided, apply it to the ground-truth and predicted depths
    if mask is not None:
        pred_depth = pred_depth * mask.squeeze(-1)
        gt_depth = gt_depth * mask.squeeze(-1)

    # Compute Mean Squared Error (MSE) loss
    depth_loss = F.mse_loss(pred_depth, gt_depth)

    # Compute smoothness loss (optional)
    # dx_pred_depth = torch.abs(pred_depth[:, :, 1:] - pred_depth[:, :, :-1])
    # dy_pred_depth = torch.abs(pred_depth[:, 1:, :] - pred_depth[:, :-1, :])
    # smoothness_loss = dx_pred_depth.mean() + dy_pred_depth.mean()

    # Combine depth loss and smoothness loss
    total_loss = depth_loss + lambda_smooth #* smoothness_loss

    return total_loss
