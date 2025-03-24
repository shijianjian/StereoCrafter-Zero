import torch
import torch.nn.functional as F
import kornia


def compute_uncertainty_mask(depth_map, threshold=0.1):
    """
    Compute an uncertainty mask for the depth map based on the variance or noise in the depth values.
    Areas with high variance are considered uncertain.

    Args:
        depth_map: Initial depth map of shape (B, N, H, W).
        threshold: Threshold to define uncertainty. Areas with a high depth gradient will be considered uncertain.
    
    Returns:
        uncertainty_mask: A binary mask (1 = uncertain, 0 = confident) indicating areas of uncertainty.
    """
    # Compute the gradient of the depth map to identify areas with high variance/noise
    depth_grad_x = F.pad(depth_map[:, :, :, :-1] - depth_map[:, :, :, 1:], (0, 1), "constant", 0)
    depth_grad_y = F.pad(depth_map[:, :, :-1, :] - depth_map[:, :, 1:, :], (0, 0, 0, 1), "constant", 0)
    
    # Compute the magnitude of the gradient
    depth_grad = torch.sqrt(depth_grad_x ** 2 + depth_grad_y ** 2)
    
    # Threshold the gradient to create an uncertainty mask
    uncertainty_mask = (depth_grad > threshold).float()

    return uncertainty_mask


def bilateral_filter_depth(depth_map, uncertainty_mask, kernel_size=5, sigma_color=50.0, sigma_space=50.0):
    """
    Apply a bilateral filter to the depth map to smooth areas of uncertainty while preserving edges.
    
    Args:
        depth_map: Initial depth map of shape (B, N, H, W).
        uncertainty_mask: A binary mask indicating uncertain regions (1 = uncertain, 0 = confident).
        kernel_size: Size of the bilateral filter kernel.
        sigma_color: Standard deviation for color-space filtering.
        sigma_space: Standard deviation for spatial filtering.
    
    Returns:
        refined_depth_map: The refined depth map after bilateral filtering.
    """
    # Apply bilateral filtering to areas of uncertainty
    depth_map_smoothed = kornia.filters.gaussian_blur2d(depth_map, (kernel_size, kernel_size), sigma=(sigma_color, sigma_space))
    
    # Combine the refined areas with the original confident regions
    refined_depth_map = depth_map * (1 - uncertainty_mask) + depth_map_smoothed * uncertainty_mask

    return refined_depth_map


def refine_depth(depth_map, threshold=0.1, kernel_size=5, sigma_color=50.0, sigma_space=50.0):
    """
    Refine the depth map by identifying uncertain areas and applying bilateral filtering.
    
    Args:
        depth_map: Initial depth map of shape (B, N, H, W).
        threshold: Threshold for defining uncertainty (higher values mean stricter filtering).
        kernel_size: Size of the bilateral filter kernel.
        sigma_color: Standard deviation for color-space filtering in bilateral filter.
        sigma_space: Standard deviation for spatial filtering in bilateral filter.
    
    Returns:
        refined_depth_map: The refined depth map.
    """
    # Step 1: Compute the uncertainty mask based on depth gradients
    uncertainty_mask = compute_uncertainty_mask(depth_map, threshold=threshold)
    
    # Step 2: Apply bilateral filtering to smooth uncertain areas while preserving edges
    refined_depth_map = bilateral_filter_depth(depth_map, uncertainty_mask, kernel_size=kernel_size, sigma_color=sigma_color, sigma_space=sigma_space)
    
    return refined_depth_map
    