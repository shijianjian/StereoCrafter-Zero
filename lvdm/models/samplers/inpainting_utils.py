
from __future__ import annotations
import torch
from scipy import spatial
import torch.nn.functional as F


class Inpainting:
    """
    Inpaints a tensor using Dirichlet interpolation.
    Works on tensors of shape (C, H, W) and a binary mask of shape (1, H, W).

    Key params:
        ps:	Patch size, it is used for creating the non-local graph.
            The default value is 7. To gain speed-up try with 3 or 5. Ideally it should be an odd value.
            For images with low spatial frequency texture, should be kept high like 11, 13 or 15 ...
        k_boundary:
            To determine the nodes at the intersection of A and dA. The default is 4.
            To gain speed-up try changing to 8 or 16.
        k_search:
            Determines the region for searching the non-local neighbors of a node. The default is 1000.
            For large size images it should be increased. To gain speed-up try with 300, 400, 500.
        k_patch:
            The KNN value for the non-local graph construction. The default is 5. Try 3 for speed-up.
            Try larger value to increase the resolution.

    Modified from: https://github.com/aGIToz/PyInpaint/tree/main
    """

    def __init__(self, ps):
        """
        ps: patch size (used for creating a dynamic non-local graph)
        """
        self.ps = ps

    def preprocess(self, latents, mask, guidance=None):
        """
        Preprocess the latents and mask to set up for inpainting.
        latents: input tensor of shape (C, H, W)
        mask: binary mask tensor of shape (1, H, W)
        """
        # Ensure latents and mask have the correct shape and type
        latents = latents.float()
        mask = mask.float()

        # Apply mask to latents (masked regions will be zero)
        latents = latents * mask

        # Store shape
        self._shape = latents.shape

        # Use guidance if provided, otherwise the latent itself is used for inpainting
        if guidance is not None:
            self.guidance = guidance.float() * (1 - mask)  # Only use guidance where the mask is zero
        else:
            self.guidance = latents.clone()  # Default to using latent features as guidance

        # Generate position feature matrix
        self._position = pmat(self._shape)

        # Flatten the latents tensor into a texture (spatial dimensions only)
        self._texture = latents.view(latents.size(0), -1).T  # Shape (H*W, C)

        # Create patches from the image/tensor
        self._patches = create_patches(latents, (self.ps, self.ps))

    def postprocess(self, fmat):
        """
        Reshape the flattened feature map back to the original latent dimensions.
        """
        return fmat.T.view(self._shape)

    def forward(self, latents, mask, guidance=None, guidance_weight=0.5, k_boundary=4, k_search=1000, k_patch=5):
        """
        Inpainting process to fill masked areas in the tensor.
        """
        self.preprocess(latents, mask, guidance)

        kdt = spatial.cKDTree(self._position.numpy())
        dA = torch.where(self._texture.any(dim=1))[0]
        A = torch.where(~self._texture.any(dim=1))[0]

        while A.size(0) >= 1:
            dmA = torch.empty(0, device=A.device, dtype=torch.long)

            for i in A:
                _, indices = kdt.query(self._position[i].numpy(), k_boundary)
                if (~torch.isin(torch.tensor(indices, device=A.device), A)).any():
                    dmA = torch.cat([dmA, i.unsqueeze(0)])
                    mask = (~(self._patches[i].flatten() == 0)).float()
                    _, indices = kdt.query(self._position[i].numpy(), k_search)
                    indices = torch.tensor(indices, device=A.device)
                    part_of_dA = indices[~torch.isin(indices, A)]
                    new_patches = mask.flatten() * self._patches[part_of_dA]
                    kdt_ = spatial.cKDTree(new_patches.cpu().numpy())
                    _, indices = kdt_.query(self._patches[i].flatten().cpu().numpy(), k_patch)
                    indices = torch.tensor(indices, device=A.device)
                    ids = part_of_dA[indices]

                    if guidance is not None:
                        # Blend the texture from guidance with the surrounding patches
                        self._texture[i] = guidance_weight * self.guidance.view(self._shape[0], -1)[:, i] + \
                                        (1 - guidance_weight) * self._texture[ids].mean(dim=0)
                    else:
                        self._texture[i] = self._texture[ids].mean(dim=0)

            self._patches = create_patches(self._texture.reshape(self._shape), (self.ps, self.ps))
            dA = torch.cat([dA, dmA])
            A = A[~torch.isin(A, dmA)]

        return self.postprocess(self._texture)


# Utility functions for pmat and create_patches

def pmat(shape):
    """
    Returns the position feature matrix for a given tensor shape.
    Assumes the shape is (C, H, W) for tensors.
    """
    h, w = shape[1], shape[2]
    
    # Create meshgrid in PyTorch
    x = torch.arange(0, w).float()
    y = torch.arange(h, 0, -1).float()

    meshx, meshy = torch.meshgrid(x, y, indexing='xy')

    # Flatten the positions
    x = meshx.reshape(-1, 1)
    y = meshy.reshape(-1, 1)

    # Concatenate x and y positions, normalize by max dimension
    pmat = torch.cat((x, y), dim=1) / max(h, w)

    return pmat


def create_patches(img, patch_shape=(3, 3)):
    """
    Creates overlapping patches from the input tensor.
    img: Tensor of shape (C, H, W) or (H, W)
    patch_shape: Tuple indicating patch size (height, width).
    """
    if img.dim() == 2:  # Handle grayscale or 2D image (no channels)
        img = img.unsqueeze(0)  # Convert to (C=1, H, W)

    d, h, w = img.shape
    r, c = patch_shape

    # Padding

    pad_h = (int((r - 0.5) / 2.), int((r + 0.5) / 2.))
    pad_w = (int((c - 0.5) / 2.), int((c + 0.5) / 2.))

    img = F.pad(img, pad=(pad_w[0], pad_w[1], pad_h[0], pad_h[1]), mode='reflect')

    # Unfold the image into patches
    patches = img.permute(1, 2, 0).unfold(0, r, 1).unfold(1, c, 1)
    # Reshape to (num_patches, patch_size)
    patches = patches.contiguous().view(h * w, r * c * d)

    return patches
