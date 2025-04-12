import torch
from einops import repeat
import torch.nn.functional as F


def calc_supervised_target_score(z, z_txt):
    """
    Calculates the supervised target score from ASAL.
    The returned score should be minimized, since we add a minus sign here.

    Parameters
    ----------
    z : torch.Tensor of shape (T, D)
        The latent representation of the images over time.
    z_txt : torch.Tensor of shape (T2, D)
        The latent representation of the text prompts over time.
    """
    T, T2 = z.shape[0], z_txt.shape[0]
    assert T % T2 == 0, "T must be divisible by T2"
    # Repeat the text representations to match the image count
    z_txt = repeat(z_txt, "T2 D -> (k T2) D", k=T // T2)
    
    # Compute the kernel similarity matrix
    kernel = z_txt @ z.T  # shape: (T, T)
    # Extract the diagonal elements and compute their mean
    diag_vals = torch.diag(kernel)
    return -diag_vals.mean()


def calc_supervised_target_softmax_score(z, z_txt, temperature_softmax=0.01):
    """
    Calculates the supervised target score from ASAL with softmax.
    This isn't part of the original ASAL, but it's a useful extension.
    The returned score should be minimized.

    Parameters
    ----------
    z : torch.Tensor of shape (T, D)
        The latent representation of the images over time.
    z_txt : torch.Tensor of shape (T2, D)
        The latent representation of the text prompts over time.
    temperature_softmax : float
        The temperature for the softmax function. For CLIP, leave it at 0.01, since that is default CLIP softmax temperature.
    """
    T, T2 = z.shape[0], z_txt.shape[0]
    assert T % T2 == 0, "T must be divisible by T2"
    z_txt = repeat(z_txt, "T2 D -> (k T2) D", k=T // T2)

    kernel = z_txt @ z.T  # shape: (T, T)
    # Apply softmax along the last dimension (axis=-1) and along the second last dimension (axis=-2)
    loss_sm1 = F.softmax(kernel / temperature_softmax, dim=-1)
    loss_sm2 = F.softmax(kernel / temperature_softmax, dim=-2)
    
    # Extract the diagonal elements and compute the negative log likelihood
    diag1 = torch.diag(loss_sm1)
    diag2 = torch.diag(loss_sm2)
    
    loss_sm1 = -torch.log(diag1)
    loss_sm2 = -torch.log(diag2)
    return (loss_sm1.mean() + loss_sm2.mean()) / 2.0


def calc_open_endedness_score(z):
    """
    Calculates the open-endedness score from ASAL.
    The returned score should be minimized.

    Parameters
    ----------
    z : torch.Tensor of shape (T, D)
        The latent representation of the images over time.
    """
    kernel = z @ z.T  # shape: (T, T)
    # Extract lower triangular part, excluding the diagonal
    tril_kernel = torch.tril(kernel, diagonal=-1)
    max_vals, _ = tril_kernel.max(dim=-1)
    return max_vals.mean()


def calc_illumination_score(zs):
    """
    Calculates the illumination score from ASAL.
    The returned score should be minimized.

    Parameters
    ----------
    zs : torch.Tensor of shape (N, D)
        The latent representation of the images from different simulation parameters.
    """
    N, D = zs.shape
    kernel = zs @ zs.T  # shape: (N, N)
    # Set the diagonal elements to -infinity so they are ignored in the max computation.
    mask = torch.eye(N, dtype=torch.bool, device=zs.device)
    kernel = torch.where(mask, float("-inf"), kernel)
    max_vals, _ = kernel.max(dim=-1)
    return max_vals.mean()
