import torch
from torch.distributions import MultivariateNormal


def compute_LL(centroids: torch.Tensor, scale_tril: torch.Tensor, labels: torch.Tensor):
    valid = ~torch.isnan(labels).any(-1)
    labels = labels[valid]
    centroids = centroids[valid]
    scale_tril = scale_tril[valid]

    dist =  MultivariateNormal(centroids, scale_tril=scale_tril)
    return dist.log_prob(labels).mean().item()

def compute_ADE(centroids: torch.Tensor, labels: torch.Tensor) -> float:
    """Calculates the average displacement (L2) error between a set of centroids and a set of labels
    over all timesteps

    Args:
        centroids: N x T x 2 tensor of predicted centroids
        labels: N x T x 2 tensor of actual gt centroids
    """
    valid = ~torch.isnan(labels).any(-1)
    return (torch.norm(centroids - labels, dim=-1)[valid]).mean().item()


def compute_FDE(centroids: torch.Tensor, labels: torch.Tensor) -> float:
    """Calculates the final (last timestep) displacement error between a set of centroids and a set of labels

    Args:
        centroids: N x T x 2 tensor of predicted centroids
        labels: N x T x 2 tensor of actual gt centroids
    """
    final_centroids = centroids[:, -1]
    final_labels = labels[:, -1]
    valid = ~torch.isnan(final_labels).any(-1)
    return (torch.norm(final_centroids - final_labels, dim=-1)[valid]).mean().item()


def compute_per_frame_err(
    centroids: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Calculates the per_frame_error

    Args:
        centroids: N x T x 2 tensor of predicted centroids
        labels: N x T x 2 tensor of actual gt centroids
    """
    valid = ~torch.isnan(labels).any(-1).any(-1)
    return (torch.norm(centroids - labels, dim=-1)[valid]).mean(0)
