from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.distributions import MultivariateNormal
from torch import Tensor, nn


def compute_l1_loss(targets: Tensor, predictions: Tensor) -> Tensor:
    """Compute the mean absolute error (MAE)/L1 loss between `predictions` and `targets`.

    Specifically, the l1-weighted MSE loss can be computed as follows:
    1. Compute a binary mask of the `targets` that are not NaN, and apply it to the `targets` and `predictions`
    2. Compute the MAE loss between `predictions` and `targets`.
        This should give us a [batch_size * num_actors x T x 2] tensor `l1_loss`.
    3. Compute the mean of `l1_loss`. This gives us our final scalar loss.

    Args:
        targets: A [batch_size * num_actors x T x 2] tensor, containing the ground truth targets.
        predictions: A [batch_size * num_actors x T x 2] tensor, containing the predictions.

    Returns:
        A scalar MAE loss between `predictions` and `targets`
    """
    # TODO: Implement.
    mask = torch.any(targets.isnan(), dim=2)
    targets_filtered = targets[~mask]
    predictions_filtered = predictions[~mask]
    
    loss = nn.L1Loss()
    return loss(predictions_filtered, targets_filtered)

def compute_nll_loss(targets, predicted_means, predicted_scaletril):
    mask = torch.any(targets.isnan(), dim=2)
    targets_filtered = targets[~mask]
    predicted_means_filtered = predicted_means[~mask]
    predicted_scaletril_filtered = predicted_scaletril[~mask]

    dist = MultivariateNormal(predicted_means_filtered, scale_tril=predicted_scaletril_filtered)

    loss = - dist.log_prob(targets_filtered).mean()
    return loss


@dataclass
class PredictionLossConfig:
    """Prediction loss function configuration.

    Attributes:
        l1_loss_weight: The multiplicative weight of the L1 loss
    """

    l1_loss_weight: float
    nll_loss_weight: float

@dataclass
class PredictionLossMetadata:
    """Detailed breakdown of the Prediction loss."""

    total_loss: torch.Tensor
    l1_loss: torch.Tensor


class PredictionLossFunction(torch.nn.Module):
    """A loss function to train a Prediction model."""

    def __init__(self, config: PredictionLossConfig) -> None:
        super(PredictionLossFunction, self).__init__()
        self._l1_loss_weight = config.l1_loss_weight
        self._nll_loss_weight = config.nll_loss_weight

    def forward(
        self, predictions: Tuple[List[Tensor], List[Tensor]], targets: List[Tensor]
    ) -> Tuple[torch.Tensor, PredictionLossMetadata]:
        """Compute the loss between the predicted Predictions and target labels.

        Args:
            predictions: A list of batch_size x [num_actors x T x 2] tensor containing the outputs of
                `PredictionModel`.
            targets:  A list of batch_size x [num_actors x T x 2] tensor containing the ground truth output.

        Returns:
            The scalar tensor containing the weighted loss between `predictions` and `targets`.
        """
        mu_predictions_tensor, sigma_predictions_tensor = torch.cat(predictions[0]), torch.cat(predictions[1])
        targets_tensor = torch.cat(targets)

        # 1. Unpack the targets tensor.
        target_centroids = targets_tensor[..., :2]  # [batch_size * num_actors x T x 2]

        # 2. Unpack the predictions tensor.
        predicted_means = mu_predictions_tensor[
            ..., :2
        ]  # [batch_size * num_actors x T x 2]

        predicted_covariances = sigma_predictions_tensor[
            ..., :4
        ]  # [batch_size * num_actors x T x 4]

        # 3. Compute individual loss terms for l1
        #l1_loss = compute_l1_loss(target_centroids, predicted_centroids)
        nll_loss = compute_nll_loss(target_centroids, predicted_means, predicted_covariances)

        # 4. Aggregate losses using the configured weights.
        #total_loss = l1_loss * self._l1_loss_weight
        total_loss = nll_loss * self._nll_loss_weight

        loss_metadata = PredictionLossMetadata(total_loss, nll_loss)
        return total_loss, loss_metadata
