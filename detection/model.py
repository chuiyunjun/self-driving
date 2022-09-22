from dataclasses import dataclass, field
from threading import local
from matplotlib.pyplot import grid

import torch
from torch import Tensor, nn

from detection.modules.loss_function import DetectionLossConfig
from detection.modules.residual_block import ResidualBlock
from detection.modules.voxelizer import VoxelizerConfig
from detection.types import Detections


@dataclass
class DetectionModelConfig:
    """Detection model configuration."""

    voxelizer: VoxelizerConfig = field(
        default_factory=lambda: VoxelizerConfig(
            x_range=(-76.0, 76.0),
            y_range=(-50.0, 50.0),
            z_range=(0.0, 10.0),
            step=0.25,
        )
    )
    loss: DetectionLossConfig = field(
        default_factory=lambda: DetectionLossConfig(
            heatmap_loss_weight=100.0,
            offset_loss_weight=10.0,
            size_loss_weight=1.0,
            heading_loss_weight=100.0,
            heatmap_threshold=0.01,
            heatmap_norm_scale=20.0,
            loss_func='mse',
        )
    )


class DetectionModel(nn.Module):
    """A basic object detection model."""

    def __init__(self, config: DetectionModelConfig) -> None:
        super(DetectionModel, self).__init__()

        D, _, _ = config.voxelizer.bev_size
        self._backbone = nn.Sequential(
            nn.Conv2d(D, 32, 3, 1, 1),  # 1x
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 2, 1),  # 2x
            ResidualBlock(64),
            nn.Conv2d(64, 128, 3, 2, 1),  # 4x
            ResidualBlock(128),
            nn.Conv2d(128, 256, 3, 2, 1),  # 8x
            ResidualBlock(256),
            nn.Conv2d(256, 512, 3, 2, 1),  # 16x
            ResidualBlock(512),
            ResidualBlock(512),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(512, 256, 3, 1, 1),  # 8x
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(256, 256, 3, 1, 1),  # 4x
        )

        self._head = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 7, 3, 1, 1),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),  # 1x
        )

    def forward(self, x: Tensor) -> Tensor:
        """Perform a forward pass of the model's neural network.

        Args:
            x: A [batch_size x D x H x W] tensor, representing the input LiDAR
                point cloud in a bird's eye view voxel representation.

        Returns:
            A [batch_size x 7 x H x W] tensor, representing the dense detection outputs.
                The 7 channels are (heatmap, offset_x, offset_y, x_size, y_size, sin_theta, cos_theta).
        """
        return self._head(self._backbone(x))

    @torch.no_grad()
    def inference(
        self, bev_lidar: Tensor, k: int = 100, score_threshold: float = 0.05
    ) -> Detections:
        """Predict a set of 2D bounding box detections from the given LiDAR point cloud.

        To predict a set of 2D bounding box detections, we use the following algorithm:
        1. Run the model's neural network to produce a [7 x H x W] prediction tensor.
            The 7 channels at each pixel (i, j) in the [H x W] BEV image are
            (heatmap, offset_x, offset_y, x_size, y_size, sin_theta, cos_theta).
        2. Find the coordinates of the local maximums in the predicted heatmap and keep the top K.
            We define the value at pixel (i, j) to be a local maximum if it is the maximum
            in a [5 x 5] window centered on (i, j). This gives us a [K x 2] tensor of
            coordinates in the [H x W] BEV image, where each row represents a detection.
            Each detection's score is given by its corresponding heatmap value.
        3. For each of the K detections, compute its centers by adding its predicted
            (offset_x, offset_y) to the detection's coordinates (i, j) from step 2.
            For example, if a detection has coordinates (100, 100) and its predicted
            offsets are (0.1, 0.2), then its center is (100.1, 100.2).
        4. For each of the K detections, set its bounding box size equal to the
            (x_size, y_size) values predicted at coordinates (i, j).
        5. For each of the K detections, set its heading equal to atan2(sin_theta, cos_theta),
            where (sin_theta, cos_theta) are the values predicted at coordinates (i, j).
        6. Remove all detections with a score less than or equal to `score_threshold`.

        Args:
            bev_lidar: A [D x H x W] tensor containing the bird's eye view voxel
                representation for one LiDAR point cloud. Note that batch inference
                is not supported!
            k: The maximum number of detections to keep; defaults to 100.
            score_threshold: Keep only detections with a score exceeding `score_threshold`.
                Defaults to 0.05.

        Returns:
            A set of 2D bounding box detections.
        """

        # 1. 
        Y = self.forward(bev_lidar.unsqueeze(0))[0]

        # 2.
        area = 25 
        pred_heatmap, offset_x, offset_y, x_size, y_size, sin_theta, cos_theta = Y
        H, W = pred_heatmap.shape
        pred_heatmaps = torch.zeros(area, H + 4, W + 4)
        for n in range(area):
            i = n // 5
            j = n % 5
            # n-th heatmap is [i: -(4-i), j: -(4-j)]
            pred_heatmaps[n] = torch.nn.functional.pad(pred_heatmap, (j ,4-j, i, 4-i), mode='constant', value=-float('Inf'))

        # delete padding
        pred_heatmaps = pred_heatmaps[:, 2: -2, 2: -2]
        max_ix = torch.argmax(pred_heatmaps, dim=0).to('cuda')
        mask = (max_ix == 12).to('cuda')
        W_coords, H_coords = torch.arange(W).to('cuda'), torch.arange(H).to('cuda')
        H_grid_coords, W_grid_coords = torch.meshgrid(H_coords, W_coords, indexing="ij")
        grid_coords = torch.stack([H_grid_coords, W_grid_coords], dim=-1).to('cuda')  # [H x W x 2]
        local_maxi_ix = grid_coords.masked_select(torch.stack([mask, mask], dim=-1)).to('cuda')
        local_maxi_ix = local_maxi_ix.reshape(len(local_maxi_ix) // 2, 2).to('cuda')
        local_maxi = pred_heatmap[local_maxi_ix[ :, 0], local_maxi_ix[ :, 1]].to('cuda')
        if len(local_maxi) < k:
            k = len(local_maxi)
        topk_value, topk_ix = torch.topk(local_maxi, k)
        topk_ix = local_maxi_ix[topk_ix, :].to('cuda')


        # 3.
        offset_xy = torch.stack([offset_x[topk_ix[:, 0], topk_ix[:, 1]], offset_y[topk_ix[:, 0], topk_ix[:, 1]]], dim=-1)
        filpped_topk_ix = topk_ix.flip([-1])
        centroids = filpped_topk_ix + offset_xy
        
        # 4.
        boxes = torch.stack([x_size[topk_ix[:, 0], topk_ix[:, 1]], y_size[topk_ix[:, 0], topk_ix[:, 1]]], dim=-1)
        
        # 5
        yaws = torch.atan2(sin_theta[topk_ix[:, 0], topk_ix[:, 1]], cos_theta[topk_ix[:, 0], topk_ix[:, 1]])
            
        # 6.
        indices = (topk_value > score_threshold).nonzero().flatten()
        scores = topk_value[indices]
        centroids = centroids[indices, :]
        boxes = boxes[indices, :]
        yaws = yaws[indices]

        
        return Detections(centroids=centroids, yaws=yaws, boxes=boxes, scores=scores)

