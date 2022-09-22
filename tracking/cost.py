import numpy as np
from shapely.geometry import Polygon
import torch
from torch import Tensor

def iou_2d(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """Computes 2D intersection over union of two sets of bounding boxes

    Args:
        bboxes1: bounding box set of shape [M, 5], each row corresponding to x, y, l, w, yaw of the bounding box
        bboxes2: bounding box set of shape [N, 5], each row corresponding to x, y, l, w, yaw of the bounding box
    Returns:
        iou_mat: matrix of shape [M, N], where iou_mat[i, j] is the 2D IoU value between bboxes[i] and bboxes[j].
        You should use the Polygon class from the shapely package to compute the area of intersection/union.
    """
    M, N = bboxes1.shape[0], bboxes2.shape[0]
    # TODO: Replace this stub code.
    iou_mat = np.zeros((M, N))

    for i in range(M):
        for j in range(N):
            x, y, l, w, yaw = bboxes1[i]
            c, s = np.cos(yaw), np.sin(yaw)
            rot_matrix = np.array(((c, -s), (s, c)))
            points = np.array([[-l/2, w/2], [l/2, w/2], [l/2, -w/2], [-l/2, -w/2]])
            points_rotated = points @ rot_matrix.T
            points_rotated = points_rotated + [x, y]

            b_1_i = Polygon(points_rotated)

            x, y, l, w, yaw = bboxes2[j]
            c, s = np.cos(yaw), np.sin(yaw)
            rot_matrix = np.array(((c, -s), (s, c)))
            points = np.array([[-l/2, w/2], [l/2, w/2], [l/2, -w/2], [-l/2, -w/2]])
            points_rotated = points @ rot_matrix.T
            points_rotated = points_rotated + [x, y]

            b_2_j = Polygon(points_rotated)

            if b_1_i.union(b_2_j).area != 0:
                iou = b_1_i.intersection(b_2_j).area / b_1_i.union(b_2_j).area
            else:
                iou = 0
            
            iou_mat[i, j] = iou


    return iou_mat


def geometry_cost(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    M, N = bboxes1.shape[0], bboxes2.shape[0]
    # TODO: Replace this stub code.
    cost_mat = torch.empty(M, N)

    for i in range(M):
        for j in range(N):
            bb1 = bboxes1[i]
            bb2 = bboxes2[j]
            diff_square = (bb1 - bb2) ** 2
            centroid_diff = diff_square[:2].sum().sqrt()
            size_diff = diff_square[2:4].sum().sqrt()
            yaw_diff = diff_square[4].sqrt()
            
            cost_mat[i, j] = centroid_diff + size_diff + yaw_diff
    print(cost_mat)
    return cost_mat.numpy()
