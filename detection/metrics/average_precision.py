from turtle import pos
from dataclasses import dataclass
from nis import match
from typing import List

import torch

from detection.metrics.types import EvaluationFrame


@dataclass
class PRCurve:
    """A precision/recall curve.

    Attributes:
        precision: [N] vector of precision values, where N is the total number of detections.
            The element at index n denotes the precision of the top n detections when ordered by
            decreasing detection scores.
        recall: [N] vector of recall values, where N is the total number of detections.
            The element at index n denotes the recall of the top n detections when ordered by
            decreasing detection scores.
    """

    precision: torch.Tensor
    recall: torch.Tensor


@dataclass
class AveragePrecisionMetric:
    """Stores average precision and its associate precision-recall curve."""

    ap: float
    pr_curve: PRCurve


def compute_precision_recall_curve(
    frames: List[EvaluationFrame], threshold: float
) -> PRCurve:
    """Compute a precision/recall curve over a batch of evaluation frames.

    The PR curve plots the trade-off between precision and recall when sweeping
    across different score thresholds for your detections. To compute precision
    and recall for a score threshold s_i, consider the set of detections with
    scores greater than or equal to s_i. A detection is a true positive if it
    matches a ground truth label; it is a false positive if it does not.

    With this, we define precision = TP / (TP + FP) and recall = TP / (TP + FN),
    where TP is the number of true positive detections, FP is the number of false
    positive detections, and FN is the number of false negative labels (i.e. the
    number of ground truth labels that did not match any detections). By varying
    the score threshold s_i over all detection scores, we have the PR curve.

    What does it mean for a detection to match a ground truth label? In this assignment, we use
    the following definition: A detection matches a ground truth label if: (1) the Euclidean
    distance between their centers is at most `threshold`; and (2) no higher scoring detection
    satisfies condition (1) with respect to the same label.

    Args:
        frames: A batch of evaluation frames, each containing a detection/label pair.
        threshold: Two bounding boxes match if their bird's eye view
            center-to-center distance is strictly less than `threshold`.

    Returns:
        A precision/recall curve.
    """
    # TODO: Replace this stub code.
    TP = torch.zeros(0)
    FN = torch.zeros(0)
    DS = torch.zeros(0)

    for frame in frames:
        detections = frame.detections.centroids
        labels = frame.labels.centroids

        TP_i = torch.zeros(len(detections))
        FN_i = torch.zeros(len(labels))

        for j in range(len(labels)):
            distances = torch.sqrt(torch.pow(detections - labels[j], 2).sum(dim=1))
            mask = distances < threshold

            if len(distances[mask] > 0):
                TP_i[distances == torch.min(distances[mask])] = 1
            else:
                FN_i[j] = 1
        
        TP = torch.cat((TP, TP_i))
        FN = torch.cat((FN, FN_i))
        DS = torch.cat((DS, frame.detections.scores[TP_i == 1]))
            
            
    TP = TP[torch.sort(DS, descending=True)[1]]
    
    precision = torch.zeros(len(TP))
    recall = torch.zeros(len(TP))

    tp_n = 0
    fp_n = 0
    fn_n = torch.sum(FN).item()
    for n in range(len(TP)):
        if TP[n] == 1:
            tp_n += 1
        else:
            fp_n += 1
        
        precision[n] = tp_n / (tp_n + fp_n)
        recall[n] = tp_n / (tp_n + fn_n)
        

    return PRCurve(precision, recall)


    all_detections_binary = torch.tensor([])
    labels_count = 0
    all_scores = torch.tensor([])
    for _, frame in enumerate(frames):
        batch_detections = frame.detections.centroids
        batch_labels = frame.labels.centroids
        batch_scores = frame.detections.scores
        d = batch_detections.shape[0]
        l = batch_labels.shape[0]
        labels_count += l

        # in binary array, 0 means unmatched and 1 means matched
        detections_match_binary = torch.zeros(d)
        labels_match_binary = torch.zeros(l)
        
        # in each frame, sort dectections with scores (high score ~ low score)
        sorted_scores, indices = torch.sort(batch_scores, descending=True)
        sorted_detections = batch_detections[indices, :]

        # distance_table[j][i] =  i-th detection ~ j-th label
        distance_table = (sorted_detections.reshape(1, d, 2) - batch_labels.reshape(l, 1, 2)).norm(dim=-1)

        # in distance_table, if distance > threshold, then set the entry as 0
        mask = distance_table <= threshold
        distance_table[~mask] = -1
        
        # find a label for i-th detection. If the label is found, mark 1 in the corresponding indexes in <detections_match_binary> and <labels_match_binary>
        # loop from detections with high scores to low
        for i in range(d):
            distances = distance_table[:, i].reshape(l)
            # If the label is matched, then set the element in distance array as 0
            labels_binary_mask = (labels_match_binary > 0)
            distances[labels_binary_mask] = -1
            # indexs of available labels
            positive_distances_indices = (distances >= 0).nonzero().flatten()
            # sorted indexs of available labels, from smallest distance to largest
            _, labels_indices = distances[positive_distances_indices].sort()
            if len(labels_indices):
                first_available_labels_ix = labels_indices[0]
                labels_match_binary[first_available_labels_ix] = 1
                detections_match_binary[i] = 1
        
        all_scores = torch.cat((all_scores, sorted_scores), dim=0)
        all_detections_binary = torch.cat((all_detections_binary, detections_match_binary), dim=0)

    detections_count = len(all_detections_binary)
    
    # sort detections with corresponding scores
    _, indices = all_scores.sort(descending=True)
    sorted_detections_binary = all_detections_binary[indices]
    
    # calculate precision and recall
    TP = sorted_detections_binary.cumsum(dim=0)
    precision = TP / torch.arange(1, detections_count + 1)
    recall = TP / labels_count

    return PRCurve(precision, recall)

def compute_area_under_curve(curve: PRCurve) -> float:
    """Return the area under the given curve.

    Given a `PRCurve` curve, this function computes the area under the curve as:
        AP = \sum_{i = 1}^{n} (r_i - r_{i - 1}) * p_i
    where r_i (resp. p_i) is the recall (resp. precision) of the top i detections,
    n is the total number of detections, and we set r_0 = 0.0. Intuitively, this
    is computing the integral of the step function defined by the PRCurve.

    Args:
        curve: The precision/recall curve.

    Returns:
        The area under the curve, as defined above.
    """
    # TODO: Replace this stub code.
    precision = curve.precision
    recall = curve.recall

    recall_shifted = torch.cat((torch.tensor([0]), recall))

    return torch.sum(precision * (recall - recall_shifted[:-1])).item()


def compute_average_precision(
    frames: List[EvaluationFrame], threshold: float
) -> AveragePrecisionMetric:
    """Compute average precision over a batch of evaluation frames.

    Args:
        frames: A batch of evaluation frames, each containing a detection/label pair.
        threshold: Two bounding boxes match if their bird's eye view
            center-to-center distance is strictly less than `threshold`.

    Returns:
        A dataclass consisting of a PRCurve and its average precision.
    """
    # TODO: Replace this stub code.
    curve = compute_precision_recall_curve(frames, threshold)
    ap = compute_area_under_curve(curve)

<<<<<<< HEAD
    return AveragePrecisionMetric(ap, curve)
=======
    return AveragePrecisionMetric(ap, curve)
>>>>>>> e87ac0dec918bcd57239ac328a607c30e0f6a459
