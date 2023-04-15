# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch, in other words, vertically stack the predictions of every image in the batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets]) # (number_of_boxes_in_batch)
        tgt_bbox = torch.cat([v["boxes"] for v in targets]) # (number_of_boxes_in_batch, 4)

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        # the indexing just remove columns/predictions of incorrect classes, possibly duplicate columns of predictions of correct classes
        # since the cost function only cares how much weight/probability the model give to the correct classes
        cost_class = -out_prob[:, tgt_ids] # [batch_size*num_queries, number_of_boxes_in_batch], 

        # Compute the L1 cost between boxes
        # basically compute the p-norm with p=2 distance between every target box's vector to every predicted boxes
        # its a vector distance metric
        # output shape is (batch_size*queries, num_boxes_in_batch)
        # each row corresponding to the distance of this box prediction to all the target box
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        # this also has shape (batch_size*num_queries, number_of_boxes_in_batch)
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        # came back to the shape of (batch_size, num_queries, number_of_boxes_in_batch)
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        # C is of shape (batch_size, num_queries, number_of_boxes_in_batch) but not every number in C has a meaning, 
        # only number at the right indices has meaning. For example, the first image's cost matrix: (0, num_queries, num_classes)
        # because the columns of the cost matrix corresponding to the true class labels of the batch, let's say 
        # in the batch you have 10 boxes with 10 class labels (labels can be of the same class for different boxes) then you'll 
        # have 10 columns. Now each of the image in the batch has a different number of boxes, let's say our batch length = 3, 
        # first image has 2 boxes, second image has 5 boxes, last image has 3 boxes, then, for the first image, only costs at indices 
        # C[0, :, :2] are the cost of predictions of the image corresponding to the 2-boxes ground true which are the columns
        # similarly, in the second image, only costs at indices C[1, :, 2:5] has meaning.
        # the number of rows of the costs matrix always divisible to num_queries, because each image has a fixed number of queries/predictions for
        # the boxes
        
        # the sizes is a list of number_of_boxes per image
        # we use this to cut the C matrix column wise
        sizes = [len(v["boxes"]) for v in targets]
        # indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = []
        for i, c in enumerate(C.split(sizes, -1)): # cut the matrix column wise (the dimension of targets)
            indices.append(linear_sum_assignment(c[i])) # only choose the C[image_index, :, num_boxes_in_this_image] costs as the cost matrix for this image
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
