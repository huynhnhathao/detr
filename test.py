import requests
from io import BytesIO
from models.matcher import HungarianMatcher

from PIL import Image, ImageDraw

from scipy.optimize import linear_sum_assignment

import numpy as np
import torch

def normalize_bbox(box: list[float], img_height: float, img_width: float) -> list[float]:
    """ assume the coordinate system where the top left cornor is 0, x axis to the right, y axis go down
    box: [x0, y0, x1, y1]: coordinates of two points: top left and bottom right of the box
    """
    assert len(box) == 4
    box[0], box[2] = box[0]/img_width, box[2]/img_width
    box[1], box[3] = box[1]/img_height, box[3]/img_height
    return box

def unnomalize_bbox(box: list[float], img_height: float, img_width: float) -> list[float]:
    """undo the normalized coordinate to pixel coordinate

    Args:
        box (list[float]): 
        img_height (float): 
        img_width (float): _description_

    Returns:
        list[float]: _description_
    """
    box[0], box[2] = box[0]*img_width, box[2]*img_width
    box[1], box[3] = box[1]*img_height, box[3]*img_height
    return box


def main():
    bytes = BytesIO(requests.get("https://upload.wikimedia.org/wikipedia/commons/4/4d/Cat_November_2010-1a.jpg").content)
    image = Image.open(bytes)

    # in the first query, the first class has logit = 0.9
    pred_logits = torch.rand((2, 5, 3))
    pred_logits[0][0][0] = 0.9
    pred_logits[1][1][0] = 0.9 # in the second image, the second class logit = 0.9


    # predicted box
    box = [250, 400, 1500, 2000]
    pred_boxes = torch.rand((2, 5, 4))
    # in the first image, the first bbox prediction has the closest coordinates to the label
    pred_boxes[0][0] = torch.tensor(normalize_bbox(box, image.size[1], image.size[0]))
    # in the second image, the fourth bbox prediction has the closest coordinates to the label
    pred_boxes[1][3] = torch.tensor(normalize_bbox(box, image.size[1], image.size[0]))
    
    preds = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}

    target_box = [300, 300, 1600, 2250]
    target0 = {"labels": torch.tensor([0]),
               "boxes": torch.tensor([normalize_bbox(target_box, image.size[1], image.size[0])]) }
    
    target1 = {"labels": torch.tensor([1]),
               "boxes": torch.tensor([normalize_bbox(target_box, image.size[1], image.size[0])])}
    matcher = HungarianMatcher()
    indices = matcher(preds, [target0, target1])
    
    print(indices)
    

if __name__ == "__main__":
    main()