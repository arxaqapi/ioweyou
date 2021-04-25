from typing import Dict, Tuple, List
# from collections import namedtuple

from ioweyou.CoordinatesHandler import CoordinatesFormat, CoordinatesValues

import sys

class BoundingBox:
    def __init__(
            self,
            x: float,
            y: float,
            w: float,
            h: float,
            confidence: float,
            coordinates_values: CoordinatesValues,
            coordinates_format: CoordinatesFormat,
            class_name: str = None,
            class_id: int = None
    ) -> None:
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.confidence = confidence
        self.coordinates_values = coordinates_values
        self.coordinates_format = coordinates_format

    def __repr__(self) -> str:
        return f"BoundingBox: ~[{self.x=}, {self.y=}, {self.w=}, {self.h=}], ~{self.confidence=}"

    def iou(self, b: 'BoundingBox') -> float:
        assert b.coordinates_format == CoordinatesFormat.XYWH and self.coordinates_format == CoordinatesFormat.XYWH
        bb_one = from_xywh_to_xyxy(self)
        bb_two = from_xywh_to_xyxy(b)
        x_a: float = max(bb_one[0], bb_two[0])
        y_a: float = max(bb_one[1], bb_two[1])
        x_b: float = min(bb_one[2], bb_two[2])
        y_b: float = min(bb_one[3], bb_two[3])
        # Compute the area of both the prediction and groud-truth
        intersec_area: float = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
        # Compute the area of both the prediction and ground-thruth
        box_one_area: float = (
            bb_one[2] - bb_one[0] + 1) * (bb_one[3] - bb_one[1] + 1)
        box_two_area: float = (
            bb_two[2] - bb_two[0] + 1) * (bb_two[3] - bb_two[1] + 1)
        return intersec_area / (box_one_area + box_two_area - intersec_area)

    def check_integrity(self, image_width: float=-1, image_height: float=-1) -> bool:
        verification = False
        # Check normalized or not
        if self.coordinates_values ==  CoordinatesValues.RELATIVE:
            verification = self.x >= 0 and self.x <= 1 and self.y >= 0 and self.y <= 1 and self.w >= 0 and self.w <= 1 and self.h >= 0 and self.h <= 1
            assert verification == True
        elif self.coordinates_values ==  CoordinatesValues.ABSOLUTE:
            if image_width == -1 and image_height == -1:
                raise ValueError("Please supply and image_width and an image_height for the integrity check")
            verification = self.x <= image_width and self.y <= image_height and self.w <= image_width and self.h <= image_height
            assert verification == True
        return verification


class Image:
    def __init__(
            self,
            filename: str,
            image_width: float = None,
            image_height: float = None
    ) -> None:
        self.filename = filename
        self.bounding_boxes: list[BoundingBox] = []
        self.image_width = image_width
        self.image_height = image_height

    def __repr__(self) -> str:
        s = f"Image: ~{self.filename=}, ~{self.image_width=}, ~{self.image_height=}"
        bb_str = [s]
        for bb in self.bounding_boxes:
            bb_str.append(bb.__repr__())
        return "\n    ".join(bb_str)

    # Getters, Setters
    def get_bb_count(self) -> int:
        return len(self.bounding_boxes)

    def verify_integrity(self, class_dict: Dict[int, str]) -> bool:
        # Check that depending on the CorrdinatesType,
        # the bb objects all have correct values
        verification = False
        for bb in self.bounding_boxes:
            verification = verification and bb.check_integrity(self.image_width, self.image_height)
        return True

    def add_bb(self, bb: BoundingBox) -> None:
        self.bounding_boxes.append(bb)

    def upscale_bounding_boxes(self) -> bool:
        verification = False
        for bb in self.bounding_boxes:
            verification = verification and bb.check_integrity()
            bb.x = bb.x * self.image_width
            bb.y = bb.y * self.image_height
            bb.w = bb.w * self.image_width
            bb.h = bb.h * self.image_height 
            bb.coordinates_values = CoordinatesValues.ABSOLUTE
        return verification
    
    def normalize_bounding_boxes(self) -> bool:
        verification = False
        for bb in self.bounding_boxes:
            verification = verification and bb.check_integrity(self.image_width, self.image_height)
            bb.x = bb.x / self.image_width
            bb.y = bb.y / self.image_height
            bb.w = bb.w / self.image_width
            bb.h = bb.h / self.image_height 
            bb.coordinates_values = CoordinatesValues.RELATIVE
        return verification


# functions operating on Image and BboundingBox classes

def from_xywh_to_xyxy(bb: 'BoundingBox') -> None:
    return [bb.x - bb.w/2, bb.y - bb.h/2, bb.x + bb.w/2, bb.y + bb.h/2]

# : list[Image]
def _get_corresponding_gt_image(gt_images: List[Image], name: str) -> Image:
    for img in gt_images:
        if img.filename == name:
            return img
    return None


def evaluate_model(
        gt_images: List[Image],
        det_images: List[Image],
        confidence_threshold: float = 0.25,
        IoU_threshold: float=0.5
    ) -> Tuple[int, int, int]:
    """[summary]
        For each detection D with confidence_score > confidence_threshold:
            Among the GTs, choose one with same class_id/name and highest IoU
            if no GT can be picked or IoU < IoU_threshold:
                FP += 1 # detection is a FP
            else:
                TP += 1 # detection is a TP
    """
    # list or dict for images ??
    # Count GT = GT per Image --  for i in images: GT += Image.get_bb_count()
    GT_count = 0
    for i in gt_images:
        GT_count += i.get_bb_count()
    TP, FP = 0, 0
    # detected bboxes
    for det_image in det_images:
        # TODO # gt for det_image
        gt_image: Image = _get_corresponding_gt_image(gt_images, det_image.filename)

        # for all bb in detected_obj
        for bb in det_image.bounding_boxes:
            if bb.confidence < confidence_threshold:
                continue
            # resize ??
            # among gtboxes, choose highest IoU
            # upscale all GT_bb and DT_bb for img(i) ?
            max_IoU = sys.float_info.min
            for gt_bb in gt_image.bounding_boxes:
                current_IoU = gt_bb.iou(bb)
                if current_IoU > max_IoU:
                    max_IoU = current_IoU
            # update (for each bb in Image det_i)
            if max_IoU > IoU_threshold:
                TP += 1
            else:
                FP += 1
    FN = GT_count - TP
    return TP, FP, FN

def get_precision_recall(TP: int, FP: int, FN: int, TN: int=0) -> Tuple[float, float]:
    p: float = 0
    r: float = 0
    try:
        p = TP / (TP + FP)
    except:
        p = 1
    try:
        r = TP / (TP / FN)
    except:
        r = 1
    return p, r

def display_confusion_matrix(TP: int, FP: int, FN: int, TN: int=0) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    data = np.array([[TP, FN], [FP, TN]])
    formatted_text = [
        [f"TP\nIoU > 0.5\n{TP}", f"FN\nnot detected\n{FN}"],
        [f"FP\nIoU < 0.5\n{FP}", f"TN\n\n{TN}"],
    ]
    ax = sns.heatmap(data, annot=formatted_text, fmt="", cmap="Blues", cbar=False)
    ax.set_title("Confusion matrix")
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("Ground truth labels")
    ax.xaxis.set_ticklabels([1, 0])
    ax.yaxis.set_ticklabels([1, 0])
    plt.show()
    # return ax