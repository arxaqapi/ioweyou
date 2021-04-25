from enum import Enum

class CoordinatesFormat(Enum):
    """Tells in wich format the coordinates are stored,
        XYWH: means centered in XY with width W and height H
        XYXY: means the box is defines with the top left and bottom right corners
    """
    XYWH = 0
    XYXY = 1

class CoordinatesValues(Enum):
    """Tells if the values of the coordinates are normalized or not
    """
    RELATIVE = 0
    ABSOLUTE = 1