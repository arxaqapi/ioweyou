
from pathlib import Path


def test_evaluate_model():
    from ioweyou.BoundingBox import BoundingBox, Image, evaluate_model
    from ioweyou.CoordinatesHandler import CoordinatesFormat, CoordinatesValues
    x, y, w, h = 100, 200, 50 , 100 
    gt1 = BoundingBox(
        x, y, w, h,
        confidence=0.7,
        coordinates_values=CoordinatesValues.ABSOLUTE,
        coordinates_format=CoordinatesFormat.XYWH,
    )
    dt1 = BoundingBox(
        x, y, 55, h,
        confidence=0.7,
        coordinates_values=CoordinatesValues.ABSOLUTE,
        coordinates_format=CoordinatesFormat.XYWH,
    )
    gt2 = BoundingBox(
        669, 359, 425, 198,
        confidence=0.7,
        coordinates_values=CoordinatesValues.ABSOLUTE,
        coordinates_format=CoordinatesFormat.XYWH,
    )
    dt2 = BoundingBox(
        670, 368, 120, 201,
        confidence=0.7,
        coordinates_values=CoordinatesValues.ABSOLUTE,
        coordinates_format=CoordinatesFormat.XYWH,
    )
    gti = Image(
        filename='test.png',
        image_width=1920,
        image_height=1080,
    )
    gti.add_bb(gt1)
    gti.add_bb(gt2)
    dti = Image(
        filename='test.png',
        image_width=1920,
        image_height=1080,
    )
    dti.add_bb(dt1)
    dti.add_bb(dt2)
    print(gti)
    print(dti)
    TP, FP, FN = evaluate_model([gti], [dti])
    print(f"[Test - test_evaluate_model()] - {TP=} {FP=} {FN=}")
    # display_confusion_matrix(TP, FP, FN)

def test_yolov4_interface():
    from ioweyou.interface import parse_yolov4_results_json
    path = Path('data/') / 'results.json'
    print(parse_yolov4_results_json(path))


def test_get_bird_dataset():
    from ioweyou.interface import get_bird_dataset
    get_bird_dataset()

test_evaluate_model()
test_yolov4_interface()
test_get_bird_dataset()