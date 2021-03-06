
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


def test_internal_from__to__():
    from ioweyou.BoundingBox import BoundingBox, Image
    from ioweyou.CoordinatesHandler import CoordinatesFormat, CoordinatesValues
    bb = BoundingBox(
        # x=500, y=300, w=400, h=200,
        x=300, y=200, w=700, h=400,
        confidence=0.7,
        coordinates_values=CoordinatesValues.ABSOLUTE,
        coordinates_format=CoordinatesFormat.XYXY,
    )
    im = Image(filename='test', image_width=1000, image_height=600)
    im.add_bb(bb)
    print(im)
    im.internal_from_xyxy_to_xywh()
    print(im)
    im.internal_from_xywh_to_xyxy()
    print(im)

def test_create_bird_dataset_index():
    from ioweyou.utils import create_bird_dataset_index
    test, train, val = create_bird_dataset_index("data/bird_dataset/")


def test_serialize_objects():
    from ioweyou.BoundingBox import BoundingBox
    from ioweyou.CoordinatesHandler import CoordinatesFormat, CoordinatesValues
    bb = BoundingBox(
            # x=500, y=300, w=400, h=200,
            x=300, y=200, w=700, h=400,
            confidence=0.7,
            coordinates_values=CoordinatesValues.ABSOLUTE,
            coordinates_format=CoordinatesFormat.XYXY,
        )

    import pickle
    with open('test.pickle', 'wb') as f:
        pickle.dump(bb, f)

    with open('test.pickle', 'rb') as f:
        returned = pickle.load(f)
        print(returned)


# test_evaluate_model()
# test_yolov4_interface()
# test_get_bird_dataset()
# test_internal_from__to__()
# test_create_bird_dataset_index()