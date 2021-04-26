# from ioweyou.CoordinatesHandler import CoordinatesFormat, CoordinatesValues
# from ioweyou.BoundingBox import BoundingBox, Image
# from ioweyou.utils import get_all_images, to_file



from ioweyou.BoundingBox import evaluate_model


PATH = "data/bird_dataset/"
# to_file(PATH + "test_index.txt", get_all_images(PATH + "test/"))
# to_file(PATH + "train_index.txt", get_all_images(PATH + "train/"))
# to_file(PATH + "eval_index.txt", get_all_images(PATH + "val/"))


def evaluate_yolov4():
    from ioweyou.interface import import_bird_dataset
    from ioweyou.interface import parse_yolov4_results_json
    from ioweyou.BoundingBox import evaluate_model, precision_recall_curve

    gt_test, gt_train, gt_val = import_bird_dataset(PATH)
    dt = parse_yolov4_results_json("data/results.json")
    for i in dt:
        i.get_wh_from_gt(gt_val)
        i.upscale_bounding_boxes()
    for i in gt_val:
        i.internal_from_xyxy_to_xywh()
    # print(evaluate_model(gt_images=gt_val, det_images=dt))
    p, r = precision_recall_curve(gt_val, dt)
    print("precision")
    print(p)
    print("recall")
    print(r)
    # print(zip(p, r))




def evaluate_yolov3():
    pass

evaluate_yolov4()