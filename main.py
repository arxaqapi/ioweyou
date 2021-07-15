from ioweyou.BoundingBox import evaluate_model

PATH = "data/bird_dataset/"
# to_file(PATH + "test_index.txt", get_all_images(PATH + "test/"))
# to_file(PATH + "train_index.txt", get_all_images(PATH + "train/"))
# to_file(PATH + "eval_index.txt", get_all_images(PATH + "val/"))


def evaluate_yolov4():
    from ioweyou.interface import import_bird_dataset
    from ioweyou.interface import parse_yolov4_results_json
    from ioweyou.BoundingBox import evaluate_model, precision_recall_curve, mAP

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
    print(f"map = {mAP(precisions=p, recalls=r)}")
    import pickle
    with open('yolov4_infered_bbxs.pickle', 'wb') as f:
        pickle.dump(dt, f)




def evaluate_yolov3():
    pass


def evaluate_efficientdet():
    from ioweyou.BoundingBox import BoundingBox, Image
    from ioweyou.CoordinatesHandler import CoordinatesFormat, CoordinatesValues
    from typing import List
    import pickle

    with open('efficiendet_infered_bbxs.pickle', 'rb') as f:
        img_list: List[Image]= pickle.load(f)

    print(img_list[0].bounding_boxes[0].coordinates_format)


# evaluate_efficientdet()
evaluate_yolov4()