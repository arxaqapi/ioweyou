from ioweyou.CoordinatesHandler import CoordinatesFormat, CoordinatesValues
from ioweyou.BoundingBox import BoundingBox, Image

from typing import List

import json
import tqdm
import requests
import os
from pathlib import Path
from glob import glob

def get_bird_dataset() -> None: # destination: str = "./data/bird_dataset.zip"
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value
        return None

    file_id = "16IQjiGu-jl2oTqr5wsp9MmJxtQiuyIWq"
    destination = "./data/bird_dataset.zip"
    if not os.path.isfile(destination) and not os.path.isdir("data/bird_dataset/"):
        # download
        URL = "https://docs.google.com/uc?export=download"
        session = requests.Session()
        response = session.get(URL, params={"id": file_id}, stream=True)
        token = get_confirm_token(response)
        if token:
            params = {"id": file_id, "confirm": token}
            response = session.get(URL, params=params, stream=True)
        # create dir if not exists
        if not os.path.exists("data/"):
            os.mkdir("data/")
        # Save
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in tqdm.tqdm(response.iter_content(CHUNK_SIZE)):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
        command = "".join(["unzip -q ", destination, " -d ", destination[:-4]])
        print(f"[Log] - {command}")
        os.system(command)


# bird dataset to Image && BB object
def import_bird_dataset(path_to_bird_dataset):
    def create_images(gt_images, jsons):
        # Test
        for json in jsons:
            im = Image(
                filename=json['frame'],
                image_width=json['imgWidth'],
                image_height=json['imgHeight']
            )
            for bb in json['objects']:
                im.add_bb(
                    BoundingBox(
                    x=bb['boundingbox'][0],
                    y=bb['boundingbox'][1],
                    w=bb['boundingbox'][2],
                    h=bb['boundingbox'][3],
                    confidence=1, # since gt, confidence = 1 (useless)
                    coordinates_format=CoordinatesFormat.XYXY,
                    coordinates_values=CoordinatesValues.ABSOLUTE,
                    # class_id=bb[],
                    class_name=bb['label'],
                    )
                )
            gt_images.append(im)

    from ioweyou.utils import get_json
    gt_images_test, gt_images_train, gt_images_val = [], [], []
    # get gt's from file/folder
    test_jsons = [get_json(e) for e in glob("".join([path_to_bird_dataset, "test/*.json"]))]
    train_jsons = [get_json(e) for e in glob("".join([path_to_bird_dataset, "train/*.json"]))]
    val_jsons = [get_json(e) for e in glob("".join([path_to_bird_dataset, "val/*.json"]))]
    # Test
    create_images(gt_images_test, test_jsons)
    # Train
    create_images(gt_images_train, test_jsons)
    # Val
    create_images(gt_images_val, val_jsons)
    
    return gt_images_test, gt_images_train, gt_images_val


def parse_yolov4_results_json(path: Path) -> List[Image]:
    with open(path, 'r') as f:
        output_data = json.load(f)  # : List[Dict]
    # test List:
    dt_images = []
    for output_image in output_data:
        # print(output_image['filename'].split('/')[-1][:-4])
        # exit()
        image = Image(output_image['filename'].split('/')[-1][:-4])
        for output_bb in output_image['objects']:
            bb = BoundingBox(
                x=output_bb['relative_coordinates']['center_x'],
                y=output_bb['relative_coordinates']['center_y'],
                w=output_bb['relative_coordinates']['width'],
                h=output_bb['relative_coordinates']['height'],
                confidence=output_bb['confidence'],
                coordinates_format=CoordinatesFormat.XYWH,
                coordinates_values=CoordinatesValues.RELATIVE,
                class_id=output_bb['class_id'],
                class_name=output_bb['name'],
            )
            image.add_bb(bb)
        dt_images.append(image)
    return dt_images
