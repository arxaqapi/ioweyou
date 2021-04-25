from ioweyou.CoordinatesHandler import CoordinatesFormat, CoordinatesValues
from ioweyou.BoundingBox import BoundingBox, Image

from typing import List

import json
import tqdm
import requests
import os
from pathlib import Path


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
def import_bird_dataset():
    train, test, val = 0, 0, 0
    # get gt's from file/folder
    #

    return train, test, val


def parse_yolov4_results_json(path: Path) -> List[Image]:
    with open(path, 'r') as f:
        output_data = json.load(f)  # : List[Dict]
    # test List:
    gt_images = []
    for output_image in output_data:
        image = Image(output_image['filename'])
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
        gt_images.append(image)
    return gt_images
