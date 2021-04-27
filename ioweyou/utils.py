import os
import json
from glob import glob
from typing import List, Tuple 

# for each directory, create a .txt file containing the list of all image paths
def get_all_images(dir):
    return glob(dir + "*.jpg") + glob(dir + "*.png")

def to_file(file_path, img_list):
    img_list = list(map(lambda i : i + '\n', img_list))
    with open(file_path, 'w') as f:
        f.writelines(img_list)
    return img_list


def create_bird_dataset_index(path_to_bird_dataset: str) -> Tuple[List[str], List[str], List[str]] :
    """test, train, val

    Args:
        path_to_bird_dataset (st): relative path to the bird_dataset/ directory

    Returns:
        Tuple[List[str], List[str], List[str]]: the list of the indexes
    """
    indexes = []
    names = ["test", "train", "val"]
    for name in names:
        fp = path_to_bird_dataset + f"{name}_index.txt"
        if os.path.exists(fp):
            os.remove(fp)
        index_list = get_all_images(path_to_bird_dataset + f"{name}/")
        indexes.append(to_file(file_path=fp, img_list=index_list))
    return indexes[0], indexes[1], indexes[2]


def get_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data