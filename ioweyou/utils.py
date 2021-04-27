import os
import json
from glob import glob 

# for each directory, create a .txt file containing the list of all image paths
def get_all_images(dir):
    return glob(dir + "*.jpg") + glob(dir + "*.png")

def to_file(file_path, img_list):
    img_list = list(map(lambda i : i + '\n', img_list))
    with open(file_path, 'w') as f:
        f.writelines(img_list)


def create_bird_dataset_index(path_to_bird_dataset):
    names = ["test", "train", "val"]
    for name in names:
        fp = path_to_bird_dataset + f"{name}_index.txt"
        if os.path.exists(fp):
            os.remove(fp)
        index_list = get_all_images(path_to_bird_dataset + f"{name}/")
        to_file(file_path=fp, img_list=index_list)


def get_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data