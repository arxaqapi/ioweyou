import os
import glob as glob

# for each directory, create a .txt file containing the list of all image paths
def get_all_images(dir):
    return glob(dir + "*.jpg") + glob(dir + "*.png")

def to_file(file_path, img_list, prefix='../'):
    if os.path.exists(file_path):
        os.remove(file_path)
    img_list = list(map(lambda x : prefix + x + '\n', img_list))
    with open(file_path, 'a+') as f:
        f.writelines(img_list)

# PATH = "bird_dataset/"
# to_file(PATH + "eval_index.txt", get_all_images(PATH + "val/"))
# to_file(PATH + "train_index.txt", get_all_images(PATH + "train/"))
# to_file(PATH + "test_index.txt", get_all_images(PATH + "test/"))