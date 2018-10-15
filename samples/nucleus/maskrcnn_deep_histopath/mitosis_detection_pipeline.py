import cv2
import os
import numpy as np
from pathlib import Path
from shutil import copyfile
from deephistopath.evaluation import add_ground_truth_mark_help
from deephistopath.visualization import Shape
from  samples.nucleus import nucleus_mitosis
import sys
import shutil

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

def reorganize_mitosis_images(input_dir, output_dir):
    input_files = [str(f) for f in Path(input_dir).glob('**/*.tif')]
    shutil.rmtree(output_dir)
    for file in input_files:
        file_basename = os.path.basename(file).split('.')[0]
        parent_dir = os.path.dirname(file)
        dir_basename = os.path.basename(parent_dir)
        new_dir = os.path.join(output_dir, "{}-{}".format(dir_basename, file_basename), 'images')
        os.makedirs(new_dir, exist_ok=True)
        copyfile(file, os.path.join(new_dir, "{}-{}.tif".format(dir_basename, file_basename)))
        print(file_basename, dir_basename)

def crop_image(input_imgs, output_dir, size=128, overlap=0):
    shutil.rmtree(output_dir)
    for input_img in input_imgs:
        basename = os.path.basename(input_img).split('.')[0]
        img = cv2.imread(input_img)

        h, w, c = img.shape
        y = overlap
        while (y < h):
            y1 = y - overlap
            y2 = y1 + size
            if (y2 > h):
                y2 = h
                y1 = h - size
            x = overlap
            while (x < w):
                x1 = x - overlap
                x2 = x1 + size
                if (x2 > w):
                    x2 = w
                    x1 = w - size
                crop_img = img[y1: y2, x1: x2]
                result_basename = os.path.join("{}_{}_{}".format(basename, y1, x1))
                result_dir = os.path.join(output_dir, result_basename, 'images')
                os.makedirs(result_dir, exist_ok=True)
                output_file = os.path.join(result_dir, result_basename + '.png')
                cv2.imwrite(output_file, crop_img)
                x = x + size
            y = y + size

def combine_images(input_dir, output_dir, size=128):
    shutil.rmtree(output_dir)
    input_files = [str(f) for f in Path(input_dir).glob('**/**/*.png')]
    input_basenames = [os.path.basename(input_file).split('.')[0].split("_")
                       for input_file in input_files]
    combined_imgs = {}
    combined_file_size = {}

    for input_basename in input_basenames:
        basename, y, x = input_basename
        y = int(y)
        x = int(x)
        if (not basename in combined_file_size):
            combined_file_size[basename] = (0,0)

        combined_file_size[basename] = \
            (max(combined_file_size[basename][0], y+size),
             max(combined_file_size[basename][1], x+size))

    for basename in combined_file_size:
        h, w = combined_file_size[basename]
        combined_imgs[basename] = np.zeros([h, w, 3], dtype=np.uint8)

    for input_file in input_files:
        img = cv2.imread(input_file)
        h, w, c = img.shape
        basename, y, x = os.path.basename(input_file).split('.')[0].split("_")
        y = int(y)
        x = int(x)
        combined_imgs[basename][y:y+h, x:x+w, :] = img
        combined_file_size[basename] = \
            (max(combined_file_size[basename][0], y*size+h),
             max(combined_file_size[basename][1], x*size+w))

    os.makedirs(output_dir, exist_ok=True)
    for combined_img in combined_imgs:
        cv2.imwrite(os.path.join(output_dir, combined_img) + '.png',
                    combined_imgs[combined_img][
                    0:combined_file_size[basename][0],
                    0:combined_file_size[basename][1], :])

def add_mark(im_dir, ground_truth_dir, hasHeader=False, shape=Shape.CROSS,
             mark_color=(0, 255, 127, 200), hasProb=False):
    im_files = [str(f) for f in Path(im_dir).glob('*.png')]
    for im_file in im_files:
        im_file_basename = os.path.basename(im_file).split('.')[0]
        ground_truth_file_path = os.path.join(ground_truth_dir,
                                              *im_file_basename.split("-"))
        ground_truth_file_path = "{}.csv".format(ground_truth_file_path)
        add_ground_truth_mark_help(im_file, ground_truth_file_path,
                                   hasHeader=hasHeader, shape=Shape.CIRCLE)

print("1. Reorganize the data structure for Mask_RCNN")
mitosis_input_dir = '../../../deep-histopath/data/mitoses/mitoses_train_image_data/'
mitosis_reorganized_dir = '../../../deep-histopath/data/mitoses/mitoses_train_image_data_new/'
reorganize_mitosis_images(mitosis_input_dir, mitosis_reorganized_dir)

print("2. Crop big images into small ones")
mitosis_files = [str(f) for f in Path(mitosis_reorganized_dir).glob('**/**/*.tif')]
inference_input_dir = 'datasets/stage1_test'
crop_image(mitosis_files, inference_input_dir, size=128, overlap=0)

print("3. Run the inference")
command = 'detect'
dataset = 'datasets/'
weights = 'models/mask_rcnn_nucleus_0380.h5'
subset = 'stage1_test'
inference_result_dir = nucleus_mitosis.run(command, dataset, weights, subset)

print("4. Combine small inference images to big ones")
#inference_result_dir = '../../results/nucleus/submit_20181015T151642'
inference_combine_result = 'datasets/stage1_combine_test/'
combine_images(inference_result_dir, inference_combine_result)

print("5. Add the ground truth masks")
ground_truth_dir = '../../../deep-histopath/data/mitoses/mitoses_train_ground_truth'
add_mark(inference_combine_result, ground_truth_dir, hasHeader=False, shape=Shape.CIRCLE)