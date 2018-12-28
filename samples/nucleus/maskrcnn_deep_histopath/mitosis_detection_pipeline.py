import cv2
import math
import os
import sys
import shutil
import logging

import numpy as np
from pathlib import Path
from shutil import copyfile
from deephistopath.evaluation import add_ground_truth_mark_help
from deephistopath.visualization import Shape
from deephistopath.evaluation import get_locations_from_csv, evaluate_global_f1
from deephistopath.detection import tuple_2_csv, dbscan_clustering
from samples.nucleus import nucleus_mitosis
import tensorflow as tf
#from tensorflow.python.data.experimental.ops.matching_files import MatchingFilesDataset
from tensorflow.python.data.ops import dataset_ops
from train_mitoses import normalize
from preprocess_mitoses import extract_patch
from preprocess_mitoses import gen_patches
from preprocess_mitoses import save_patch
from eval_mitoses import evaluate


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
        y = 0
        while (y < h):
            y1 = y
            y2 = y1 + size
            if (y2 > h):
                y2 = h
                y1 = h - size
            x = 0
            while (x < w):
                x1 = x
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
                x = x + size - overlap
            y = y + size - overlap

def combine_images(input_dir, output_dir, size=128, clean_output_dir=False):
    if clean_output_dir:
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

def combine_csvs(input_dir, output_dir, hasHeader=True, clean_output_dir=False):
    if clean_output_dir:
        shutil.rmtree(output_dir)
    input_files = [str(f) for f in Path(input_dir).glob('**/**/*.csv')]
    combine_csvs = {}
    for input_file in input_files:
        points = get_locations_from_csv(input_file, hasHeader=hasHeader,
                                     hasProb=False)
        basename, y, x = os.path.basename(input_file).split('.')[0].split("_")
        if not basename in combine_csvs:
            combine_csvs[basename] = []
        y = int(y)
        x = int(x)
        for i in range(len(points)):
            points[i] = (points[i][0]+y, points[i][1]+x)
        combine_csvs[basename].extend(points)

    os.makedirs(output_dir, exist_ok=True)
    for combined_csv in combine_csvs:
        tuple_2_csv(combine_csvs[combined_csv],
                    os.path.join(output_dir, combined_csv) + '.csv',
                    columns=['Y', 'X'])

def add_groundtruth_mark(im_dir, ground_truth_dir, hasHeader=False, shape=Shape.CROSS,
                         mark_color=(0, 255, 127, 200), hasProb=False):
    im_files = [str(f) for f in Path(im_dir).glob('*.png')]
    for im_file in im_files:
        im_file_basename = os.path.basename(im_file).split('.')[0]
        ground_truth_file_path = os.path.join(ground_truth_dir,
                                              *im_file_basename.split("-"))
        ground_truth_file_path = "{}.csv".format(ground_truth_file_path)
        if not os.path.exists(ground_truth_file_path):
            print("{} doestn't exist".format(ground_truth_file_path))
            continue
        add_ground_truth_mark_help(im_file, ground_truth_file_path,
                                   hasHeader=hasHeader, shape=shape)

def add_mark(img_file, csv_file, hasHeader=False, shape=Shape.CROSS,
             mark_color=(0, 255, 127, 200), hasProb=False):
    add_ground_truth_mark_help(img_file, csv_file, hasHeader=hasHeader,
                               shape=shape, mark_color=mark_color,
                               hasProb=hasProb)

def is_inside(x1, y1, x2, y2, radius):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist <= radius

def check_nucleius_inference(inference_dir, ground_truth_dir):
    ground_truth_csvs = [str(f) for f in Path(ground_truth_dir).glob('*/*.csv')]
    matched_count = 0
    total_count = 0
    for ground_truth_csv in ground_truth_csvs:
        ground_truth_dir, base = os.path.split(ground_truth_csv)
        sub_dir = os.path.split(ground_truth_dir)[1]
        inference_csv = os.path.join(inference_dir, "{}-{}".format(sub_dir, base))
        ground_truth_locations = get_locations_from_csv(
            ground_truth_csv, hasHeader=False, hasProb=False)
        inference_locations = get_locations_from_csv(
            inference_csv, hasHeader=True, hasProb=False)
        for (x1, y1) in ground_truth_locations:
            total_count = total_count + 1
            for (x2, y2) in inference_locations:
                if (is_inside(x2, y2, x1, y1, 32)):
                    matched_count = matched_count + 1
                    break
    print("There are {} ground truth points, found {} of them.".format(
        total_count, matched_count))


def extract_patches(img_dir, location_csv_dir, output_patch_basedir):
    location_csv_files = [str(f) for f in Path(location_csv_dir).glob('*.csv')]
    if len(location_csv_files) == 0:
        raise ValueError(
            "Please check the input dir for the location csv files.")

    for location_csv_file in location_csv_files:
        print("Processing {} ......".format(location_csv_file))
        points = get_locations_from_csv(location_csv_file, hasHeader=True,
                                        hasProb=False)
        # Get the image file name.
        subfolder = os.path.basename(location_csv_file) \
            .replace('-', '/') \
            .replace('.csv', '')
        img_file = os.path.join(img_dir, "{}.tif".format(subfolder))
        print("Processing {} ......".format(img_file))
        img = cv2.imread(img_file)
        img = np.asarray(img)[:, :, ::-1]

        output_patch_dir = os.path.join(output_patch_basedir, subfolder)
        if not os.path.exists(output_patch_dir):
            os.makedirs(output_patch_dir, exist_ok=True)

        for (row, col) in points:
            patch = extract_patch(img, row, col, 64)
            save_patch(patch, path=output_patch_dir, lab=0, case=0, region=0,
                row=row, col=col, rotation=0, row_shift=0, col_shift=0,
                suffix=0, ext="png")

def get_image_tf(filename):
  """Get image from filename.

  Args:
    filename: String filename of an image.

  Returns:
    TensorFlow tensor containing the decoded and resized image with
    type float32 and values in [0, 1).
  """
  image_string = tf.read_file(filename)
  # shape (h,w,c), uint8 in [0, 255]:
  image = tf.image.decode_png(image_string, channels=3)
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  return image

def get_location_from_file_name(filename):
    basename = os.path.basename(str(filename))
    filename_comps = basename.split('_')
    row = int(filename_comps[3])
    col = int(filename_comps[4])
    return row, col

def run_inference(batch_size,
                  input_dir_path,
                  output_dir_path,
                  model_file,
                  num_parallel_calls=1,
                  prob_thres=0.5,
                  eps=64, min_samples=1,
                  isWeightedAvg=False):
    # create session
    config = tf.ConfigProto(
        allow_soft_placement=True)  # , log_device_placement=True)
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)
    input_file_paths = [str(f) for f in Path(input_dir_path).glob('*.png')]
    input_files = np.asarray(input_file_paths, dtype=np.str)

    input_file_dataset = tf.data.Dataset.from_tensor_slices(input_files)
    img_dataset = input_file_dataset.map(lambda file: get_image_tf(file),
                                         num_parallel_calls=1)
    img_dataset = img_dataset\
        .map(lambda img: normalize(img, "resnet_custom"))\
        .batch(batch_size=batch_size)
    img_iterator = img_dataset.make_one_shot_iterator()
    next_batch = img_iterator.get_next()

    # load the model and add the sigmoid layer
    base_model = tf.keras.models.load_model(model_file, compile=False)

    # specify the name of the added activation layer to avoid the name
    # conflict in ResNet
    probs = tf.keras.layers.Activation('sigmoid', name="sigmoid")(
        base_model.output)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=probs)

    prob_result = np.empty((0, 1))
    while True:
        try:
            img_batch = sess.run(next_batch)
            pred_np = model.predict(img_batch, batch_size)
            prob_result = np.concatenate((prob_result, pred_np), axis=0)
        except tf.errors.OutOfRangeError:
            print("prediction result size: {}".format(prob_result.shape))
            break

    assert prob_result.shape[0] == input_files.shape[0]
    mitosis_probs = prob_result[prob_result > prob_thres]
    input_files = input_files.reshape(-1, 1)
    mitosis_patch_files = input_files[prob_result > prob_thres]
    inference_result = []
    for i in range(mitosis_patch_files.size):
        row, col = get_location_from_file_name(mitosis_patch_files[i])
        prob = mitosis_probs[i]
        inference_result.append((row, col, prob))

    clustered_pred_locations = dbscan_clustering(
        inference_result, eps=eps, min_samples=min_samples,
        isWeightedAvg=isWeightedAvg)

    tuple_2_csv(inference_result,
                os.path.join(output_dir_path, 'mitosis_locations.csv'))
    tuple_2_csv(clustered_pred_locations,
                os.path.join(output_dir_path, 'clustered_mitosis_locations.csv'))

def run_reference_in_batch(batch_size,
                           input_dir_basepath ='datasets/sample_patches/',
                           output_dir_basepath ='datasets/inference_results/',
                           model_file='../../../deep-histopath/experiments/models/deep_histopath_model.hdf5',
                           num_parallel_calls=1,
                           prob_thres=0.5,
                           eps=64, min_samples=1,
                           isWeightedAvg=False):
    re = '[0-9]'*2 + '/' + '[0-9]'*2
    input_patch_dirs = [str(f) for f in Path(input_dir_basepath).glob(re)]
    for input_patch_dir in input_patch_dirs:
        print("Run the inference on {} ......".format(input_patch_dir))
        input_patch_path = Path(input_patch_dir)
        subfolder = os.path.join(input_patch_path.parent.name,
                                 input_patch_path.name)
        reference_output_path = os.path.join(output_dir_basepath, subfolder)
        run_inference(batch_size, input_patch_path, reference_output_path,
                      model_file, num_parallel_calls, prob_thres, eps,
                      min_samples, isWeightedAvg)

print("1. Reorganize the data structure for Mask_RCNN")
mitosis_input_dir = '../../../deep-histopath/data/mitoses/mitoses_train_image_data/'
mitosis_reorganized_dir = '../../../deep-histopath/data/mitoses/mitoses_train_image_data_new/'
#reorganize_mitosis_images(mitosis_input_dir, mitosis_reorganized_dir)

print("2. Crop big images into small ones")
mitosis_files = [str(f) for f in Path(mitosis_reorganized_dir).glob('**/**/*.tif')]
inference_input_dir = 'datasets/stage1_test'
#crop_image(mitosis_files, inference_input_dir, size=128, overlap=16)

print("3. Run the inference")
command = 'detect'
dataset = 'datasets/'
weights = 'models/mask_rcnn_nucleus_0380.h5'
subset = 'stage1_test'
#inference_result_dir = nucleus_mitosis.run(command, dataset, weights, subset)

print("4. Combine small inference images to big ones")
# inference_result_dir = '../../results/nucleus/submit_20181016T152921'
inference_combine_result = 'datasets/stage1_combine_test/'
#combine_images(inference_result_dir, inference_combine_result)
#combine_csvs(inference_result_dir, inference_combine_result, clean_output_dir=False)

print("5. Add the ground truth masks")
ground_truth_dir = '../../../deep-histopath/data/mitoses/mitoses_train_ground_truth'
# add_groundtruth_mark(inference_combine_result, ground_truth_dir, hasHeader=False, shape=Shape.CIRCLE)
# add_mark('/Users/fei/Documents/Github/Mask_RCNN/samples/nucleus/datasets/stage1_combine_test/01-01.png',
#          '/Users/fei/Documents/Github/Mask_RCNN/samples/nucleus/datasets/stage1_combine_test/01-01.csv',
#          hasHeader=True, shape=Shape.CIRCLE, mark_color=(255,0,0,50))

print("6. Check the inference result")
inference_dir = "datasets/stage1_combine_test/"
ground_truth_dir = "../../../deep-histopath/data/mitoses/mitoses_train_ground_truth"
#check_nucleius_inference(inference_dir, ground_truth_dir)

print("7. Extract patches according to the inference result")
img_dir = '../../../deep-histopath/data/mitoses/mitoses_train_image_data/'
location_csv_dir = 'datasets/stage1_combine_test/'
output_patch_basedir = 'datasets/sample_patches'
#extract_patches(img_dir, location_csv_dir, output_patch_basedir)

print("8. Run inference")
batch_size = 128
input_dir_basepath = 'datasets/sample_patches/'
output_dir_basepath = 'datasets/inference_results/'
model_file = '../../../deep-histopath/experiments/models/deep_histopath_model.hdf5'
num_parallel_calls = 1
prob_thres = 0.5,
eps = 64
min_samples = 1
isWeightedAvg = False
run_reference_in_batch(
    128,
    input_dir_basepath=input_dir_basepath,
    output_dir_basepath=output_dir_basepath,
    model_file=model_file,
    num_parallel_calls=num_parallel_calls,
    prob_thres=prob_thres,
    eps=eps,
    min_samples=min_samples,
    isWeightedAvg=isWeightedAvg)

print("9. Compute F1 score")
"""
f1, precision, recall, over_detected, non_detected, FP, TP, FN = \
    evaluate_global_f1(output_dir_basepath, ground_truth_dir, threshold=30,
                       prob_threshold=None)
print("F1: {} \n"
      "Precision: {} \n"
      "Recall: {} \n"
      "Over_detected: {} \n"
      "Non_detected: {} \n"
      "FP: {} \n"
      "TP: {} \n"
      "FN: {} \n".format(f1, precision, recall, over_detected, non_detected,
                         FP, TP, FN))
"""

