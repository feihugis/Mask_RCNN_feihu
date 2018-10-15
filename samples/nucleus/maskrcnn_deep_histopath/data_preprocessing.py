import cv2
import os
import numpy as np
from pathlib import Path
from shutil import copyfile


def reorganize_mitosis_images(input_dir, output_dir):
    input_files = [str(f) for f in Path(input_dir).glob('**/*.tif')]
    for file in input_files:
        file_basename = os.path.basename(file).split('.')[0]
        parent_dir = os.path.dirname(file)
        dir_basename = os.path.basename(parent_dir)
        new_dir = os.path.join(output_dir, "{}-{}".format(dir_basename, file_basename), 'images')
        os.makedirs(new_dir, exist_ok=True)
        copyfile(file, os.path.join(new_dir, "{}-{}.tif".format(dir_basename, file_basename)))
        print(file_basename, dir_basename)

def crop_image(input_imgs, output_dir, size=128):
    for input_img in input_imgs:
        basename = os.path.basename(input_img).split('.')[0]
        img = cv2.imread(input_img)

        h, w, c = img.shape
        n_w = int(np.ceil(w / size))
        n_h = int(np.ceil(h / size))

        for i in range(n_h):
            for j in range(n_w):
                crop_img = img[size * i: size * (i + 1), size * j: size * (j + 1)]
                result_basename = os.path.join("{}_{}_{}".format(basename, i, j))
                result_dir = os.path.join(output_dir, result_basename, 'images')
                os.makedirs(result_dir, exist_ok=True)
                output_file = os.path.join(result_dir, result_basename + '.png')
                cv2.imwrite(output_file, crop_img)

def combine_images(input_dir, output_dir, size=128):
    input_files = [str(f) for f in Path(input_dir).glob('**/**/*.png')]
    input_basenames = [os.path.basename(input_file).split('.')[0].split("_")
                       for input_file in input_files]
    combined_files = list(set([input_base[0]
                               for input_base in input_basenames]))
    combined_tile_size = {combined_file : (0,0)
                               for combined_file in combined_files}
    for input_basename in input_basenames:
        basename, y, x = input_basename
        (y_tile, x_tile) = combined_tile_size[basename]
        x_tile = max(int(x), x_tile)
        y_tile = max(int(y), y_tile)
        combined_tile_size[basename] = (y_tile, x_tile)

    combined_imgs = {}
    combined_file_size = {}
    for file in combined_tile_size:
        tile_size = combined_tile_size[file]
        combined_imgs[file] = np.zeros([size*(tile_size[0]+1),
                                        size*(tile_size[1]+1), 3],
                                       dtype=np.uint8)
        combined_file_size[file] = (0,0)

    for input_file in input_files:
        img = cv2.imread(input_file)
        h, w, c = img.shape
        print(img.shape)
        basename, y, x = os.path.basename(input_file).split('.')[0].split("_")
        x = int(x)
        y = int(y)
        combined_imgs[basename][y*size:y*size+h, x*size:x*size+w, :] = img
        combined_file_size[basename] = \
            (max(combined_file_size[basename][0], y*size+h),
             max(combined_file_size[basename][1], x*size+w))


    os.makedirs(output_dir, exist_ok=True)
    for combined_img in combined_imgs:
        cv2.imwrite(os.path.join(output_dir, combined_img) + '.png',
                    combined_imgs[combined_img][
                    0:combined_file_size[basename][0],
                    0:combined_file_size[basename][1], :])

mitosis_input_dir = '../../../../deep-histopath/data/mitosis/mitoses_train_image_data/'
mitosis_reorganized_dir = '../../../../deep-histopath/data/mitosis/mitoses_train_image_data_new/'
reorganize_mitosis_images(mitosis_input_dir, mitosis_reorganized_dir)
mitosis_files = [str(f) for f in Path(mitosis_reorganized_dir).glob('**/**/*.tif')]
inference_dir = '/Users/fei/Documents/Github/Mask_RCNN/samples/nucleus/datasets/stage1_test'
crop_image(mitosis_files, inference_dir)
#combine_images(input_dir, '/Users/fei/Documents/Github/Mask_RCNN/samples/nucleus/datasets/stage1_combine_test/')