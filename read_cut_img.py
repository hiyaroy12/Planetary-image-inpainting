from matplotlib import pyplot as plt
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import os
import glob
import json
from tqdm import tqdm

IMAGE_FILE_PATH = os.path.expanduser("~/Data/hirise-map-proj-v3/map-proj-v3/")
SAVE_FILE_PATH = './image_array.npy'


def add_lines(img, lines):
    if len(lines == 0) > len(lines > 0):
        return img * lines
    else:
        return img * (1 - lines)


def add_lines_batch(img_list, lines_list):
    list_lined_imgs = []
    list_true_imgs = []

    for img in img_list:
        for label in lines_list:
            line_added_img = add_lines(img, label)
            list_lined_imgs.append(line_added_img)
            list_true_imgs.append(img)

    np_true_imgs = np.asarray(list_true_imgs, dtype='uint8')
    np_true_imgs = np_true_imgs[..., np.newaxis]

    np_lined_imgs = np.asarray(list_lined_imgs, dtype='uint8')
    np_lined_imgs = np_lined_imgs[..., np.newaxis]

    return np_true_imgs, np_lined_imgs


def detect_lines(img, thres=7):
    thres_img = np.asarray((img < thres), dtype='uint8')
    display_img = 255 * thres_img
    # Set up the detector with default parameters.
    return thres_img, display_img


def display_sidebyside(img1, img2, verbose=True):
    print('Image 1:', img1.shape)
    print('Image 2:', img2.shape)
    display_img = np.concatenate((img1, img2), axis=1)
    if verbose:
        # plt.imshow(display_img, cmap='gray')
        cv2.imshow('frame', display_img)
        cv2.waitKey(40)
    return display_img


class ImageReader(object):
    def __init__(self):
        pass

    def save_image(self):
        img = np.asarray(Image.open(IMAGE_FILE_PATH), dtype='uint8')
        np.save(SAVE_FILE_PATH, img)

    def cut_image(self, verbose=True):
        list_imgs_no_lines = []
        list_imgs_lines = []

        files = glob.glob(IMAGE_FILE_PATH + '*.jpg', recursive=True)

        block_size = 227
        low_thres = block_size * block_size * 0.01
        high_thres = block_size * block_size * 0.3

        for filename in tqdm(files):
            sub_img = cv2.imread(filename)
            thres_img, binary_i = detect_lines(sub_img)
            sum_i = np.sum(thres_img > 0)
            if np.mean(sub_img[sub_img > 0]) > 64:
                if (sum_i > low_thres) and (sum_i < high_thres):
                    list_imgs_lines.append(filename)
                    # print('Added line image...')
                    # cv2.imshow('frame', sub_img)
                    # cv2.waitKey(10)
                elif (sum_i <= low_thres):
                    list_imgs_no_lines.append(filename)
                    # print('Added NO line image')

        num_lines_imgs = len(list_imgs_lines)
        num_no_lines_imgs = len(list_imgs_no_lines)
        total_imgs = num_lines_imgs + num_no_lines_imgs
        print('Statistics:')
        print('Total no of patches:', total_imgs)
        print('Total no of patches with lines:', num_lines_imgs)
        print('Total no of patches without lines:', num_no_lines_imgs)

        os.makedirs('./result/', exist_ok=True)
        json_obj = {'no_lines': list_imgs_no_lines, 'lines': list_imgs_lines}
        with open("./result/list_ims.json", 'w') as fp:
            json.dump(json_obj, fp)

    def load_cut_images(self, block_size=32, stride=16):
        file_save_name = 'img_blocksize_{}_stride_{}.npy'.format(block_size, stride)
        np_imgs = np.load(file_save_name)
        return np_imgs

def read_images():
    with open("./result/list_ims.json", 'r') as fp:
        json_obj = json.load(fp)

    list_imgs_no_lines = json_obj['no_lines']
    list_imgs_lines = json_obj['lines']
    batchsize = 128
    rand_idx_lines = np.random.choice(len(list_imgs_lines), batchsize, replace=False).tolist()
    rand_idx_no_lines = np.random.choice(len(list_imgs_no_lines), batchsize, replace=False).tolist()

    for lineidx, nolineidx in tqdm(zip(rand_idx_lines, rand_idx_no_lines)):

        img_line = cv2.imread(list_imgs_lines[lineidx], 0)
        img_no_line = cv2.imread(list_imgs_no_lines[nolineidx], 0)
        thres_img, binary_i = detect_lines(img_line, thres=10)

        img_superimposed_lines = add_lines(img_no_line.copy(), thres_img)
        # display_sidebyside(img_no_line, img_superimposed_lines)
        display_sidebyside(np.concatenate((img_no_line, img_superimposed_lines), axis=1),
                           np.concatenate((img_line, binary_i), axis=1))

        print(img_line.shape)
        print(img_superimposed_lines.shape)

if __name__ == '__main__':
    reader = ImageReader()
    print('Reading image...')
    # img = reader.cut_image()
    read_images()