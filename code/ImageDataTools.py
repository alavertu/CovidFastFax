"""
ImageDataTools.py
Written by Adam Lavertu
Stanford University
"""

# Mainly data handling and representation
import struct
import random

import numpy as np
import pandas as pd

from skimage import io
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import torch
from torchvision import transforms


def get_centered_checkbox(xmi, xma, ymi, yma, buffer=25):
    x_cent = ((xma - xmi) // 2) + xmi
    new_xmi = x_cent - buffer
    new_xma = x_cent + buffer

    y_cent = ((yma - ymi) // 2) + ymi
    new_ymi = y_cent - buffer
    new_yma = y_cent + buffer

    return [new_xmi, new_xma, new_ymi, new_yma]


def get_dense_dim(temp_in, axis=0, max_gap=50):
    col_sums = np.sum(temp_in, axis=axis)
    col_prop = col_sums / np.max(col_sums)
    border_start = 0
    border_end = 0
    runs_found = []
    dense_cols = np.where(col_prop != np.max(col_prop))[0]
    running = False
    for j, x in enumerate(dense_cols[:-1]):
        if running:
            if (dense_cols[(j + 1)] - x) <= max_gap:
                border_end = x
            else:
                runs_found.append([border_end - border_start, border_start, border_end])
                running = False
        else:
            running = True
            border_start = x
    runs_found.append([border_end - border_start, border_start, border_end])
    return sorted(runs_found, reverse=True)[0][1:]


def auto_crop(temp_im):
    crop_row_start, crop_row_end = get_dense_dim(temp_im, 0)
    crop_col_start, crop_col_end = get_dense_dim(temp_im, 1)
    return temp_im[crop_col_start:crop_col_end, crop_row_start:crop_row_end]


class Crop(object):
    def __init__(self, height, width):
        self.width = width
        self.height = height

    def __call__(self, img):
        width, _ = img.size
        return transforms.functional.crop(
            img, 0, width - self.width, self.height, self.width
        )


def prep_image_data(images, transforms):
    out_tensors = []
    for k in images:
        for k in images:
            out_tensors.append(transforms(k))
            temp_im = 1.0 - k
            temp_im = temp_im.astype(np.float32)
            out_tensors.append(transforms(temp_im))
    return torch.stack(out_tensors)


def load_mnist_data(labels_path, images_path):
    with open(labels_path, "rb") as lbpath:
        magic, n = struct.unpack(">II", lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, "rb") as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return (labels, images)


def grid_images(image_list, num_cols=2):

    dim_size = len(image_list) // num_cols

    if len(image_list) % num_cols != 0:
        dim_size += 1

    fig = plt.figure(figsize=(8, 12 * num_cols))
    grid = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(dim_size, num_cols),  # creates 2x2 grid of axes
        axes_pad=0.1,  # pad between axes in inch.
    )

    for ax, im in zip(grid, image_list):
        # Iterating over the grid returns the Axes.
        _ = ax.imshow(im, cmap="Greys")
    plt.show()


def plot_sample_images(X, y, images_to_show=10, random=True, cmap="Greys"):

    fig = plt.figure(1)

    images_to_show = min(X.shape[0], images_to_show)

    # Set the canvas based on the numer of images
    fig.set_size_inches(18.5, images_to_show * 0.3)

    # Generate random integers (non repeating)
    if random == True:
        idx = np.random.choice(range(X.shape[0]), images_to_show, replace=False)
    else:
        idx = np.arange(images_to_show)

    # Print the images with labels
    for i in range(images_to_show):
        plt.subplot(images_to_show / 10 + 1, 10, i + 1)
        plt.title(str(y[idx[i]]))
        plt.imshow(X[idx[i], :, :], cmap=cmap, vmin=0, vmax=1)


def plot_big_image(X, cmap="Greys"):
    plt.figure(figsize=(12, 12))
    plt.imshow(X, cmap=cmap)
    plt.show()


def plot_image(X, cmap="Greys"):
    plt.imshow(X, cmap=cmap)


def plot_image_box(im, xmi, xma, ymi, yma):
    plt.imshow(im[xmi:xma, ymi:yma])
    plt.show()


def get_coord_data(image, coords):
    xmi, xma, ymi, yma = coords
    return image[xmi:xma, ymi:yma]


def get_coord_data(image, coords):
    xmi, xma, ymi, yma = coords
    return image[xmi:xma, ymi:yma]


def read_image(imPath):
    temp = np.asarray(io.imread(imPath), dtype=float)
    if np.max(temp) > 1.0:
        temp /= 255.0
    return temp
