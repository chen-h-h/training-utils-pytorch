import os
import sys
import math
import random

import json
import yaml
import cv2
import torch
import argparse
import numpy as np
import matplotlib.pylab as plt
from typing import List, Dict
from PIL import Image

def makedirs(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def split_list(list4split : List, sub_list_ratio: List[float] = None) -> List[List]:
    
    random.shuffle(list4split)
    length = len(list4split)

    list_splits = []
    for i, ratio in enumerate(sub_list_ratio):
        if i==0:
            list_splits.append(list4split[:int(ratio * length)])

        elif i==len(sub_list_ratio) - 1:
            list_splits.append(list4split[int(sum(sub_list_ratio[:i]) * length):])
        
        else:
            list_splits.append(list4split[int(sum(sub_list_ratio[:i]) * length):int(sum(sub_list_ratio[:i+1]) * length)])

    return list_splits

def visualize_image(img: np.ndarray, max_num: int = 10):
    """
    img shape :CHW / HW, 可视化一个通道
    max_num:最大可视化数量
    """
    assert 2<=len(img.shape)<=3, f'shape: {img.shape}'
    print('shape:',img.shape)

    if len(img.shape)==2:
        plt.figure(figsize=(3,3))
        plt.imshow(img,cmap='gray')
        plt.show()
    else:
        columns = 5
        rows = math.ceil(min(max_num, img.shape[0])/columns)
        plt.figure(figsize=(10,2*rows))
        for plt_index in range(1, min(max_num, img.shape[0])+1):
            plt.subplot(rows, columns, plt_index)
            plt.imshow(img[plt_index-1,:,:], cmap='gray')

        plt.tight_layout()
        plt.show()

def inv_normalize(img: np.ndarray) -> np.ndarray:
    """
    image[-1.0, 1.0] float => image[0, 255] uint8
    """
    return ((img*0.5+0.5)*255).astype(np.uint8)

def save_image4visualize(save_path: str, save_name: str, img: np.ndarray):
    """
    保存灰度图片(无需逆normalize，数据格式不限)，仅限用于可视化，考虑废弃
    """
    makedirs(save_path)
    plt.imsave(os.path.join(save_path, save_name), img, cmap='gray')

def save_image(save_path: str, save_name: str, img: np.ndarray):
    """
    保存图片为png.
    img: HW/HWC, np.uint8 [0,255]
    """
    assert img.dtype==np.uint8, 'The data type should be np.uint8.'
    Image.fromarray(img).save(os.path.join(save_path, save_name))

def save_image_dpi(save_path: str, save_name: str, img: np.ndarray, dpi: int=300):
    """
    保存灰度图片(无需逆normalize，数据格式不限)，仅限用于可视化，但是可以设定dpi(原图像尺寸将失效)
    """
    makedirs(save_path)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(7.0/3,7.0/3) #dpi = 300, output = 700*700 pixels
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    fig.savefig(os.path.join(save_path, save_name), format='png', transparent=True, dpi=dpi, pad_inches = 0)

def image_with_contours(img: np.array, mask: np.array, num_class: int) -> np.array:
    """
    A function that add contours to image.
    :param img: HW, np.uint8([0, 255])/np.float([0, 1.0])
    :param mask: HW, 最大值为num_class-1, np.uint8
    :param num_class: 类别数
    :return: HWC
    """
    assert img.shape==mask.shape
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    color_rgb=[(255,0,0),(0,255,0),(0,0,255),(0,255,255),(255,0,255)]
    for i in range(num_class-1):
        _, thresh = cv2.threshold(mask, i, 255, 0)
        contours, im = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #第一个参数是轮廓
        cv2.drawContours(image=img, contours=contours, contourIdx=-1, color=color_rgb[i], thickness=1)

    return img

def image_with_masks(img: np.array, mask: np.array, num_class: int, beta: float = 0.4) -> np.array:
    """
    A function that add mask to image.
    :param img: HW, np.uint8([0, 255])/np.float32([0, 1.0])
    :param mask: HW, 最大值为num_class-1, np.uint8
    :param num_class: 类别数
    :param beta: 透明度
    :return: HWC
    """
    assert img.shape==mask.shape

    if img.max()<=1.0:
        img = (img*255).astype(np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    mask_rgb = np.zeros_like(img)
    color_rgb=[(255,0,0),(0,255,0),(0,0,255),(0,255,255),(255,0,255)]
    
    for i in range(1, num_class):
        mask_rgb[mask==i] = np.array(color_rgb[i-1])
    
    img = cv2.addWeighted(img, 1, mask_rgb, beta, 0)

    return img    

class HiddenPrints:
    """
    A function that hidden the print.

    Example::

        # If you don't want display the information of `function()`
        with HiddenPrints():
            function()...
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
