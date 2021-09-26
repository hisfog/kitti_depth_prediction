from PIL import Image
import torch
import numpy as np
import random

def random_crop(img, height, width):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    assert img.shape[0] == depth.shape[0]
    assert img.shape[1] == depth.shape[1]
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y:y+height , x:x+width, :]
    #confirmed final input size is height*width
    return img
    #np.ndarray to np.ndarray

def random_flip(image):
    # do random flipping
    do_flip = random.random()
    if do_flip > 0.5:
        image = (image[:, ::-1, :]).copy()
    return image

def gamma_augment(image):
    gamma = random.uniform(0.9, 1.1)
    image = image ** gamma
    return image

def brightness_augment(image):
    brightness = random.uniform(0.9, 1.1)
    image = image * brightness
    return image

def color_augment(image, colors):
    colors = np.random.uniform(0.9, 1.1, size=3)
    white = np.ones((image.shape[0], image.shape[1]))
    color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
    image *= color_image
    image = np.clip(image, 0, 1)
    return image

def random_rotate(image, angle, flag=Image.BILINEAR):
    #PIL.Image -> PIL.Image
    random_angle = (random.random() - 0.5) * 2 * self.args.degree
    result = image.rotate(angle, resample=flag)
    return result
