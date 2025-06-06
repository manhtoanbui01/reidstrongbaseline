# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import torchvision.transforms as T

from .transforms import RandomErasing

def to_grayscale_3ch(img):
        img = T.Grayscale(num_output_channels=1)(img)
        return img.convert("RGB")  # Duplicate grayscale to 3 channels

def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            to_grayscale_3ch,
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    else:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            to_grayscale_3ch,
            T.ToTensor(),
            normalize_transform
        ])

    return transform

# from PIL import Image
# import torchvision.transforms.functional as F
# import torch

# class ResizeAndPad:
#     def __init__(self, size, fill=0):
#         """
#         size: (height, width) tuple for final output size
#         fill: padding color (default black)
#         """
#         self.size = size
#         self.fill = fill

#     def __call__(self, img):
#         # Resize while preserving aspect ratio (shorter side = target shorter side)
#         img = self.resize_preserve_aspect_ratio(img, self.size)

#         # Pad image to target size
#         img = self.pad_to_target_size(img, self.size, self.fill)

#         return img

#     def resize_preserve_aspect_ratio(self, img, target_size):
#         w, h = img.size
#         target_h, target_w = target_size

#         scale = min(target_w / w, target_h / h)  # scale so image fits inside target size
#         new_w, new_h = int(w * scale), int(h * scale)
#         return img.resize((new_w, new_h), Image.BILINEAR)

#     def pad_to_target_size(self, img, target_size, fill):
#         new_w, new_h = img.size
#         target_h, target_w = target_size

#         pad_left = (target_w - new_w) // 2
#         pad_top = (target_h - new_h) // 2
#         pad_right = target_w - new_w - pad_left
#         pad_bottom = target_h - new_h - pad_top

#         padding = (pad_left, pad_top, pad_right, pad_bottom)
#         return F.pad(img, padding, fill=fill)
    
# def build_transforms(cfg, is_train=True):
#     normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
#     target_size = cfg.INPUT.SIZE_TRAIN if is_train else cfg.INPUT.SIZE_TEST

#     if is_train:
#         transform = T.Compose([
#             ResizeAndPad(target_size),
#             T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
#             T.ToTensor(),
#             normalize_transform,
#             RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
#         ])
#     else:
#         transform = T.Compose([
#             ResizeAndPad(target_size),
#             T.ToTensor(),
#             normalize_transform
#         ])

#     return transform
