import os 
import shutil
import torch

# input_path = "/mnt/disk1/data/Market1501-All/LowLight/Market1501_LowLight/Market-1501-v15.09.15/"
# output_path = "/mnt/disk1/data/Market1501-All/LowLight/Market1501_LowLight/market1501_x1/"

# for folder in os.listdir(input_path):
#   output_folder = os.path.join(output_path, folder)
#   os.makedirs(output_folder, exist_ok=True)
#   for image in os.listdir(os.path.join(input_path, folder)):
#     if image[-5] == '0':
#       shutil.copy(os.path.join(input_path, folder, image), os.path.join(output_folder, image))

# pretrain_path = "results/market1501_lowlight/resnet50_checkpoint_0.9983.pt"
# ckpt = torch.load(pretrain_path, map_location='cpu')
# state_dict = ckpt['model']
# print(ckpt.keys())
# print(state_dict["classifier.weight"].shape)

import re

img_name = "0101L4C5678"
pattern = re.compile(r'([-\d]+)([RL])(\d)')
print()