import pdb
import os
import json
from tqdm import tqdm
import torch

import cv2
import pytesseract


from PIL import Image
import requests

input_folder = "/data/chris/CRAFT-pytorch/result/raw_frames_idx/output_1250"
output_file = os.path.join(input_folder, "ocr_result.json")

image_list_base = [x for x in os.listdir(input_folder) if x.endswith(".png") and "cropped_" in x]
image_list = [os.path.join(input_folder, x) for x in image_list_base]

with torch.no_grad():
    results = {}
    for i, img in enumerate(tqdm(image_list)):
        image = cv2.imread(img)
        result = pytesseract.image_to_string(img)
        # pdb.set_trace()
        results[image_list_base[i]] = result

json.dump(results, open(output_file, "w"), indent=4)


pdb.set_trace()
    # for idx in range(len(result)):
    #     res = result[idx]
    #     for line in res:
    #         print(line)