from paddleocr import PaddleOCR, draw_ocr
import pdb
import os
import json
from PIL import Image
import file_utils
from tqdm import tqdm

model = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
# img_path = './imgs_en/img_12.jpg'
# result = model.ocr(img_path, cls=True)
# for idx in range(len(result)):
#     res = result[idx]
#     for line in res:
#         print(line)
pdb.set_trace()



input_folder = "/data/chris/CRAFT-pytorch/result/keyframe_519"
output_file = os.path.join(input_folder, "ocr_result.json")

image_list_base = [x for x in os.listdir(input_folder) if x.endswith(".png") and "cropped_" in x]
image_list = [os.path.join(input_folder, x) for x in image_list_base]


results = {}
for i, img in enumerate(tqdm(image_list)):
    result = model.ocr(img, cls=True)
    results[image_list_base[i]] = result
json.dump(results, open(output_file, "w"), indent=4)


pdb.set_trace()
    # for idx in range(len(result)):
    #     res = result[idx]
    #     for line in res:
    #         print(line)