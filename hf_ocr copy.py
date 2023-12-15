import argparse
import json
import os
import pdb

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import cv2
from PIL import Image
import torch
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed').to(device)

pdb.set_trace()
def main(result_folder):
    image_folders = [os.path.join(result_folder, f) for f in os.listdir(result_folder) if os.path.isdir(os.path.join(result_folder, f))]

    for input_folder in image_folders:
        output_file = os.path.join(input_folder, "ocr_result.json")
        image_list_base = [x for x in os.listdir(input_folder) if x.endswith(".png") and "cropped_" in x]
        image_list = [os.path.join(input_folder, x) for x in image_list_base]

        with torch.no_grad():
            results = {}
            for i, img in enumerate(tqdm(image_list)):
                image = Image.open(img).convert("RGB")
                pixel_values = processor(images=image, return_tensors="pt").pixel_values

                generated_ids = model.generate(pixel_values.to(device))
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                results[image_list_base[i]] = generated_text

        json.dump(results, open(output_file, "w"), indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",type=str, help="input file path (.mp4)")
    parser.add_argument('--video', type=str, help='Path to video file')
    args = vars(parser.parse_args())

    result_folder = "/data/chris/CRAFT-pytorch/result/us_frames_idx"
    main(result_folder)
    # pdb.set_trace()

# python hf_ocr.py --input "/data/chris/CRAFT-pytorch/result/russia_frames_idx" --video "/data/chris/CRAFT-pytorch/russia_police/russia_police.mp4"
# ffmpeg -i test.mp4 -vf "fps=1" -q:v 2 -vsync 0 -f image2 raw_frames/output_%d.png
# import os, shutil
# fps = 30
# imgs = [x for x in os.listdir("raw_frames/") if x.endswith(".png")]
# img_ids = [int(x.split("_")[1].replace(".png", "")) for x in imgs]
# sorted_imgs = sorted(list(zip(imgs, img_ids)), key=lambda x:x[1])
# new_names = [f"{x[0].split('_')[0]}_{x[1]*fps}.png" for x in sorted_imgs]
# old_imgs = [os.path.join("raw_frames/", x[0]) for x in sorted_imgs]
# new_names = [os.path.join("us_frames_idx/", x) for x in new_names]
# for o, n in zip(old_imgs, new_names):
#     shutil.copy(o, n)