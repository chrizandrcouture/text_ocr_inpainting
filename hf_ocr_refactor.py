import argparse
import json
import os
import pdb

from PIL import Image
import torch
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

import subprocess


# def extract_frames(video_file, output_dir):
#     """Run the ffmpeg command to extract frames from the video file."""
#     command = f"ffmpeg -i {video_file} -vf \"fps=1\" -q:v 2 -vsync 0 -f image2 {output_dir}/output_%d.png"
#     subprocess.run(command, shell=True)


class OCR:
    def __init__(self, device, processor, model):
        self.device = device
        self.processor = processor
        self.model = model

    def main(self, result_folder):
        image_folders = [os.path.join(result_folder, f)
                         for f in os.listdir(result_folder)
                         if os.path.isdir(os.path.join(result_folder, f))]

        for input_folder in image_folders:
            output_file = os.path.join(input_folder, "ocr_result.json")
            image_list_base = [x for x in os.listdir(input_folder)
                               if x.endswith(".png") and "cropped_" in x]
            image_list = [os.path.join(input_folder, x) for x in image_list_base]

            with torch.no_grad():
                results = {}
                for i, img in enumerate(tqdm(image_list)):
                    image = Image.open(img).convert("RGB")
                    pixel_values = self.processor(images=image, return_tensors="pt").pixel_values

                    generated_ids = self.model.generate(pixel_values.to(self.device))
                    generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    results[image_list_base[i]] = generated_text

            json.dump(results, open(output_file, "w"), indent=4)

    def run(self, result_folder):
        """Run the OCR on the given result folder."""
        self.main(result_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--frames_folder', type=str, help='Path to extracted frames folder')
    parser.add_argument('--result_folder', type=str, help='Path to results folder')
    args = vars(parser.parse_args())

    video_file =args["video"]
    # frame_dir = args["frames_folder"]
    output_dir = args["result_folder"]

    # extract_frames(video_file, frame_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed').to(device)
    ocr = OCR(device, processor, model)
    ocr.run(output_dir)

    # USAGE:
    # python hf_ocr.py --result_folder "/data/chris/CRAFT-pytorch/result/us_frames_idx" --video "/data/chris/CRAFT-pytorch/frames/us_air/us_air_force.mp4"
    # python hf_ocr_refactor.py --result_folder "/data/chris/CRAFT-pytorch/result/test/us_frames_idx" --video "/data/chris/CRAFT-pytorch/frames/us_air/us_air_force.mp4"