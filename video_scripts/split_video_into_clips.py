import os
import subprocess
import pdb
import argparse
import json


target_languages = {
    	"hi-IN": 4, #hindi
    	"bn-IN": 7, #bengali
    	"gu-IN": 3, #gujarati
    	"mr-IN": 6, #marathi
    	"ta-IN": 5, #tamil
    	"te-IN": 2, #telugu
    	"kn-IN": 0, #kannada
    	"ml-IN": 1, #malayalam
}

sts_maps = {
        "hi-IN": "hindi",
        "bn-IN": "bengali",
        "gu-IN": "gujarati",
        "mr-IN": "marathi",
        "ta-IN": "tamil",
        "te-IN": "telugu",
        "kn-IN": "kannada",
        "ml-IN": "malayalam"
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="input video path (mp4)")
    args = vars(parser.parse_args())

    input_file = args["input_file"] #'LSGvsMI_highlights/LSGvsMI_highlights.mp4'

    base_dir = os.path.dirname(input_file)
    transcripts_folder = os.path.join(base_dir, "parsed_transcripts")

    for lang in target_languages.keys():
        transcripts_file = json.load(open(os.path.join(transcripts_folder, f"transcripts_{sts_maps[lang]}.json"), "rb"))
        timestamps = transcripts_file["timestamps"]
        # pdb.set_trace()

        output_dir = os.path.join(base_dir, "source_clips", lang)
        os.makedirs(output_dir, exist_ok=True)
        for i, (start_time, end_time) in enumerate(timestamps, start=1):
            output_file = os.path.join(output_dir, "clip"+str(i)+".mp4")
            # pdb.set_trace()
            command = [
                'ffmpeg', "-y", '-ss', start_time, '-to', end_time, '-i', input_file, '-c', 'copy', output_file
                ]
            # pdb.set_trace()
            subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f'Clip {i} created: {output_file}')
