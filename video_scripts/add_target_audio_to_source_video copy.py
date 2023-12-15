import os
import subprocess
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input",type=str, help="input file path (.mp4)")
args = vars(parser.parse_args())

input_file = args["input"] #'adani_group_regain_investor/adani_group_regain_investor.mp4'

base_dir = os.path.dirname(input_file)

source_clips = os.listdir(os.path.join(base_dir, "source_clips"))
source_clips = [x for x in source_clips if x.startswith("clip") if x.endswith("mp4")]
source_clips = [os.path.join(base_dir, "source_clips", "clip"+str(i+1)+".mp4") for i in range(len(source_clips))]
print("source clips: ", source_clips)

dubbed_clips_dir = os.path.join(base_dir, "dubbed_clips_with_sst")
if os.path.exists(dubbed_clips_dir):
    shutil.rmtree(dubbed_clips_dir)
os.makedirs(dubbed_clips_dir, exist_ok=True)

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

for lang in target_languages.keys():
    target_clips = os.listdir(os.path.join(base_dir, "output", lang))
    target_clips = sorted([os.path.join(base_dir, "output", lang, x) for x in target_clips])
    print("target clips: ", lang, target_clips)

    dub_out_dir = os.path.join(dubbed_clips_dir, lang)
    os.makedirs(dub_out_dir, exist_ok=True)

    for i, (src, trg) in enumerate(zip(source_clips, target_clips), start=1):
        output_file = os.path.join(dub_out_dir, "output"+str(i)+".mp4")

        #ffmpeg -i clip1.mp4 -i chunk1.mp3 -c:v copy -c:a aac -map 0:v -map 1:a -shortest output1.mp4
        command = [
            "ffmpeg", "-i", src, "-i", trg, "-c:v", "copy", "-c:a", "aac",\
            "-map", "0:v", "-map", "1:a", output_file
        ]
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f'Clip {i} created: {output_file}')

    # ffmpeg -i output1.mp4 -i output2.mp4 -i output3.mp4 -i output4.mp4 -i output5.mp4 -i output6.mp4 -i output7.mp4 -i output8.mp4 -i output9.mp4 -i output10.mp4 -i output11.mp4 \
    # -filter_complex "[0:v][0:a][1:v][1:a][2:v][2:a][3:v][3:a][4:v][4:a][5:v][5:a][6:v][6:a][7:v][7:a][8:v][8:a][9:v][9:a][10:v][10:a]concat=n=11:v=1:a=1[outv][outa]" \
    # -map "[outv]" -map "[outa]" -c:v libx264 -c:a aac merged_output.mp4

    dubbed_clips = os.listdir(dub_out_dir)
    dubbed_clips = [x for x in dubbed_clips if x.startswith("output")]
    dubbed_clips = [os.path.join(dub_out_dir, "output"+str(i+1)+".mp4") for i in range(len(dubbed_clips))]
    print("dubbed clips: ", lang, dubbed_clips)

    merged_output_file = input_file.replace(".mp4", "_"+str(target_languages[lang])+".mp4")

    # Build the filter_complex argument
    filter_complex = ''
    for i, file in enumerate(dubbed_clips):
        filter_complex += f'[{i}:v][{i}:a]'

    filter_complex += f'concat=n={len(dubbed_clips)}:v=1:a=1[outv][outa]'
    # Build the command
    command = [
        'ffmpeg', '-y'
    ]

    for file in dubbed_clips:
        command.extend(['-i', file])

    command.extend([
        '-filter_complex', filter_complex,
        '-map', '[outv]',
        '-map', '[outa]',
        '-c:v', 'libx264',
        '-c:a', 'aac',
        merged_output_file
    ])

    # Execute the command
    code = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(code)
    print('Merged output created: ', merged_output_file)
