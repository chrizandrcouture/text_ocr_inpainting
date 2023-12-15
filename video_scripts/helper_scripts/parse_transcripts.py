# -*- coding: utf-8 -*-
import json
import os
import codecs
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, help="transcripts input directory")
parser.add_argument("--output_dir", type=str, help="transcripts output directory")
args = vars(parser.parse_args())


def float_to_time(time):
    total=int(time/1)
    remainder=time*100-total*100
    remainder=remainder/100
    total,seconds=divmod(total,60)
    hours,minutes=divmod(total,60)

    # Format the time values with leading zeros
    formatted_time = "{:02d}:{:02d}:{:02d}.{:02d}".format(
        int(hours), int(minutes), int(seconds), int(remainder*100)
    )
    return formatted_time


if __name__=="__main__":

    sts_maps = {
        "0":"kannada",
        "1":"malayalam",
        "2":"telugu",
        "3":"gujarati",
        "4":"hindi",
        "5":"tamil",
        "6":"marathi",
        "7":"bengali"
    }

    json_files = os.listdir(args["input_dir"])
    json_files = [os.path.join(args["input_dir"], x) for x in json_files]

    os.makedirs(args["output_dir"], exist_ok=True)
    for json_file in json_files:
        print(f"Parsing {json_file} ...")
        with codecs.open(json_file,'r','utf-8') as f:
            in_dict=json.load(f)

        segments=in_dict['segments']
        out={"transcripts":[],"timestamps":[]}
        for i in segments:
            out['transcripts'].append(i['target']['text'])
            out["timestamps"].append((float_to_time(i['target']["timestamp_start"]), float_to_time(i['target']["timestamp_end"])))

        file_ind = str((os.path.splitext(json_file)[0]).split("_")[-1])
        with codecs.open(os.path.join(args["output_dir"], "transcripts_"+sts_maps[file_ind]+".json"), 'w', 'utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=4)

        print(f"Parsing {json_file} complete.")
