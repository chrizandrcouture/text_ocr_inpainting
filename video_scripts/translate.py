import requests
import json
import uuid
import os
import pdb
import argparse
from tqdm import tqdm

# Add your key and endpoint
# key = "cde8702cf79b4cfca5e9ca9c96b3f11d"
key = "3e779e02a76541f6b59472d29d83695f"

endpoint = "https://api.cognitive.microsofttranslator.com"
location = "eastus"
path = '/translate'
constructed_url = endpoint + path

langs = {
        "hi": "hindi",
        "bn": "bengali",
        "gu": "gujarati",
        "mr": "marathi",
        "ta": "tamil",
        "te": "telugu",
        "kn": "kannada",
        "ml": "malayalam"
}

headers = {
    'Ocp-Apim-Subscription-Key': key,
    # location required if you're using a multi-service or regional (not global) resource.
    'Ocp-Apim-Subscription-Region': location,
    'Content-type': 'application/json',
    'X-ClientTraceId': str(uuid.uuid4())
}

params = {
    'api-version': '3.0',
    'from': 'en',
    'to': list(langs.keys())
}

def translate(input_transcript, output_folder):
    global langs
    with open(input_transcript, "r") as f:
        data = json.load(f)

    translated = {k:{"transcripts": [], "timestamps": []} for k in langs.keys()}

    for timestamp, text in tqdm(data.items()):
        text = text.lower()
        text = text.replace("\n", " ")
        body = [{'text': text}]

        request = requests.post(constructed_url, params=params, headers=headers, json=body)
        response = request.json()
        st, et = timestamp.split("-")
        st = st + ".00"
        et = et + ".00"

        for translation in response[0]["translations"]:
            lang = translation["to"]
            transcript = translation["text"]
            translated[lang]["transcripts"].append(transcript)
            translated[lang]["timestamps"].append([st, et])

    for lang, transcripts in translated.items():
        output_file = os.path.join(output_folder, f"transcripts_{langs[lang]}.json")
        json.dump(transcripts, open(output_file, "w", encoding='utf8'), indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript", type=str, default="transcripts.json", help="transcripts file (json)")
    parser.add_argument("--output_folder", type=str, default="tts_output/", help="directory to store generated speech")
    args = vars(parser.parse_args())

    input_transcript = "/data/chris/CRAFT-pytorch/result/us_frames_idx/generated_transcript.json"
    output_folder = "/data/lucky/video_dubbing/data/us_air/parsed_transcripts"
    translate(input_transcript, output_folder)