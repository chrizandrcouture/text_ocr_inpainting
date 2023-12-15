import os
import codecs
import json
import argparse


def translate_text(target: str, text: str) -> dict:
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """
    from google.cloud import translate_v2 as translate

    translate_client = translate.Client()

    if isinstance(text, bytes):
        text = text.decode("utf-8")

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(text, target_language=target)

    print("Text: {}".format(result["input"]))
    print("Translation: {}".format(result["translatedText"]))
    print("Detected source language: {}".format(result["detectedSourceLanguage"]))

    return result


target_languages = {
    "hi-IN":(4,"Hindi (हिन्दी)"),       #hindi
    "bn-IN":(7,"Bengali (বাংলা)"),     #bengali
    "gu-IN":(3,"Gujarati (ગુજરાતી)"),    #gujarati
    "mr-IN":(6,"Marathi (मराठी)"),       #marathi
    "ta-IN":(5,"Tamil (தமிழ்)"),          #tamil
    "te-IN":(2,"Telugu (తెలుగు)"),        #telugu
    "kn-IN":(0,"Kannada (ಕನ್ನಡ)"),       #kannada
    "ml-IN":(1,"Malayalam (മലയാളം)"),     #malayalam
}

parser = argparse.ArgumentParser()
parser.add_argument("--transcripts",type=str, help="source transcripts file path (.json)")
args = vars(parser.parse_args())

input_file = args["transcripts"] #'adani_group_regain_investor/parsed_transcripts/transcripts_english.json'
input_basename = os.path.dirname(input_file).split("/")[0]

output_dir = input_basename
output_dir = os.path.join(output_dir, "transcripts")
os.makedirs(output_dir, exist_ok=True)


input_transcripts = json.load(open(input_file, "rb"))
for lang in target_languages.keys():
    target_lang_transcripts = {
                                "source":{"language":"English"},
                                "target":{"language":target_languages[lang][1]},
                                "segments":[]
                            }
    for segment in input_transcripts["segments"]:
        ttt = segment["source"]["text"]
        tt = translate_text(lang, ttt)
        tt = tt["translatedText"]
        ind_ts = {
                    "source": segment["source"], 
                    "target": {
                        "text":tt,
                        "index":segment["source"]["index"], 
                        "timestamp_start": segment["source"]["timestamp_start"],
                        "timestamp_end": segment["source"]["timestamp_end"],
                        "duration": segment["source"]["duration"]
                    }
                }
        target_lang_transcripts["segments"].append(ind_ts)
    with codecs.open(os.path.join(output_dir, input_basename+"_"+str(target_languages[lang][0])+".json"), 'w', 'utf-8') as f:
        json.dump(target_lang_transcripts, f, ensure_ascii=False, indent=4)

