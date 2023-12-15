import os
import argparse
import json

from google.cloud import texttospeech

parser = argparse.ArgumentParser()
parser.add_argument("--transcripts", type=str, default="transcripts.json", help="transcripts file (json)")
parser.add_argument("--output_dir", type=str, default="tts_output", help="directory to store generated speech")
parser.add_argument("--lang", type=str, default="hi", help="glcoud tts sdk language code")
parser.add_argument("--speaker", type=str, default="hi-IN-Standard-C", help="glcoud tts sdk speaker type")
parser.add_argument("--gender", type=str, default="male", help="glcoud tts sdk language gender choice")
args = vars(parser.parse_args())


def translatedspeech(x, language, speaker_type, gender, chunk_num, output_dir):

    # Instantiates a client
    client = texttospeech.TextToSpeechClient()

    # Set the text input to be synthesized
    synthesis_input = texttospeech.types.SynthesisInput(text=x)

    # Build the voice request, select the language code ("en-US") and the ssml
    # voice gender ("neutral")
    if gender=="male" or gender=="MALE" or gender=="Male":
    	ssml_gender = texttospeech.enums.SsmlVoiceGender.MALE
    elif gender=="female" or gender=="FEMALE" or gender=="FeMale":
    	ssml_gender = texttospeech.enums.SsmlVoiceGender.FEMALE
    voice = texttospeech.types.VoiceSelectionParams(
        language_code=language,
        name=speaker_type,
        ssml_gender=ssml_gender)

    # Select the type of audio file you want returned
    audio_config = texttospeech.types.AudioConfig(
        audio_encoding=texttospeech.enums.AudioEncoding.MP3)

    # Perform the text-to-speech request on the text input with the selected
    # voice parameters and audio file type
    response = client.synthesize_speech(synthesis_input, voice, audio_config)

    # The response's audio_content is binary.
    output_filename = os.path.join(output_dir, "chunk"+chunk_num+".mp3")
    with open(output_filename, 'wb+') as out:
        # Write the response to the output file.
        out.write(response.audio_content)
        #print('Audio content written to file "output.mp3"')
    out.close()



if __name__ == "__main__":
	transcripts = json.load(open(args["transcripts"],"rb"))
	output_dir = os.path.join(args["output_dir"], "target_clips", args["lang"])
	os.makedirs(output_dir, exist_ok=True)
	for i,chunk_text in enumerate(transcripts["transcripts"]):
		# translatedspeech(t2, 'te', 'te-IN-Standard-B', str(i))
		# translatedspeech(chunk_text, 'hi', 'hi-IN-Standard-C', str(i))
		# translatedspeech(chunk_text, 'bn', 'bn-IN-Standard-A', str(i))
		# translatedspeech(chunk_text, 'bn', 'bn-IN-Wavenet-B', str(i))
		translatedspeech(chunk_text, args["lang"], args["speaker"], args["gender"], str(i+1), output_dir)

