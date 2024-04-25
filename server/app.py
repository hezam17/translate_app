from datetime import datetime
from flask import Flask, request, jsonify
import os
import speech_recognition as sr
from googletrans import Translator
from scipy.io.wavfile import read as read_wav
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import base64

app = Flask(__name__)

# Load model and configuration
config = XttsConfig()
config.load_json("model/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="model")

@app.route('/process_clip', methods=['POST'])
def process_clip():
    data = {}
    # Get audio clip from request
    audio_clip = request.files['audio_clip']
    target_language = request.form['target_language']  # Get target language from request
    source_language = request.form['source_language']  # Get source language from request

    # Get the current timestamp
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Construct input and output file names using the obtained source and target languages
    input_clip_filename = f"Lang_{source_language}_{current_time}_input.wav"
    output_clip_filename = f"Lang_{target_language}_{current_time}_output.wav"
    
    # Define the file paths using raw string literals
    input_clip_folder = "input_clips/"
    output_clip_folder = "output_clips/"
    input_clip_path = os.path.join(input_clip_folder, input_clip_filename)
    output_clip_path = os.path.join(output_clip_folder, output_clip_filename)

    # Save the audio file to the input folder
    with open(input_clip_path, 'wb') as f:
        f.write(audio_clip.read())
    
    # Perform speech recognition
    recognized_text = recognize_speech(input_clip_path)

    # Translate recognized text
    translated_text = translate_text(recognized_text, target_language)

    # Synthesize audio and save to the output folder
    output_file = synthesize_audio(translated_text, input_clip_path, source_language, target_language, output_clip_path)

    with open(output_file, 'rb') as audio_file:
        data["audio_clip"] = base64.b64encode(audio_file.read()).decode('utf-8')

    data["translated_text"] = translated_text
    data["recognized_text"] = recognized_text

    # Return the recognized and translated text in the response
    return jsonify(data)

def recognize_speech(audio_clip):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_clip) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
    return text

def translate_text(text, target_language):
    translator = Translator()
    translated_text = translator.translate(text, dest=target_language).text
    return translated_text

def synthesize_audio(text, audio_file, source_language, target_language, output_clip_path):
    # Compute speaker latents
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_file)

    # Inference
    chunks = model.inference_stream(text, source_language, gpt_cond_latent, speaker_embedding)

    wav_chunks = []
    for i, chunk in enumerate(chunks):
        wav_chunks.append(chunk)
    
    wav = torch.cat(wav_chunks, dim=0)
    
    # Save the audio file to the output folder
    torchaudio.save(output_clip_path, wav.squeeze().unsqueeze(0).cpu(), 24000)
    print("saved")

    return output_clip_path

if __name__ == '__main__':
    app.run(debug=True)
