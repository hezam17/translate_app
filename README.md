# Server for Speech Translation and Synthesis

This project provides a Flask server for performing speech translation and synthesis using machine learning models.

## Features

- Accepts audio clips for translation and synthesis via HTTP POST requests.
- Supports multiple source and target languages.
- Utilizes a pre-trained speech recognition and translation model.
- Generates synthesized audio clips based on translated text.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/hezam17/translate_app.git

## Dependencies

The server relies on the following Python packages:

- **Flask**: A lightweight web framework for building web applications.
- **speech\_recognition**: Library for performing speech recognition.
- **googletrans**: Google Translate API wrapper for Python.
- **torch**: PyTorch, a deep learning framework.
- **torchaudio**: Audio processing library for PyTorch.
- **TTS**: Text-to-Speech library for generating synthesized audio.
- **base64**: Encoding and decoding functions for base64 data.
- **datetime**: Module for working with dates and times.
- **os**: Module for interacting with the operating system.
- **scipy**: Library for scientific computing.
- **numpy**: Fundamental package for numerical computing.
- **requests**: Library for making HTTP requests.

## Testing the Server
To test the server, you can use the provided Python script test_server.py.
Modify the script to set the URL of your Flask API endpoint and the path to the audio file you want to send. Then, run the script:

