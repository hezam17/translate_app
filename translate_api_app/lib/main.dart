import 'dart:io';
import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: AudioPage(),
    );
  }
}

class AudioPage extends StatefulWidget {
  @override
  _AudioPageState createState() => _AudioPageState();
}

class _AudioPageState extends State<AudioPage> {
  FlutterSoundRecorder? _recorder;
  FlutterSoundPlayer? _player;
  bool _isRecording = false;
  bool _isPlaying = false;
  String _path = '';
  String _recognizedText = '';
  String _translatedText = '';
  Uint8List? _audioClipBytes;
  String _selectedSourceLanguage = 'English';
  String _selectedTargetLanguage = 'Arabic';

  List<String> _languages = [
    'English',
    'Arabic',
    'Russian',
    'Spanish',
  ];
  List<String> _languagesCode = [
    'en',
    'ar',
    'ru',
    'es',
  ];

  String _responseData = '';
  @override
  void initState() {
    super.initState();
    _recorder = FlutterSoundRecorder();
    _player = FlutterSoundPlayer();
    initRecorder();
    initPlayer();
    requestPermissions();
  }

  Future<void> requestPermissions() async {
    await [
      Permission.microphone,
      Permission.storage,
    ].request();
  }

  Future<void> initRecorder() async {
    await _recorder!.openRecorder();
    _recorder!.setSubscriptionDuration(const Duration(milliseconds: 500));
  }

  Future<void> initPlayer() async {
    await _player!.openPlayer();
  }

  Future<void> startRecording() async {
    try {
      final directory = await getApplicationDocumentsDirectory();
      final path = directory.path + '/flutter_sound_record.wav';

      await _recorder!.startRecorder(
        toFile: path,
        codec: Codec.pcm16WAV,
      );
      setState(() {
        _isRecording = true;
        _path = path; // Store the full path
      });
    } catch (e) {
      print('Failed to start recording: $e');
      // Optionally, handle the error e.g., by showing a notification or alert
    }
  }

  Future<void> stopRecording() async {
    await _recorder!.stopRecorder();
    setState(() {
      _isRecording = false;
    });
    sendFileToApi(_path);
  }

  Future<void> startPlaying() async {
    await _player!.startPlayer(
        fromURI: _path,
        codec: Codec.pcm16WAV,
        whenFinished: () {
          setState(() {
            _isPlaying = false;
          });
        });
    setState(() {
      _isPlaying = true;
    });
  }

  Future<void> stopPlaying() async {
    await _player!.stopPlayer();
    setState(() {
      _isPlaying = false;
    });
  }

  Future<void> sendFileToApi(String filePath) async {
    try {
      var uri = Uri.parse('http://10.0.2.2:5000/process_clip');
      var request = http.MultipartRequest('POST', uri)
        ..files.add(await http.MultipartFile.fromPath('audio_clip', filePath))
        ..fields['source_language'] =
            _languagesCode[_languages.indexOf(_selectedSourceLanguage)]
        ..fields['target_language'] =
            _languagesCode[_languages.indexOf(_selectedTargetLanguage)];

      // Debug: Log request payload
      print('Sending request:');
      print('Files: ${request.files}');
      print('Fields: ${request.fields}');

      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);

      // Debug: Log response
      print('Received response:');
      print('Status code: ${response.statusCode}');
      print('Body: ${response.body}');

      if (response.statusCode == 200) {
        print('Upload successful');
        await _processResponse(response);
      } else {
        print('Failed to upload. Status: ${response.statusCode}');
      }
    } catch (e) {
      print('Failed to connect: $e');
    }
  }

  Future<void> _processResponse(http.Response response) async {
    var _responseData = response.body;

    try {
      final decodedResponse = json.decode(_responseData);
      print('Decoded Response: $decodedResponse');
      _updateUI(decodedResponse);
    } catch (e) {
      print('Error parsing response: $e');
    }
  }

  void _updateUI(Map<String, dynamic> decodedResponse) {
    setState(() {
      _recognizedText = decodedResponse['recognized_text'];
      _translatedText = decodedResponse['translated_text'];
      _audioClipBytes = base64Decode(decodedResponse['audio_clip']);
    });
    playAudioFromBytes(_audioClipBytes!); // Play the audio
  }

  void playAudioFromBytes(Uint8List bytes) async {
    try {
      await _player!.startPlayer(
        codec: Codec.pcm16WAV,
        fromDataBuffer: bytes,
        whenFinished: () {
          print("Finished playing");
        },
      );
    } catch (e) {
      print('Error playing audio: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        centerTitle: true,
        elevation: 0, // Removes the shadow under the AppBar for a flat design.
        title: const Text(
          'Translate',
          style: TextStyle(
            fontWeight: FontWeight.bold,
            fontSize: 28,
          ),
        ),
      ),
      backgroundColor: Colors.grey[200],
      body: Column(
        mainAxisAlignment: MainAxisAlignment.start, // Align content to the top
        crossAxisAlignment: CrossAxisAlignment.center,
        children: <Widget>[
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // Source language dropdown
              Expanded(
                child: DropdownButtonHideUnderline(
                  child: DropdownButton<String>(
                    isExpanded: true,
                    value: _selectedSourceLanguage,
                    items: _languages
                        .map<DropdownMenuItem<String>>((String value) {
                      return DropdownMenuItem<String>(
                        value: value,
                        child: Container(
                          padding: const EdgeInsets.symmetric(horizontal: 10),
                          child: Text(value, textAlign: TextAlign.center),
                        ),
                      );
                    }).toList(),
                    onChanged: (String? newValue) {
                      setState(() {
                        _selectedSourceLanguage = newValue!;
                      });
                    },
                  ),
                ),
              ),
              IconButton(
                icon: Icon(Icons.swap_horiz),
                onPressed: () {
                  setState(() {
                    final temp = _selectedSourceLanguage;
                    _selectedSourceLanguage = _selectedTargetLanguage;
                    _selectedTargetLanguage = temp;
                  });
                },
              ),
              // Target language dropdown
              Expanded(
                child: DropdownButtonHideUnderline(
                  child: DropdownButton<String>(
                    isExpanded: true,
                    value: _selectedTargetLanguage,
                    items: _languages
                        .map<DropdownMenuItem<String>>((String value) {
                      return DropdownMenuItem<String>(
                        value: value,
                        child: Container(
                          padding: const EdgeInsets.symmetric(horizontal: 10),
                          child: Text(value, textAlign: TextAlign.center),
                        ),
                      );
                    }).toList(),
                    onChanged: (String? newValue) {
                      setState(() {
                        _selectedTargetLanguage = newValue!;
                      });
                    },
                  ),
                ),
              ),
            ],
          ),

          SizedBox(
              height:
                  20), // Add spacing between language selection and other content
          if (_isRecording)
            ElevatedButton(
              onPressed: stopRecording,
              child: Text('Stop Recording'),
            )
          else
            ElevatedButton(
              onPressed: startRecording,
              child: Text('Start Recording'),
            ),
          SizedBox(height: 20),
          Text('Recognized Text: $_recognizedText'),
          SizedBox(height: 10),
          if (_isPlaying)
            ElevatedButton(
              onPressed: stopPlaying,
              child: Text('Stop Playing'),
            )
          else
            ElevatedButton(
              onPressed: startPlaying,
              child: Text('Start Playing'),
            ),
          SizedBox(height: 20),
          Text('Translated Text: $_translatedText'),
          SizedBox(height: 20),
          if (_audioClipBytes != null)
            ElevatedButton(
              onPressed: () => playAudioFromBytes(_audioClipBytes!),
              child: Text('Play Translated Audio'),
            ),
        ],
      ),
    );
  }
}
