
import io
import queue
import time
from datetime import datetime
import json

import pyaudio
import numpy as np
import openwakeword
from openwakeword.model import Model
from faster_whisper import WhisperModel
import soundfile as sf

class SpeechProcessor():
    def __init__(self, status_callback, callback_context):
        self.status_callback = status_callback
        self.callback_context = callback_context

        self.log_prefix = f'{self.__class__.__name__}:'

        self.queue = queue.Queue()

        self.oww_model = None
        self.oww_model_framework = 'onnx'

        self.input_stream = None
        self.def_audio_dev_name = 'ReSpeaker'
        self.channels = 1
        self.audio_chunk_size = 1280
        self.sample_rate = 16000

        self.chunk_period_s = self.audio_chunk_size/self.sample_rate

        self.max_record_duration_s = 10.0
        self.max_record_chunks = int(self.max_record_duration_s/self.chunk_period_s)
        print(f'{self.log_prefix} Max record chunks: {self.max_record_chunks}')

        self.stop_record_delay_after_no_vad_s = 2.0
        self.stop_record_delay_after_no_vad_chuck_cnt = int(self.stop_record_delay_after_no_vad_s/self.chunk_period_s)
        print(f'{self.log_prefix} Stop record delay after no vad (chunks): {self.stop_record_delay_after_no_vad_chuck_cnt}')

        self.speech_recog_active = False

        self.whisper_language = 'en'
        self.whisper_model_path = '/opt/whisper/models/turbo'
        self.whisper_model = WhisperModel(self.whisper_model_path, device="cuda", compute_type="int8_float16")

    def open_wakeword_model(self, ww_name, ww_model_path):
        self.oww_model = Model(wakeword_models=[ww_model_path], inference_framework=self.oww_model_framework)
        print(f"{self.log_prefix} opened ww mode, name: {ww_name}, path: {ww_model_path}")

    def open_input_stream(self, device_name):
        # Open microphone stream
        FORMAT = pyaudio.paInt16
        audio = pyaudio.PyAudio()

        device_index = None
        if device_name is not None:
            for i in range(audio.get_device_count()):
                info = audio.get_device_info_by_index(i)
                print(f'{self.log_prefix} audio dev info: {info}')
                if device_name in info['name'] and info['maxInputChannels'] > 0:
                    device_index = info['index']
                    print(f'{self.log_prefix} Using device {info["name"]}, index {device_index} chans: {info["maxInputChannels"]}')
                    break

        self.input_stream = audio.open(format=FORMAT, channels=self.channels, rate=self.sample_rate, input=True, input_device_index=device_index)

    def start_speech_to_text(self, audio):
        print(f'{self.log_prefix} starting speech to text, num samples: {len(audio)}')

        success = False
        try:
            audio_data_array: np.ndarray = np.frombuffer(audio, np.int16).astype(np.float32) / 0x7fff
            sf.write("/jetson_ws/speech.wav", audio_data_array, 16000)

            segments, info = self.whisper_model.transcribe(audio_data_array, language=self.whisper_language, beam_size=5)
            #print(f"{self.log_prefix} Detected language '{info.language}' with probability {info.language_probability}")

            text = ""
            for segment in segments:
                #print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
                text += segment.text
            print(f'{self.log_prefix} finished speech-to-text: {text}')
            self.status_callback(self.callback_context, {"msg": "stt_ok", "text": text})

        except Exception as ex:
            print(f'{self.log_prefix} STT exception: {ex}')
            self.status_callback(self.callback_context, {"msg": "stt_failed", "error_data": str(ex)})

        self.speech_recog_active = False

    def reset_recording(self):
        self.recording_chunk_cnt = 0
        self.recording_no_vad_chunk_cnt = 0
        self.recording_stop_listening = False
        self.recording = False
        self.recording_buffer = b""
        self.vad_during_recording = False

    def worker(self):
        print("{self.log_prefix} worker thread started")

        vad = openwakeword.VAD()

        self.open_input_stream(self.def_audio_dev_name)

        vad_active = False
        ww_active = False

        self.reset_recording()

        while True:
            try:
                cmd = self.queue.get_nowait()
                print(f"{self.log_prefix} Received cmd '{str(cmd)}'...")
                          
                if cmd['cmd'] == 'set_wake_word':
                  self.open_wakeword_model(cmd['args'].ww_name, cmd['args'].ww_model_path)

                elif cmd['cmd'] == 'speech_recognizer_start':
                    if self.speech_recog_active:
                        print('{self.log_prefix} Speech recog already active')
                    else:
                        self.max_record_chunks = int(cmd['timeout']/self.chunk_period_s)
                        self.speech_recog_active = True
                        self.reset_recording()
                        self.recording = True
                        print('{self.log_prefix} Speech recog recording start')
                elif cmd['cmd'] == 'speech_recognizer_cancel':
                    if self.speech_recog_active and self.recording:
                        self.speech_recog_active = False
                        self.reset_recording()
                        print('{self.log_prefix} Speech recog cancelled')

                elif cmd['cmd'] == 'speech_recognizer_finish':
                    if self.speech_recog_active and self.recording:
                        self.recording_stop_listening = True
                        print('{self.log_prefix} Speech recog recording finished')

                self.queue.task_done()
            except queue.Empty:
                pass

            try:
                chunk = self.input_stream.read(self.audio_chunk_size, exception_on_overflow=False)
            except Exception as ex:
                print(f"{self.log_prefix} Exception while reading stream: {ex}")
                continue

            audio = np.frombuffer(chunk, dtype=np.int16) [0::self.channels]

            try:
                vad(audio)
                # Consider last 10 frames (1280/16000*10= 800ms)
                vad_frames = list(vad.prediction_buffer)[-10:]
                vad_max_score = np.max(vad_frames) if len(vad_frames) > 0 else 0

                cur_vad = vad_max_score > 0.5
                if cur_vad != vad_active:
                    vad_active = cur_vad
                    print(f'{self.log_prefix} vad change: {cur_vad}')
                    self.status_callback(self.callback_context, {"msg": "vad", "active": bool(vad_active)})

                if self.recording:
                    if cur_vad:
                        self.vad_during_recording = True

                    self.recording_buffer += audio.tobytes()
                    self.recording_chunk_cnt += 1

                    if cur_vad:
                        self.recording_no_vad_chunk_cnt = 0
                    else:
                        self.recording_no_vad_chunk_cnt += 1
                    
                    if self.recording_stop_listening or \
                        self.recording_chunk_cnt >= self.max_record_chunks or \
                        self.recording_no_vad_chunk_cnt > self.stop_record_delay_after_no_vad_chuck_cnt:
                        
                        if not self.vad_during_recording:
                            print(f'{self.log_prefix} finished speech-to-text (no speech detected): {""}')
                            self.status_callback(self.callback_context, {"msg": "stt_ok", "text": ""})
                            self.speech_recog_active = False
                        else:
                            self.start_speech_to_text(self.recording_buffer)
                        self.recording = False
                        self.recording_buffer = []

                # Feed to openWakeWord model
                if self.oww_model is not None:
                    prediction = self.oww_model.predict(audio, threshold={"elsabot": 0.5}, debounce_time=1.0)
                    for mdl in prediction.keys():
                        if prediction[mdl] > 0.5:
                            print(f'{self.log_prefix} ww detected: {mdl}')
                            self.status_callback(self.callback_context, {"msg": "wakeword_detected", "wakeword": mdl})
            except Exception as ex:
                print(f"{self.log_prefix} Exception {ex}")
                
    def new_command(self, cmd):
        self.queue.put(cmd)