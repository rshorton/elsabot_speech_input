
import os
import io
import queue
import time
from datetime import datetime
import json
from pathlib import Path

import pyaudio
import numpy as np
from numpy_ringbuffer import RingBuffer

import scipy.signal as signal

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
        self.wake_word_thresh = 0.8

        self.vad_thresh = 0.8

        self.input_stream = None
        self.def_audio_dev_name = 'ReSpeaker'
        # channels to use out of total channels in stream (6 for respeaker)
        self.channels = 1
        self.audio_chunk_size = 1280
        self.sample_rate = 16000

        self.chunk_period_s = self.audio_chunk_size/self.sample_rate
        print(f'{self.log_prefix} Chunk period(sec): {self.chunk_period_s}')

        self.max_speech_duration = 25       # Max duration of recorded speech
        self.def_pre_speech_timeout = 2     # Initial record timeout if no speech
        self.def_post_speech_timeout = 2    # Record timeout after hearing some speech

        self.pre_speech_timeout = self.def_pre_speech_timeout
        self.post_speech_timeout = self.def_post_speech_timeout

        self.speech_recog_active = False

        self.whisper_language = 'en'
        self.whisper_model_path = '/opt/whisper/models/turbo'
        self.whisper_model = WhisperModel(self.whisper_model_path, device="cuda", compute_type="int8_float16")

        self.test_rec_file = "/jetson_ws/test_audio.wav"

        self.init_prompt_file = "initial_prompt.wav"

        from pathlib import Path
        script_dir = Path(__file__).parent.resolve()
        initial_prompt_path = os.path.join(script_dir, self.init_prompt_file)

        audio_data, sample_rate = sf.read(initial_prompt_path, dtype='float32')

        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        segments, info = self.whisper_model.transcribe(audio_data, initial_prompt="your name is elsabot, elsabot, elsabot, elsabot.", temperature=1.0,
                                                       language=self.whisper_language, beam_size=8, no_speech_threshold=0.2, vad_filter=True, repetition_penalty=1.2)
        text = ""
        for segment in segments:
            #print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            text += segment.text
        print(f'{self.log_prefix} Initial prompt output: {text}')

        self.vad_filter = False
        if self.vad_filter:
            self.filter = signal.butter(6, [200, 2200], btype='bandpass', fs=self.sample_rate, output='sos')
            samp_freq = 1000  # Sample frequency (Hz)
            notch_freq = 700.0  # Frequency to be removed from signal (Hz)
            quality_factor = 3.0  # Quality factor
            self.b_notch, self.a_notch = signal.iirnotch(notch_freq, quality_factor, self.sample_rate)

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

            self.status_callback(self.callback_context, {"msg": "stt_recognizing"})
            sf.write("/jetson_ws/speech.wav", audio_data_array, self.sample_rate)

            segments, info = self.whisper_model.transcribe(audio_data_array, temperature=1.0, language=self.whisper_language,
                                                           beam_size=8, no_speech_threshold=0.2, vad_filter=True,
                                                           vad_parameters=dict(threshold=0.8, min_silence_duration_ms=500), repetition_penalty=1.2)
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
        self.recording_delay_chunk_cnt = 0

        self.max_record_chunks = int(self.max_speech_duration/self.chunk_period_s)
        print(f'{self.log_prefix} Max record chunks: {self.max_record_chunks}')

        self.pre_speech_vad_timeout_chunk_cnt = int(self.pre_speech_timeout/self.chunk_period_s)
        print(f'{self.log_prefix} Stop record if no initial speech after (chunks): {self.pre_speech_vad_timeout_chunk_cnt}')

        self.post_speech_vad_timeout_chunk_cnt = int(self.post_speech_timeout/self.chunk_period_s)
        print(f'{self.log_prefix} Stop record if no speech after (chunks): {self.post_speech_vad_timeout_chunk_cnt}')

        self.recording_no_vad_chunk_cnt = int(self.pre_speech_vad_timeout_chunk_cnt)
        self.recording_stop_listening = False
        self.recording = False
        self.recording_buffer = b""
        self.vad_during_recording = False

    def notify_recording_started(self, using_pre_buffered_data):
        self.status_callback(self.callback_context, {"msg": "recording_started", "using_pre_buffered_data": using_pre_buffered_data})

    def notify_recording_stopped(self):
        self.status_callback(self.callback_context, {"msg": "recording_stopped"})

    def audio_snapshot(self, buffer):
        to_save = np.array(buffer)
        sf.write(self.test_rec_file, to_save, self.sample_rate)

    def worker(self):
        print("{self.log_prefix} worker thread started")

        vad = openwakeword.VAD()

        self.open_input_stream(self.def_audio_dev_name)

        vad_active = False
        ww_active = False

        self.reset_recording()

        ring_buf_len = self.sample_rate*6
        audio_ring_buf = RingBuffer(capacity=ring_buf_len, dtype=np.int16)

        test_rec = True
        test_rec_save_after_chunks = 0

        while True:
            try:
                cmd = self.queue.get_nowait()
                print(f"{self.log_prefix} Received cmd '{str(cmd)}'...")
                          
                if cmd['cmd'] == 'set_wake_word':
                  self.open_wakeword_model(cmd['args'].ww_name, cmd['args'].ww_model_path)

                elif cmd['cmd'] == 'speech_recognizer_start':
                    if self.speech_recog_active:
                        print(f'{self.log_prefix} Speech recog already active')
                    else:
                        start_delay = cmd['args'].start_delay
                        self.pre_speech_timeout = cmd['args'].pre_speech_timeout if cmd['args'].pre_speech_timeout > 0 \
                            else self.def_pre_speech_timeout
                        self.post_speech_timeout = cmd['args'].post_speech_timeout if cmd['args'].post_speech_timeout > 0 \
                            else self.def_post_speech_timeout
                        self.max_speech_duration = cmd['args'].max_speech_duration

                        self.speech_recog_active = True
                        self.reset_recording()
                        self.recording = True

                        self.recording_delay_chunk_cnt = 0
                        if start_delay > 0:
                            self.recording_delay_chunk_cnt = int(start_delay/self.chunk_period_s)
                            print(f'Record with start_delay: {start_delay}, num chunks: {self.recording_delay_chunk_cnt}')
                        elif start_delay < 0:
                            num_samples = int(start_delay*self.sample_rate)
                            print(f'Record with pre-start data, start_delay: {start_delay}, num samples: {num_samples}')
                            self.recording_buffer += (np.array(audio_ring_buf)[num_samples:]).tobytes()
                            # Since the start_delay is earlier, assume VAD was used to trigger
                            self.vad_during_recording = True
                        self.notify_recording_started(start_delay < 0)

                elif cmd['cmd'] == 'speech_recognizer_cancel':
                    if self.speech_recog_active and self.recording:
                        self.speech_recog_active = False
                        self.reset_recording()
                        self.notify_recording_stopped()
                        print(f'{self.log_prefix} Speech recog cancelled')

                elif cmd['cmd'] == 'speech_recognizer_finish':
                    if self.speech_recog_active and self.recording:
                        self.recording_stop_listening = True
                        print(f'{self.log_prefix} Speech recog recording finished')

                elif cmd['cmd'] == 'audio_snapshot':
                    self.audio_snapshot(audio_ring_buf)

                self.queue.task_done()
            except queue.Empty:
                pass

            try:
                chunk = self.input_stream.read(self.audio_chunk_size, exception_on_overflow=False)
            except Exception as ex:
                print(f"{self.log_prefix} Exception while reading stream: {ex}")
                continue

            audio = np.frombuffer(chunk, dtype=np.int16) [0::self.channels]

            audio_ring_buf.extend(audio)

            try:
                if self.vad_filter:
                    audio_float = audio.astype(np.float32)/32768.0
                    #audio_float_filtered = signal.sosfiltfilt(self.filter, audio_float)
                    #audio_float_filtered = signal.sosfiltfilt(self.filter, audio_float)
                    audio_float_filtered = signal.filtfilt(self.b_notch, self.a_notch, audio_float)
                    audio_vad = (audio_float_filtered * 32767).astype(np.int16)
                    vad(audio_vad)
                else:                    
                    vad(audio)

                # Consider last 10 frames (1280/16000*10= 800ms)
                vad_frames = list(vad.prediction_buffer)[-10:]
                vad_max_score = np.max(vad_frames) if len(vad_frames) > 0 else 0

                num_frames_above_thresh = sum(f > self.vad_thresh for f in vad_frames)
                #print(f'num_frames_above_thresh {num_frames_above_thresh}')

                cur_vad = vad_max_score > self.vad_thresh and (self.recording or
                          (num_frames_above_thresh > 2))
                if cur_vad != vad_active:
                    vad_active = cur_vad
                    print(f'{self.log_prefix} vad change: {cur_vad}')
                    self.status_callback(self.callback_context, {"msg": "vad", "active": bool(vad_active), "cnt": int(num_frames_above_thresh)})

                    if test_rec and cur_vad:
                        test_rec_save_after_chunks = 2/self.chunk_period_s

                if test_rec and test_rec_save_after_chunks > 0:
                    test_rec_save_after_chunks -= 1
                    if test_rec_save_after_chunks == 0:
                        to_save = np.array(audio_ring_buf)
                        sf.write(self.test_rec_file, to_save, self.sample_rate)

                if self.recording:
                    if self.recording_delay_chunk_cnt > 0:
                        self.recording_delay_chunk_cnt -= 1

                        if self.recording_delay_chunk_cnt == 0:
                            vad = openwakeword.VAD()
                            vad_active = False
                    else:
                        if cur_vad and not self.vad_during_recording:
                            self.vad_during_recording = True
                            print('VAD during recording')

                        self.recording_buffer += audio.tobytes()
                        self.recording_chunk_cnt += 1

                        if cur_vad:
                            self.recording_no_vad_chunk_cnt = self.post_speech_vad_timeout_chunk_cnt
                        else:
                            self.recording_no_vad_chunk_cnt -= 1
                        
                        if self.recording_stop_listening or \
                            self.recording_chunk_cnt >= self.max_record_chunks or \
                            self.recording_no_vad_chunk_cnt <= 0:

                            self.notify_recording_stopped()

                            if not self.vad_during_recording:
                                print(f'{self.log_prefix} finished speech-to-text (no speech detected): {""}')
                                self.status_callback(self.callback_context, {"msg": "stt_ok", "text": ""})
                                self.speech_recog_active = False
                                self.audio_snapshot(audio_ring_buf)
                            else:
                                self.start_speech_to_text(self.recording_buffer)
                            self.recording = False
                            self.recording_buffer = []

                # Feed to openWakeWord model
                if self.oww_model is not None:
                    prediction = self.oww_model.predict(audio, threshold={"elsabot": self.wake_word_thresh}, debounce_time=1.0)
                    for mdl in prediction.keys():
                        if prediction[mdl] > self.wake_word_thresh:
                            print(f'{self.log_prefix} ww detected: {mdl}')
                            self.status_callback(self.callback_context, {"msg": "wakeword_detected", "wakeword": mdl})
            except Exception as ex:
                print(f"{self.log_prefix} Exception {ex}")
                
    def new_command(self, cmd):
        self.queue.put(cmd)