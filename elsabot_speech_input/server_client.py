import os
import asyncio
import threading
import websockets
import httpx
import json
import time
from pathlib import Path

from .tuning import Tuning, find

class SpeechInputServerClient():
    def __init__(self, logger, server_host_and_port, connected_cb, wakeword_cb, vad_cb,
                 speech_recog_finished_cb, speech_recog_failed_cb, recording_cb):
        self.logger = logger
        self.server_host_and_port = server_host_and_port

        self.loop = asyncio.new_event_loop()
        self.client = httpx.AsyncClient()
        
        self.connected_cb = connected_cb
        self.wakeword_cb = wakeword_cb
        self.vad_cb = vad_cb
        self.speech_recog_finished_cb = speech_recog_finished_cb
        self.speech_recog_failed_cb = speech_recog_failed_cb
        self.recording_cb = recording_cb

        self.seeed_mic_dev = find()
        if self.seeed_mic_dev is None:
            self.logger.error(f'Error, failed to find Seeed Mic device')
        self.configure_seed_mic_dev()

    def run(self):
        self.logger.info(f'Starting client thread')
        # Start the network thread
        self.logger.info(f'Starting server listener')
        self.thread = threading.Thread(target=self._run_async_loop, daemon=True).start()

    def read_mic_array_aoa(self):
        return self.seeed_mic_dev.read('DOAANGLE')

    def read_mic_array_vad(self):
        return self.seeed_mic_dev.read('VOICEACTIVITY')

    def configure_seed_mic_dev(self):
        self.seeed_mic_dev.write('GAMMAVAD_SR', 2)
        self.seeed_mic_dev.write('AGCGAIN', 15)
        self.seeed_mic_dev.write('AGCONOFF', 0)
        return

    def _run_async_loop(self):
        # Runs the asyncio loop in a dedicated background thread.
        self.logger.info(f'Running loop')
        asyncio.set_event_loop(self.loop)
      
        self.loop.create_task(self._listen_to_server())
        self.loop.run_forever()
        self.logger.info(f'Finisheding running loop')

    async def _listen_to_server(self):
        # Receives status from Speech Input Server
        self.logger.info(f'Connecting ws to {self.server_host_and_port}')
        url = f'ws://{self.server_host_and_port}/ws'
        async for websocket in websockets.connect(url):
            try:
                self.logger.info("Connected to Speech Input Server WebSocket")
                self.connected_cb()

                while True:
                    data_raw = await websocket.recv()
                    self.logger.debug(f"Received WS Event from server: {str(data_raw)}")
                    data = json.loads(data_raw)  
                  
                    if data['msg'] == 'stt_ok':
                        self.speech_recog_finished_cb(data['text'])
                    elif data['msg'] == 'stt_failed':
                        self.speech_recog_failed_cb(data['error_data'])
                    elif data['msg'] == 'stt_recognizing':
                        pass
                    elif data['msg'] == 'wakeword_detected':
                        self.wakeword_cb(data['wakeword'])
                    elif data['msg'] == 'vad':
                        self.logger.info(f"VAD: active: {data['active']}, cnt: {data['cnt']}")
                        self.vad_cb(data['active'])
                    elif data['msg'] == 'recording_started':
                        self.recording_cb(True, data['using_pre_buffered_data'])
                    elif data['msg'] == 'recording_stopped':
                        self.recording_cb(False, False)
                    elif data['msg'] == 'heartbeat':
                        pass
                    else:
                        self.logger.error(f'Unrecognized status: {data['msg']}')
                  
            except websockets.ConnectionClosed:
                self.logger.warn("WebSocket connection lost. Retrying...")
                await asyncio.sleep(2)

    async def _send_http_post(self, command, data):
        # Sends an async HTTP POST request to the server.
        try:
            url = f'http://{self.server_host_and_port}/{command}'
            response = await self.client.post(url, json=data)
            self.logger.info(f"Server Response: {response.json()}")
        except Exception as e:
            self.logger.error(f"HTTP Request failed: {e}")

        self.speech_server_client.start_speech_recognizer(goal_handle.request.max_speech_duration, goal_handle.request.start_delay,
            goal_handle.request.pre_speech_timeout, goal_handle.request.post_speech_timeout)


    def start_speech_recognizer(self, max_speech_duration, start_delay, pre_speech_timeout, post_speech_timeout):
        args = {'max_speech_duration': max_speech_duration,
                'start_delay': start_delay,
                'pre_speech_timeout': pre_speech_timeout,
                'post_speech_timeout': post_speech_timeout}
        self.logger.info(f"Recog speech, start_delay: {start_delay}, max_speech_duration: {max_speech_duration},                         pre_speech_timeout: {pre_speech_timeout},  post_speech_timeout: {post_speech_timeout}")
        asyncio.run_coroutine_threadsafe(self._send_http_post('speech_recognizer_start', args), self.loop)

    def finish_speech_recognizer(self):
        asyncio.run_coroutine_threadsafe(self._send_http_post('speech_recognizer_finish', None), self.loop)

    def cancel_speech_recognizer(self):
        asyncio.run_coroutine_threadsafe(self._send_http_post('speech_recognizer_cancel', None), self.loop)

    def set_wakeword(self, ww_name, ww_model_path):
        args = {'ww_name': ww_name, 'ww_model_path': ww_model_path}
        asyncio.run_coroutine_threadsafe(self._send_http_post('wake_word_set', args), self.loop)

    def unset_wakeword(self, ww_name, ww_model_path):
        args = {'ww_name': ww_name}
        asyncio.run_coroutine_threadsafe(self._send_http_post('wake_word_unset', args), self.loop)

    def load_wake_words(self, dir, speech_server_ww_model_dir):
        p = Path(dir)

        for entry in p.iterdir():
            if entry.is_file():
                name = Path(entry.name).stem
                if name != 'elsabot':
                    continue
                self.logger.info(f'Loading wake word model: {entry.name}')
                model_path = os.path.join(speech_server_ww_model_dir, entry.name)
                asyncio.run_coroutine_threadsafe(self._send_http_post('wake_word_set', {"ww_name": name, "ww_model_path": model_path}), self.loop)
