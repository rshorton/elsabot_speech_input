import asyncio
import threading
import queue
import time
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import uvicorn
import json
import argparse

from .processor import SpeechProcessor

app = FastAPI()
speech_processor = None

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, data: dict):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(data)
            except Exception as ex:
                print(f'Connection dropped: {ex}')
                self.disconnect(connection)

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(send_heartbeat())
    
    loop = asyncio.get_running_loop()

    global speech_processor
    speech_processor = SpeechProcessor(speech_processor_callback, loop)

    threading.Thread(target=speech_processor.worker, daemon=True).start()

def speech_processor_callback(loop, data):
    asyncio.run_coroutine_threadsafe(manager.broadcast(data), loop)

async def send_heartbeat():
    while True:
        await manager.broadcast({
            "msg": "heartbeat",
            "timestamp": datetime.now().isoformat()
        })
        await asyncio.sleep(5)

class SpeechRecogStartArgs(BaseModel):
    timeout: float
    delay: float

@app.post("/speech_recognizer_start")
def receive_command(args: SpeechRecogStartArgs):
    print(f"HTTP Post: Received speech_recognizer_start")
    speech_processor.new_command({'cmd': 'speech_recognizer_start', 'args': args})
    
    return {
        "result": "queued",
        "details": None}

@app.post("/speech_recognizer_finish")
def receive_command():
    print(f"HTTP Post: Received speech_recognizer_finish")
    speech_processor.new_command({'cmd': 'speech_recognizer_finish'})
    
    return {
        "result": "queued",
        "details": None}

@app.post("/speech_recognizer_cancel")
def receive_command():
    print(f"HTTP Post: Received speech_recognizer_cancel")
    speech_processor.new_command({'cmd': 'speech_recognizer_cancel'})
    
    return {
        "result": "queued",
        "details": None}

class WakeWordSetArgs(BaseModel):
    ww_name: str
    ww_model_path: str

@app.post("/wake_word_set")
def receive_command(args: WakeWordSetArgs):
    print(f"HTTP Post: Received set wake word")
    speech_processor.new_command({'cmd': 'set_wake_word', 'args': args})
    
    return {
        "result": "queued",
        "details": None}

class WakeWordUnsetArgs(BaseModel):
    ww_name: str

@app.post("/wake_word_unset")
def receive_command(args: WakeWordUnsetArgs):
    print(f"HTTP Post: Received unset wake word")
    speech_processor.new_command({'cmd': 'unset_wake_word', 'args': args})
    
    return {
        "result": "queued",
        "details": None}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Handle incoming JSON from client if needed
            await websocket.receive_json()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speech Input Processor.")

    parser.add_argument('--host', type=str, default='127.0.0.1', help="host")
    parser.add_argument('--port', type=str, default='8800', help="port")

    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=int(args.port))
