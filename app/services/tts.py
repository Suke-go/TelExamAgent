import asyncio
import websockets
import json
import base64
import os
from typing import AsyncGenerator

class TTSService:
    def __init__(self, api_key: str, voice_id: str = "21m00Tcm4TlvDq8ikWAM"): # Rachel (American) - Change to a Japanese voice ID
        # For Japanese, we should use a multi-lingual model and a Japanese voice.
        # Let's use a placeholder generic Japanese voice ID if known, or default to Rachel and hope for Multilingual v2 to handle it.
        # Better yet, let's assume the user will provide a valid Voice ID.
        # Recommended for Japanese: "21m00Tcm4TlvDq8ikWAM" is Rachel, but we need a Japanese one or Multilingual.
        # "eleven_turbo_v2_5" supports Japanese.
        self.api_key = api_key
        self.voice_id = "21m00Tcm4TlvDq8ikWAM" # Placeholder, user should update
        self.model_id = "eleven_turbo_v2_5" 
        self.websocket = None

    async def connect(self):
        uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream-input?model_id={self.model_id}&output_format=ulaw_8000"
        try:
            # Add timeout for connection
            print(f"Connecting to ElevenLabs WS: {uri}")
            self.websocket = await asyncio.wait_for(websockets.connect(uri), timeout=10.0)
            
            # Send initial configuration (optional but good practice)
            # BOS message
            await self.websocket.send(json.dumps({
                "text": " ",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.8
                },
                "xi_api_key": self.api_key
            }))
            print("Connected to ElevenLabs TTS")
        except asyncio.TimeoutError:
            print("Timeout connecting to ElevenLabs TTS")
            raise
        except Exception as e:
            print(f"Error connecting to ElevenLabs: {e}")
            raise

    async def stream_text(self, text_iterator: AsyncGenerator[str, None]) -> AsyncGenerator[bytes, None]:
        """
        Takes an async generator of text (from LLM) and yields audio bytes (u-law 8000).
        """
        if not self.websocket:
            await self.connect()

        # Task to send text to websocket
        async def send_text():
            try:
                async for text_chunk in text_iterator:
                    if self.websocket:
                        await self.websocket.send(json.dumps({
                            "text": text_chunk,
                            "try_trigger_generation": True
                        }))
                # Send EOS
                if self.websocket:
                    await self.websocket.send(json.dumps({"text": ""})) 
            except Exception as e:
                print(f"Error sending text to TTS: {e}")

        # Start sending task
        sender_task = asyncio.create_task(send_text())

        # Receive audio
        try:
            while True:
                if not self.websocket:
                    break
                try:
                    message = await self.websocket.recv()
                    data = json.loads(message)
                    
                    if data.get("audio"):
                        # Decode base64 audio
                        audio_chunk = base64.b64decode(data["audio"])
                        if audio_chunk:
                            yield audio_chunk
                            
                    if data.get("isFinal"):
                        break
                except websockets.exceptions.ConnectionClosed:
                    print("ElevenLabs connection closed")
                    break
        except Exception as e:
            print(f"Error receiving audio from TTS: {e}")
        finally:
             # Make sure sender task is done
             if not sender_task.done():
                 sender_task.cancel()

    async def close(self):
        if self.websocket:
            await self.websocket.close()
            self.websocket = None

