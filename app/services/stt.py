from deepgram import DeepgramClient, DeepgramClientOptions, LiveTranscriptionEvents, LiveOptions
import os
import json
import asyncio
from typing import Callable, Optional
import inspect
import threading

class STTService:
    def __init__(self, api_key: str, callback: Callable[[str], None]):
        self.api_key = api_key
        self.callback = callback
        self.connection = None
        self.deepgram = DeepgramClient(api_key, DeepgramClientOptions(verbose=False))
        # Check if callback is async
        self.is_async_callback = inspect.iscoroutinefunction(callback)
        # Store the event loop reference
        self._loop = None

    async def start(self):
        try:
            # Store the current event loop for use in callbacks
            self._loop = asyncio.get_running_loop()
            
            # Create a websocket connection to Deepgram
            self.connection = self.deepgram.listen.live.v("1")

            # Define event handlers
            def on_message(self_dg, result, **kwargs):
                try:
                    sentence = result.channel.alternatives[0].transcript
                    if len(sentence) > 0:
                        if result.is_final:
                            if self.is_async_callback and self._loop:
                                # Use call_soon_threadsafe to schedule in the correct event loop
                                self._loop.call_soon_threadsafe(
                                    lambda: asyncio.create_task(self.callback(sentence))
                                )
                            elif not self.is_async_callback:
                                self.callback(sentence)
                except Exception as e:
                    print(f"Error in on_message callback: {e}")

            def on_metadata(self_dg, metadata, **kwargs):
                pass

            def on_error(self_dg, error, **kwargs):
                print(f"Deepgram Error: {error}")

            def on_close(self_dg, close, **kwargs):
                print("Deepgram connection closed")

            # Register handlers
            self.connection.on(LiveTranscriptionEvents.Transcript, on_message)
            self.connection.on(LiveTranscriptionEvents.Metadata, on_metadata)
            self.connection.on(LiveTranscriptionEvents.Error, on_error)
            self.connection.on(LiveTranscriptionEvents.Close, on_close)

            # Configure options
            options = LiveOptions(
                model="nova-2",
                language="ja", 
                smart_format=True,
                encoding="mulaw",
                channels=1, 
                sample_rate=8000,
                interim_results=True,
                endpointing=2000,  # Increased to 2000ms to reduce mid-sentence splits in Japanese
                vad_events=True
            )

            # Start connection
            if self.connection.start(options) is False:
                print("Failed to start Deepgram connection")
                return False
            
            print("Deepgram STT started.")
            return True

        except Exception as e:
            print(f"Error starting STT: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def send_audio(self, audio_data: bytes):
        """Send audio bytes to Deepgram"""
        if self.connection:
            try:
                self.connection.send(audio_data)
            except Exception as e:
                print(f"Error sending audio to Deepgram: {e}")

    async def stop(self):
        if self.connection:
            try:
                self.connection.finish()
            except Exception as e:
                print(f"Error stopping Deepgram: {e}")
            finally:
                self.connection = None
                self._loop = None
                print("Deepgram STT stopped.")
