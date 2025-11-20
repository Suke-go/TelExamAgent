from deepgram import DeepgramClient, DeepgramClientOptions, LiveTranscriptionEvents, LiveOptions
import os
import json
import asyncio
from typing import Callable, Optional
import inspect

class STTService:
    def __init__(self, api_key: str, callback: Callable[[str], None]):
        self.api_key = api_key
        self.callback = callback
        self.connection = None
        self.deepgram = DeepgramClient(api_key, DeepgramClientOptions(verbose=False))
        # Check if callback is async
        self.is_async_callback = inspect.iscoroutinefunction(callback)

    async def start(self):
        try:
            # Create a websocket connection to Deepgram
            self.connection = self.deepgram.listen.live.v("1")

            # Define event handlers
            def on_message(self_dg, result, **kwargs):
                sentence = result.channel.alternatives[0].transcript
                if len(sentence) > 0:
                    # We only care about final results for now for simplicity, 
                    # but for lower latency we might want to use interim results
                    if result.is_final:
                        if self.is_async_callback:
                            # Schedule async callback
                            asyncio.create_task(self.callback(sentence))
                        else:
                            # Call sync callback directly
                            self.callback(sentence)

            def on_metadata(self_dg, metadata, **kwargs):
                pass
                # print(f"Metadata: {metadata}")

            def on_error(self_dg, error, **kwargs):
                print(f"Deepgram Error: {error}")

            # Register handlers
            self.connection.on(LiveTranscriptionEvents.Transcript, on_message)
            self.connection.on(LiveTranscriptionEvents.Metadata, on_metadata)
            self.connection.on(LiveTranscriptionEvents.Error, on_error)

            # Configure options
            options = LiveOptions(
                model="nova-2",
                language="ja", 
                smart_format=True,
                encoding="mulaw", # Twilio sends mulaw
                channels=1, 
                sample_rate=8000, # Twilio sends 8000Hz
                interim_results=True,
                endpointing=300, # Wait 300ms of silence to determine end of utterance
                vad_events=True
            )

            # Start connection
            # Note: Deepgram SDK v3 start() might be synchronous or async depending on usage.
            # If it returns bool directly, await will fail.
            # Try without await first based on error "object bool can't be used in 'await' expression"
            if self.connection.start(options) is False:
                print("Failed to start Deepgram connection")
                return False
            
            print("Deepgram STT started.")
            return True

        except Exception as e:
            print(f"Error starting STT: {e}")
            return False

    async def send_audio(self, audio_data: bytes):
        """Send audio bytes to Deepgram"""
        if self.connection:
            await self.connection.send(audio_data)

    async def stop(self):
        if self.connection:
            await self.connection.finish()
            self.connection = None
            print("Deepgram STT stopped.")

