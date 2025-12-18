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
        # Buffer for accumulating is_final transcripts until speech_final
        self._final_buf: list[str] = []

    def _emit(self, sentence: str):
        """Emit the finalized sentence via callback"""
        if not sentence:
            return
        if self.is_async_callback and self._loop:
            # Use call_soon_threadsafe to schedule in the correct event loop
            self._loop.call_soon_threadsafe(
                lambda s=sentence: asyncio.create_task(self.callback(s))
            )
        elif not self.is_async_callback:
            self.callback(sentence)

    async def start(self):
        try:
            # Store the current event loop for use in callbacks
            self._loop = asyncio.get_running_loop()
            # Clear buffer on start
            self._final_buf.clear()
            
            # Create a websocket connection to Deepgram
            self.connection = self.deepgram.listen.live.v("1")

            # Define event handlers
            def on_message(self_dg, result, **kwargs):
                try:
                    text = result.channel.alternatives[0].transcript.strip()
                    if not text:
                        return

                    if result.is_final:
                        # Accumulate is_final results in buffer
                        self._final_buf.append(text)

                        # Only emit when speech_final is True (end of utterance)
                        if getattr(result, "speech_final", False):
                            sentence = "".join(self._final_buf)
                            self._final_buf.clear()
                            self._emit(sentence)
                    # else: interim results - ignore for now (could be used for UI display)
                except Exception as e:
                    print(f"Error in on_message callback: {e}")

            def on_utterance_end(*args, **kwargs):
                """Fallback flush when UtteranceEnd event is received (insurance)"""
                try:
                    if self._final_buf:
                        sentence = "".join(self._final_buf)
                        self._final_buf.clear()
                        self._emit(sentence)
                except Exception as e:
                    print(f"Error in on_utterance_end callback: {e}")

            def on_metadata(self_dg, metadata, **kwargs):
                pass

            def on_error(self_dg, error, **kwargs):
                print(f"Deepgram Error: {error}")

            def on_close(self_dg, close, **kwargs):
                print("Deepgram connection closed")

            # Register handlers
            self.connection.on(LiveTranscriptionEvents.Transcript, on_message)
            self.connection.on(LiveTranscriptionEvents.Metadata, on_metadata)
            self.connection.on(LiveTranscriptionEvents.UtteranceEnd, on_utterance_end)
            self.connection.on(LiveTranscriptionEvents.Error, on_error)
            self.connection.on(LiveTranscriptionEvents.Close, on_close)

            # Configure options
            # NOTE: Browser sends audio as mulaw 8kHz (converted in main.py)
            options = LiveOptions(
                model="nova-3",          # 後述
                language="ja",
                encoding="mulaw",        # Changed from linear16 to match browser audio format
                sample_rate=8000,        # Changed from 16000 to match browser audio
                channels=1,
                interim_results=True,
                punctuate=True,
                smart_format=True,
                utterance_end_ms=1000,   # UtteranceEndイベントを有効化（1秒の無音で発火）
            )
            # options = LiveOptions(
            #     model="nova-2",
            #     language="ja", 
            #     smart_format=True,
            #     encoding="mulaw",
            #     channels=1, 
            #     sample_rate=8000,
            #     interim_results=True,
            #     endpointing=500,  # Reduced for faster response, combined with dB-level detection
            #     vad_events=True
            # )

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
