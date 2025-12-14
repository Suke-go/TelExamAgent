import asyncio
import websockets
import json
import base64
import os
import audioop
from typing import AsyncGenerator, Optional
import logging
import numpy as np
from scipy import signal
import aiohttp

logger = logging.getLogger(__name__)

class TTSService:
    """
    ElevenLabs WebSocket TTS Service for ultra-low latency streaming.
    Falls back to OpenAI TTS if ElevenLabs is not configured.
    """
    
    def __init__(self, api_key: str, voice_id: str = "nova", elevenlabs_api_key: Optional[str] = None):
        self.openai_api_key = api_key
        self.elevenlabs_api_key = elevenlabs_api_key or os.getenv("ELEVENLABS_API_KEY")
        
        # ElevenLabs voice ID - Japanese voice
        # Default: Kozy - Male Japanese Narrative Voice (Tokyo Standard Accent)
        # You can find other voices at https://elevenlabs.io/voice-library
        self.elevenlabs_voice_id = os.getenv("ELEVENLABS_VOICE_ID", "GxxMAMfQkDlnqjpzjLHH")
        
        # OpenAI voice as fallback
        self.openai_voice = voice_id
        
        # Use ElevenLabs if API key is available
        self.use_elevenlabs = bool(self.elevenlabs_api_key)
        
        if self.use_elevenlabs:
            logger.info(f"Using ElevenLabs TTS (voice: {self.elevenlabs_voice_id})")
        else:
            logger.info(f"Using OpenAI TTS (voice: {self.openai_voice})")

    async def connect(self):
        """Initialize connection (no persistent connection needed)"""
        logger.info("TTS service ready")

    async def stream_text(self, text_iterator: AsyncGenerator[str, None]) -> AsyncGenerator[bytes, None]:
        """
        Stream text to TTS and yield audio chunks.
        Uses ElevenLabs WebSocket for low latency if available.
        """
        if self.use_elevenlabs:
            async for chunk in self._stream_elevenlabs(text_iterator):
                yield chunk
        else:
            async for chunk in self._stream_openai(text_iterator):
                yield chunk

    async def _stream_elevenlabs(self, text_iterator: AsyncGenerator[str, None]) -> AsyncGenerator[bytes, None]:
        """
        Stream text via ElevenLabs WebSocket API for ultra-low latency.
        """
        # ElevenLabs WebSocket URL
        # Using eleven_multilingual_v2 for best Japanese quality
        # Alternative: eleven_flash_v2_5 for lower latency but less quality
        model_id = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
        ws_url = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.elevenlabs_voice_id}/stream-input?model_id={model_id}&output_format=ulaw_8000&optimize_streaming_latency=3"
        
        logger.info(f"Connecting to ElevenLabs WebSocket: model={model_id}, voice={self.elevenlabs_voice_id}")
        
        headers = {
            "xi-api-key": self.elevenlabs_api_key
        }
        
        # Collect text first to ensure we have something to send
        collected_text = []
        async for text_chunk in text_iterator:
            if text_chunk:
                collected_text.append(text_chunk)
        
        full_text = "".join(collected_text)
        if not full_text.strip():
            logger.warning("No text to convert to speech")
            return
        
        logger.info(f"ElevenLabs TTS: Converting {len(full_text)} chars")
        
        try:
            async with websockets.connect(ws_url, extra_headers=headers) as ws:
                logger.info("ElevenLabs WebSocket connected")
                
                # Send initial message with voice settings and text
                init_message = {
                    "text": full_text,
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                        "style": 0.0,
                        "use_speaker_boost": True
                    },
                    "xi_api_key": self.elevenlabs_api_key
                }
                await ws.send(json.dumps(init_message))
                
                # Send end of stream signal
                await ws.send(json.dumps({"text": ""}))
                
                # Receive audio chunks
                audio_chunk_count = 0
                try:
                    async for message in ws:
                        try:
                            data = json.loads(message)
                            
                            if "audio" in data and data["audio"]:
                                # Decode base64 audio (already in ulaw_8000 format)
                                audio_bytes = base64.b64decode(data["audio"])
                                if audio_bytes:
                                    audio_chunk_count += 1
                                    logger.debug(f"ElevenLabs audio chunk {audio_chunk_count}: {len(audio_bytes)} bytes")
                                    yield audio_bytes
                            
                            if data.get("isFinal"):
                                logger.info(f"ElevenLabs TTS completed: {audio_chunk_count} chunks")
                                break
                            
                            # Check for errors
                            if "error" in data:
                                logger.error(f"ElevenLabs error: {data['error']}")
                                break
                                
                        except json.JSONDecodeError:
                            # Binary audio data
                            logger.debug(f"Received binary data: {len(message)} bytes")
                            yield message
                            
                except websockets.exceptions.ConnectionClosed as e:
                    logger.warning(f"ElevenLabs connection closed: {e}")
                    
        except Exception as e:
            logger.error(f"ElevenLabs WebSocket error: {e}", exc_info=True)
            # Fallback to OpenAI
            logger.info("Falling back to OpenAI TTS")
            async for chunk in self._openai_tts(full_text):
                yield chunk

    async def _stream_openai(self, text_iterator: AsyncGenerator[str, None]) -> AsyncGenerator[bytes, None]:
        """
        OpenAI TTS with sentence-level batching (higher latency fallback).
        """
        sentence_buffer = ""
        sentence_endings = ('。', '！', '？', '!', '?', '\n')
        
        async for text_chunk in text_iterator:
            sentence_buffer += text_chunk
            
            # Check for complete sentences
            while any(ending in sentence_buffer for ending in sentence_endings):
                earliest_pos = len(sentence_buffer)
                for ending in sentence_endings:
                    pos = sentence_buffer.find(ending)
                    if pos != -1 and pos < earliest_pos:
                        earliest_pos = pos
                
                if earliest_pos < len(sentence_buffer):
                    sentence = sentence_buffer[:earliest_pos + 1]
                    sentence_buffer = sentence_buffer[earliest_pos + 1:]
                    
                    async for audio_chunk in self._openai_tts(sentence):
                        yield audio_chunk
                else:
                    break
        
        # Process remaining text
        if sentence_buffer.strip():
            async for audio_chunk in self._openai_tts(sentence_buffer):
                yield audio_chunk

    async def _openai_tts(self, text: str) -> AsyncGenerator[bytes, None]:
        """Generate audio for a single text segment using OpenAI TTS."""
        if not text.strip():
            return
            
        url = "https://api.openai.com/v1/audio/speech"
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "tts-1",
            "input": text,
            "voice": self.openai_voice,
            "response_format": "pcm",
            "speed": 1.0
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"OpenAI TTS error: {response.status} - {error_text}")
                        return
                    
                    async for chunk in response.content.iter_chunked(4096):
                        if chunk:
                            audio_bytes = await self._convert_pcm_to_ulaw(chunk)
                            if audio_bytes:
                                yield audio_bytes
        except Exception as e:
            logger.error(f"OpenAI TTS error: {e}")

    async def _convert_pcm_to_ulaw(self, pcm16_data: bytes) -> bytes:
        """Convert PCM16 24kHz to u-law 8kHz."""
        try:
            audio_np = np.frombuffer(pcm16_data, dtype=np.int16)
            if len(audio_np) == 0:
                return b""
            
            # Resample 24kHz -> 8kHz
            num_samples = int(len(audio_np) * 8000 / 24000)
            if num_samples > 0:
                resampled = signal.resample(audio_np, num_samples)
                resampled_pcm = resampled.astype(np.int16).tobytes()
                return audioop.lin2ulaw(resampled_pcm, 2)
            return b""
        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            return b""

    async def close(self):
        """Cleanup (no persistent connections to close)."""
        pass
