import asyncio
import aiohttp
import base64
import os
from typing import AsyncGenerator
import logging

logger = logging.getLogger(__name__)

class TTSService:
    def __init__(self, api_key: str, voice_id: str = "alloy"):
        """
        OpenAI TTS Service
        voice_id options: alloy, echo, fable, onyx, nova, shimmer
        For Japanese, 'nova' or 'shimmer' work well.
        """
        self.api_key = api_key
        self.voice_id = voice_id
        self.model = "tts-1"  # Use tts-1 for lower latency, tts-1-hd for higher quality
        self.client = None

    async def connect(self):
        """No connection needed for OpenAI TTS (HTTP API)"""
        logger.info("OpenAI TTS service ready (no connection needed)")

    async def stream_text(self, text_iterator: AsyncGenerator[str, None]) -> AsyncGenerator[bytes, None]:
        """
        Takes an async generator of text (from LLM) and yields audio bytes (PCM16, 24000Hz).
        Note: OpenAI TTS returns PCM16 at 24000Hz, but we need u-law at 8000Hz for our pipeline.
        We'll convert it.
        """
        # Collect all text first (OpenAI TTS doesn't support streaming input)
        full_text = ""
        async for text_chunk in text_iterator:
            full_text += text_chunk
        
        if not full_text.strip():
            return
        
        logger.info(f"Generating TTS for: {full_text[:50]}... (length: {len(full_text)})")
        
        # OpenAI TTS API endpoint
        url = "https://api.openai.com/v1/audio/speech"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "input": full_text,
            "voice": self.voice_id,
            "response_format": "pcm",  # PCM16 format
            "speed": 1.0
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"OpenAI TTS API error: {response.status} - {error_text}")
                        return
                    
                    # Read audio data in chunks
                    async for chunk in response.content.iter_chunked(4096):
                        if chunk:
                            # Convert PCM16 (24000Hz) to u-law (8000Hz)
                            # First, resample to 8000Hz, then convert to u-law
                            audio_bytes = await self._convert_pcm16_to_ulaw_8k(chunk)
                            if audio_bytes:
                                yield audio_bytes
                    
                    logger.info("OpenAI TTS audio generation completed")
        except Exception as e:
            logger.error(f"Error in OpenAI TTS: {e}", exc_info=True)

    async def _convert_pcm16_to_ulaw_8k(self, pcm16_data: bytes) -> bytes:
        """
        Convert PCM16 (24000Hz) to u-law (8000Hz).
        This is a simplified conversion - for production, use proper resampling.
        """
        try:
            # Convert bytes to numpy array (int16)
            audio_np = np.frombuffer(pcm16_data, dtype=np.int16)
            
            if len(audio_np) == 0:
                return b""
            
            # Resample from 24000Hz to 8000Hz
            original_rate = 24000
            target_rate = 8000
            num_samples = int(len(audio_np) * target_rate / original_rate)
            
            if num_samples > 0:
                resampled_audio = signal.resample(audio_np, num_samples)
                resampled_pcm = resampled_audio.astype(np.int16).tobytes()
                
                # Convert PCM16 to u-law
                ulaw_data = audioop.lin2ulaw(resampled_pcm, 2)
                return ulaw_data
            else:
                return b""
        except Exception as e:
            logger.error(f"Error converting audio format: {e}")
            return b""

    async def close(self):
        """No cleanup needed for OpenAI TTS"""
        pass
