import audioop
import numpy as np
import scipy.signal

def ulaw_to_pcm16(ulaw_data: bytes) -> bytes:
    """Convert u-law encoded bytes to 16-bit PCM bytes."""
    return audioop.ulaw2lin(ulaw_data, 2)

def pcm16_to_ulaw(pcm_data: bytes) -> bytes:
    """Convert 16-bit PCM bytes to u-law encoded bytes."""
    return audioop.lin2ulaw(pcm_data, 2)

def resample_audio(pcm_data: bytes, original_rate: int = 8000, target_rate: int = 16000) -> np.ndarray:
    """
    Resample audio from original_rate to target_rate.
    Returns numpy array of float32, normalized to [-1, 1] for Silero VAD.
    """
    if not pcm_data:
        return np.array([], dtype=np.float32)

    # Convert bytes to numpy array of int16
    audio_np = np.frombuffer(pcm_data, dtype=np.int16)
    
    # Calculate number of samples after resampling
    num_samples = int(len(audio_np) * target_rate / original_rate)
    
    # Resample
    # Note: scipy.signal.resample uses FFT, which might be slow for large chunks but okay for small ones.
    # For real-time, simple interpolation or decimation might be faster if ratios are integer.
    # But 8k -> 16k is exactly 2x.
    resampled_audio = scipy.signal.resample(audio_np, num_samples)
    
    # Normalize to float32 [-1, 1]
    resampled_float = resampled_audio.astype(np.float32) / 32768.0
    
    return resampled_float

