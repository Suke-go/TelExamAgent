import torch
import numpy as np
import os
import time

class VADService:
    # dB threshold for silence detection (adjust as needed)
    SILENCE_DB_THRESHOLD = -40  # dB below which is considered silence
    # Number of consecutive silent chunks needed to confirm speech end
    SILENCE_CHUNKS_REQUIRED = 3
    
    def __init__(self):
        # Load Silero VAD model
        # force_reload=True helps if the cache is corrupted, but generally False is faster
        self.model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                           model='silero_vad',
                                           force_reload=False,
                                           trust_repo=True)
        
        (self.get_speech_timestamps,
         self.save_audio,
         self.read_audio,
         self.VADIterator,
         self.collect_chunks) = utils
        
        self.vad_iterator = self.VADIterator(self.model)
        
        # State for multi-signal detection
        self._is_speaking = False
        self._silence_chunk_count = 0
        self._last_speech_time = 0.0
        self._current_db = -100.0
        
        print("Silero VAD model loaded successfully.")

    def reset(self):
        """Reset the VAD iterator state for a new call."""
        self.vad_iterator.reset_states()
        self._is_speaking = False
        self._silence_chunk_count = 0
        self._last_speech_time = 0.0
        self._current_db = -100.0

    @staticmethod
    def calculate_db_level(audio_float32: np.ndarray) -> float:
        """
        Calculate the dB level of an audio chunk.
        audio_float32: numpy array of float32, normalized to [-1, 1]
        Returns: dB level (typically -100 to 0, where 0 is max)
        """
        if len(audio_float32) == 0:
            return -100.0
        
        # Calculate RMS (Root Mean Square)
        rms = np.sqrt(np.mean(audio_float32 ** 2))
        
        # Avoid log(0)
        if rms < 1e-10:
            return -100.0
        
        # Convert to dB (reference is 1.0 for normalized audio)
        db = 20 * np.log10(rms)
        return db

    def is_audio_silent(self, audio_float32: np.ndarray, threshold: float = None) -> bool:
        """
        Check if audio chunk is silent based on dB level.
        """
        if threshold is None:
            threshold = self.SILENCE_DB_THRESHOLD
        db = self.calculate_db_level(audio_float32)
        self._current_db = db
        return db < threshold

    def process_audio_chunk(self, audio_float32: np.ndarray) -> dict:
        """
        Process a chunk of audio and return VAD status.
        audio_float32: numpy array of float32, normalized to [-1, 1]
        """
        # Convert numpy array to torch tensor
        if len(audio_float32) == 0:
             return {"speech_start": False, "speech_end": False, "db_level": -100.0, "is_speaking": False}

        tensor_chunk = torch.from_numpy(audio_float32)
        
        # If chunk is stereo (2 channels), convert to mono by averaging or picking one
        if len(tensor_chunk.shape) > 1 and tensor_chunk.shape[1] > 1:
             tensor_chunk = torch.mean(tensor_chunk, dim=1)

        # Calculate dB level
        db_level = self.calculate_db_level(audio_float32)
        self._current_db = db_level
        is_silent = db_level < self.SILENCE_DB_THRESHOLD

        # Get speech dict from VAD iterator
        # It returns a dict like {'start': timestamp} or {'end': timestamp} or None
        speech_dict = self.vad_iterator(tensor_chunk, return_seconds=True)
        
        result = {
            "speech_start": False,
            "speech_end": False,
            "db_level": db_level,
            "is_speaking": self._is_speaking
        }
        
        if speech_dict:
            if 'start' in speech_dict:
                result["speech_start"] = True
                self._is_speaking = True
                self._silence_chunk_count = 0
                self._last_speech_time = time.time()
            if 'end' in speech_dict:
                result["speech_end"] = True
                self._is_speaking = False
        
        # Track silence chunks when we think speech might be ending
        if is_silent:
            self._silence_chunk_count += 1
        else:
            self._silence_chunk_count = 0
            self._last_speech_time = time.time()
        
        result["is_speaking"] = self._is_speaking or not is_silent
                
        return result

    def is_user_done_speaking(self, min_silence_duration: float = 0.5) -> bool:
        """
        Multi-signal check: Is the user truly done speaking?
        Combines VAD state + dB level + time since last speech.
        
        Returns True only when:
        1. VAD says not speaking (_is_speaking = False)
        2. Audio is silent (dB below threshold)
        3. Enough time has passed since last detected speech
        """
        if self._is_speaking:
            return False
        
        if self._current_db >= self.SILENCE_DB_THRESHOLD:
            return False
        
        time_since_speech = time.time() - self._last_speech_time
        if time_since_speech < min_silence_duration:
            return False
        
        return True
    
    def get_current_state(self) -> dict:
        """Get current VAD state for debugging/monitoring."""
        return {
            "is_speaking": self._is_speaking,
            "db_level": self._current_db,
            "silence_chunks": self._silence_chunk_count,
            "time_since_speech": time.time() - self._last_speech_time if self._last_speech_time > 0 else 0
        }

