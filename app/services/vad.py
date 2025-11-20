import torch
import numpy as np
import os

class VADService:
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
        print("Silero VAD model loaded successfully.")

    def reset(self):
        """Reset the VAD iterator state for a new call."""
        self.vad_iterator.reset_states()

    def process_audio_chunk(self, audio_float32: np.ndarray) -> dict:
        """
        Process a chunk of audio and return VAD status.
        audio_float32: numpy array of float32, normalized to [-1, 1]
        """
        # Convert numpy array to torch tensor
        if len(audio_float32) == 0:
             return {"speech_start": False, "speech_end": False}

        tensor_chunk = torch.from_numpy(audio_float32)
        
        # If chunk is stereo (2 channels), convert to mono by averaging or picking one
        if len(tensor_chunk.shape) > 1 and tensor_chunk.shape[1] > 1:
             tensor_chunk = torch.mean(tensor_chunk, dim=1)

        # Get speech dict from VAD iterator
        # It returns a dict like {'start': timestamp} or {'end': timestamp} or None
        speech_dict = self.vad_iterator(tensor_chunk, return_seconds=True)
        
        result = {
            "speech_start": False,
            "speech_end": False
        }
        
        if speech_dict:
            if 'start' in speech_dict:
                result["speech_start"] = True
            if 'end' in speech_dict:
                result["speech_end"] = True
                
        return result

