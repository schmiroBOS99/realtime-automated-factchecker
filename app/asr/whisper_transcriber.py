"""
Whisper-based audio transcription for realtime processing.

Uses faster-whisper for efficient transcription of audio chunks.
"""
import logging
import io
import wave
from typing import Optional

try:
    from faster_whisper import WhisperModel  # type: ignore[import]
except ImportError:
    WhisperModel = None

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """
    Whisper-based audio transcriber with model caching.
    
    Loads model once and reuses for multiple transcriptions.
    """
    
    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        """
        Initialize Whisper transcriber.
        
        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large-v2")
                       Default: "base" (good balance of speed/quality)
            device: Device to run on ("cpu" or "cuda")
            compute_type: Computation type ("int8", "float16", "float32")
                         Default: "int8" (faster, slightly lower quality)
        """
        self.model = None
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        
        if WhisperModel is None:
            logger.warning(
                "faster-whisper not installed. Audio transcription will not work. "
                "Install with: pip install faster-whisper"
            )
            return
        
        try:
            logger.info(f"Loading Whisper model: {model_size} (device={device}, compute_type={compute_type})")
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}", exc_info=True)
            self.model = None
    
    def is_available(self) -> bool:
        """Check if transcriber is available (model loaded)."""
        return self.model is not None
    
    def transcribe_pcm_or_wav(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        """
        Transcribe audio bytes (PCM or WAV format).
        
        Args:
            audio_bytes: Audio data as bytes (WAV file bytes or raw PCM)
            sample_rate: Sample rate for PCM data (ignored if WAV format)
                        Default: 16000 Hz (Whisper's native rate)
        
        Returns:
            Transcribed text string, or empty string on error
        """
        if not self.is_available():
            logger.warning("Whisper model not available, cannot transcribe")
            return ""
        
        if not audio_bytes:
            return ""
        
        try:
            # Try to parse as WAV first
            audio_data, sr = self._parse_audio(audio_bytes, sample_rate)
            
            if audio_data is None or len(audio_data) == 0:
                logger.warning("Empty or invalid audio data")
                return ""
            
            # Transcribe using faster-whisper
            # Faster-whisper expects float32 numpy array with values in [-1, 1]
            segments, info = self.model.transcribe(
                audio_data,
                beam_size=5,
                language="de",  # German (can be made configurable)
                task="transcribe",
            )
            
            # Collect all segments into text
            text_parts = []
            for segment in segments:
                text = segment.text.strip()
                if text:
                    # CRITICAL: Add period at end of each segment to mark sentence boundaries
                    # This helps with sentence segmentation for claim detection
                    # Only add period if it doesn't already end with punctuation
                    if not text.endswith(('.', '!', '?', ',', ';', ':')):
                        text += '.'
                    text_parts.append(text)
            
            # Join segments with space (each segment already has punctuation)
            result = " ".join(text_parts).strip()
            
            logger.debug(f"Transcribed {len(audio_data)} samples ({sr} Hz) -> '{result[:50]}...'")
            
            return result
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}", exc_info=True)
            return ""
    
    def _parse_audio(self, audio_bytes: bytes, default_sample_rate: int = 16000):
        """
        Parse audio bytes (WAV or raw PCM).
        
        Args:
            audio_bytes: Audio data as bytes
            default_sample_rate: Sample rate to assume for raw PCM data
        
        Returns:
            Tuple of (audio_data as numpy array, sample_rate)
        """
        try:
            import numpy as np  # type: ignore[import]
        except ImportError:
            logger.error("numpy not installed, cannot process audio")
            return None, default_sample_rate
        
        # Try to parse as WAV
        try:
            wav_file = io.BytesIO(audio_bytes)
            with wave.open(wav_file, 'rb') as wf:
                sample_rate = wf.getframerate()
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                n_frames = wf.getnframes()
                
                # Read audio data
                audio_frames = wf.readframes(n_frames)
                
                # Convert to numpy array based on sample width
                if sample_width == 1:
                    # 8-bit unsigned
                    audio_data = np.frombuffer(audio_frames, dtype=np.uint8)
                    audio_data = (audio_data.astype(np.float32) - 128) / 128.0
                elif sample_width == 2:
                    # 16-bit signed
                    audio_data = np.frombuffer(audio_frames, dtype=np.int16)
                    audio_data = audio_data.astype(np.float32) / 32768.0
                elif sample_width == 4:
                    # 32-bit float or int
                    audio_data = np.frombuffer(audio_frames, dtype=np.int32)
                    audio_data = audio_data.astype(np.float32) / 2147483648.0
                else:
                    logger.warning(f"Unsupported sample width: {sample_width}")
                    return None, sample_rate
                
                # Handle mono/stereo
                if n_channels == 2:
                    # Convert stereo to mono (average channels)
                    audio_data = audio_data.reshape(-1, 2).mean(axis=1)
                elif n_channels > 2:
                    logger.warning(f"Unsupported channel count: {n_channels}, using first channel")
                    audio_data = audio_data.reshape(-1, n_channels)[:, 0]
                
                logger.debug(f"Parsed WAV: {n_frames} frames, {sample_rate} Hz, {n_channels} channels")
                return audio_data, sample_rate
                
        except Exception as wav_error:
            # Not WAV format, try as raw PCM
            logger.debug(f"Not WAV format, trying raw PCM: {wav_error}")
            
            try:
                # Assume 16-bit signed PCM, mono
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_data = audio_data.astype(np.float32) / 32768.0
                
                logger.debug(f"Parsed raw PCM: {len(audio_data)} samples (assuming {default_sample_rate} Hz)")
                return audio_data, default_sample_rate
                
            except Exception as pcm_error:
                logger.error(f"Failed to parse audio (neither WAV nor PCM): {pcm_error}")
                return None, default_sample_rate

