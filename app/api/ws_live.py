"""
WebSocket endpoint for live text-based and audio-based fact-checking.

Protocol supports both text and audio input.
"""
import json
import uuid
import logging
import time
import base64
import asyncio
from typing import Dict, Any, Optional

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.routing import APIRouter

from app.realtime.orchestrator import process_sentence
from app.realtime.segmenter import split_sentences
from app.asr.whisper_transcriber import WhisperTranscriber

logger = logging.getLogger(__name__)

router = APIRouter()

# WebSocket timeout configuration
WS_RECEIVE_TIMEOUT = 60.0  # Timeout for receive operations (60 seconds)
WS_SEND_TIMEOUT = 10.0     # Timeout for send operations (10 seconds)

# Audio buffer configuration
AUDIO_BUFFER_DURATION_SECONDS = 1.5  # Buffer ~1.5 seconds before transcribing
AUDIO_CHUNK_DURATION_SECONDS = 0.5   # Expected chunk duration (for buffer estimation)
SAMPLE_RATE = 16000  # Whisper default sample rate
BYTES_PER_SAMPLE = 2  # 16-bit = 2 bytes
MONO_CHANNELS = 1
AUDIO_BUFFER_BYTES_THRESHOLD = int(AUDIO_BUFFER_DURATION_SECONDS * SAMPLE_RATE * BYTES_PER_SAMPLE * MONO_CHANNELS)

# Global Whisper transcriber instance (singleton)
_transcriber: Optional[WhisperTranscriber] = None


def get_transcriber() -> Optional[WhisperTranscriber]:
    """Get or create the global Whisper transcriber instance."""
    global _transcriber
    if _transcriber is None:
        _transcriber = WhisperTranscriber(model_size="base", device="cpu", compute_type="int8")
        if not _transcriber.is_available():
            logger.warning("Whisper transcriber not available, audio transcription will not work")
    return _transcriber


@router.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """
    WebSocket endpoint for live text-based and audio-based fact-checking.
    
    Protocol:
    Client → Server: 
        - { "type": "text", "text": "..." }
        - { "type": "audio", "format": "wav", "chunk_b64": "..." }
    Server → Client: Events (transcript_final, transcript_partial, claim_detected, factcheck_result)
    
    Each event is sent as a JSON object.
    """
    await websocket.accept()
    logger.info("WebSocket client connected")
    
    # Audio buffer for chunk-based transcription
    audio_buffer = bytearray()
    
    try:
        while True:
            # Receive message from client (can be text or binary) with timeout
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=WS_RECEIVE_TIMEOUT
                )
            except asyncio.TimeoutError:
                # Send ping to check if connection is still alive
                try:
                    await safe_send_json(websocket, {"type": "ping"})
                    continue
                except:
                    # Connection lost, break out of loop
                    break
            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected (receive_text)")
                break
            except Exception:
                # If text receive fails, try binary (for audio chunks)
                try:
                    binary_data = await asyncio.wait_for(
                        websocket.receive_bytes(),
                        timeout=WS_RECEIVE_TIMEOUT
                    )
                    # Handle binary audio data (could be raw PCM or WAV)
                    audio_buffer.extend(binary_data)
                    
                    # Check if buffer is large enough to transcribe
                    if len(audio_buffer) >= AUDIO_BUFFER_BYTES_THRESHOLD:
                        await process_audio_buffer(websocket, audio_buffer, "wav")
                        # Don't clear buffer completely - keep last 0.5 seconds for continuity
                        keep_bytes = int(AUDIO_BUFFER_BYTES_THRESHOLD * 0.3)
                        if len(audio_buffer) > keep_bytes:
                            audio_buffer = audio_buffer[-keep_bytes:]
                        else:
                            audio_buffer.clear()
                    
                    continue
                except WebSocketDisconnect:
                    logger.info("WebSocket client disconnected (receive_bytes)")
                    break
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error receiving message: {e}")
                    continue
            
            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                await safe_send_json(websocket, {
                    "type": "error",
                    "error": "Invalid JSON format"
                })
                continue
            
            # Handle text message
            if message.get("type") == "text":
                await handle_text_message(websocket, message)
            
            # Handle audio message (base64-encoded chunks)
            elif message.get("type") == "audio":
                await handle_audio_message(websocket, message, audio_buffer)
            
            else:
                # Unknown message type
                await safe_send_json(websocket, {
                    "type": "error",
                    "error": f"Unknown message type: {message.get('type')}"
                })
    
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        try:
            await websocket.close()
        except:
            pass
        logger.info("WebSocket connection closed")


async def safe_send_json(websocket: WebSocket, data: Dict[str, Any]) -> bool:
    """
    Safely send JSON data through WebSocket with timeout and error handling.
    
    Returns:
        True if send succeeded, False if connection is closed or error occurred
    """
    try:
        await asyncio.wait_for(
            websocket.send_json(data),
            timeout=WS_SEND_TIMEOUT
        )
        return True
    except (WebSocketDisconnect, ConnectionError, RuntimeError):
        # Connection closed or error
        return False
    except asyncio.TimeoutError:
        logger.warning("WebSocket send_json timeout")
        return False
    except Exception as e:
        logger.error(f"Error sending JSON: {e}")
        return False


async def handle_text_message(websocket: WebSocket, message: Dict[str, Any]):
    """Handle text message from client."""
    text = message.get("text", "")
    
    if not text or not text.strip():
        return
    
    # CRITICAL: Ensure text ends with punctuation for proper sentence segmentation
    # This helps split sentences correctly for claim detection
    processed_text = text.strip()
    if processed_text and not processed_text.endswith(('.', '!', '?', ',', ';', ':')):
        processed_text += '.'
    
    # Emit transcript_final event
    if not await safe_send_json(websocket, {
        "type": "transcript_final",
        "text": processed_text,
        "t_start": 0.0,
        "t_end": 0.0,
    }):
        return  # Connection closed
    
    # Process text through pipeline
    await process_text(websocket, processed_text)


async def handle_audio_message(websocket: WebSocket, message: Dict[str, Any], audio_buffer: bytearray):
    """Handle audio message (base64-encoded chunk) from client."""
    audio_format = message.get("format", "wav")
    chunk_b64 = message.get("chunk_b64", "")
    
    if not chunk_b64:
        return
    
    try:
        # Decode base64 to audio bytes
        chunk_bytes = base64.b64decode(chunk_b64)
        
        # Add to buffer
        audio_buffer.extend(chunk_bytes)
        
        # Check if buffer is large enough to transcribe
        # For WebM format, we need to accumulate more data before transcribing
        # (WebM chunks are typically smaller than WAV chunks)
        buffer_threshold = AUDIO_BUFFER_BYTES_THRESHOLD
        if audio_format == "webm":
            # WebM chunks are typically smaller, so we might need to wait longer
            # But for real-time transcription, we still want to process frequently
            buffer_threshold = int(AUDIO_BUFFER_BYTES_THRESHOLD * 0.5)  # Lower threshold for WebM
        
        if len(audio_buffer) >= buffer_threshold:
            await process_audio_buffer(websocket, audio_buffer, audio_format)
            # Don't clear buffer completely - keep last 0.5 seconds for continuity
            keep_bytes = int(AUDIO_BUFFER_BYTES_THRESHOLD * 0.3)
            if len(audio_buffer) > keep_bytes:
                audio_buffer = audio_buffer[-keep_bytes:]
        else:
            # Emit partial transcript indication (optional, for UI feedback)
            await safe_send_json(websocket, {
                "type": "transcript_partial",
                "status": "buffering",
                "buffer_size": len(audio_buffer),
            })
    
    except Exception as e:
        logger.error(f"Error processing audio chunk: {e}", exc_info=True)
        await safe_send_json(websocket, {
            "type": "error",
            "error": f"Error processing audio: {str(e)}"
        })


async def process_audio_buffer(websocket: WebSocket, audio_buffer: bytearray, audio_format: str = "wav"):
    """Process audio buffer through Whisper and emit transcription events."""
    transcriber = get_transcriber()
    
    if not transcriber or not transcriber.is_available():
        logger.warning("Whisper transcriber not available, skipping audio transcription")
        return
    
    try:
        # Convert bytearray to bytes
        audio_bytes = bytes(audio_buffer)
        
        # Convert WebM to WAV if needed (for Whisper compatibility)
        if audio_format == "webm":
            audio_bytes = await convert_webm_to_wav(audio_bytes)
            if not audio_bytes:
                logger.warning("Failed to convert WebM to WAV, skipping transcription")
                return
        
        # Transcribe audio
        transcript = transcriber.transcribe_pcm_or_wav(audio_bytes)
        
        if not transcript or not transcript.strip():
            logger.debug("Empty transcription result")
            return
        
        # CRITICAL: Ensure transcript ends with punctuation for proper sentence segmentation
        # This helps split sentences correctly for claim detection
        processed_transcript = transcript.strip()
        if processed_transcript and not processed_transcript.endswith(('.', '!', '?', ',', ';', ':')):
            processed_transcript += '.'
        
        # Emit transcript_final event
        if not await safe_send_json(websocket, {
            "type": "transcript_final",
            "text": processed_transcript,
            "t_start": 0.0,  # TODO: calculate from buffer timing
            "t_end": 0.0,
        }):
            return  # Connection closed
        
        # Process transcript through pipeline
        await process_text(websocket, processed_transcript)
    
    except Exception as e:
        logger.error(f"Error transcribing audio buffer: {e}", exc_info=True)
        await safe_send_json(websocket, {
            "type": "error",
            "error": f"Transcription error: {str(e)}"
        })


async def convert_webm_to_wav(webm_bytes: bytes) -> Optional[bytes]:
    """
    Convert WebM audio bytes to WAV format for Whisper compatibility.
    
    Args:
        webm_bytes: WebM audio data as bytes
        
    Returns:
        WAV audio data as bytes, or None if conversion fails
    """
    try:
        import subprocess
        import tempfile
        import os
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as webm_file:
            webm_file.write(webm_bytes)
            webm_path = webm_file.name
        
        wav_path = webm_path.replace('.webm', '.wav')
        
        try:
            # Use ffmpeg to convert WebM to WAV
            # ffmpeg -i input.webm -ar 16000 -ac 1 -f wav output.wav
            subprocess.run(
                [
                    'ffmpeg', '-i', webm_path,
                    '-ar', '16000',  # Sample rate: 16kHz (Whisper default)
                    '-ac', '1',      # Mono channel
                    '-f', 'wav',     # WAV format
                    '-y',            # Overwrite output file
                    wav_path
                ],
                check=True,
                capture_output=True,
                timeout=5  # 5 second timeout
            )
            
            # Read WAV file
            with open(wav_path, 'rb') as wav_file:
                wav_bytes = wav_file.read()
            
            return wav_bytes
            
        finally:
            # Clean up temporary files
            try:
                if os.path.exists(webm_path):
                    os.unlink(webm_path)
                if os.path.exists(wav_path):
                    os.unlink(wav_path)
            except Exception as cleanup_error:
                logger.warning(f"Error cleaning up temp files: {cleanup_error}")
    
    except subprocess.CalledProcessError as e:
        logger.warning(f"ffmpeg conversion failed: {e}")
        return None
    except FileNotFoundError:
        logger.warning("ffmpeg not found. WebM conversion requires ffmpeg to be installed.")
        return None
    except Exception as e:
        logger.error(f"Error converting WebM to WAV: {e}", exc_info=True)
        return None


async def process_text(websocket: WebSocket, text: str):
    """
    Process text through the fact-checking pipeline.
    
    Segments text, processes sentences through orchestrator, and emits events.
    """
    if not text or not text.strip():
        return
    
    # Segment text into sentences
    sentences = split_sentences(text)
    
    if not sentences:
        return
    
    # Track claim_id mapping (normalized_claim -> claim_id) across all sentences
    claim_id_map: Dict[str, str] = {}
    
    # Process each sentence through orchestrator
    for sentence in sentences:
        # Track start time for latency measurement
        factcheck_start_time = time.perf_counter()
        
        # Get events from orchestrator
        events = await process_sentence(sentence)
        
        # Process events and emit to client
        for event in events:
            event_type = event.get("event_type")
            
            if event_type == "claim_detected":
                # Generate UUID for claim_id
                normalized_claim = event.get("normalized_claim", "")
                if normalized_claim and normalized_claim not in claim_id_map:
                    claim_id_map[normalized_claim] = str(uuid.uuid4())
                
                claim_id = claim_id_map.get(normalized_claim, str(uuid.uuid4()))
                
                # Emit claim_detected event
                if not await safe_send_json(websocket, {
                    "type": "claim_detected",
                    "claim_id": claim_id,
                    "claim_text": event.get("claim_text", ""),
                    "detector_confidence": event.get("confidence", 0.0),
                }):
                    return  # Connection closed, stop processing
            
            elif event_type == "factcheck_result":
                # Get claim_id for this normalized_claim
                normalized_claim = event.get("normalized_claim", "")
                claim_id = claim_id_map.get(normalized_claim, str(uuid.uuid4()))
                
                # Extract result data (handle None case)
                result = event.get("result")
                if result is None:
                    logger.warning(f"factcheck_result event has None result for claim_id={claim_id}")
                    # Skip this event or emit error
                    continue
                
                label = result.get("label", "NOT_ENOUGH_INFO")
                answer = result.get("answer", "")  # Natural language answer
                best_evidence = result.get("best_evidence")
                
                # Build evidence array (single item for now, can be extended)
                evidence = []
                if best_evidence:
                    evidence.append({
                        "title": best_evidence.get("title", ""),
                        "url": best_evidence.get("url", ""),
                        "snippet": best_evidence.get("snippet", ""),
                        "source": best_evidence.get("source", ""),
                    })
                
                # Calculate confidence from scores if available
                scores = result.get("scores", [])
                confidence = 0.0
                if scores:
                    # Use max entailment from scores as confidence
                    confidence = max(
                        (s.get("entailment", 0.0) for s in scores),
                        default=0.0
                    )
                
                # Calculate latency (milliseconds)
                factcheck_end_time = time.perf_counter()
                latency_ms = int((factcheck_end_time - factcheck_start_time) * 1000)
                
                # Emit factcheck_result event
                event_data = {
                    "type": "factcheck_result",
                    "claim_id": claim_id,
                    "label": label,
                    "confidence": round(confidence, 2),
                    "evidence": evidence,
                    "latency_ms": latency_ms,
                }
                # Add answer if available
                if answer:
                    event_data["answer"] = answer
                
                if not await safe_send_json(websocket, event_data):
                    return  # Connection closed, stop processing
            
            elif event_type == "factcheck_error":
                # Emit error event (optional, for debugging)
                normalized_claim = event.get("normalized_claim", "")
                claim_id = claim_id_map.get(normalized_claim, str(uuid.uuid4()))
                
                logger.warning(
                    f"Fact-check error for claim_id={claim_id}: {event.get('error', 'Unknown error')}"
                )
                if not await safe_send_json(websocket, {
                    "type": "factcheck_error",
                    "claim_id": claim_id,
                    "error": event.get("error", "Unknown error"),
                }):
                    return  # Connection closed, stop processing

