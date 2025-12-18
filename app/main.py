import os
import json
import asyncio
import base64
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from app.services.stt import STTService
from app.services.llm import LLMService
from app.services.tts import TTSService
from app.services.vad import VADService
from app.services.session_storage import SessionStorage
from app.services.report import ReportService
from app.services.pdf_generator import generate_pdf
from app.models.session import ConversationSession
from app.utils.audio import ulaw_to_pcm16, resample_audio, pcm16_to_ulaw

load_dotenv()

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Configuration
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Using OpenAI TTS instead of ElevenLabs (ElevenLabs free tier blocked on Railway)
# Voice options: alloy, echo, fable, onyx, nova, shimmer (nova or shimmer work well for Japanese)
OPENAI_TTS_VOICE = "nova"  # Good for Japanese 

@app.get("/")
async def get():
    return FileResponse("app/static/index.html")

@app.get("/tts-test")
async def tts_test():
    return FileResponse("app/static/tts-test.html")

# Greeting audio cache
_greeting_audio_cache = None
# 挨拶テキスト（表示用と読み上げ用を分離）
_greeting_display = "こんにちは。定期検診システムのサクラです。本日もお電話いただきありがとうございます。お体の調子はいかがですか？"
_greeting_speech = "こんにちは。ていきけんしんシステムの サクラです。ほんじつも おでんわいただき ありがとうございます。おからだの ちょうしは いかがですか？"

@app.get("/api/preload-greeting")
async def preload_greeting():
    """プリロード用の挨拶音声を生成して返す"""
    global _greeting_audio_cache
    
    if _greeting_audio_cache is None:
        import aiohttp
        import audioop
        from scipy import signal
        import websockets
        
        elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        elevenlabs_voice_id = os.getenv("ELEVENLABS_VOICE_ID", "GxxMAMfQkDlnqjpzjLHH")
        elevenlabs_model = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
        
        if elevenlabs_api_key:
            # Use ElevenLabs for consistent voice
            try:
                ws_url = f"wss://api.elevenlabs.io/v1/text-to-speech/{elevenlabs_voice_id}/stream-input?model_id={elevenlabs_model}&output_format=ulaw_8000&optimize_streaming_latency=3"
                headers = {"xi-api-key": elevenlabs_api_key}
                
                async with websockets.connect(ws_url, extra_headers=headers) as ws:
                    # Send text with voice settings
                    init_message = {
                        "text": _greeting_speech,
                        "voice_settings": {
                            "stability": 0.5,
                            "similarity_boost": 0.75,
                            "style": 0.0,
                            "use_speaker_boost": True
                        },
                        "xi_api_key": elevenlabs_api_key
                    }
                    await ws.send(json.dumps(init_message))
                    await ws.send(json.dumps({"text": ""}))
                    
                    # Collect audio chunks
                    audio_chunks = []
                    async for message in ws:
                        try:
                            data = json.loads(message)
                            if "audio" in data and data["audio"]:
                                audio_chunks.append(base64.b64decode(data["audio"]))
                            if data.get("isFinal"):
                                break
                        except:
                            pass
                    
                    if audio_chunks:
                        full_audio = b"".join(audio_chunks)
                        _greeting_audio_cache = base64.b64encode(full_audio).decode('utf-8')
                        print(f"Preloaded greeting with ElevenLabs: {len(full_audio)} bytes")
                    else:
                        raise Exception("No audio received from ElevenLabs")
                        
            except Exception as e:
                print(f"ElevenLabs preload failed: {e}, falling back to OpenAI")
                elevenlabs_api_key = None  # Fall through to OpenAI
        
        if not elevenlabs_api_key or _greeting_audio_cache is None:
            # Fallback to OpenAI TTS
            url = "https://api.openai.com/v1/audio/speech"
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "tts-1",
                "input": _greeting_speech,
                "voice": OPENAI_TTS_VOICE,
                "response_format": "pcm",
                "speed": 1.0
            }
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, headers=headers) as response:
                        if response.status == 200:
                            pcm_data = await response.read()
                            audio_np = np.frombuffer(pcm_data, dtype=np.int16)
                            num_samples = int(len(audio_np) * 8000 / 24000)
                            resampled = signal.resample(audio_np, num_samples)
                            resampled_pcm = resampled.astype(np.int16).tobytes()
                            ulaw_data = audioop.lin2ulaw(resampled_pcm, 2)
                            _greeting_audio_cache = base64.b64encode(ulaw_data).decode('utf-8')
                            print(f"Preloaded greeting with OpenAI: {len(ulaw_data)} bytes")
            except Exception as e:
                print(f"Error preloading greeting: {e}")
                return {"status": "error", "message": str(e)}
    
    return {
        "status": "ok",
        "text": _greeting_display,
        "audio": _greeting_audio_cache
    }

# --- Report Endpoints ---
@app.get("/reports")
async def list_reports():
    """レポート一覧を取得"""
    from app.services.session_storage import SessionStorage
    sessions = SessionStorage.list_sessions(limit=50)
    
    reports = []
    for session in sessions:
        # レポートファイルを読み込み
        from pathlib import Path
        report_path = Path("data/reports") / f"{session.session_id}.json"
        if report_path.exists():
            with open(report_path, "r", encoding="utf-8") as f:
                report_data = json.load(f)
                reports.append({
                    "session_id": session.session_id,
                    "examination_date": report_data.get("examination_date", session.start_time.isoformat()),
                    "summary": report_data.get("summary", ""),
                    "phone_number": session.phone_number
                })
    
    return {"reports": reports}

@app.get("/reports/{session_id}")
async def get_report(session_id: str):
    """レポート詳細を取得"""
    from pathlib import Path
    report_path = Path("data/reports") / f"{session_id}.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    
    return report

@app.get("/reports/{session_id}/pdf")
async def get_report_pdf(session_id: str):
    """レポートをPDFでダウンロード"""
    from pathlib import Path
    report_path = Path("data/reports") / f"{session_id}.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    
    pdf_buffer = generate_pdf(report)
    
    return Response(
        content=pdf_buffer.read(),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="report_{session_id}.pdf"'
        }
    )

@app.get("/reports/{session_id}/html")
async def get_report_html(session_id: str):
    """レポートをHTMLで表示"""
    return FileResponse("app/static/report.html")

@app.get("/reports-list")
async def reports_list_page():
    """レポート一覧ページ"""
    return FileResponse("app/static/reports.html")

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Browser WebSocket Endpoint ---
@app.websocket("/browser-stream")
async def browser_websocket_endpoint(websocket: WebSocket):
    logger.info("WebSocket connection request received")
    await websocket.accept()
    logger.info("Browser connected.")
    
    # Initialize session
    session = ConversationSession()
    
    # Reuse the same logic but adapt audio formats
    # Browser sends JSON with Base64 PCM16 (8kHz ideally, or we downsample)
    # We send back JSON with Base64 u-law (or PCM16 if we change browser code, but let's stick to u-law for consistency with Twilio flow logic for now, browser will decode)

    vad_service = VADService()
    llm_service = LLMService(OPENAI_API_KEY)
    tts_service = TTSService(OPENAI_API_KEY, voice_id=OPENAI_TTS_VOICE)
    
    # VAD buffer for accumulating audio samples
    vad_audio_buffer = []
    VAD_MIN_SAMPLES = 512  # Silero VAD minimum requirement
    browser_sample_rate = 44100  # Default, can be updated by client
    
    stt_service = None
    is_ai_speaking = False
    interrupt_event = asyncio.Event()
    
    # Turn management
    current_conversation_task = None
    conversation_lock = asyncio.Lock()
    pending_user_text = None
    
    # Transcript buffering for VAD-based processing
    transcript_buffer = []
    vad_check_task = None
    speculative_llm_task = None  # Background LLM task
    speculative_result = None  # Store speculative LLM result
    last_processed_text = ""  # 最後に処理したテキストを追跡
    user_speaking_detected = False  # Track if user started speaking
    last_transcript_time = 0.0  # Track when last transcript was received
    VAD_CHECK_INTERVAL = 0.05  # Check VAD state every 50ms (faster polling)
    MAX_WAIT_TIME = 8.0  # Max time to wait for speech end (safety limit)
    
    def get_adaptive_wait_time(db_level: float) -> float:
        """Calculate wait time based on dB level.
        Lower dB = more confident speech ended = shorter wait."""
        if db_level < -50:
            return 0.3  # Very quiet, almost certainly done
        elif db_level < -45:
            return 0.5  # Quiet, probably done
        elif db_level < -40:
            return 0.8  # Moderate, might be done
        else:
            return 1.5  # Still audible, wait longer

    async def run_speculative_llm(text: str):
        """Run LLM speculatively in background."""
        nonlocal speculative_result
        try:
            # Get full response (non-streaming for speculative)
            full_display = ""
            full_speech = ""
            async for display_text, speech_text in llm_service.process_text(text):
                full_display = display_text
                full_speech = speech_text
            speculative_result = (full_display, full_speech)
            logger.info(f"Speculative LLM completed: {full_display[:50]}...")
        except asyncio.CancelledError:
            logger.info("Speculative LLM cancelled (user continued speaking)")
            speculative_result = None
            raise
        except Exception as e:
            logger.error(f"Speculative LLM error: {e}")
            speculative_result = None

    async def check_vad_and_process():
        """Adaptive timing + speculative processing.
        1. Start LLM speculatively
        2. Wait with adaptive timing based on dB
        3. If user continues speaking, cancel speculative LLM
        4. If user done, use speculative result for faster response"""
        nonlocal transcript_buffer, is_ai_speaking, current_conversation_task
        nonlocal last_processed_text, user_speaking_detected
        nonlocal speculative_llm_task, speculative_result
        
        import time
        start_time = time.time()
        initial_transcript = "".join(transcript_buffer).strip()
        
        if not initial_transcript:
            return
        
        # Start speculative LLM immediately
        speculative_result = None
        if speculative_llm_task and not speculative_llm_task.done():
            speculative_llm_task.cancel()
            try:
                await speculative_llm_task
            except asyncio.CancelledError:
                pass
        speculative_llm_task = asyncio.create_task(run_speculative_llm(initial_transcript))
        logger.info(f"Started speculative LLM for: {initial_transcript[:30]}...")
        
        # Adaptive wait loop with dual inference:
        # 1. dB drop detection (fast path)
        # 2. Silence duration (fallback)
        silence_start_time = None
        required_silence_duration = 0.5  # Reduced from 1.5s since we have dB drop detection
        
        while True:
            elapsed = time.time() - start_time
            
            # Safety: max wait time
            if elapsed > MAX_WAIT_TIME:
                logger.info(f"Max wait time reached ({MAX_WAIT_TIME}s)")
                break
            
            # Get current dB level
            current_db = vad_service._current_db
            is_silent = current_db < -40
            
            # Update dB history for drop detection
            vad_service.update_db_history(current_db)
            
            # FAST PATH: Check for significant dB drop
            if vad_service.detect_db_drop():
                drop_info = vad_service.get_db_drop_info()
                logger.info(f"dB drop detected! peak={drop_info['recent_peak']:.1f}dB -> current={drop_info['current_db']:.1f}dB (relative drop={drop_info['relative_drop']:.0%})")
                break
            
            # FALLBACK: Traditional silence duration check
            if is_silent:
                if silence_start_time is None:
                    silence_start_time = time.time()
                
                silence_duration = time.time() - silence_start_time
                
                # Check if we've had enough silence
                if silence_duration >= required_silence_duration:
                    logger.info(f"Silence detected: {silence_duration:.2f}s at {current_db:.1f}dB")
                    break
            else:
                # Reset silence timer - user is still speaking
                if silence_start_time is not None:
                    logger.debug(f"Silence interrupted at {current_db:.1f}dB")
                silence_start_time = None
            
            await asyncio.sleep(VAD_CHECK_INTERVAL)
        
        # Check if more text was added while waiting
        current_transcript = "".join(transcript_buffer).strip()
        
        if current_transcript != initial_transcript:
            # User continued speaking! Cancel speculative LLM and restart
            logger.info(f"User continued speaking: '{initial_transcript[:20]}...' -> '{current_transcript[:20]}...'")
            if speculative_llm_task and not speculative_llm_task.done():
                speculative_llm_task.cancel()
                try:
                    await speculative_llm_task
                except asyncio.CancelledError:
                    pass
            speculative_result = None
            # Restart the process with updated transcript
            vad_check_task = asyncio.create_task(check_vad_and_process())
            return
        
        # User is done - process the transcript
        full_transcript = current_transcript
        transcript_buffer.clear()
        user_speaking_detected = False
        
        if not full_transcript:
            return
        
        # 同じテキストを重複処理しない
        if full_transcript == last_processed_text:
            logger.info(f"Skipping duplicate transcript: {full_transcript[:30]}...")
            return
        
        last_processed_text = full_transcript
        logger.info(f"Processing transcript: {full_transcript}")
        
        # 最終的なトランスクリプトをUIに送信
        try:
            await websocket.send_json({
                "event": "transcript",
                "text": full_transcript,
                "is_final": True
            })
        except Exception as e:
            logger.error(f"Error sending final transcript: {e}")
        
        # セッションにメッセージを記録
        session.add_message("user", full_transcript)
        
        # AIが話している場合は割り込み
        if is_ai_speaking:
            logger.info("User interrupted AI, canceling current response")
            await handle_interruption()
            await asyncio.sleep(0.1)
        
        # 前の会話タスクがあればキャンセル
        if current_conversation_task and not current_conversation_task.done():
            current_conversation_task.cancel()
            try:
                await current_conversation_task
            except asyncio.CancelledError:
                pass
        
        # Use speculative result if available, otherwise start normal conversation
        if speculative_result and speculative_llm_task and speculative_llm_task.done():
            logger.info("Using speculative LLM result for faster response!")
            current_conversation_task = asyncio.create_task(
                process_conversation_with_result(full_transcript, speculative_result)
            )
        else:
            # Speculative result not ready, wait for it or start fresh
            if speculative_llm_task and not speculative_llm_task.done():
                try:
                    await asyncio.wait_for(speculative_llm_task, timeout=2.0)
                    if speculative_result:
                        logger.info("Speculative LLM finished just in time!")
                        current_conversation_task = asyncio.create_task(
                            process_conversation_with_result(full_transcript, speculative_result)
                        )
                        return
                except asyncio.TimeoutError:
                    logger.info("Speculative LLM taking too long, starting fresh")
            current_conversation_task = asyncio.create_task(process_conversation(full_transcript))

    async def process_conversation_with_result(user_text: str, llm_result: tuple):
        """Process conversation using pre-computed LLM result."""
        nonlocal is_ai_speaking
        
        async with conversation_lock:
            is_ai_speaking = True
            interrupt_event.clear()
            
            try:
                await websocket.send_json({"event": "ai_started"})
            except:
                pass

            try:
                full_display_response, full_speech_response = llm_result
                
                if not full_display_response or interrupt_event.is_set():
                    return
                
                # 表示テキストを即座にUIに送信
                try:
                    await websocket.send_json({
                        "event": "ai_text",
                        "text": full_display_response,
                        "is_partial": False
                    })
                except Exception as e:
                    logger.error(f"Error sending AI text: {e}")
                
                # TTS streaming
                async def sentence_generator():
                    sentences = []
                    current = ""
                    for char in full_speech_response:
                        current += char
                        if char in ['。', '！', '？', '!', '?', '\n']:
                            sentences.append(current.strip())
                            current = ""
                    if current.strip():
                        sentences.append(current.strip())
                    
                    for sentence in sentences:
                        if interrupt_event.is_set():
                            break
                        if sentence:
                            yield sentence

                audio_stream = tts_service.stream_text(sentence_generator())
                
                async for audio_chunk in audio_stream:
                    if interrupt_event.is_set():
                        break
                    
                    audio_b64 = base64.b64encode(audio_chunk).decode('utf-8')
                    await websocket.send_json({
                        "event": "media",
                        "media": {"payload": audio_b64}
                    })
                
                if full_display_response and not interrupt_event.is_set():
                    session.add_message("assistant", full_display_response)
                    
            except asyncio.CancelledError:
                logger.info("Conversation task cancelled")
                raise
            except Exception as e:
                logger.error(f"Error in conversation: {e}")
            finally:
                is_ai_speaking = False
                if not interrupt_event.is_set():
                    try:
                        await websocket.send_json({"event": "ai_finished"})
                    except:
                        pass

    async def stt_callback(transcript: str):
        nonlocal transcript_buffer, vad_check_task, user_speaking_detected, last_transcript_time
        if not transcript.strip():
            return
        logger.info(f"User (Browser) fragment: {transcript}")
        
        import time
        last_transcript_time = time.time()
        
        # Reset VAD speech state
        vad_service._last_speech_time = time.time()
        vad_service._is_speaking = True
        
        # Add to buffer
        transcript_buffer.append(transcript)
        user_speaking_detected = True
        
        # ブラウザにリアルタイム文字起こしを送信（プレビュー用）
        try:
            await websocket.send_json({
                "event": "transcript",
                "text": transcript,
                "is_final": False,
                "buffered": "".join(transcript_buffer)
            })
        except Exception as e:
            logger.error(f"Error sending transcript: {e}")
        
        # Start VAD check task only if not already running
        if vad_check_task is None or vad_check_task.done():
            vad_check_task = asyncio.create_task(check_vad_and_process())

    async def handle_interruption():
        nonlocal is_ai_speaking
        interrupt_event.set()
        is_ai_speaking = False
        try:
            await websocket.send_json({"event": "clear"})
            # AIが停止したことを通知
            await websocket.send_json({
                "event": "ai_stopped",
                "reason": "interrupted"
            })
        except Exception as e:
            logger.error(f"Error sending interrupt: {e}")
        logger.info("Browser Interruption handled.")

    async def process_conversation(user_text: str):
        nonlocal is_ai_speaking
        
        async with conversation_lock:
            is_ai_speaking = True
            interrupt_event.clear()
            
            # AIが話し始めることを通知
            try:
                await websocket.send_json({
                    "event": "ai_started"
                })
            except:
                pass

            try:
                text_stream = llm_service.process_text(user_text)
                
                full_display_response = ""
                full_speech_response = ""
                
                # LLMからの応答を取得
                async for display_text, speech_text in text_stream:
                    if interrupt_event.is_set():
                        break
                    full_display_response = display_text
                    full_speech_response = speech_text
                
                if not full_display_response or interrupt_event.is_set():
                    return
                
                # 表示テキストを即座にUIに送信
                try:
                    await websocket.send_json({
                        "event": "ai_text",
                        "text": full_display_response,
                        "is_partial": False
                    })
                except Exception as e:
                    logger.error(f"Error sending AI text: {e}")
                
                # TTSはspeechテキストを文単位でストリーミング
                async def sentence_generator():
                    """speechテキストを文単位で分割してyield"""
                    sentences = []
                    current = ""
                    for char in full_speech_response:
                        current += char
                        if char in ['。', '！', '？', '!', '?', '\n']:
                            sentences.append(current.strip())
                            current = ""
                    if current.strip():
                        sentences.append(current.strip())
                    
                    for sentence in sentences:
                        if interrupt_event.is_set():
                            break
                        if sentence:
                            yield sentence

                # TTS streaming
                audio_stream = tts_service.stream_text(sentence_generator())
                
                async for audio_chunk in audio_stream:
                    if interrupt_event.is_set():
                        break
                    
                    # Send as Base64 u-law
                    audio_b64 = base64.b64encode(audio_chunk).decode('utf-8')
                    await websocket.send_json({
                        "event": "media",
                        "media": {"payload": audio_b64}
                    })
                
                # セッションにAIの応答を記録（表示用テキスト）
                if full_display_response and not interrupt_event.is_set():
                    session.add_message("assistant", full_display_response)
                    
            except asyncio.CancelledError:
                logger.info("Conversation task cancelled")
                raise
            except Exception as e:
                logger.error(f"Error in conversation: {e}")
            finally:
                is_ai_speaking = False
                if not interrupt_event.is_set():
                    # AIが話し終わったことを通知
                    try:
                        await websocket.send_json({
                            "event": "ai_finished"
                        })
                    except:
                        pass

    # Initialize STT (Deepgram)
    logger.info("Initializing STT service...")
    stt_service = STTService(DEEPGRAM_API_KEY, stt_callback)
    if await stt_service.start() is False:
        logger.error("Failed to start STT service")
        await websocket.close()
        return
    logger.info("STT service started successfully")

    # Connect TTS service
    logger.info("Connecting to TTS service...")
    try:
        await tts_service.connect()
        logger.info("TTS service connected")
    except Exception as e:
        logger.error(f"Failed to connect to TTS service: {e}")

    # Flag to track if greeting was sent
    greeting_sent = False
    
    async def send_greeting():
        nonlocal greeting_sent
        if greeting_sent:
            return
        greeting_sent = True
        try:
            # TTS用にひらがな多めのテキストを使用
            greeting_speech = "こんにちは。ていきけんしんシステムです。おはなしください。"
            logger.info(f"Generating TTS for greeting")
            async def greeting_generator():
                yield greeting_speech
            
            audio_stream = tts_service.stream_text(greeting_generator())
            chunk_count = 0
            total_bytes = 0
            async for audio_chunk in audio_stream:
                if audio_chunk:
                    audio_b64 = base64.b64encode(audio_chunk).decode('utf-8')
                    total_bytes += len(audio_chunk)
                    chunk_count += 1
                    await websocket.send_json({
                        "event": "media",
                        "media": {"payload": audio_b64}
                    })
            logger.info(f"Initial greeting sent: {chunk_count} chunks, {total_bytes} total bytes")
        except Exception as e:
            logger.error(f"Error in send_greeting: {e}", exc_info=True)

    # Message receiving loop - must be outside the TTS connection try-except
    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            logger.info(f"Received message: {data.get('event', 'unknown')}")

            if data['event'] == 'media':
                payload = data['media']['payload']
                # Browser sends PCM16, but sample rate is usually 44100Hz or 48000Hz (not 8000Hz)
                pcm_data = base64.b64decode(payload)
                
                # Browser AudioContext is typically 44100Hz or 48000Hz, not 8000Hz
                # We need to resample to 8000Hz for Deepgram
                audio_np = np.frombuffer(pcm_data, dtype=np.int16)
                
                # Use configurable sample rate
                original_rate = browser_sample_rate
                target_rate = 8000
                
                # Resample to 8000Hz
                num_samples = int(len(audio_np) * target_rate / original_rate)
                if num_samples > 0:
                    from scipy import signal
                    resampled_audio = signal.resample(audio_np, num_samples)
                    resampled_pcm = resampled_audio.astype(np.int16).tobytes()
                    
                    # 1. Send to STT (convert PCM16 -> u-law)
                    # IMPORTANT: Skip STT while AI is speaking to prevent echo
                    if not is_ai_speaking:
                        ulaw_data = pcm16_to_ulaw(resampled_pcm)
                        asyncio.create_task(stt_service.send_audio(ulaw_data))
                    
                    # 2. VAD Check (needs Float32 normalized to [-1, 1])
                    # Buffer audio samples until we have enough for VAD
                    float_data = resampled_audio.astype(np.float32) / 32768.0
                    vad_audio_buffer.extend(float_data.tolist())
                    
                    # Process VAD when buffer has enough samples
                    if len(vad_audio_buffer) >= VAD_MIN_SAMPLES:
                        try:
                            buffer_array = np.array(vad_audio_buffer, dtype=np.float32)
                            vad_result = vad_service.process_audio_chunk(buffer_array)
                            if vad_result['speech_start']:
                                logger.info("VAD (Browser): Speech start detected.")
                                if is_ai_speaking:
                                    await handle_interruption()
                        except Exception as vad_error:
                            logger.debug(f"VAD error: {vad_error}")
                        finally:
                            # Clear buffer after processing
                            vad_audio_buffer.clear()
            
            elif data['event'] == 'config':
                # Allow browser to send its sample rate and greeting status
                if 'sampleRate' in data:
                    browser_sample_rate = data['sampleRate']
                    logger.info(f"Browser sample rate set to: {browser_sample_rate}Hz")
                
                # Check if greeting was preloaded by browser
                greeting_preloaded = data.get('greetingPreloaded', False)
                if not greeting_preloaded:
                    # Browser didn't preload greeting, so server sends it
                    logger.info("Greeting not preloaded, sending from server")
                    asyncio.create_task(send_greeting())
                else:
                    logger.info("Greeting already preloaded by browser, skipping server greeting")

    except WebSocketDisconnect:
        logger.info("Browser disconnected")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        await stt_service.stop()
        await tts_service.close()
        # セッションを終了して保存
        session.finish()
        SessionStorage.save_session(session)
        # レポートを生成して保存
        try:
            report_service = ReportService(OPENAI_API_KEY)
            report = await report_service.generate_report(session)
            # レポートも保存
            from pathlib import Path
            reports_dir = Path("data/reports")
            reports_dir.mkdir(parents=True, exist_ok=True)
            with open(reports_dir / f"{session.session_id}.json", "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"Report generated for session: {session.session_id}")
        except Exception as e:
            print(f"Error generating report: {e}")

# --- Twilio Endpoint (Keep existing) ---
@app.websocket("/media-stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Twilio connected.")

    # Initialize session
    session = ConversationSession()
    
    # Initialize services for this session
    vad_service = VADService()
    llm_service = LLMService(OPENAI_API_KEY)
    # TTS Service needs to be re-instantiated or connected per turn? 
    # ElevenLabs WS usually handles one stream. We might keep it open or reconnect.
    # For simplicity/robustness, let's keep one instance but manage connection state.
    tts_service = TTSService(OPENAI_API_KEY, voice_id=OPENAI_TTS_VOICE)
    
    # State
    stream_sid = None
    stt_service = None
    
    # Queues and Flags
    audio_queue = asyncio.Queue()
    is_ai_speaking = False
    interrupt_event = asyncio.Event()

    async def stt_callback(transcript: str):
        nonlocal is_ai_speaking
        if not transcript.strip():
            return
        
        print(f"User: {transcript}")
        # セッションにメッセージを記録
        session.add_message("user", transcript)
        
        # If user spoke, we process it.
        # If AI was speaking, we should have already interrupted via VAD, 
        # but if not, we do it here too.
        if is_ai_speaking:
            print("Interrupting AI due to transcript...")
            await handle_interruption()

        # Start response generation
        asyncio.create_task(process_conversation(transcript))

    async def handle_interruption():
        nonlocal is_ai_speaking
        interrupt_event.set()
        is_ai_speaking = False
        
        # Send clear message to Twilio
        if stream_sid:
            await websocket.send_json({
                "event": "clear",
                "streamSid": stream_sid
            })
        print("Interruption handled.")

    async def process_conversation(user_text: str):
        nonlocal is_ai_speaking
        is_ai_speaking = True
        interrupt_event.clear()

        try:
            # LLM Stream
            text_stream = llm_service.process_text(user_text)
            
            full_display_response = ""
            full_speech_response = ""
            
            # LLMからの応答を取得
            async for display_text, speech_text in text_stream:
                if interrupt_event.is_set():
                    print("LLM generation interrupted.")
                    break
                full_display_response = display_text
                full_speech_response = speech_text
            
            if not full_display_response or interrupt_event.is_set():
                return
            
            # TTSはspeechテキストを文単位でストリーミング
            async def sentence_generator():
                """speechテキストを文単位で分割してyield"""
                sentences = []
                current = ""
                for char in full_speech_response:
                    current += char
                    if char in ['。', '！', '？', '!', '?', '\n']:
                        sentences.append(current.strip())
                        current = ""
                if current.strip():
                    sentences.append(current.strip())
                
                for sentence in sentences:
                    if interrupt_event.is_set():
                        break
                    if sentence:
                        yield sentence

            # TTS Output
            audio_stream = tts_service.stream_text(sentence_generator())
            
            async for audio_chunk in audio_stream:
                if interrupt_event.is_set():
                    print("TTS playback interrupted.")
                    break
                
                # Convert back to Base64
                audio_b64 = base64.b64encode(audio_chunk).decode('utf-8')
                
                media_message = {
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {
                        "payload": audio_b64
                    }
                }
                await websocket.send_json(media_message)
            
            # セッションにAIの応答を記録（表示用テキスト）
            if full_display_response and not interrupt_event.is_set():
                session.add_message("assistant", full_display_response)
                
        except Exception as e:
            print(f"Error in conversation loop: {e}")
        finally:
            if not interrupt_event.is_set():
                is_ai_speaking = False

    # Initialize STT
    stt_service = STTService(DEEPGRAM_API_KEY, stt_callback)
    if await stt_service.start() is False:
        await websocket.close()
        return

    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)

            if data['event'] == 'start':
                stream_sid = data['start']['streamSid']
                print(f"Stream started: {stream_sid}")
                # 電話番号があればセッションに記録
                if 'callSid' in data.get('start', {}):
                    # Twilioからの電話番号情報があれば取得できる
                    pass
                
                # Optional: Initial greeting
                asyncio.create_task(process_conversation("こんにちは。"))

            elif data['event'] == 'media':
                payload = data['media']['payload']
                # Decode audio (u-law)
                audio_data = base64.b64decode(payload)
                
                # 1. Send to STT
                await stt_service.send_audio(audio_data)
                
                # 2. VAD Check (for interruption)
                # Convert to PCM for VAD
                pcm_data = ulaw_to_pcm16(audio_data)
                # Normalize to float32
                float_data = resample_audio(pcm_data, 8000, 8000) 
                
                vad_result = vad_service.process_audio_chunk(float_data)
                
                if vad_result['speech_start']:
                    print("VAD: Speech start detected.")
                    if is_ai_speaking:
                        await handle_interruption()

            elif data['event'] == 'stop':
                print("Stream stopped.")
                break

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await stt_service.stop()
        await tts_service.close()
        # セッションを終了して保存
        session.finish()
        SessionStorage.save_session(session)
        # レポートを生成して保存
        try:
            report_service = ReportService(OPENAI_API_KEY)
            report = await report_service.generate_report(session)
            # レポートも保存
            from pathlib import Path
            reports_dir = Path("data/reports")
            reports_dir.mkdir(parents=True, exist_ok=True)
            with open(reports_dir / f"{session.session_id}.json", "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"Report generated for session: {session.session_id}")
        except Exception as e:
            print(f"Error generating report: {e}")
