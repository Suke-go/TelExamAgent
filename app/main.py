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
    
    stt_service = None
    is_ai_speaking = False
    interrupt_event = asyncio.Event()

    async def stt_callback(transcript: str):
        nonlocal is_ai_speaking
        if not transcript.strip():
            return
        logger.info(f"User (Browser): {transcript}")
        # セッションにメッセージを記録
        session.add_message("user", transcript)
        if is_ai_speaking:
            await handle_interruption()
        asyncio.create_task(process_conversation(transcript))

    async def handle_interruption():
        nonlocal is_ai_speaking
        interrupt_event.set()
        is_ai_speaking = False
        await websocket.send_json({"event": "clear"})
        logger.info("Browser Interruption handled.")

    async def process_conversation(user_text: str):
        nonlocal is_ai_speaking
        is_ai_speaking = True
        interrupt_event.clear()

        try:
            text_stream = llm_service.process_text(user_text)
            
            full_response = ""
            async def interruptible_text_generator():
                nonlocal full_response
                async for text in text_stream:
                    if interrupt_event.is_set():
                        break
                    full_response += text
                    yield text

            # ElevenLabs returns u-law 8000Hz (as per our TTSService config)
            audio_stream = tts_service.stream_text(interruptible_text_generator())
            
            async for audio_chunk in audio_stream:
                if interrupt_event.is_set():
                    break
                
                # Send as Base64 u-law
                audio_b64 = base64.b64encode(audio_chunk).decode('utf-8')
                await websocket.send_json({
                    "event": "media",
                    "media": {"payload": audio_b64}
                })
            
            # セッションにAIの応答を記録
            if full_response and not interrupt_event.is_set():
                session.add_message("assistant", full_response)
                
        except Exception as e:
            logger.error(f"Error in conversation: {e}")
        finally:
            if not interrupt_event.is_set():
                is_ai_speaking = False

    # Initialize STT (Deepgram)
    logger.info("Initializing STT service...")
    stt_service = STTService(DEEPGRAM_API_KEY, stt_callback)
    if await stt_service.start() is False:
        logger.error("Failed to start STT service")
        await websocket.close()
        return
    logger.info("STT service started successfully")

    # Connect TTS service before sending greeting
    logger.info("Connecting to TTS service...")
    try:
        await tts_service.connect()
        logger.info("TTS service connected")
        
        # Initial Greeting - send directly without LLM for faster response
        logger.info("Sending initial greeting: こんにちは。定期検診システムです。お話しください。")
        
        async def send_greeting():
            try:
                greeting_text = "こんにちは。定期検診システムです。お話しください。"
                logger.info(f"Generating TTS for: {greeting_text}")
                # Create a simple async generator for the greeting text
                async def greeting_generator():
                    yield greeting_text
                
                # Stream TTS audio
                audio_stream = tts_service.stream_text(greeting_generator())
                chunk_count = 0
                total_bytes = 0
                async for audio_chunk in audio_stream:
                    if audio_chunk:
                        audio_b64 = base64.b64encode(audio_chunk).decode('utf-8')
                        total_bytes += len(audio_chunk)
                        chunk_count += 1
                        logger.info(f"Sending audio chunk {chunk_count} ({len(audio_chunk)} bytes, base64: {len(audio_b64)} chars)")
                        await websocket.send_json({
                            "event": "media",
                            "media": {"payload": audio_b64}
                        })
                    else:
                        logger.warning("Received empty audio chunk from TTS")
                logger.info(f"Initial greeting sent: {chunk_count} chunks, {total_bytes} total bytes")
            except Exception as e:
                logger.error(f"Error in send_greeting: {e}", exc_info=True)
        
        asyncio.create_task(send_greeting())
    except Exception as e:
        logger.error(f"Failed to connect to TTS service: {e}")
        # Continue without TTS (might fail later but keeps connection open)

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
                # First, convert bytes to numpy array, resample, then convert back
                import numpy as np
                audio_np = np.frombuffer(pcm_data, dtype=np.int16)
                
                # Detect or assume sample rate (browser typically uses 44100Hz)
                # For now, assume 44100Hz (most common)
                original_rate = 44100
                target_rate = 8000
                
                # Resample to 8000Hz
                num_samples = int(len(audio_np) * target_rate / original_rate)
                if num_samples > 0:
                    from scipy import signal
                    resampled_audio = signal.resample(audio_np, num_samples)
                    resampled_pcm = resampled_audio.astype(np.int16).tobytes()
                    
                    # 1. Send to STT (convert PCM16 -> u-law)
                    ulaw_data = pcm16_to_ulaw(resampled_pcm)
                    # Note: send_audio is async but Deepgram's send() is synchronous
                    # We call it without await since it's fire-and-forget
                    asyncio.create_task(stt_service.send_audio(ulaw_data))
                    
                    # 2. VAD Check (needs Float32 normalized to [-1, 1])
                    # Silero VAD requires at least ~4000 samples (0.5s at 8kHz) to work properly
                    # Skip VAD for very short chunks to avoid "Input audio chunk is too short" error
                    if len(resampled_audio) >= 4000:
                        float_data = resampled_audio.astype(np.float32) / 32768.0
                        try:
                            vad_result = vad_service.process_audio_chunk(float_data)
                            if vad_result['speech_start']:
                                logger.info("VAD (Browser): Speech start detected.")
                                if is_ai_speaking:
                                    await handle_interruption()
                        except Exception as vad_error:
                            # Silently skip VAD errors for short chunks
                            pass

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
            
            full_response = ""
            # TTS Stream
            # We need to wrap text_stream to respect interrupt_event
            async def interruptible_text_generator():
                nonlocal full_response
                async for text in text_stream:
                    if interrupt_event.is_set():
                        print("LLM generation interrupted.")
                        break
                    full_response += text
                    yield text

            # TTS Output
            # We need to stream audio chunks back to Twilio
            audio_stream = tts_service.stream_text(interruptible_text_generator())
            
            async for audio_chunk in audio_stream:
                if interrupt_event.is_set():
                    print("TTS playback interrupted.")
                    break
                
                # Convert back to Base64
                # ElevenLabs sends us u-law 8000Hz if requested, so just base64 encode
                audio_b64 = base64.b64encode(audio_chunk).decode('utf-8')
                
                media_message = {
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {
                        "payload": audio_b64
                    }
                }
                await websocket.send_json(media_message)
            
            # セッションにAIの応答を記録
            if full_response and not interrupt_event.is_set():
                session.add_message("assistant", full_response)
                
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
