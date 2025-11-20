# Diabetes Checkup Voicebot (Telephony AI)

This is a real-time voicebot for diabetes checkups using Twilio, Deepgram, OpenAI GPT-4o, and ElevenLabs.

## Architecture

- **Twilio**: Telephony Gateway (Media Streams)
- **FastAPI**: WebSocket Server (Orchestrator)
- **Deepgram**: Speech-to-Text (Streaming)
- **OpenAI GPT-4o**: Intelligence & Medical Dialogue
- **ElevenLabs**: Text-to-Speech (Streaming)
- **Silero VAD**: Barge-in detection

## Prerequisites

- Python 3.10+
- `uv` (Package Manager)
- `ngrok` (For local tunneling)
- API Keys for:
  - Twilio (Account SID, Auth Token)
  - Deepgram
  - OpenAI
  - ElevenLabs

## Setup

1. **Install Dependencies**
   Using `uv` to sync dependencies from `pyproject.toml`:
   ```bash
   uv sync
   ```

2. **Environment Variables**
   Copy `env.example` to `.env` and fill in your API keys.
   ```bash
   cp env.example .env
   ```

3. **Run the Server**
   ```bash
   uv run uvicorn app.main:app --reload --port 8000
   ```

4. **Expose to Internet (ngrok)**
   In a new terminal:
   ```bash
   ngrok http 8000
   ```
   Copy the forwarding URL (e.g., `https://xxxx-xxxx.ngrok-free.app`).

5. **Configure Twilio**
   - Go to Twilio Console > Phone Numbers > Manage > Active Numbers.
   - Click on your number.
   - Under "Voice & Fax", configure "A Call Comes In":
     - **Webhook**: TwiML Bin (We need to create one) or direct URL?
     - Actually, for Media Streams, it's best to use TwiML.

   **Create a TwiML Bin in Twilio Console:**
   ```xml
   <?xml version="1.0" encoding="UTF-8"?>
   <Response>
       <Connect>
           <Stream url="wss://xxxx-xxxx.ngrok-free.app/media-stream" />
       </Connect>
   </Response>
   ```
   Replace `wss://xxxx-xxxx.ngrok-free.app` with your ngrok URL (note `wss://`).

## Notes

- **Voice ID**: The default voice ID in `app/main.py` is "Rachel". For better Japanese results, clone a Japanese voice in ElevenLabs and replace the `ELEVENLABS_VOICE_ID`.
- **Latency**: The system uses streaming for STT, LLM, and TTS to minimize latency.
- **Barge-in**: Silero VAD detects user speech to interrupt the AI.


