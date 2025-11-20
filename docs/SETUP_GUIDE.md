# API & Service Setup Guide

This document provides detailed instructions on how to set up the necessary external services (Twilio, Deepgram, OpenAI, ElevenLabs) and configure the local environment to run the Diabetes Checkup Voicebot.

## 1. ngrok Setup (Local Tunneling)

Since Twilio needs to connect to your local WebSocket server, we use `ngrok` to expose your local port (8000) to the internet.

1.  **Install ngrok**:
    -   Download and install from [ngrok.com](https://ngrok.com/download).
    -   Sign up for a free account to get an authtoken.
2.  **Authenticate**:
    ```bash
    ngrok config add-authtoken <YOUR_AUTH_TOKEN>
    ```
3.  **Start Tunnel**:
    Run this in a separate terminal window (keep it running):
    ```bash
    ngrok http 8000
    ```
4.  **Copy URL**:
    Note the forwarding URL that looks like `https://xxxx-xxxx.ngrok-free.app`.
    **Important**: For WebSockets, we will replace `https://` with `wss://`.

---

## 2. Twilio Setup (Telephony Gateway)

Twilio handles the phone connection and streams audio to our server.

1.  **Create Account**: Sign up at [twilio.com](https://www.twilio.com/).
2.  **Get a Phone Number**:
    -   Go to **Phone Numbers** > **Manage** > **Buy a number**.
    -   Search for a number (Japan region recommended for latency, but US works too) and buy/claim it.
3.  **Create TwiML Bin**:
    -   Go to **Runtime** (or **Developer Tools**) > **TwiML Bins**.
    -   Click **Create New TwiML Bin**.
    -   **Friendly Name**: `Voicebot Stream`
    -   **TwiML**:
        ```xml
        <?xml version="1.0" encoding="UTF-8"?>
        <Response>
            <Say language="ja-JP">検診システムにお繋ぎします。</Say>
            <Connect>
                <Stream url="wss://YOUR-NGROK-URL.ngrok-free.app/media-stream" />
            </Connect>
        </Response>
        ```
        *Replace `YOUR-NGROK-URL.ngrok-free.app` with the actual URL from Step 1.*
    -   Click **Create** (or Save).
4.  **Configure Phone Number**:
    -   Go to **Phone Numbers** > **Manage** > **Active Numbers**.
    -   Click on your purchased number.
    -   Scroll down to the **Voice & Fax** section.
    -   **A Call Comes In**: Select **TwiML Bin**.
    -   Select the `Voicebot Stream` bin you just created.
    -   Click **Save**.

---

## 3. API Keys Setup

You need to get API keys from the AI providers and add them to your `.env` file.

### A. Deepgram (STT)
1.  Sign up at [console.deepgram.com](https://console.deepgram.com/).
2.  Create a New API Key with `Member` permissions.
3.  Copy the key to `.env` as `DEEPGRAM_API_KEY`.

### B. OpenAI (LLM)
1.  Sign up at [platform.openai.com](https://platform.openai.com/).
2.  Go to **API Keys** and create a new secret key.
3.  Ensure you have credits/billing set up (GPT-4o requires a paid tier usually).
4.  Copy the key to `.env` as `OPENAI_API_KEY`.

### C. ElevenLabs (TTS)
1.  Sign up at [elevenlabs.io](https://elevenlabs.io/).
2.  Click on your profile icon > **Profile + API Key**.
3.  Click the "Eye" icon to reveal and copy the API Key.
4.  Paste it into `.env` as `ELEVENLABS_API_KEY`.

#### Voice ID Configuration
By default, the code uses a placeholder Voice ID. To use a high-quality Japanese voice:
1.  Go to **VoiceLab** in ElevenLabs.
2.  You can add a "Generative" voice or "Clone" a voice.
3.  Alternatively, go to **Voice Library**, filter by "Japanese", find a voice you like, and click "Add to VoiceLab".
4.  Back in VoiceLab, find the Voice ID (it's usually a button or in the ID details).
5.  Update `ELEVENLABS_VOICE_ID` in `app/main.py` with this ID.

---

## 4. Running the Application

1.  Ensure `.env` is filled out.
2.  Ensure `ngrok` is running.
3.  Update the TwiML Bin with the current ngrok URL (if it changed).
4.  Start the server:
    ```bash
    uv run uvicorn app.main:app --reload --port 8000
    ```
5.  Call your Twilio phone number.

### Troubleshooting
-   **No Audio?** Check the server logs. If you see "Twilio connected", the WebSocket is working.
-   **Latency?** Ensure you are using `uvicorn` locally and that your internet connection is stable. Deepgram and ElevenLabs servers are generally fast.
-   **Permissions?** Make sure your Mac/PC allows microphone input if testing via softphone, though real phone calls rely on Twilio's connection.

