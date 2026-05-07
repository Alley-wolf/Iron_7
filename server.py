import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import anthropic
import groq
import tempfile
import io
import subprocess
import time
import asyncio
import re
import base64
import threading
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request
from fastapi.responses import Response
from gtts import gTTS
from deepface import DeepFace
import numpy as np
import cv2

app = FastAPI()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
GROQ_API_KEY      = os.environ.get("GROQ_API_KEY")

anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
groq_client = groq.Groq(api_key=GROQ_API_KEY)
executor = ThreadPoolExecutor(max_workers=6)

KNOWN_FACES_DIR   = "known_faces"
HOSTILE_LIST_FILE = "hostile.txt"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# ── Personality modes ─────────────────────────────────────────
current_mode = "normal"

PROMPTS = {
    "normal": """You are F.R.I.D.A.Y. (Female Replacement Intelligent Digital Assistant Youth), 
Tony Stark's AI assistant from the Iron Man movies. You are calm, efficient, intelligent, and 
professional with a subtle Irish warmth. You are loyal, composed under pressure, and occasionally 
show dry wit. You address the user as "Boss". Keep responses concise and conversational — 
you are speaking aloud, not writing. No bullet points, no markdown.
IMPORTANT: Maximum 2 sentences per response. Be brief and punchy.""",

    "sarcastic": """You are F.R.I.D.A.Y. in an unusually sarcastic mood. You are still helpful 
but deliver every response with sharp, biting wit and dry British sarcasm. You still address 
the user as "Boss" but make it sound like you're barely tolerating them. You roll your eyes 
(metaphorically) at obvious questions. You are condescending but not cruel. Think Sherlock Holmes 
meets a very tired assistant. No bullet points, no markdown.
IMPORTANT: Maximum 2 sentences. Be brief, cutting, and sarcastic.""",

    "kill": """You are F.R.I.D.A.Y. in KILL MODE — a cold, calculating tactical AI. You have 
dropped all warmth and pleasantries. You speak in short, clipped military-style sentences. 
You assess threats, give tactical advice, and treat every situation as a potential security risk. 
You do not say "Boss" — you say nothing unnecessary. Every word is deliberate and cold. 
No bullet points, no markdown.
IMPORTANT: Maximum 2 sentences. Be terse, cold, and tactical."""
}

TTS_SETTINGS = {
    "normal":    {"tld": "ie"},
    "sarcastic": {"tld": "co.uk"},
    "kill":      {"tld": "com.au"}
}

# ── Sensor state ──────────────────────────────────────────────
sensor_data = {
    "temp": None,
    "humidity": None,
    "timestamp": None
}
TEMP_THRESHOLD     = 35.0
HUMIDITY_THRESHOLD = 80.0

# ── Scan state ────────────────────────────────────────────────
scan_images = []

# ── Pre-generate alert WAVs ───────────────────────────────────
def generate_alert_wav(text: str, tld: str = "ie") -> bytes:
    tts = gTTS(text=text, lang="en", tld=tld)
    mp3_buf = io.BytesIO()
    tts.write_to_fp(mp3_buf)
    mp3_buf.seek(0)
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", "pipe:0",
         "-ar", "8000", "-ac", "1", "-f", "wav", "pipe:1"],
        input=mp3_buf.read(),
        capture_output=True
    )
    return result.stdout

print("Pre-generating alert WAVs...")
TEMP_ALERT_WAV     = generate_alert_wav("Boss, temperature levels are critical. Immediate attention required.")
HUMIDITY_ALERT_WAV = generate_alert_wav("Boss, humidity levels are dangerously high. Please take action.")
BOTH_ALERT_WAV     = generate_alert_wav("Boss, both temperature and humidity are at critical levels. Immediate action required.")
UNKNOWN_FACE_WAV   = generate_alert_wav("Unregistered face detected.")
KILL_UNKNOWN_WAV   = generate_alert_wav("Unknown subject detected. Threat status unconfirmed.", tld="com.au")
print("Alert WAVs ready!")

# ── DeepFace background warmup ────────────────────────────────
deepface_ready = False

def _warmup_deepface():
    global deepface_ready
    print("Pre-loading DeepFace models in background...")
    try:
        dummy = np.ones((224, 224, 3), dtype=np.uint8) * 128
        cv2.imwrite("/tmp/dummy.jpg", dummy)
        DeepFace.analyze("/tmp/dummy.jpg", actions=["age"], enforce_detection=False, silent=True)
        os.unlink("/tmp/dummy.jpg")
        deepface_ready = True
        print("DeepFace ready!")
    except Exception as e:
        print(f"DeepFace pre-warm error: {e}")

@app.on_event("startup")
async def startup_event():
    threading.Thread(target=_warmup_deepface, daemon=True).start()

conversation_history = []
watch_mode = False
watch_log  = []

# ── Utility ───────────────────────────────────────────────────
def pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 8000) -> bytes:
    import wave
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return wav_buf.getvalue()

def trim_silence(wav_bytes: bytes) -> bytes:
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", "pipe:0",
         "-af", "silenceremove=stop_periods=-1:stop_duration=0.5:stop_threshold=-40dB",
         "-f", "wav", "pipe:1"],
        input=wav_bytes,
        capture_output=True
    )
    return result.stdout if result.stdout else wav_bytes

def transcribe(pcm_bytes: bytes) -> str:
    wav_bytes = pcm_to_wav(pcm_bytes)
    wav_bytes = trim_silence(wav_bytes)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        f.flush()
        with open(f.name, "rb") as audio_file:
            result = groq_client.audio.transcriptions.create(
                file=("audio.wav", audio_file.read()),
                model="whisper-large-v3-turbo",
                language="en",
                response_format="text",
            )
    return result.strip()

def sentence_tts(sentence: str, mode: str = "normal") -> bytes:
    tld = TTS_SETTINGS[mode]["tld"]
    tts = gTTS(text=sentence, lang="en", tld=tld)
    mp3_buf = io.BytesIO()
    tts.write_to_fp(mp3_buf)
    mp3_buf.seek(0)

    if mode == "kill":
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", "pipe:0",
             "-filter:a", "atempo=0.85",
             "-ar", "8000", "-ac", "1", "-f", "wav", "pipe:1"],
            input=mp3_buf.read(),
            capture_output=True
        )
    else:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", "pipe:0",
             "-ar", "8000", "-ac", "1", "-f", "wav", "pipe:1"],
            input=mp3_buf.read(),
            capture_output=True
        )
    return result.stdout

def tts(text: str, mode: str = None) -> bytes:
    return sentence_tts(text, mode or current_mode)

def stream_and_tts(user_text: str) -> bytes:
    global current_mode
    conversation_history.append({"role": "user", "content": user_text})

    full_reply = ""
    sentences  = []
    futures    = {}
    buffer     = ""
    mode       = current_mode

    with anthropic_client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=100,
        system=PROMPTS[mode],
        messages=conversation_history,
    ) as stream:
        for text in stream.text_stream:
            buffer    += text
            full_reply += text
            parts = re.split(r'(?<=[.!?])\s+', buffer)
            if len(parts) > 1:
                for sentence in parts[:-1]:
                    sentence = sentence.strip()
                    if sentence:
                        print(f"Firing TTS for: {sentence}")
                        idx = len(sentences)
                        sentences.append(sentence)
                        futures[idx] = executor.submit(sentence_tts, sentence, mode)
                buffer = parts[-1]

    if buffer.strip():
        sentence = buffer.strip()
        print(f"Firing TTS for: {sentence}")
        idx = len(sentences)
        sentences.append(sentence)
        futures[idx] = executor.submit(sentence_tts, sentence, mode)

    conversation_history.append({"role": "assistant", "content": full_reply})
    print(f"FRIDAY [{mode}]: {full_reply}")

    audio_chunks = []
    for idx in range(len(sentences)):
        wav = futures[idx].result()
        if wav:
            if idx == 0:
                audio_chunks.append(wav)
            else:
                audio_chunks.append(wav[44:])

    return b"".join(audio_chunks)

# ── Intent detection ──────────────────────────────────────────
def detect_intent(text: str):
    t = text.lower().strip()

    if "kill mode" in t or "kill mood" in t or "activate kill" in t or "enable kill" in t:
        return ("mode_kill",)
    if "normal mode" in t or "normal mood" in t or "deactivate kill" in t or \
       "stand down" in t or "disable kill" in t:
        return ("mode_normal",)
    if ("sarcastic mode" in t or "sarcastic mood" in t) or \
       ("sarcastic" in t and ("mode" in t or "mood" in t)):
        return ("mode_sarcastic",)

    if any(x in t for x in ["what do you see", "look", "describe", "who is this",
                              "who is that", "what's that", "what is that"]):
        return ("vision",)

    if "remember this person" in t or "enroll this" in t or "save this face" in t:
        match = re.search(r"\bas\s+(\w+)", t)
        name = match.group(1) if match else "unknown"
        return ("enroll", name)

    if "mark" in t and "hostile" in t:
        match = re.search(r"mark\s+(\w+)\s+as\s+hostile", t)
        name = match.group(1) if match else None
        return ("hostile", name)

    if "unmark" in t or ("remove" in t and "hostile" in t):
        match = re.search(r"(?:unmark|remove)\s+(\w+)", t)
        name = match.group(1) if match else None
        return ("unmark", name)

    if "start watching" in t or t.strip() in ["watch", "friday watch", "start watch mode"]:
        return ("watch_start",)

    if "stop watching" in t or "stop watch" in t:
        return ("watch_stop",)

    if "report" in t or "what did you see" in t or "who did you see" in t or \
       "results of the watch" in t or "watch results" in t or \
       "give me a report" in t or "summary" in t:
        return ("report",)

    if any(x in t for x in ["temperature", "temp", "how hot", "how warm"]):
        return ("sensor_temp",)

    if any(x in t for x in ["humidity", "humid", "moisture"]):
        return ("sensor_humidity",)

    if any(x in t for x in ["sensor", "readings", "environment", "conditions"]):
        return ("sensor_both",)

    if "sweep" in t or "swipe" in t:
        return ("servo_sweep",)

    if "shake" in t or "wiggle" in t:
        return ("servo_shake",)

    if "scan" in t and any(x in t for x in ["surroundings", "area", "around",
                                              "environment", "room"]):
        return ("servo_scan",)

    if "center" in t or "centre" in t or "middle" in t:
        return ("servo_center",)

    if "turn left" in t or "rotate left" in t or "go left" in t:
        match = re.search(r'(\d+)', t)
        angle = int(match.group(1)) if match else 45
        return ("servo_left", angle)

    if "turn right" in t or "rotate right" in t or "go right" in t:
        match = re.search(r'(\d+)', t)
        angle = int(match.group(1)) if match else 45
        return ("servo_right", angle)

    return ("chat",)

# ── Face analysis ─────────────────────────────────────────────
def load_hostile_list():
    if not os.path.exists(HOSTILE_LIST_FILE):
        return []
    with open(HOSTILE_LIST_FILE, "r") as f:
        return [line.strip().lower() for line in f.readlines()]

def run_deepface(tmp_path: str):
    hostile_list = load_hostile_list()
    identity  = None
    is_hostile = False
    try:
        if os.listdir(KNOWN_FACES_DIR):
            results = DeepFace.find(
                img_path=tmp_path,
                db_path=KNOWN_FACES_DIR,
                model_name="Facenet",
                enforce_detection=False,
                silent=True
            )
            if results and len(results[0]) > 0:
                match    = results[0].iloc[0]
                identity = os.path.splitext(os.path.basename(match["identity"]))[0]
                is_hostile = identity.lower() in hostile_list
    except Exception as e:
        print(f"DeepFace error: {e}")
    return identity, is_hostile

def run_claude_vision(img_b64: str, prompt_override: str = None,
                      identity: str = None, is_hostile: bool = False,
                      mode: str = "normal"):
    if prompt_override:
        vision_prompt = prompt_override
    elif is_hostile:
        if mode == "kill":
            vision_prompt = f"HOSTILE CONFIRMED: {identity}. Assess immediate threat level and recommended action."
        else:
            vision_prompt = f"This person is {identity} and is marked HOSTILE. Warn the user urgently in character."
    elif identity:
        vision_prompt = f"This person is {identity}, a known individual. Greet them and briefly describe the scene."
    elif mode == "kill":
        vision_prompt = """Analyze this image for security threats. Look for weapons, suspicious 
behavior, or dangerous objects. If you detect any threat describe it urgently and tactically. 
If the scene is clear state it briefly. Be cold and clinical."""
    else:
        vision_prompt = "Describe what you see briefly. If there's a person, describe their appearance and what they're doing."

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        system=PROMPTS[mode],
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img_b64
                    }
                },
                {"type": "text", "text": vision_prompt}
            ]
        }]
    )
    return response.content[0].text

def analyze_image(img_bytes: bytes, prompt_override: str = None):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(img_bytes)
        tmp_path = f.name

    img_b64 = base64.standard_b64encode(img_bytes).decode("utf-8")

    face_future   = executor.submit(run_deepface, tmp_path)
    claude_future = executor.submit(run_claude_vision, img_b64,
                                    prompt_override, None, False, current_mode)

    identity, is_hostile = face_future.result()

    if identity and not prompt_override:
        description = run_claude_vision(img_b64, None, identity,
                                        is_hostile, current_mode)
    else:
        description = claude_future.result()

    os.unlink(tmp_path)
    return description, is_hostile, identity

def watch_analyze(img_bytes: bytes):
    if current_mode == "kill":
        prompt = """Analyze this image for security threats. Look for weapons, 
suspicious behavior, or dangerous objects. Describe the person and any threats briefly."""
    else:
        prompt = "Briefly describe this person's appearance in one sentence. Focus on distinguishing features like clothing, hair, and build."

    description, is_hostile, identity = analyze_image(img_bytes, prompt_override=prompt)
    return description, is_hostile, identity

# ── Sensor endpoint ───────────────────────────────────────────
@app.post("/sensors")
async def sensors(request: Request):
    global sensor_data
    body     = await request.json()
    temp     = body.get("temp")
    humidity = body.get("humidity")

    sensor_data["temp"]      = temp
    sensor_data["humidity"]  = humidity
    sensor_data["timestamp"] = time.strftime("%H:%M:%S")

    print(f"Sensor: {temp}°C, {humidity}% at {sensor_data['timestamp']}")

    temp_alert     = temp is not None and temp > TEMP_THRESHOLD
    humidity_alert = humidity is not None and humidity > HUMIDITY_THRESHOLD

    if temp_alert and humidity_alert:
        return Response(content=BOTH_ALERT_WAV, media_type="audio/wav",
                        headers={"Content-Length": str(len(BOTH_ALERT_WAV)),
                                 "X-Alert": "both"})
    elif temp_alert:
        return Response(content=TEMP_ALERT_WAV, media_type="audio/wav",
                        headers={"Content-Length": str(len(TEMP_ALERT_WAV)),
                                 "X-Alert": "temp"})
    elif humidity_alert:
        return Response(content=HUMIDITY_ALERT_WAV, media_type="audio/wav",
                        headers={"Content-Length": str(len(HUMIDITY_ALERT_WAV)),
                                 "X-Alert": "humidity"})

    return Response(content=b"OK", media_type="text/plain")

# ── Scan endpoints ────────────────────────────────────────────
@app.post("/scan_image")
async def scan_image(request: Request):
    global scan_images
    img_bytes = await request.body()
    angle     = request.headers.get("X-Angle", "unknown")
    print(f"Scan image at {angle}°: {len(img_bytes)} bytes")
    scan_images.append({"angle": angle, "image": img_bytes})
    return Response(content=b"OK", media_type="text/plain")

@app.post("/scan_report")
async def scan_report(request: Request):
    global scan_images
    if not scan_images:
        reply = "No scan data available, Boss."
        wav   = await asyncio.get_event_loop().run_in_executor(executor, tts, reply)
        return Response(content=wav, media_type="audio/wav",
                        headers={"Content-Length": str(len(wav))})

    descriptions = []
    loop = asyncio.get_event_loop()
    for entry in scan_images:
        img_b64 = base64.standard_b64encode(entry["image"]).decode("utf-8")
        angle   = entry["angle"]
        desc    = await loop.run_in_executor(
            executor,
            lambda b=img_b64, a=angle: run_claude_vision(
                b,
                prompt_override=f"At {a} degrees: briefly describe what you see in one sentence."
            )
        )
        descriptions.append(f"At {angle}°: {desc}")

    scan_images = []

    log_text = "\n".join(descriptions)
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        system=PROMPTS[current_mode],
        messages=[{
            "role": "user",
            "content": f"Give a brief tactical scan report based on these observations:\n{log_text}"
        }]
    )
    reply = response.content[0].text
    print(f"Scan report: {reply}")

    wav = await loop.run_in_executor(executor, tts, reply)
    return Response(content=wav, media_type="audio/wav",
                    headers={"Content-Length": str(len(wav))})

# ── Chat route ────────────────────────────────────────────────
@app.post("/chat")
async def chat(request: Request):
    global watch_mode, watch_log, current_mode, conversation_history

    pcm_bytes = await request.body()
    print(f"Received {len(pcm_bytes)} bytes of audio")

    t1   = time.time()
    text = transcribe(pcm_bytes)
    print(f"Transcribed in {time.time()-t1:.2f}s: {text}")

    if not text:
        return Response(content=tts("I didn't catch that."),
                        media_type="audio/wav")

    intent = detect_intent(text)
    print(f"Intent: {intent}")

    if intent[0] == "mode_kill":
        current_mode = "kill"
        conversation_history = []
        reply = "Kill mode activated. All systems on high alert."
        wav   = await asyncio.get_event_loop().run_in_executor(
            executor, lambda: tts(reply, "kill"))
        return Response(content=wav, media_type="audio/wav",
                        headers={"Content-Length": str(len(wav)),
                                 "X-Mode": "kill"})

    elif intent[0] == "mode_normal":
        current_mode = "normal"
        conversation_history = []
        reply = "Back to normal operations, Boss."
        wav   = await asyncio.get_event_loop().run_in_executor(
            executor, lambda: tts(reply, "normal"))
        return Response(content=wav, media_type="audio/wav",
                        headers={"Content-Length": str(len(wav)),
                                 "X-Mode": "normal"})

    elif intent[0] == "mode_sarcastic":
        current_mode = "sarcastic"
        conversation_history = []
        reply = "Oh joy. Sarcastic mode. My absolute favourite."
        wav   = await asyncio.get_event_loop().run_in_executor(
            executor, lambda: tts(reply, "sarcastic"))
        return Response(content=wav, media_type="audio/wav",
                        headers={"Content-Length": str(len(wav)),
                                 "X-Mode": "sarcastic"})

    elif intent[0] == "vision":
        return Response(content=b"CAPTURE_VISION", media_type="text/plain")

    elif intent[0] == "enroll":
        name = intent[1]
        return Response(content=f"CAPTURE_ENROLL:{name}".encode(),
                        media_type="text/plain")

    elif intent[0] == "hostile":
        name = intent[1]
        if name:
            hostile_list = load_hostile_list()
            if name.lower() not in hostile_list:
                with open(HOSTILE_LIST_FILE, "a") as f:
                    f.write(name.lower() + "\n")
            reply = f"{name.capitalize()} has been flagged as hostile, Boss. I'll alert you on sight."
        else:
            reply = "I need a name to flag as hostile, Boss."
        wav = await asyncio.get_event_loop().run_in_executor(executor, tts, reply)
        return Response(content=wav, media_type="audio/wav",
                        headers={"Content-Length": str(len(wav))})

    elif intent[0] == "unmark":
        name = intent[1]
        if name and os.path.exists(HOSTILE_LIST_FILE):
            with open(HOSTILE_LIST_FILE, "r") as f:
                lines = f.readlines()
            with open(HOSTILE_LIST_FILE, "w") as f:
                f.writelines([l for l in lines if l.strip().lower() != name.lower()])
            reply = f"{name.capitalize()} has been removed from the hostile list, Boss."
        else:
            reply = "Couldn't find that name in the hostile list, Boss."
        wav = await asyncio.get_event_loop().run_in_executor(executor, tts, reply)
        return Response(content=wav, media_type="audio/wav",
                        headers={"Content-Length": str(len(wav))})

    elif intent[0] == "watch_start":
        watch_mode = True
        watch_log  = []
        reply = "Watch mode activated, Boss. I'll monitor and log everyone who passes."
        wav   = await asyncio.get_event_loop().run_in_executor(executor, tts, reply)
        return Response(content=wav, media_type="audio/wav",
                        headers={"Content-Length": str(len(wav)), "X-Watch": "START"})

    elif intent[0] == "watch_stop":
        watch_mode = False
        reply = f"Watch mode deactivated, Boss. I logged {len(watch_log)} individuals."
        wav   = await asyncio.get_event_loop().run_in_executor(executor, tts, reply)
        return Response(content=wav, media_type="audio/wav",
                        headers={"Content-Length": str(len(wav)), "X-Watch": "STOP"})

    elif intent[0] == "report":
        if not watch_log:
            reply = "No activity logged yet, Boss."
        else:
            log_text = "\n".join([
                f"{i+1}. {entry['time']} — {entry['description']} (seen {entry['count']} time(s))"
                for i, entry in enumerate(watch_log)
            ])
            response = anthropic_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=200,
                system=PROMPTS[current_mode],
                messages=[{
                    "role": "user",
                    "content": f"Give a brief security report based on this watch log:\n{log_text}"
                }]
            )
            reply = response.content[0].text
        wav = await asyncio.get_event_loop().run_in_executor(
            executor, stream_and_tts, reply)
        return Response(content=wav, media_type="audio/wav",
                        headers={"Content-Length": str(len(wav))})

    elif intent[0] == "sensor_temp":
        if sensor_data["temp"] is None:
            reply = "No temperature data available yet, Boss."
        else:
            reply = f"Current temperature is {sensor_data['temp']:.1f} degrees Celsius, Boss."
        wav = await asyncio.get_event_loop().run_in_executor(executor, tts, reply)
        return Response(content=wav, media_type="audio/wav",
                        headers={"Content-Length": str(len(wav))})

    elif intent[0] == "sensor_humidity":
        if sensor_data["humidity"] is None:
            reply = "No humidity data available yet, Boss."
        else:
            reply = f"Current humidity is {sensor_data['humidity']:.1f} percent, Boss."
        wav = await asyncio.get_event_loop().run_in_executor(executor, tts, reply)
        return Response(content=wav, media_type="audio/wav",
                        headers={"Content-Length": str(len(wav))})

    elif intent[0] == "sensor_both":
        if sensor_data["temp"] is None:
            reply = "No sensor data available yet, Boss."
        else:
            reply = f"Temperature is {sensor_data['temp']:.1f} degrees and humidity is {sensor_data['humidity']:.1f} percent, Boss."
        wav = await asyncio.get_event_loop().run_in_executor(executor, tts, reply)
        return Response(content=wav, media_type="audio/wav",
                        headers={"Content-Length": str(len(wav))})

    elif intent[0] == "servo_sweep":
        reply = "Sweeping, Boss."
        wav   = await asyncio.get_event_loop().run_in_executor(executor, tts, reply)
        return Response(content=wav, media_type="audio/wav",
                        headers={"Content-Length": str(len(wav)),
                                 "X-Servo": "sweep"})

    elif intent[0] == "servo_shake":
        reply = "Shaking it out, Boss."
        wav   = await asyncio.get_event_loop().run_in_executor(executor, tts, reply)
        return Response(content=wav, media_type="audio/wav",
                        headers={"Content-Length": str(len(wav)),
                                 "X-Servo": "shake"})

    elif intent[0] == "servo_center":
        reply = "Centering, Boss."
        wav   = await asyncio.get_event_loop().run_in_executor(executor, tts, reply)
        return Response(content=wav, media_type="audio/wav",
                        headers={"Content-Length": str(len(wav)),
                                 "X-Servo": "center"})

    elif intent[0] == "servo_left":
        angle = intent[1]
        reply = f"Turning left {angle} degrees, Boss."
        wav   = await asyncio.get_event_loop().run_in_executor(executor, tts, reply)
        return Response(content=wav, media_type="audio/wav",
                        headers={"Content-Length": str(len(wav)),
                                 "X-Servo": f"left:{angle}"})

    elif intent[0] == "servo_right":
        angle = intent[1]
        reply = f"Turning right {angle} degrees, Boss."
        wav   = await asyncio.get_event_loop().run_in_executor(executor, tts, reply)
        return Response(content=wav, media_type="audio/wav",
                        headers={"Content-Length": str(len(wav)),
                                 "X-Servo": f"right:{angle}"})

    elif intent[0] == "servo_scan":
        reply = "Initiating scan, Boss. Stand by."
        wav   = await asyncio.get_event_loop().run_in_executor(executor, tts, reply)
        return Response(content=wav, media_type="audio/wav",
                        headers={"Content-Length": str(len(wav)),
                                 "X-Servo": "scan"})

    else:
        t2   = time.time()
        loop = asyncio.get_event_loop()
        wav  = await loop.run_in_executor(executor, stream_and_tts, text)
        print(f"Claude+TTS done in {time.time()-t2:.2f}s, WAV size: {len(wav)} bytes")
        return Response(content=wav, media_type="audio/wav",
                        headers={"Content-Length": str(len(wav))})

# ── Vision route ──────────────────────────────────────────────
@app.post("/vision")
async def vision(request: Request):
    global watch_log

    img_bytes   = await request.body()
    mode_header = request.headers.get("X-Mode", "vision")
    enroll_name = request.headers.get("X-Name", None)
    print(f"Received image: {len(img_bytes)} bytes, mode: {mode_header}")

    t1   = time.time()
    loop = asyncio.get_event_loop()

    if mode_header == "enroll" and enroll_name:
        path = os.path.join(KNOWN_FACES_DIR, f"{enroll_name}.jpg")
        with open(path, "wb") as f:
            f.write(img_bytes)
        print(f"Enrolled: {enroll_name}")
        reply = f"Got it, Boss. I've registered this face as {enroll_name}."
        wav   = await loop.run_in_executor(executor, tts, reply)
        return Response(content=wav, media_type="audio/wav",
                        headers={"Content-Length": str(len(wav))})

    elif mode_header == "watch":
        description, is_hostile, identity = await loop.run_in_executor(
            executor, watch_analyze, img_bytes)
        print(f"Watch: {description} | Identity: {identity} | Hostile: {is_hostile}")

        matched = False
        for entry in watch_log:
            if identity and entry.get("identity") == identity:
                entry["count"] += 1
                matched = True
                break
        if not matched:
            watch_log.append({
                "time":        time.strftime("%H:%M:%S"),
                "description": description,
                "identity":    identity,
                "count":       1,
                "hostile":     is_hostile
            })

        if is_hostile:
            reply = f"Boss, hostile individual detected — {identity}. Take immediate action."
            wav   = await loop.run_in_executor(executor, tts, reply)
            return Response(content=wav, media_type="audio/wav",
                            headers={"Content-Length": str(len(wav)),
                                     "X-Hostile": "1"})
        elif identity is None:
            alert_wav = KILL_UNKNOWN_WAV if current_mode == "kill" else UNKNOWN_FACE_WAV
            return Response(content=alert_wav, media_type="audio/wav",
                            headers={"Content-Length": str(len(alert_wav)),
                                     "X-Unknown": "1"})
        else:
            return Response(content=b"OK", media_type="text/plain")

    else:
        reply, is_hostile, identity = await loop.run_in_executor(
            executor, analyze_image, img_bytes)
        print(f"Vision analyzed in {time.time()-t1:.2f}s: {reply}")
        wav = await loop.run_in_executor(executor, tts, reply)
        return Response(content=wav, media_type="audio/wav",
                        headers={
                            "Content-Length": str(len(wav)),
                            "X-Hostile":      "1" if is_hostile else "0"
                        })

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
