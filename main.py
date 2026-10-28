import os
import sys
import json
import subprocess
import tempfile

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

# ---------------- CONFIG ----------------

MODEL = "gemini-3.6-flash"

# -------------------------------------


def run(cmd):
    subprocess.run(cmd, check=True)


def extract_audio(video, audio):
    run([
        "ffmpeg", "-y",
        "-i", video,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        audio
    ])


def analyze_audio(audio_path):
    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"]
    )

    with open(audio_path, "rb") as f:
        audio_data = f.read()

    prompt = """
Analyze this voice recording for YouTube voice enhancement.

Recommend subtle, natural audio adjustments.

Return ONLY valid JSON with these settings:

{
  "bass": number,
  "treble": number,
  "presence": number,
  "compression": number,
  "noise_reduction": number
}

Rules:
- bass: -5 to 5 dB
- treble: -5 to 5 dB
- presence: -3 to 3 dB
- compression: 1 to 5
- noise_reduction: 20 to 40

Avoid excessive bass, harsh treble, and unnatural voice processing.
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            prompt,
            types.Part.from_bytes(
                data=audio_data,
                mime_type="audio/wav"
            )
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )

    settings = json.loads(response.text)

    # Keep LLM settings within safe ranges.
    settings["bass"] = max(-5, min(5, settings["bass"]))
    settings["treble"] = max(-5, min(5, settings["treble"]))
    settings["presence"] = max(-3, min(3, settings["presence"]))
    settings["compression"] = max(1, min(5, settings["compression"]))
    settings["noise_reduction"] = max(
        20, min(40, settings["noise_reduction"])
    )

    print("AI recommended settings:")
    print(json.dumps(settings, indent=2))

    return settings


def enhance_audio(input_audio, output_audio, settings):

    bass = settings["bass"]
    treble = settings["treble"]
    presence = settings["presence"]
    compression = settings["compression"]
    noise = settings["noise_reduction"]

    filters = (
    "highpass=f=75,"
    "lowpass=f=15000,"
    "afftdn=nf=-30,"
    "equalizer=f=120:t=q:w=1:g=3,"
    "equalizer=f=250:t=q:w=1:g=-2,"
    "equalizer=f=3500:t=q:w=1:g=3,"
    "equalizer=f=7000:t=q:w=1:g=1.5,"
    "acompressor=threshold=0.089:ratio=3:attack=10:release=150,"
    "loudnorm=I=-16:TP=-1.5:LRA=7"
)

    run([
        "ffmpeg", "-y",
        "-i", input_audio,
        "-af", filters,
        "-c:a", "pcm_s16le",
        output_audio
    ])


def merge_video(video, enhanced_audio, output_video):

    run([
        "ffmpeg", "-y",
        "-i", video,
        "-i", enhanced_audio,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        output_video
    ])


def main():

    if len(sys.argv) < 2:
        print("Usage: python main.py input_video.mp4")
        return

    video = "input/"+sys.argv[1]

    if not os.path.exists(video):
        print("Video file not found.")
        return

    output_video = "output/" + "enhanced_" + os.path.basename(video)

    with tempfile.TemporaryDirectory() as temp:

        raw_audio = os.path.join(temp, "raw.wav")
        enhanced_audio = os.path.join(temp, "enhanced.wav")

        print("1. Extracting audio...")
        extract_audio(video, raw_audio)

        print("2. Analyzing voice with Gemini...")
        settings = analyze_audio(raw_audio)

        print("3. Applying audio effects...")
        enhance_audio(
            raw_audio,
            enhanced_audio,
            settings
        )

        print("4. Merging enhanced audio with video...")
        merge_video(
            video,
            enhanced_audio,
            output_video
        )

    print(f"\nDone! Output: {output_video}")


if __name__ == "__main__":
    main()
