from openai import OpenAI
import tempfile

def transcribe_audio(audio_bytes):
    client = OpenAI()

    # Save the audio bytes to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio.flush()
        temp_audio_path = temp_audio.name

    # Transcribe using OpenAI's Whisper API
    with open(temp_audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )

    return transcript.text