# Script para convertir M4A a WAV
from pydub import AudioSegment

# Convertir M4A a WAV
audio = AudioSegment.from_file("test_audio.m4a", format="m4a")
audio.export("test_audio.wav", format="wav")