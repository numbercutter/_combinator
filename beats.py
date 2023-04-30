import random
import numpy as np
from PIL import Image

from pydub import AudioSegment  # Ensure this import is here

from audio import generate_bass_pattern, sine_wave_synthesizer, generate_drum_pattern, generate_drum_pattern_high_res

def generate_audio(duration):
    # Generate the drum loop
    drum_loop = generate_drum_pattern(tempo=190, filename="drum_pattern.wav", bars=16)

    # Generate the high-resolution drum loop
    drum_loop_high_res = generate_drum_pattern_high_res(tempo=190, filename="drum_pattern_high_res.wav", bars=16)

    # Generate the bass line
    bass_line = generate_bass_pattern(tempo=190, duration=duration, bars=16)

    # Determine the length of the longest loop
    max_loop_length = len(drum_loop)

    # Repeat the drum loops until they reach the desired duration
    while len(drum_loop) < duration * 1000:
        drum_loop += drum_loop

    while len(drum_loop_high_res) < duration * 1000:
        drum_loop_high_res += drum_loop_high_res

    # Trim the drum loops if necessary
    drum_loop = drum_loop[:duration * 1000]
    drum_loop_high_res = drum_loop_high_res[:duration * 1000]

    # Mix the bass line with the drum loops
    mixed_audio = drum_loop.overlay(bass_line)
    mixed_audio = mixed_audio.overlay(drum_loop_high_res)

    # Export the mixed audio to a WAV file
    mixed_audio.export("audio_.wav", format="wav")
if __name__ == "__main__":
    duration = 20


    generate_audio(duration)
