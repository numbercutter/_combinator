import os
import random
import numpy as np
from PIL import Image
from pydub import AudioSegment
from moviepy.editor import AudioFileClip, ImageSequenceClip
import cairo
from pydub.generators import Sine
from pydub.effects import normalize
# Determine the triads
triads = {
    "major": [0, 4, 7],
    "minor": [0, 3, 7],
    "diminished": [0, 3, 6],
    "augmented": [0, 4, 8],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    "dominant7": [0, 4, 7, 10],
    "major7": [0, 4, 7, 11],
    "minor7": [0, 3, 7, 10],
    "halfdiminished7": [0, 3, 6, 10],
    "diminished7": [0, 3, 6, 9]
}

def generate_chord(scale, chord_type, duration):
    triad = triads[chord_type]

    # Choose a random chord from the triads
    root = random.choice(scale)
    notes = [root + i for i in triad]

    # Generate sine wave for each note in the chord
    chord = []
    sample_rate = 44100
    t = np.linspace(0, duration, int(duration * sample_rate), False)

    for note in notes:
        freq = 440 * 2 ** ((note - 69) / 12)  # Convert MIDI note number to frequency
        sine_wave = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        chord.append(sine_wave)

    # Combine sine waves to create chord
    chord_data = np.zeros_like(chord[0])
    for note_data in chord:
        chord_data += note_data

    # Normalize the combined chord_data
    chord_data = chord_data / np.max(np.abs(chord_data))

    # Apply a volume scaling factor (optional)
    volume_scale = 0.5  # Reduce volume by 50%
    chord_data = chord_data * volume_scale

    # Convert to int16 format
    chord_data_int16 = (chord_data * (2 ** 15 - 1)).astype(np.int16)

    # Create an AudioSegment from the chord data
    chord_audio = AudioSegment(chord_data_int16.tobytes(), frame_rate=sample_rate, sample_width=2, channels=1)

    return chord_audio


def generate_audio(duration):
    # Define major scales
    scales = {
    "C major": [60, 62, 64, 65, 67, 69, 71],
    "C# major": [61, 63, 65, 66, 68, 70, 72],
    "D major": [62, 64, 66, 67, 69, 71, 73],
    "D# major": [63, 65, 67, 68, 70, 72, 74],
    "E major": [64, 66, 68, 69, 71, 73, 75],
    "F major": [65, 67, 69, 70, 72, 74, 76],
    "F# major": [66, 68, 70, 71, 73, 75, 77],
    "G major": [67, 69, 71, 72, 74, 76, 78],
    "G# major": [68, 70, 72, 73, 75, 77, 79],
    "A major": [69, 71, 73, 74, 76, 78, 80]
    }
    # Choose a random scale
    scale_name = random.choice(list(scales.keys()))
    scale = scales[scale_name]

    # Choose a random chord type
    chord_type = random.choice(list(triads.keys()))

    # Generate chords
    chord1 = generate_chord(scale, chord_type, duration)
    chord2 = generate_chord(scale, chord_type, duration)

    # Apply a pulsing wave-like volume effect to each chord
    for i, chord in enumerate([chord1, chord2]):
        if i == 0:
            audio = chord
        else:
            audio = audio.overlay(chord)

    # Export the audio to a WAV file
    audio.export("audio.wav", format="wav")



def generate_visuals(duration, img_size, num_frames):
    frames = []
    for _ in range(num_frames):
        img = np.random.randint(0, 256, size=(img_size, img_size, 3), dtype=np.uint8)
        frames.append(Image.fromarray(img))

    return frames

def generate_video(duration, img_size, fps):
    audio_path = "audio.wav"
    video_path = "output.mp4"

    num_frames = duration * fps
    generate_audio(duration)
    frames = generate_visuals(duration, img_size, num_frames)

    audio = AudioFileClip(audio_path)
    video = ImageSequenceClip([np.array(img) for img in frames], fps=fps)
    video = video.set_audio(audio)
    video.write_videofile(video_path, fps=fps)

if __name__ == "__main__":
    duration = 5  # seconds
    img_size = 1080  # Instagram square dimensions (1080x1080)
    fps = 30  # frames per second
    generate_video(duration, img_size, fps)