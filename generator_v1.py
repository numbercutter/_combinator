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
    "major": [0, 2, 4],
    "minor": [0, 2, 3],
    "diminished": [0, 1, 3],
    "augmented": [0, 2, 4],
    "sus2": [0, 1, 3],
    "sus4": [0, 3, 4],
    "dominant7": [0, 2, 4, 6],
    "major7": [0, 2, 4, 7],
    "minor7": [0, 2, 3, 5],
    "halfdiminished7": [0, 2, 3, 6],
    "diminished7": [0, 1, 3, 4]
}
def generate_chord(scale, chord_type, duration):
    
    triad = triads[chord_type]

    # Choose a random chord from the triads
    root = random.choice(scale)
    notes = [root + i for i in triad]

    # Add extensions
    extensions = {
        "7": [0, 2, 4, 6],
        "9": [0, 2, 4, 6, 8],
        "11": [0, 2, 4, 6, 8, 10],
        "13": [0, 2, 4, 6, 8, 10, 12],
        "maj7": [0, 2, 4, 7],
        "add9": [0, 2, 4, 8]
    }
    extension_type = random.choice(list(extensions.keys()))
    extension = extensions[extension_type]
    notes += [root + i for i in extension]

    # Randomly invert the chord
    inversion = random.choice([0, 1, 2])
    notes = notes[inversion:] + notes[:inversion]

    # Generate sine wave for each note in the chord
    chord = []
    for note in notes:
        freq = 440 * 2**((note-69)/12)  # Convert MIDI note number to frequency
        velocity = random.randint(80, 120)  # Generate random velocity between 80 and 120
        sine_wave = Sine(freq).to_audio_segment(duration=duration*1000)
        sine_wave = sine_wave.normalize()  # Normalize volume
        sine_wave = sine_wave - 60  # Adjust volume level; lower the number for a louder sound
        sine_wave = sine_wave + (120 - velocity)  # Adjust volume according to velocity
        # Add sine wave to chord
        chord.append(np.frombuffer(sine_wave.raw_data, dtype=np.int16))


    # Combine sine waves to create chord
    chord_data = np.zeros_like(chord[0])
    for note_data in chord:
        chord_data += note_data

    # Create an AudioSegment from the chord data
    chord_data = (chord_data * (2**15 - 1) / np.max(np.abs(chord_data))).astype(np.int16)
    chord_audio = AudioSegment(chord_data.tobytes(), frame_rate=44100, sample_width=2, channels=1)

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