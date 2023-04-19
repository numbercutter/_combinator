import os
import random
import numpy as np
from PIL import Image
from pydub import AudioSegment
from moviepy.editor import AudioFileClip, ImageSequenceClip

# Determine the triads
triads = {
    "major": [0, 4, 7],
    "minor": [0, 3, 7],

}

def generate_chord(scale, chord_type, duration, extensions=None, inversion=0, fm_intensity=0.01, fm_speed=0.5):
    triad = triads[chord_type]

    # Add extensions if provided
    if extensions:
        extended_chord = triad + extensions
    else:
        extended_chord = triad

    # Choose a random chord from the extended chords
    root = random.choice(scale)
    notes = [root + i for i in extended_chord]

    # Apply inversion if specified
    if inversion > 0:
        for _ in range(inversion):
            notes.append(notes.pop(0) + 12)

    # Generate sine wave for each note in the chord
    chord = []
    sample_rate = 44100
    t = np.linspace(0, duration, int(duration * sample_rate), False)

    for note in notes:
        freq = 440 * 2 ** ((note - 69) / 12)  # Convert MIDI note number to frequency

        # Generate sine waves for harmonics
        harmonics = np.zeros_like(t)
        max_harmonic = int(1000 / freq)
        for harmonic in range(1, max_harmonic + 1):
            sine_wave = (0.5 * np.sin(2 * np.pi * freq * harmonic * t) / harmonic).astype(np.float32)
            harmonics += sine_wave

        # Apply frequency modulation
        fm = np.sin(2 * np.pi * fm_speed * t) * fm_intensity * freq
        modulated_freq = freq + fm
        modulated_sine_wave = 0.5 * np.sin(2 * np.pi * modulated_freq * t).astype(np.float32)
        harmonics = 0.5 * harmonics + 0.5 * modulated_sine_wave

        chord.append(harmonics)

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

    # Create an AudioSegment from the modulated chord data
    modulated_chord_audio = AudioSegment(chord_data_int16.tobytes(), frame_rate=sample_rate, sample_width=2, channels=1)

    return modulated_chord_audio


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

    # Limit the range of root notes
    min_root = 60  # C4
    max_root = 72  # C5
    scale = [note for note in scale if min_root <= note <= max_root]

    # Choose a random chord type (only major and minor)
    chord_type = random.choice(list(triads.keys()))

    # Randomly decide whether to add extensions and/or inversions
    use_extensions = random.choice([True, False])
    use_inversions = random.choice([True, False])

    if use_extensions:
        extensions = [9, 11]
    else:
        extensions = None

    if use_inversions:
        inversion = random.choice([0, 1, 2])
    else:
        inversion = 0

    # Randomly decide whether to apply modulation
    apply_modulation = random.random() < 0.3  # 30% chance of modulation

    if apply_modulation:
        fm_intensity = 0.01
        fm_speed = 0.5
    else:
        fm_intensity = 0
        fm_speed = 0

    # Generate the chord with or without extensions and inversions
    chord = generate_chord(scale, chord_type, duration, extensions=extensions, inversion=inversion, fm_intensity=fm_intensity, fm_speed=fm_speed)

    # Export the audio to a WAV file
    chord.export("audio_.wav", format="wav")




def generate_visuals(duration, img_size, num_frames):
    frames = []
    for _ in range(num_frames):
        img = np.random.randint(0, 256, size=(img_size, img_size, 3), dtype=np.uint8)
        frames.append(Image.fromarray(img))

    return frames

def generate_video(duration, img_size, fps):
    audio_path = "audio_.wav"
    video_path = "output_.mp4"

    num_frames = duration * fps

    min_chords = 1
    max_chords = 5
    num_chords = random.randint(min_chords, max_chords)
    chord_durations = np.random.dirichlet(np.ones(num_chords), size=1) * duration

    audio_segments = []
    for chord_duration in chord_durations[0]:
        generate_audio(chord_duration)
        audio_segment = AudioSegment.from_wav(audio_path)
        audio_segments.append(audio_segment)

    full_audio = sum(audio_segments)
    full_audio.export(audio_path, format="wav")

    frames = generate_visuals(duration, img_size, num_frames)

    audio = AudioFileClip(audio_path)
    video = ImageSequenceClip([np.array(img) for img in frames], fps=fps)
    video = video.set_audio(audio)
    video.write_videofile(video_path, fps=fps)

if __name__ == "__main__":
    duration = 10  # seconds
    img_size = 1080  # Instagram square dimensions (1080x1080)
    fps = 10  # frames per second
    generate_video(duration, img_size, fps)