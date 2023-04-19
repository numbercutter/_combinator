import os
import random
import numpy as np
from PIL import Image
from pydub import AudioSegment
from moviepy.editor import AudioFileClip, ImageSequenceClip
import cairo

# Determine the triads
triads = {
    "major": [0, 4, 7],
    "minor": [0, 3, 7],

}
solfeggio_freqs = {
    "UT": 396 / 4,
    "RE": 417 / 4,
    "MI": 528 / 4,
    "FA": 639 / 4,
    "SOL": 741 / 4,
    "LA": 852 / 4,
}


def generate_chord(start_freqs, end_freqs, duration, fm_intensity=0.01, fm_speed=0.1):
    num_notes = len(start_freqs)
    assert num_notes == len(end_freqs), "Start and end frequencies should have the same length."

    # Generate the time vector
    sample_rate = 44100
    t = np.linspace(0, duration, int(duration * sample_rate), False)

    # Linearly interpolate the frequencies
    freqs = [np.linspace(start_freqs[i], end_freqs[i], int(duration * sample_rate)) for i in range(num_notes)]

    # Generate sine wave for each note in the chord
    chord = []

    for note_freqs in freqs:
        # Generate sine waves for harmonics
        harmonics = np.zeros_like(t)
        max_harmonic = int(1000 / np.max(note_freqs))
        for harmonic in range(1, max_harmonic + 1):
            sine_wave = (0.5 * np.sin(2 * np.pi * note_freqs * harmonic * t) / harmonic).astype(np.float32)
            harmonics += sine_wave

        # Apply frequency modulation
        fm = np.sin(2 * np.pi * fm_speed * t) * fm_intensity * note_freqs
        modulated_freq = note_freqs + fm
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


def midi_to_freq(midi_note):
    return 440 * 2 ** ((midi_note - 69) / 12)


def generate_audio(duration, num_segments=2):
    segment_duration = duration / num_segments

    # Create a list of all solfeggio frequencies
    freq_list = list(solfeggio_freqs.values())
    
    # Generate the audio segments
    audio_segments = []
    for _ in range(num_segments):
        start_freqs = random.sample(freq_list, 3)
        end_freqs = random.sample(freq_list, 3)
        chord = generate_chord(start_freqs, end_freqs, segment_duration)
        audio_segments.append(chord)
    
    # Concatenate audio segments
    full_audio = audio_segments[0]
    for segment in audio_segments[1:]:
        full_audio = full_audio.append(segment)

    # Export the audio to a WAV file
    full_audio.export("audio_.wav", format="wav")


def generate_visuals(duration, img_size, num_frames):
    frames = []

    # Create a surface for drawing the shapes
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, img_size, img_size)
    ctx = cairo.Context(surface)

    # Set background color
    ctx.set_source_rgb(0.2, 0.2, 0.2)
    ctx.paint()

    # Generate random shapes
    for i in range(num_frames):
        # Set random colors
        r, g, b = np.random.rand(3)
        ctx.set_source_rgb(r, g, b)

        # Set random shape parameters
        shape_type = np.random.choice(['rectangle', 'circle', 'line'])
        if shape_type == 'rectangle':
            x, y = np.random.randint(0, img_size, size=2)
            width, height = np.random.randint(img_size//4, img_size//2, size=2)
            ctx.rectangle(x, y, width, height)
            ctx.fill()
        elif shape_type == 'circle':
            x, y = np.random.randint(0, img_size, size=2)
            radius = np.random.randint(img_size//4, img_size//2)
            ctx.arc(x, y, radius, 0, 2 * np.pi)
            ctx.fill()
        elif shape_type == 'line':
            x1, y1 = np.random.randint(0, img_size, size=2)
            x2, y2 = np.random.randint(0, img_size, size=2)
            ctx.move_to(x1, y1)
            ctx.line_to(x2, y2)
            ctx.set_line_width(np.random.randint(1, 10))
            ctx.stroke()

        # Convert surface to PIL image and append to list of frames
        img = Image.frombuffer("RGBA", (img_size, img_size), surface.get_data(), "raw", "BGRA", 0, 1)
        frames.append(img)

    return frames


def generate_video(duration, img_size, fps):
    audio_path = "audio_.wav"
    video_path = "output_.mp4"
    num_frames = duration * fps

    audio_segment = AudioSegment.from_wav(audio_path)
    audio_segment.export(audio_path, format="wav")

    frames = generate_visuals(duration, img_size, num_frames)

    audio = AudioFileClip(audio_path)
    video = ImageSequenceClip([np.array(img) for img in frames], fps=fps)
    video = video.set_audio(audio)
    video.write_videofile(video_path, fps=fps)

    
if __name__ == "__main__":
    duration = 30  # seconds
    img_size = 600  # Instagram square dimensions (1080x1080)
    fps = 30  # frames per second

    generate_audio(duration)
    generate_video(duration, img_size, fps)

