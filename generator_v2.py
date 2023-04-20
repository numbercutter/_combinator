import os
import random
import numpy as np
from PIL import Image
from pydub import AudioSegment
from moviepy.editor import AudioFileClip, ImageSequenceClip
from moviepy.editor import TextClip, CompositeVideoClip
from moviepy.editor import ImageClip

import string
import cairo
from moviepy.editor import concatenate_videoclips
import librosa
import soundfile as sf

solfeggio_freqs = {
    "UT": 396 / 4,
    "RE": 417 / 4,
    "MI": 528 / 4,
    "FA": 639 / 4,
    "SOL": 741 / 4,
    "LA": 852 / 4,
}

def generate_sine_wave(freq, duration, sample_rate, amplitude=1.0):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * freq * t)

def apply_fade(audio, fade_in_samples, fade_out_samples):
    audio[:fade_in_samples] *= np.linspace(0, 1, fade_in_samples)
    audio[-fade_out_samples:] *= np.linspace(1, 0, fade_out_samples)
    return audio

def generate_soothing_sound_bath(duration, output_file='deep_ambient_sound.wav'):
    sample_rate = 44100

    # Set the frequencies, amplitudes, and fade times for the deep ambient sound
    freqs = [150, 225, 300, 360]  # Increased frequencies
    amps = [0.4, 0.3, 0.2, 0.1]  # Increased amplitudes
    fade_time = int(0.25 * sample_rate)  # 0.25 seconds

    # Generate sine waves with the given frequencies and amplitudes
    sine_waves = []
    for freq, amp in zip(freqs, amps):
        sine_wave = generate_sine_wave(freq, duration, sample_rate, amplitude=amp)
        sine_waves.append(sine_wave)

    # Create the combined deep ambient sound
    lush_sound = np.zeros_like(sine_waves[0])
    for sine_wave in sine_waves:
        lush_sound += sine_wave

    # Apply fade in and fade out to create smooth transitions
    lush_sound = apply_fade(lush_sound, fade_time, fade_time)

    # Normalize the audio
    lush_sound = librosa.util.normalize(lush_sound, norm=np.inf, axis=None)

    # Save the audio to a file
    sf.write(output_file, lush_sound, sample_rate, format='wav')

    return output_file


def generate_chord(start_freqs, end_freqs, duration, fm_intensity=0.01, fm_speed=0.1):
    num_notes = len(start_freqs)
    assert num_notes == len(
        end_freqs
    ), "Start and end frequencies should have the same length."

    # Generate the time vector
    sample_rate = 44100
    t = np.linspace(0, duration, int(duration * sample_rate), False)

    # Linearly interpolate the frequencies
    freqs = [
        np.linspace(start_freqs[i], end_freqs[i], int(duration * sample_rate))
        for i in range(num_notes)
    ]

    # Generate sine wave for each note in the chord
    chord = []

    for note_freqs in freqs:
        # Generate sine waves for harmonics
        harmonics = np.zeros_like(t)
        max_harmonic = int(1000 / np.max(note_freqs))
        for harmonic in range(1, max_harmonic + 1):
            sine_wave = (
                0.5 * np.sin(2 * np.pi * note_freqs * harmonic * t) / harmonic
            ).astype(np.float32)
            harmonics += sine_wave

        # Apply frequency modulation
        fm = np.sin(2 * np.pi * fm_speed * t) * fm_intensity * note_freqs
        modulated_freq = note_freqs + fm
        modulated_sine_wave = 0.5 * np.sin(2 * np.pi * modulated_freq * t).astype(
            np.float32
        )
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
    chord_data_int16 = (chord_data * (2**15 - 1)).astype(np.int16)

    # Create an AudioSegment from the modulated chord data
    modulated_chord_audio = AudioSegment(
        chord_data_int16.tobytes(), frame_rate=sample_rate, sample_width=2, channels=1
    )

    return modulated_chord_audio


def midi_to_freq(midi_note):
    return 440 * 2 ** ((midi_note - 69) / 12)


def generate_audio(duration, num_segments=1):
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

    # Create a soothing sound bath for the hook
    hook_audio_file = generate_soothing_sound_bath(5)
    hook_audio = AudioSegment.from_wav(hook_audio_file)  # Read the file back as an AudioSegment

    # Concatenate the hook audio with the rest of the audio
    full_audio = hook_audio.append(full_audio)  # Changed this line

    # Export the audio to a WAV file
    full_audio.export("audio_.wav", format="wav")

def generate_background(img_size):
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, img_size, img_size)
    ctx = cairo.Context(surface)

    # Generate random gradient
    gradient = cairo.LinearGradient(0, 0, img_size, img_size)
    gradient.add_color_stop_rgb(0, np.random.rand(), np.random.rand(), np.random.rand())
    gradient.add_color_stop_rgb(1, np.random.rand(), np.random.rand(), np.random.rand())

    # Draw rectangle with gradient
    ctx.rectangle(0, 0, img_size, img_size)
    ctx.set_source(gradient)
    ctx.fill()

    # Create SurfacePattern from surface
    pattern = cairo.SurfacePattern(surface)
    pattern.set_extend(cairo.EXTEND_REPEAT)

    return pattern

def mutate_shape(ctx, shape_type, x, y, width, height):
    """
    Modifies a shape to create a face.
    """
    if shape_type == 'rectangle':
        # Create a square face
        width = height = np.random.randint(img_size//4, img_size//2)
        ctx.rectangle(x, y, width, height)
    elif shape_type == 'circle':
        # Add eyes and a mouth to the circle
        radius = np.random.randint(img_size//4, img_size//2)
        ctx.arc(x, y, radius, 0, 2 * np.pi)
        eye_radius = radius // 5
        eye_x_offset = radius // 2
        eye_y_offset = radius // 2
        mouth_width = radius // 2
        mouth_height = radius // 3
        mouth_y_offset = radius // 2
        ctx.arc(x - eye_x_offset, y - eye_y_offset, eye_radius, 0, 2 * np.pi)
        ctx.arc(x + eye_x_offset, y - eye_y_offset, eye_radius, 0, 2 * np.pi)
        ctx.move_to(x - mouth_width//2, y + mouth_y_offset)
        ctx.line_to(x + mouth_width//2, y + mouth_y_offset)
        ctx.line_to(x + mouth_width//2, y + mouth_y_offset + mouth_height)
        ctx.line_to(x - mouth_width//2, y + mouth_y_offset + mouth_height)
        ctx.line_to(x - mouth_width//2, y + mouth_y_offset)
    elif shape_type == 'triangle':
        # Add eyes and a mouth to the triangle
        x1, y1 = x, y - height//2
        x2, y2 = x - width//2, y + height//2
        x3, y3 = x + width//2, y + height//2
        ctx.move_to(x1, y1)
        ctx.line_to(x2, y2)
        ctx.line_to(x3, y3)
        ctx.line_to(x1, y1)
        eye_radius = width // 10
        eye_x_offset = width // 4
        eye_y_offset = height // 4
        mouth_width = width // 2
        mouth_height = height // 3
        mouth_y_offset = height // 2
        ctx.arc(x - eye_x_offset, y - eye_y_offset, eye_radius, 0, 2 * np.pi)
        ctx.arc(x + eye_x_offset, y - eye_y_offset, eye_radius, 0, 2 * np.pi)
        ctx.move_to(x - mouth_width//2, y + mouth_y_offset)
        ctx.line_to(x + mouth_width//2, y + mouth_y_offset)
        ctx.line_to(x + mouth_width//2, y + mouth_y_offset + mouth_height)
        ctx.line_to(x - mouth_width//2, y + mouth_y_offset + mouth_height)
        ctx.line_to(x - mouth_width//2, y + mouth_y_offset)

    return x, y, width, height


def generate_visuals(duration, img_size, num_frames):
    frames = []
    # Generate random background
    background = generate_background(img_size)
    # Create a surface for drawing the shapes
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, img_size, img_size)
    ctx = cairo.Context(surface)


    ctx.paint()

    # Define possible shape types
    shape_types = ['rectangle', 'circle', 'line', 'triangle', 'ellipse', 'star']

    # Define possible movement types
    movements = ['linear', 'circular', 'bezier', 'zigzag', 'bounce']

    # Generate random shapes
    for i in range(num_frames):
        # Draw background
        ctx.set_source(background)
        ctx.paint()
        # Set random colors
        r, g, b = np.random.rand(3)
        ctx.set_source_rgb(r, g, b)

        # Set random shape parameters
        shape_type = np.random.choice(shape_types)
        if shape_type == 'rectangle':
            x, y = np.random.randint(0, img_size, size=2)
            width, height = np.random.randint(img_size//4, img_size//2, size=2)
            ctx.rectangle(x, y, width, height)
        elif shape_type == 'circle':
            x, y = np.random.randint(0, img_size, size=2)
            radius = np.random.randint(img_size//4, img_size//2)
            ctx.arc(x, y, radius, 0, 2 * np.pi)
        elif shape_type == 'line':
            x1, y1 = np.random.randint(0, img_size, size=2)
            x2, y2 = np.random.randint(0, img_size, size=2)
            ctx.move_to(x1, y1)
            ctx.line_to(x2, y2)
            ctx.set_line_width(np.random.randint(1, 10))
        elif shape_type == 'triangle':
            x1, y1 = np.random.randint(0, img_size, size=2)
            x2, y2 = np.random.randint(0, img_size, size=2)
            x3, y3 = np.random.randint(0, img_size, size=2)
            ctx.move_to(x1, y1)
            ctx.line_to(x2, y2)
            ctx.line_to(x3, y3)
            ctx.line_to(x1, y1)
        elif shape_type == 'ellipse':
            x, y = np.random.randint(0, img_size, size=2)
            width, height = np.random.randint(img_size//4, img_size//2, size=2)
            ctx.save()
            ctx.translate(x, y)
            ctx.scale(width / 2, height / 2)
            ctx.arc(0, 0, 1, 0, 2 * np.pi)
            ctx.restore()
        elif shape_type == 'star':
            x, y = np.random.randint(0, img_size, size=2)
            outer_radius = np.random.randint(img_size//4, img_size//2)
            inner_radius = outer_radius // 2
            num_points = np.random.randint(5, 10)
            angle = np.pi / num_points
            ctx.save()
            ctx.translate(x, y)
            ctx.move_to(outer_radius, 0)
            for i in range(num_points):
                ctx.rotate(angle)
                ctx.line_to(inner_radius, 0)
                ctx.rotate(angle)
                ctx.line_to(outer_radius, 0)
            ctx.restore()

        # Apply random movement to shape
        movement = np.random.choice(movements)
        speed = np.random.randint(1, 10)
        if movement == 'linear':
            ctx.translate(np.sin(i * speed) * img_size / 2, np.cos(i * speed) * img_size / 2)
        elif movement == 'circular':
            ctx.translate(img_size / 2, img_size / 2)
            ctx.rotate(np.random.random() * 2 * np.pi)
            ctx.translate(-img_size / 2, -img_size / 2)
            ctx.translate(np.sin(i * speed) * img_size / 2, np.cos(i * speed) * img_size / 2)
        elif movement == 'bezier':
            cp1x, cp1y = np.random.randint(0, img_size, size=2)
            cp2x, cp2y = np.random.randint(0, img_size, size=2)
            x, y = np.random.randint(0, img_size, size=2)
            ctx.curve_to(cp1x, cp1y, cp2x, cp2y, x, y)
        elif movement == 'zigzag':
            x, y = np.random.randint(0, img_size, size=2)
            freq = np.random.randint(5, 20)
            amp = np.random.randint(5, 20)
            offset = np.random.randint(-img_size//4, img_size//4)
            x_offset = (np.sin(i / freq) * amp) + offset
            ctx.translate(x_offset, 0)
        elif movement == 'bounce':
            x, y = np.random.randint(0, img_size, size=2)
            freq = np.random.randint(5, 20)
            amp = np.random.randint(5, 20)
            offset = np.random.randint(-img_size//4, img_size//4)
            y_offset = (np.sin(i / freq) * amp) + offset
            ctx.translate(0, y_offset)

        # Fill or stroke the shape
        if shape_type == 'line':
            ctx.stroke()
        else:
            ctx.fill()

        # Convert surface to PIL image and append to list of frames
        img = Image.frombuffer("RGBA", (img_size, img_size), surface.get_data(), "raw", "BGRA", 0, 1)
        frames.append(img)

    return frames



def generate_random_text():
    random_text = " ".join(
        "".join(
            random.choices(
                string.ascii_letters + string.digits, k=random.randint(1, 10)
            )
        )
        for _ in range(random.randint(1, 4))  # Change 6 to 4 to reserve 2 slots for "harmony" and "simapsee"
    )

    # Add "harmony" and "simapsee" to the random text
    random_text = f"harmony {random_text} simapsee"

    # Add "get off instagram" or "focus on your goals" to the random text with a certain probability
    if random.random() < 0.3:
        random_text = random.choice(["get off instagram", "focus on your goals"])

    return random_text

def generate_video(duration, img_size, fps, text_duration, num_generations=30, crossfade_duration=0):
    audio_path = "audio_.wav"
    video_path = "output_.mp4"
    num_frames = duration * fps

    audio_segment = AudioSegment.from_wav(audio_path)
    audio_segment.export(audio_path, format="wav")

    # Generate multiple generations of visuals
    all_frames = []
    for generation in range(num_generations):
        generation_duration = duration / num_generations
        generation_frames = generate_visuals(generation_duration, img_size, num_frames // num_generations)
        all_frames.extend(generation_frames)

    # Combine frames with crossfades
    crossfade_frames = int(crossfade_duration * fps)
    num_frames = len(all_frames)
    fade_in = np.linspace(0, 1, crossfade_frames)
    fade_out = np.linspace(1, 0, crossfade_frames)
    for i in range(num_generations - 1):
        start_frame = i * num_frames // num_generations
        end_frame = (i + 1) * num_frames // num_generations
        for j in range(crossfade_frames):
            alpha = fade_out[j]
            all_frames[start_frame + j] = Image.blend(
                all_frames[start_frame + j],
                all_frames[end_frame - crossfade_frames + j],
                alpha,
            )
            alpha = fade_in[j]
            all_frames[end_frame - crossfade_frames + j] = Image.blend(
                all_frames[start_frame + j],
                all_frames[end_frame - crossfade_frames + j],
                alpha,
            )

   # Combine frames into video
    audio = AudioFileClip(audio_path)
    video = ImageSequenceClip([np.array(img) for img in all_frames], fps=fps)


    # Add text frames at random intervals
    text_frames = int(text_duration * fps)

    # Create a 5-second visual hook
    hook_text = (
        "WARNING: This content contains WILL flashing images IMPROVE YOUR LIFE. "
        "Enter at your own risk."
    )
    hook_text_clip = (TextClip(hook_text, fontsize=30, font="Arial", color="white", size=video.size, method='caption')
                  .set_position("center")
                  .set_duration(5))



    # Initialize clips as an empty list
    clips = [hook_text_clip]

    for i in range(0, num_frames, text_frames):
        random_text = generate_random_text()

        text_clip = (TextClip(random_text, fontsize=30, font="Blox2.ttf", color="white", size=video.size)
                     .set_position("center")
                     .set_duration(text_duration))

        if random.random() < 0.5:
            clips.append(text_clip.set_start(5 + i / fps))

    # Combine the video and the text clips
    video = CompositeVideoClip(clips + [ImageClip(np.array(img)).set_duration(1 / fps).set_start(5 + i / fps) for i, img in enumerate(all_frames)])

    # Add audio and save the video
    audio = AudioFileClip(audio_path)
    video = video.set_audio(audio)
    video.write_videofile(video_path, fps=fps)






if __name__ == "__main__":
    duration = 25  # seconds
    img_size = 800  # Instagram square dimensions (1080x1080)
    fps = 30  # frames per second

    num_generations = random.randint(20, 50)
    crossfade_duration = random.uniform(0.5, 2)
    text_interval = random.randint(2, 6)
    text_duration = random.uniform(0.2, 1)
    num_segments = 1

    generate_audio(duration, num_segments)
    generate_video(
        duration,
        img_size,
        fps,
        num_generations=num_generations,
        crossfade_duration=crossfade_duration,
        text_duration=text_duration,
    )
