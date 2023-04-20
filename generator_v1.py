import os
import random
import numpy as np
from PIL import Image
from pydub import AudioSegment
from moviepy.editor import AudioFileClip, ImageSequenceClip
from moviepy.editor import TextClip, CompositeVideoClip
import string
import cairo
from moviepy.editor import concatenate_videoclips

solfeggio_freqs = {
    "UT": 396 / 4,
    "RE": 417 / 4,
    "MI": 528 / 4,
    "FA": 639 / 4,
    "SOL": 741 / 4,
    "LA": 852 / 4,
}

# New function to generate soothing sound bath
def generate_soothing_sound_bath(duration):
    freqs = [200, 300, 400]  # You can modify these frequencies to create a more soothing sound bath
    chord_data = generate_chord(freqs, freqs, duration)
    return chord_data

# New function to create the 5-second visual hook
def create_hook_text_clip(duration, text, fontsize, font, color, size, position="center"):
    text_clip = (
        TextClip(text, fontsize=fontsize, font=font, color=color, size=size)
        .set_position(position)
        .set_duration(duration)
    )
    return text_clip

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
    hook_audio = generate_soothing_sound_bath(5)

    # Concatenate the hook audio with the rest of the audio
    full_audio = hook_audio + full_audio

    # Export the audio to a WAV file
    full_audio.export("audio_.wav", format="wav")




def generate_visuals(duration, img_size, num_frames, update_interval=5):
    frames = []

    # Create a surface for drawing the shapes
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, img_size, img_size)
    ctx = cairo.Context(surface)

    # Define possible shape types
    shape_types = ["rectangle", "circle", "line", "triangle", "arc"]

    # Define possible movement types
    movements = ["linear", "circular", "bezier", "scale"]

    # Set up variables for the previous and current shapes
    prev_shapes = []
    curr_shapes = []

    # Generate random shapes
    for i in range(num_frames):
        # Clear the canvas
        ctx.save()
        ctx.set_operator(cairo.Operator.CLEAR)
        ctx.paint()
        ctx.restore()

        # Check if it's time to update the visuals
        if i % update_interval == 0:
            prev_shapes = curr_shapes
            curr_shapes = []

            # Generate random shapes
            num_shapes = np.random.randint(1, 5)
            for _ in range(num_shapes):
                shape = {
                    "type": np.random.choice(shape_types),
                    "params": np.random.uniform(0, img_size, size=4),
                    "color": np.random.uniform(0.3, 0.7, size=4),
                    "movement": np.random.choice(movements),
                    "speed": np.random.randint(1, 10),
                }
                curr_shapes.append(shape)

        # Interpolate between the previous and the current shapes
        t = (i % update_interval) / update_interval
        shapes = []
        for prev_shape, curr_shape in zip(prev_shapes, curr_shapes):
            shape = {}
            for key in prev_shape.keys():
                if key == "params":
                    shape[key] = prev_shape[key] * (1 - t) + curr_shape[key] * t
                else:
                    shape[key] = curr_shape[key]
            shapes.append(shape)

        # Draw the interpolated shapes
        for shape in shapes:
            ctx.set_source_rgba(*shape["color"])

            if shape["type"] == "rectangle":
                x, y, width, height = shape["params"]
                ctx.rectangle(x, y, width, height)
            elif shape["type"] == "circle":
                x, y, radius, _ = shape["params"]
                ctx.arc(x, y, radius, 0, 2 * np.pi)
            elif shape["type"] == "line":
                x1, y1, x2, y2 = shape["params"]
                ctx.move_to(x1, y1)
                ctx.line_to(x2, y2)
                ctx.set_line_width(np.random.randint(1, 10))
            elif shape["type"] == "triangle":
                x1, y1, x2, y2 = shape["params"]
                x3, y3 = np.random.randint(0, img_size, size=2)
                ctx.move_to(x1, y1)
                ctx.line_to(x2, y2)
                ctx.line_to(x3, y3)
                ctx.close_path()
            elif shape["type"] == "arc":
                x, y, radius, angle = shape["params"]
                ctx.arc(x, y, radius, 0, angle)
        # Apply random movement to shape
            movement = np.random.choice(movements)
            speed = np.random.randint(1, 10)
            if movement == "linear":
                ctx.translate(
                    np.sin(i * speed) * img_size / 2, np.cos(i * speed) * img_size / 2
                )
            elif movement == "circular":
                ctx.translate(img_size / 2, img_size / 2)
                ctx.rotate(np.random.random() * 2 * np.pi)
                ctx.translate(-img_size / 2, -img_size / 2)
                ctx.translate(
                    np.sin(i * speed) * img_size / 2, np.cos(i * speed) * img_size / 2
                )

            elif movement == "bezier":
                cp1x, cp1y = np.random.randint(0, img_size, size=2)
                cp2x, cp2y = np.random.randint(0, img_size, size=2)
                x, y = np.random.randint(0, img_size, size=2)
                ctx.curve_to(cp1x, cp1y, cp2x, cp2y, x, y)
            elif movement == "scale":
                scale_factor = np.random.rand() * 2
                ctx.translate(img_size / 2, img_size / 2)
                ctx.scale(scale_factor, scale_factor)
                ctx.translate(-img_size / 2, -img_size / 2)
            # Fill or stroke the shape
            if shape["type"] == "line":
                ctx.stroke()
            else:
                ctx.fill()

        # Convert surface to PIL image and append to list of frames
        img = Image.frombuffer(
            "RGBA", (img_size, img_size), surface.get_data(), "raw", "BGRA", 0, 1
        )
        frames.append(img)

    return frames


def generate_video(
    duration,
    img_size,
    fps,
    text_interval,
    text_duration,
    num_generations=30,
    crossfade_duration=0,
):
    audio_path = "audio_.wav"
    video_path = "output_.mp4"
    num_frames = duration * fps

    audio_segment = AudioSegment.from_wav(audio_path)
    audio_segment.export(audio_path, format="wav")

    # Generate multiple generations of visuals
    all_frames = []
    for generation in range(num_generations):
        generation_duration = duration / num_generations
        generation_frames = generate_visuals(
            generation_duration, img_size, num_frames // num_generations
        )
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

    # Your other settings and configurations...


    # Add text frames at random intervals
    text_duration = 0.5  # seconds
    text_frames = int(text_duration * fps)

    # Create a 5-second visual hook
    hook_text = (
        "WARNING: This content will improve your life. "
        "Enter at your own risk."
    )
    hook_text_clip = create_hook_text_clip(
        5, hook_text, 30, "Arial", "white", video.size
    )

    # Add the hook text to the beginning of the clips array
    clips = [hook_text_clip] + clips

    # Add text clips to the array at random intervals
    for i in range(0, num_frames, text_frames):
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

        text_clip = (
            TextClip(random_text, fontsize=30, font="Blox2.ttf", color="white", size=video.size)
            .set_position("center")
            .set_duration(text_duration)
        )

        # Only add the text clip to the array with a certain probability
        if random.random() < 0.5:
            clips.append(text_clip.set_start(i / fps))

    # Combine the video and the text clips
    video = CompositeVideoClip(clips)

    # Add audio and save the video
    video = video.set_audio(audio)
    video.write_videofile(video_path, fps=fps)



if __name__ == "__main__":
    duration = 30  # seconds
    img_size = 800  # Instagram square dimensions (1080x1080)
    fps = 30  # frames per second

    num_generations = random.randint(20, 50)
    crossfade_duration = random.uniform(0.5, 2)
    text_interval = random.randint(2, 15)
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
        text_interval=text_interval,
    )
