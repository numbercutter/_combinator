import os
import random
import numpy as np
from math import sqrt, sin, cos, radians
from PIL import Image, ImageDraw, ImageFont, ImageChops
from pydub import AudioSegment
from moviepy.editor import AudioFileClip, ImageSequenceClip
from moviepy.editor import TextClip, CompositeVideoClip
from moviepy.editor import ImageClip
import re
import string
import cairo
from moviepy.editor import concatenate_videoclips
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
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
    freqs = [50, 100, 150, 225, 300, 360]  # Added lower frequencies
    amps = [0.5, 0.4, 0.4, 0.3, 0.2, 0.1]  # Adjusted amplitudes
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

    # Generate binaural beat (e.g., 5 Hz difference for theta waves)
    binaural_freq = 5
    left_ear = generate_sine_wave(freqs[0] - binaural_freq / 2, duration, sample_rate, amplitude=0.5)
    right_ear = generate_sine_wave(freqs[0] + binaural_freq / 2, duration, sample_rate, amplitude=0.5)

    # Combine lush sound with binaural beat
    lush_sound_left = lush_sound + left_ear
    lush_sound_right = lush_sound + right_ear

    # Create stereo audio
    stereo_audio = np.vstack((lush_sound_left, lush_sound_right)).T

    # Normalize the audio
    stereo_audio = librosa.util.normalize(stereo_audio, norm=np.inf, axis=None)

    # Save the audio to a file
    sf.write(output_file, stereo_audio, sample_rate, format='wav', subtype='PCM_24')

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


def generate_visuals(duration, img_size, num_frames):
    frames = []
    # Generate random background
    background = generate_background(img_size)

    # Create a surface for drawing the shapes
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, img_size, img_size)
    ctx = cairo.Context(surface)


    ctx.paint()

    # Define possible shape types
    shape_types = ['rectangle', 'circle', 'line']

    # Define possible movement types
    movements = ['linear', 'circular', 'bezier']

    # Generate random shapes
    for i in range(num_frames):
        # Draw background
        #ctx.set_source(background)
        #ctx.paint()
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


def save_instagram_caption(caption, hashtags, filename="instagram_caption.txt"):
    with open(filename, "w") as file:
        file.write(caption)
        file.write("\n\n. . .\n\n")
        file.write(" ".join(hashtags))

def get_random_hashtags(hashtags, num=7):
    return random.sample(hashtags, num)

def generate_title(hashtags, num_words=2):
    words = []
    for hashtag in hashtags:
        words += re.findall(r'\w+', hashtag)

    combined_words = []
    for _ in range(num_words):
        word1 = random.choice(words)
        word2 = random.choice(words)
        word1_part = word1[:len(word1) // 2]
        word2_part = word2[len(word2) // 2:]
        combined_word = word1_part + word2_part
        combined_words.append(combined_word)

    title = " ".join(combined_words) + " " + str(random.randint(1, 999))
    return title

def mandelbrot(c, max_iter):
    z = c
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z * z + c
    return max_iter

def generate_fractal_layer(width, height):
    zoom_factor = random.uniform(0.6, 1.5)
    center_x = random.uniform(-2, 0.5)
    center_y = random.uniform(-1.5, 1.5)

    min_x, max_x = center_x - zoom_factor, center_x + zoom_factor
    min_y, max_y = center_y - zoom_factor, center_y + zoom_factor
    max_iter = 1000

    img = Image.new("RGB", (width, height))
    pixels = img.load()

    color_shifts = (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10))

    for x in range(width):
        for y in range(height):
            real = min_x + (max_x - min_x) * x / (width - 1)
            imag = min_y + (max_y - min_y) * y / (height - 1)
            c = complex(real, imag)
            color = mandelbrot(c, max_iter)
            r = (color * color_shifts[0]) % 256
            g = (color * color_shifts[1]) % 256
            b = (color * color_shifts[2]) % 256
            pixels[x, y] = (r, g, b)

    return img

def draw_circle(draw, center, radius, fill):
    draw.ellipse([center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius], outline=fill)

def draw_regular_polygon(draw, center, num_sides, radius, fill):
    angle = 360 / num_sides
    points = []
    for i in range(num_sides):
        x = center[0] + radius * cos(radians(i * angle))
        y = center[1] + radius * sin(radians(i * angle))
        points.append((x, y))
    draw.polygon(points, outline=fill)

def draw_vesica_piscis(draw, center, radius, fill):
    draw_circle(draw, center, radius, fill)
    draw_circle(draw, (center[0] + radius, center[1]), radius, fill)

def draw_seed_of_life(draw, center, radius, fill):
    draw_circle(draw, center, radius, fill)
    for i in range(6):
        x = center[0] + radius * cos(radians(i * 60))
        y = center[1] + radius * sin(radians(i * 60))
        draw_circle(draw, (x, y), radius, fill)

def draw_sacred_geometry(img, emblem_size=100):
    center = (img.size[0] // 2, img.size[1] // 2)
    radius = emblem_size // 2
    draw = ImageDraw.Draw(img)
    fill = (255, 255, 255)

    shapes = [
        lambda: draw_circle(draw, center, radius, fill),
        lambda: draw_regular_polygon(draw, center, 3, radius, fill),
        lambda: draw_regular_polygon(draw, center, 4, radius, fill),
        lambda: draw_regular_polygon(draw, center, 5, radius, fill),
        lambda: draw_regular_polygon(draw, center, 6, radius, fill),
        lambda: draw_vesica_piscis(draw, center, radius, fill),
        lambda: draw_seed_of_life(draw, center, radius, fill)
    ]

    num_shapes = random.randint(1, 3)
    selected_shapes = random.sample(shapes, num_shapes)
    for shape in selected_shapes:
        shape()

def draw_text(img, text="Focal Point", font_path="Blox2.ttf", font_size=48):
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, font_size)
    text_width, text_height = draw.textsize(text, font=font)

    text_x = (img.size[0] - text_width) // 2
    text_y = img.size[1] // 2 + 100

    draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255))

def generate_image(width, height, num_layers=3, filename="fractal_image.png"):
    base_image = generate_fractal_layer(width, height)

    for _ in range(num_layers - 1):
        layer = generate_fractal_layer(width, height)
        base_image = ImageChops.blend(base_image, layer, alpha=0.5)

    draw_sacred_geometry(base_image)
    draw_text(base_image)

    base_image.save(filename)

if __name__ == "__main__":
    hashtags = [
    "#mindfulness", "#meditation", "#awareness", "#consciousness", "#presence", 
    "#brainwaves", "#neuroscience", "#brainpower", "#cognitive", "#neural",
    "#sacredgeometry", "#mandala", "#fractal", "#goldenratio", "#floweroflife",
    "#ancientwisdom", "#spirituality", "#ancientknowledge", "#esoteric", "#mysticism",
    "#soundscience", "#binauralbeats", "#soundhealing", "#frequency", "#vibration",
    "#mindfulthinking", "#mindfulmovement", "#mindfulbreathing", "#mindfulawareness", "#mindfultravel",
    "#meditationteachertraining", "#meditationtipsandtricks", "#meditationinspiration", "#meditationpractice",
    "#consciousnessshift", "#consciouscommunity", "#consciousawakening", "#consciousliving", "#consciouslife",
    "#brainhacks", "#braintrainingtips", "#cognitiverehabilitation", "#neuralplasticitytraining", "#neuroplasticityresearch",
    "#sacredgeometryjewelry", "#sacredgeometrytattoo", "#mandalapainting", "#fractalartwork", "#goldenratiopattern",
    "#floweroflifemandalas", "#ancientegypt", "#ancientcivilizations", "#spiritualgrowth", "#spiritualenlightenment",
    "#esotericphilosophy", "#mysticalteachings", "#soundscapes", "#binauralbeatsfrequency", "#soundtherapyhealing",
    "#frequencyhealing", "#vibrationtherapy", "#mindfulbeauty", "#mindfulwellness", "#mindfulnutrition", "#mindfulfitness",
    "#meditationmusician", "#meditationretreats", "#meditationcenter", "#consciousnesscoaching", "#consciousparenting",
    "#brainboosters", "#brainimprovement", "#cognitiveenhancement", "#neuralscience", "#sacredgeometrydesigns",
    "#mandaladesign", "#fractalgeometry", "#goldenratioart", "#floweroflifeartwork", "#ancientknowledgekeeper",
    "#spiritualjourneys", "#esotericmysticism", "#soundhealingtherapy", "#binauralbeatsmeditations", "#vibrationmedicine",
    "#mindfulleadership", "#mindfulentrepreneur", "#mindfulmanagement", "#meditationbenefitsresearch", "#consciousrelationships",
    "#braintraininggames", "#neuralstimulation", "#sacredgeometryartwork", "#mandalawallpaper", "#fractalnature",
    "#goldenratiophotography", "#floweroflifepatterns", "#ancienthealing", "#spiritualawakeningprocess",
    "#esotericwisdom", "#mysticalhealing", "#soundhealingmeditation", "#binauralbeatsmusic", "#vibrationhealing",
    "#mindfularttherapy", "#mindfulhiking", "#mindfulyoga", "#meditationfestival", "#consciouscommunitybuilding",
    "#brainwaveentrainment", "#neuroplasticityexercise", "#sacredgeometryartist", "#mandalastones", "#fractalpatterns",
    "#goldenratiogarden", "#floweroflifetattoos", "#ancienttexts", "#spiritualunderstanding", "#esoterictraditions",
    "#mysticalliving", "#soundhealingtherapy", "#binauralbeatstherapy"
    ]
    random_hashtags = get_random_hashtags(hashtags)
    print("Random Hashtags:", random_hashtags)

    random_title = generate_title(hashtags)
    print("Random Title:", random_title)
    
    
    
    save_instagram_caption(random_title, random_hashtags)
    generate_image(1080, 1080)
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
