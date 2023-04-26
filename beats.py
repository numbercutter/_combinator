import random
import numpy as np
from PIL import Image
from moviepy.editor import AudioFileClip, ImageSequenceClip
from moviepy.editor import TextClip, CompositeVideoClip
from moviepy.editor import ImageClip
from moviepy.editor import concatenate_videoclips
from pydub import AudioSegment  # Ensure this import is here
import librosa
import soundfile as sf
from audio import apply_high_pass_filter, generate_chord, solfeggio_freqs, generate_soothing_sound_bath, generate_drum_pattern
from visual import generate_visuals, generate_image, generate_random_text, get_random_hashtags, generate_title, save_instagram_caption



def generate_audio(duration, num_segments=1):
    segment_duration = duration / num_segments
    patterns = {
        "kick": ["x", "-", "-", "-", "-", "-", "-", "-", "-", "-", "x", "-", "-", "-", "-", "-"],
        "snare": ["-", "-", "-", "-", "x", "-", "-", "-", "-", "-", "-", "-", "x", "-", "-", "-"],
        "hihat": ["x", "-", "x", "-", "x", "-", "x", "-", "x", "-", "x", "-", "x", "-", "x", "-"]
    }

    tempo = 190
    filename = "drum_pattern.wav"
    drum_loop = generate_drum_pattern(patterns, tempo, filename)
    drum_loop.export("audio_.wav", format="wav")
    
if __name__ == "__main__":
    duration = 12
    img_size = 1000
    fps = 30
    num_generations = random.randint(20, 50)
    crossfade_duration = random.uniform(0.5, 2)
    text_interval = random.randint(2, 6)
    text_duration = random.uniform(0.2, 1)
    num_segments = 1

    generate_audio(duration, num_segments)