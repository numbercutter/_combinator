import random
import numpy as np
from PIL import Image

from pydub import AudioSegment  # Ensure this import is here
from audio import generate_bass_pattern, sine_wave_synthesizer, generate_drum_pattern, generate_drum_pattern_high_res, generate_soothing_sound_bath, generate_simple_chord, apply_high_pass_filter

chords = [
    # A minor - C major - D major - E major - G major (already provided)
    [[110, 165, 220], [130, 195, 260], [146, 220, 293], [164, 246, 329], [196, 294, 392]],

    # C major - D major - E major - G major (already provided)
    [[130, 195, 260], [146, 220, 293], [164, 246, 329], [196, 294, 392]],

    # C major - A minor - F major - G major
    [[130, 195, 260], [110, 165, 220], [174, 261, 348], [196, 294, 392]],

    # D minor - Bb major - A minor - G major
    [[146, 220, 293], [116, 174, 232], [110, 165, 220], [196, 294, 392]],

    # E minor - G major - A minor - B minor
    [[164, 246, 329], [196, 294, 392], [110, 165, 220], [123, 185, 246]],

    # F major - G major - E minor - C major
    [[174, 261, 348], [196, 294, 392], [164, 246, 329], [130, 195, 260]]
]

    
def generate_full_audio(duration, num_segments=2):

    # Choose a random chord progression
    chord_progression = random.choice(chords)
    print(chord_progression)
    # Generate the drum loop
    tempo = 100
    drum_loop = generate_drum_pattern(tempo=tempo, filename="drum_pattern.wav", bars=4)

    # Generate the high-resolution drum loop
    drum_loop_high_res = generate_drum_pattern_high_res(tempo=tempo, filename="drum_pattern_high_res.wav", bars=16)

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

    segment_duration = duration / num_segments

    # Generate the bass line segments using the chord progression
    bass_segments = []
    for chord in chord_progression:
        bass_line = generate_bass_pattern(chord, tempo=190, duration=segment_duration, bars=16)
        bass_segments.append(bass_line)

    # Concatenate bass line segments
    full_bass_line = bass_segments[0]
    for segment in bass_segments[1:]:
        full_bass_line = full_bass_line.append(segment)


    # Mix the full bass line with the drum loops
    mixed_audio = drum_loop.overlay(full_bass_line)
    mixed_audio = mixed_audio.overlay(drum_loop_high_res)

    # Generate the audio segments using the chord progression
    audio_segments = []
    for _ in range(num_segments):
        chord_file = generate_simple_chord(chord_progression, duration=segment_duration)
        chord = AudioSegment.from_wav(chord_file)
        audio_segments.append(chord)

    # Concatenate audio segments
    full_audio = audio_segments[0]
    for segment in audio_segments[1:]:
        full_audio = full_audio.append(segment)

    # Create a soothing sound bath for the hook
    hook_audio_file = generate_soothing_sound_bath(3)
    hook_audio = AudioSegment.from_wav(hook_audio_file)  # Read the file back as an AudioSegment

    filtered_audio = apply_high_pass_filter(full_audio)

    # Mix the filtered audio with the mixed_audio
    mixed_audio = mixed_audio.overlay(filtered_audio)

    # Concatenate the hook audio with the mixed audio
    mixed_audio = hook_audio + mixed_audio

    # Export the mixed audio to a WAV file
    mixed_audio.export("audio_.wav", format="wav")

if __name__ == "__main__":
    duration = 20
    generate_full_audio(duration)
