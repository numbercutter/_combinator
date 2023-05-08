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
    [[174, 261, 348], [196, 294, 392], [164, 246, 329], [130, 195, 260]],
    
    # C major - D major - E major - G major (already provided)
    [[130, 195, 260], [146, 220, 293], [164, 246, 329], [196, 294, 392]],

    # C major - A minor - F major - G major
    [[130, 195, 260], [110, 165, 220], [174, 261, 348], [196, 294, 392]],

    # D minor - Bb major - A minor - G major
    [[146, 220, 293], [116, 174, 232], [110, 165, 220], [196, 294, 392]],

    # E minor - G major - A minor - B minor
    [[164, 246, 329], [196, 294, 392], [110, 165, 220], [123, 185, 246]],

    # F major - G major - E minor - C major
    [[174, 261, 348], [196, 294, 392], [164, 246, 329], [130, 195, 260]],

    # G major - D major - A minor - C major
    [[196, 294, 392], [146, 220, 293], [110, 165, 220], [130, 195, 260]],

    # A minor - E minor - F major - G major
    [[110, 165, 220], [164, 246, 329], [174, 261, 348], [196, 294, 392]],

    # C major - G major - A minor - F major
    [[130, 195, 260], [196, 294, 392], [110, 165, 220], [174, 261, 348]],

    # D major - B minor - G major - A major
    [[146, 220, 293], [123, 185, 246], [196, 294, 392], [220, 330, 440]]
]

    
def generate_full_audio(duration, num_segments=2):
    chord_progression = random.choice(chords)
    print(chord_progression)

    tempo = 190
    drum_loop = generate_drum_pattern(tempo=tempo, filename="drum_pattern.wav", bars=4)
    drum_loop_high_res = generate_drum_pattern_high_res(tempo=100, filename="drum_pattern_high_res.wav", bars=16)

    while len(drum_loop) < duration * 1000:
        drum_loop += drum_loop

    while len(drum_loop_high_res) < duration * 1000:
        drum_loop_high_res += drum_loop_high_res

    drum_loop = drum_loop[:duration * 1000]
    drum_loop_high_res = drum_loop_high_res[:duration * 1000]

    segment_duration = duration / num_segments

    bass_motifs = [generate_bass_pattern(chord, tempo=tempo, duration=segment_duration, bars=4) for chord in chord_progression]

    full_bass_line = AudioSegment.silent(duration=0)
    while len(full_bass_line) < duration * 1000:
        for motif in bass_motifs:
            full_bass_line += motif

    full_bass_line = full_bass_line[:duration * 1000]

    mixed_audio = drum_loop.overlay(full_bass_line)
    mixed_audio = mixed_audio.overlay(drum_loop_high_res)
    

    audio_segments = []
    for _ in range(num_segments):
        chord_file = generate_simple_chord(chord_progression, duration=segment_duration)
        chord = AudioSegment.from_wav(chord_file)
        audio_segments.append(chord)

    full_audio = audio_segments[0]
    for segment in audio_segments[1:]:
        full_audio = full_audio.append(segment)

    hook_audio_file = generate_soothing_sound_bath(3)
    hook_audio = AudioSegment.from_wav(hook_audio_file)

    filtered_audio = apply_high_pass_filter(full_audio)
    mixed_audio = mixed_audio.overlay(filtered_audio)
    mixed_audio = hook_audio + mixed_audio
    mixed_audio.export("audio_.wav", format="wav")

if __name__ == "__main__":
    duration = 20
    generate_full_audio(duration)
