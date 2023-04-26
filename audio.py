import os
import random
import numpy as np

from pydub import AudioSegment

import librosa
import soundfile as sf
from pydub.effects import high_pass_filter  # Added import for high_pass_filter


solfeggio_freqs = {
    "UT": 396 / 4,
    "RE": 417 / 4,
    "MI": 528 / 4,
    "FA": 639 / 4,
    "SOL": 741 / 4,
    "LA": 852 / 4,
}

def apply_high_pass_filter(audio_segment, cutoff_freq=150):
    filtered_audio = high_pass_filter(audio_segment, cutoff_freq)
    return filtered_audio

def generate_sine_wave(freq, duration, sample_rate, amplitude=1.0):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * freq * t)

def apply_fade(audio, fade_in_samples, fade_out_samples):
    audio[:fade_in_samples] *= np.linspace(0, 1, fade_in_samples)
    audio[-fade_out_samples:] *= np.linspace(1, 0, fade_out_samples)
    return audio




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


def generate_soothing_sound_bath(duration, output_file='deep_ambient_sound.wav'):
    sample_rate = 44100

    # Define the different chords to use
    chords = [
        [110, 165, 220],  # A minor
        [130, 195, 260],  # C major
        [146, 220, 293],  # D major
        [164, 246, 329],  # E major
        [196, 294, 392],  # G major
    ]

    # Select a random chord to use
    selected_chord = random.choice(chords)

    # Set the frequencies, amplitudes, and fade times for the deep ambient sound
    freqs = [f * random.uniform(0.9, 1.1) for f in selected_chord]  # Randomize frequency slightly
    amps = [0.2, 0.5, 0.3]  # Adjusted amplitudes
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

    # Map brainwave states to corresponding binaural frequencies
    binaural_freqs = {
        "theta": 5,
        "alpha": 10,
        "beta": 20,
        "gamma": 30,
    }

    # Choose a random brainwave state
    brainwave_state = random.choice(list(binaural_freqs.keys()))

    # Generate binaural beat with the selected frequency
    binaural_freq = binaural_freqs[brainwave_state]
    left_ear = generate_sine_wave(freqs[0] - binaural_freq / 2, duration, sample_rate, amplitude=0.2)
    right_ear = generate_sine_wave(freqs[0] + binaural_freq / 2, duration, sample_rate, amplitude=0.2)

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

def generate_amen_break_drum_loop(tempo=160, pattern_length=16):
    # Load Amen Break samples (replace with your own file paths)
    kick = AudioSegment.from_file("kick.wav")
    snare = AudioSegment.from_file("snare.wav")
    hihat = AudioSegment.from_file("hihat.wav")

    samples = [kick, snare, hihat]

    # Generate a random drum pattern
    pattern = [random.choice(samples) for _ in range(pattern_length)]

    # Create an empty audio segment
    drum_loop = AudioSegment.silent(duration=0)

    # Calculate the duration of each drum hit based on the tempo
    beat_duration = 60000 / tempo  # in milliseconds

    # Sequence the drum pattern
    for hit in pattern:
        drum_loop += hit[:beat_duration] + AudioSegment.silent(duration=beat_duration - len(hit))

    return drum_loop