import os
import random
import numpy as np
import librosa
import soundfile as sf
from pydub.effects import high_pass_filter  # Added import for high_pass_filter
from pydub import AudioSegment, effects
from pydub.generators import WhiteNoise
from pydub.generators import Sine
from scipy.signal import butter, lfilter
from scipy import signal


solfeggio_freqs = {
    "UT": 396 / 4,
    "RE": 417 / 4,
    "MI": 528 / 4,
    "FA": 639 / 4,
    "SOL": 741 / 4,
    "LA": 852 / 4,
}

def get_note_duration(tempo, steps_per_beat):
    return (60000 / tempo) / steps_per_beat

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def low_pass_filter(data, cutoff_freq, sample_rate, order=4):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data)
    return filtered_data

def compress_audio(audio, threshold=-20, ratio=4, attack=5, release=50):
    # Convert AudioSegment to NumPy array
    audio_samples = np.array(audio.get_array_of_samples())
    audio_samples = audio_samples.astype(np.float32)

    # Compress the audio samples
    gain_reduction = np.zeros_like(audio_samples)
    gain = 0
    for i in range(len(audio_samples)):
        amplitude_db = 20 * np.log10(np.abs(audio_samples[i]) / 2**15)
        if amplitude_db > threshold:
            gain_reduction[i] = (amplitude_db - threshold) / ratio
            target_gain = gain - gain_reduction[i]
            attack_coeff = 1 - np.exp(-1 / (attack * audio.frame_rate / 1000))
            release_coeff = 1 - np.exp(-1 / (release * audio.frame_rate / 1000))
            if target_gain < gain:
                gain = gain * (1 - attack_coeff) + target_gain * attack_coeff
            else:
                gain = gain * (1 - release_coeff) + target_gain * release_coeff
            audio_samples[i] *= 10 ** (gain / 20)

    # Convert the NumPy array back to an AudioSegment
    audio_samples_int16 = (audio_samples * (2**15 - 1)).astype(np.int16)
    compressed_audio = AudioSegment(audio_samples_int16.tobytes(), frame_rate=audio.frame_rate, sample_width=2, channels=audio.channels)

    return compressed_audio


def equalize_audio(audio, low_freq_boost=6, mid_freq_cut=6, high_freq_boost=6):
    audio = audio.low_pass_filter(400).apply_gain(low_freq_boost)
    audio = audio.high_pass_filter(400).low_pass_filter(2500).apply_gain(-mid_freq_cut)
    audio = audio.high_pass_filter(2500).apply_gain(high_freq_boost)
    return audio

def apply_lfo_filter(audio, lfo_freq=0.5, lfo_amp=0.05):
    sample_rate = audio.frame_rate
    duration = audio.duration_seconds
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    lfo = 1 + lfo_amp * np.sin(2 * np.pi * lfo_freq * t)
    audio_samples = audio.get_array_of_samples()
    audio_lfo_filtered = np.multiply(audio_samples, lfo)
    audio_lfo_filtered = audio_lfo_filtered.astype(np.int16)
    audio_filtered = AudioSegment(audio_lfo_filtered.tobytes(), frame_rate=sample_rate, sample_width=2, channels=1)
    return audio_filtered



def generate_simple_chord(chord_progression, duration, output_file='simple_chord.wav', fm_intensity=0.01, fm_speed=0.1):
    
    selected_chord_progression = chord_progression
    sample_rate = 44100
    duration_ms = duration * 1000  # Convert duration to milliseconds

    # Generate the time vector
    t = np.linspace(0, duration, int(duration * sample_rate), False)
    
    # Generate sine wave for each note in the chord
    chord = []

    for note_set in selected_chord_progression:
        harmonics_set = []
        for note_freq in note_set:
            # Generate sine waves for harmonics
            harmonics = np.zeros_like(t)
            max_harmonic = int(1000 / note_freq)
            for harmonic in range(1, max_harmonic + 1):
                sine_wave = (
                    0.5 * np.sin(2 * np.pi * note_freq * harmonic * t) / harmonic
                ).astype(np.float32)
                harmonics += sine_wave

            # Apply frequency modulation
            fm = np.sin(2 * np.pi * fm_speed * t) * fm_intensity * note_freq
            modulated_freq = note_freq + fm
            modulated_sine_wave = 0.5 * np.sin(2 * np.pi * modulated_freq * t).astype(
                np.float32
            )
            harmonics = 0.5 * harmonics + 0.5 * modulated_sine_wave

            harmonics_set.append(harmonics)
            
            # Print note frequency
            print(f"Note frequency: {note_freq} Hz")

        # Combine harmonics to create a chord
        chord_data = np.zeros_like(harmonics_set[0])
        for note_data in harmonics_set:
            chord_data += note_data

        chord.append(chord_data)

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

    # Apply a smoother fade-in and fade-out effect
    fade_duration = int(duration_ms * 0.1)  # Fade duration is 10% of the total duration
    chord_audio_faded = modulated_chord_audio.fade_in(fade_duration).fade_out(fade_duration)


    # Save the audio to a file
    chord_audio_faded.export(output_file, format="wav")

    return output_file




def apply_high_pass_filter(audio_segment, cutoff_freq=150):
    filtered_audio = high_pass_filter(audio_segment, cutoff_freq)
    return filtered_audio

def generate_sine_wave(freq, duration, sample_rate, amplitude=1, phase=0):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    sine_wave = amplitude * np.sin(2 * np.pi * freq * t + phase)
    return sine_wave

def apply_fade_bath(audio, fade_in_samples, fade_out_samples):
    audio[:fade_in_samples] *= np.linspace(0, 1, fade_in_samples)
    audio[-fade_out_samples:] *= np.linspace(1, 0, fade_out_samples)
    return audio

def generate_chime_sound(freq, duration, sample_rate, num_sinusoids=10):
    chime_wave = np.zeros(int(sample_rate * duration))
    for _ in range(num_sinusoids):
        amplitude = random.uniform(0.01, 0.1)
        phase = random.uniform(0, 2 * np.pi)
        chime_wave += generate_sine_wave(freq, duration, sample_rate, amplitude, phase)
    return chime_wave

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


def apply_fade_bath(audio, fade_in_time, fade_out_time):
    fade_in = np.linspace(0, 1, fade_in_time)
    fade_out = np.linspace(1, 0, fade_out_time)

    audio[:fade_in_time] *= fade_in
    audio[-fade_out_time:] *= fade_out

    return audio

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
    fade_time = int(0.25 * sample_rate)  # 0.25 seconds

    # Generate chime sounds with the given frequencies
    chime_waves = []
    for freq in freqs:
        chime_wave = generate_chime_sound(freq, duration, sample_rate)
        chime_waves.append(chime_wave)

    # Create the combined deep ambient sound
    lush_sound = np.zeros_like(chime_waves[0])
    for chime_wave in chime_waves:
        lush_sound += chime_wave

    # Apply fade in and fade out to create smooth transitions
    lush_sound = apply_fade_bath(lush_sound, fade_time, fade_time)

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
    left_ear = generate_sine_wave(freqs[0] - binaural_freq / 2, duration, sample_rate, amplitude=0.025)
    right_ear = generate_sine_wave(freqs[0] + binaural_freq / 2, duration, sample_rate, amplitude=0.025)
    
    # Combine lush sound with binaural beat
    lush_sound_left = lush_sound + left_ear
    lush_sound_right = lush_sound + right_ear

    # Create stereo audio
    stereo_audio = np.vstack((lush_sound_left, lush_sound_right)).T

    # Apply a limiter to prevent clipping
    def limiter(audio, threshold=0.9):
        audio_clipped = np.where(audio > threshold, threshold, audio)
        audio_clipped = np.where(audio_clipped < -threshold, -threshold, audio_clipped)
        return audio_clipped

    stereo_audio = limiter(stereo_audio)

    # Normalize the audio
    stereo_audio = librosa.util.normalize(stereo_audio, norm=np.inf, axis=None)

    # Save the audio to a file
    sf.write(output_file, stereo_audio, sample_rate, format='wav', subtype='PCM_24')

    return output_file


def generate_random_pattern(samples, length, probability=0.1):
    pattern = ["-"] * length
    for i in range(length):
        if random.random() < probability:
            pattern[i] = random.choice(list(samples.keys()))
    return pattern

def apply_random_glitch(samples, pattern, probability=0.1):
    glitch_pattern = generate_random_pattern(samples, len(pattern), probability)
    glitched_pattern = pattern.copy()
    for i in range(len(pattern)):
        if glitch_pattern[i] != "-":
            glitched_pattern[i] = glitch_pattern[i]
    return glitched_pattern

def generate_drum_pattern(tempo=190, filename="drum_pattern.wav", bars=8):
    kick = AudioSegment.from_file("samples/kick.wav")
    snare = AudioSegment.from_file("samples/snare.wav")
    hihat = AudioSegment.from_file("samples/hihat.wav")
    drum_pattern = AudioSegment.silent(duration=0)

    patterns = {
        "kick": ["x", "-", "-", "-", "-", "-", "-", "-", "-", "-", "x", "-", "-", "-", "-", "-"],
        "snare": ["-", "-", "-", "-", "x", "-", "-", "-", "-", "-", "-", "-", "x", "-", "-", "-"],
        "hihat": ["x", "-", "x", "-", "x", "-", "x", "-", "x", "-", "x", "-", "x", "-", "x", "-"]
    }

    steps_per_beat = 4
    beat_duration = get_note_duration(tempo, steps_per_beat)

    priority_order = ['kick', 'snare', 'hihat']
    samples = {'kick': kick, 'snare': snare, 'hihat': hihat}

    for bar in range(bars):
        bar_segment = AudioSegment.silent(duration=0)
        for i in range(len(patterns["kick"])):
            beat_segment = AudioSegment.silent(duration=int(beat_duration))

            glitched_patterns = {key: apply_random_glitch(samples, patterns[key], 0.2) for key in patterns}

            for beat_step in range(steps_per_beat):
                sample_added = False
                for priority_sample in priority_order:
                    if glitched_patterns[priority_sample][i] == "x" and not sample_added:
                        sample = samples[priority_sample]
                        beat_segment = beat_segment.overlay(sample - 6)
                        sample_added = True

            bar_segment += beat_segment

        drum_pattern += bar_segment

    drum_pattern.export(filename, format="wav")
    return drum_pattern

def generate_drum_pattern_high_res(tempo=190, filename="drum_pattern.wav", bars=8):
    kick = AudioSegment.from_file("samples/kick.wav")
    snare = AudioSegment.from_file("samples/snare.wav")
    hihat = AudioSegment.from_file("samples/hihat.wav")
    drum_pattern = AudioSegment.silent(duration=0)

    def generate_random_pattern(length=64, probability=0.1):
        return ["x" if random.random() < probability else "-" for _ in range(length)]

    patterns = {
        "kick": generate_random_pattern(64, 0.02),
        "snare": generate_random_pattern(64, 0.02),
        "hihat": generate_random_pattern(64, 0.04)
    }

    steps_per_beat = 16
    beat_duration = get_note_duration(tempo, steps_per_beat)

    priority_order = ['kick', 'snare', 'hihat']
    samples = {'kick': kick, 'snare': snare, 'hihat': hihat}

    for bar in range(bars):
        bar_segment = AudioSegment.silent(duration=0)
        for i in range(64):
            beat_segment = AudioSegment.silent(duration=int(beat_duration))

            for priority_sample in priority_order:
                if patterns[priority_sample][i] == "x":
                    sample = samples[priority_sample]
                    beat_segment = beat_segment.overlay(sample - 6)

            bar_segment += beat_segment

        drum_pattern += bar_segment

    drum_pattern.export(filename, format="wav")
    return drum_pattern

def generate_bass_pattern(chord, tempo, duration, bars):
    bass_notes = [note_freq / 2 for note_freq in chord]

    steps_per_beat = 1  # Change this value to match the drum pattern
    beat_duration_ms = 60000 / tempo
    note_duration = int(beat_duration_ms / steps_per_beat)

    bass_line = AudioSegment.silent(duration=0)

    # Modify the bass pattern to have the same length as the drum pattern (64 steps)
    pattern = [0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1]

    for bar in range(bars):
        for step in range(steps_per_beat * 4):
            bass_note_index = pattern[step % len(pattern)]

            if bass_note_index != -1:
                bass_note = bass_notes[bass_note_index]
                sine_wave_data = sine_wave_synthesizer(bass_note, beat_duration_ms / 1000, 0.5)
                sine_wave_int16 = (sine_wave_data * (2**15 - 1)).astype(np.int16)
                bass_audio_segment = AudioSegment(
                    sine_wave_int16.tobytes(), frame_rate=44100, sample_width=2, channels=1
                )
                bass_line += bass_audio_segment[:note_duration]
            else:
                bass_line += AudioSegment.silent(duration=note_duration)

    return bass_line[:duration * 1000]





def sine_wave_synthesizer(freq, duration, amplitude):
    sample_rate = 44100
    t = np.linspace(0, duration, int(duration * sample_rate), False)

    # Use a Hann window for a more balanced attack and release
    hann_window = np.hanning(len(t))

    sine_wave = (amplitude * np.sin(2 * np.pi * freq * t) * hann_window).astype(np.float32)
    return sine_wave
