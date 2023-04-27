import random
from pydub import AudioSegment  # Ensure this import is here
from audio import generate_soothing_sound_bath, generate_drum_pattern, generate_ambient_pattern, generate_bass_pattern, generate_cymbal_pattern





def generate_audio(duration, num_segments=1):

    # Create a soothing sound bath for the hook
    hook_audio_file = generate_soothing_sound_bath(3)
    hook_audio = AudioSegment.from_wav(hook_audio_file)  # Read the file back as an AudioSegment


    drum_loop = generate_drum_pattern(tempo=190, filename="drum_pattern.wav", bars=8)
    bass_loop = generate_bass_pattern(tempo=190, filename="bass_pattern.wav", bars=8)
    ambient_loop = generate_ambient_pattern(tempo=190, filename="ambient_pattern.wav", bars=8)
    cymbal_loop = generate_cymbal_pattern(tempo=190, filename="cymbal_pattern.wav", bars=8)

    # Determine the length of the longest loop
    max_loop_length = max(len(drum_loop), len(bass_loop), len(ambient_loop), len(cymbal_loop))

    # Repeat each loop until it reaches the length of the longest loop
    while len(drum_loop) < max_loop_length:
        drum_loop += drum_loop
    drum_loop = drum_loop[:max_loop_length]

    while len(bass_loop) < max_loop_length:
        bass_loop += bass_loop
    bass_loop = bass_loop[:max_loop_length]

    while len(ambient_loop) < max_loop_length:
        ambient_loop = ambient_loop[:max_loop_length]

    while len(cymbal_loop) < max_loop_length:
        cymbal_loop += cymbal_loop
    cymbal_loop = cymbal_loop[:max_loop_length]

    # Mix the loops together
    mixed_audio = drum_loop.overlay(bass_loop).overlay(ambient_loop).overlay(cymbal_loop)

    # Repeat the mixed audio until it reaches the desired duration
    while len(mixed_audio) < duration * 1000:
        mixed_audio += mixed_audio

    # Trim the mixed audio if necessary
    mixed_audio = mixed_audio[:duration * 1000]

    # Concatenate the hook audio with the mixed audio
    mixed_audio = hook_audio + mixed_audio

    # Export the mixed audio to a WAV file
    mixed_audio.export("audio.wav", format="wav")

    
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