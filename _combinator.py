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
from audio import apply_high_pass_filter, generate_soothing_sound_bath, generate_drum_pattern, generate_bass_pattern, generate_chord, solfeggio_freqs, generate_simple_chord, sine_wave_synthesizer, generate_drum_pattern_high_res
from visual import generate_visuals, generate_image, generate_random_text, get_random_hashtags, generate_title, save_instagram_caption


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
        bass_line = generate_bass_pattern(chord, tempo=190, duration=segment_duration, bars=4)
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
            if start_frame + j < len(all_frames) and end_frame - crossfade_frames + j < len(all_frames):
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
                  .set_duration(3))



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
    video = CompositeVideoClip(clips + [ImageClip(np.array(img)).set_duration(1 / fps).set_start(3 + i / fps) for i, img in enumerate(all_frames)])

    # Add audio and save the video
    audio = AudioFileClip(audio_path)
    video = video.set_audio(audio)
    video.write_videofile(video_path, fps=fps)
    
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
    "#mysticalliving", "#soundhealingtherapy", "#binauralbeatstherapy","#sobriety", "#sobrietyrocks", "#sobrietymatters", "#soberliving", "#sobersquad",
    "#soberaf", "#soberissexy", "#sobercommunity", "#sobermovement", "#sobrietyispossible",
    "#sobrietyjourney", "#soberlife", "#sobrietyquotes", "#sobercurious", "#soberafnation",
    "#goals", "#goalsetting", "#goalgetter", "#goalchaser", "#goaloriented",
    "#achievegoals", "#mindset", "#successmindset", "#motivation", "#inspiration",
    "#mindpower", "#determination", "#focus", "#hardworkpaysoff", "#nevergiveup",
    "#positivethinking", "#selfimprovement", "#personaldevelopment", "#lifegoals", "#dreambig",
    "#goalplanning", "#goaldigger", "#ambition", "#success", "#goalcrusher",
    "#goalplanner", "#goalinspiration", "#goaldriven", "#motivationmonday", "#successquotes",
    "#mindsetiseverything", "#mindsetquotes", "#motivationalquotes", "#inspirationalquotes", "#mindsetcoach",
    "#mindsetmatters", "#motivationalspeaker", "#successmindsettraining", "#inspirationalthoughts", "#motivationalthoughts",
    "#mindsetshift", "#mindsetofgreatness", "#goalsettingtips", "#achieveyourgoals", "#personalgrowth"
    ]



    random_hashtags = get_random_hashtags(hashtags)
    print("Random Hashtags:", random_hashtags)

    random_title = generate_title(hashtags)
    print("Random Title:", random_title)
    
    
    
    save_instagram_caption(random_title, random_hashtags)
    generate_image(1080, 1080)
    duration = 20  # seconds
    img_size = 1000  # Instagram square dimensions (1080x1080)
    fps = 30  # frames per second

    num_generations = random.randint(20, 50)
    crossfade_duration = random.uniform(0.5, 2)
    text_interval = random.randint(2, 6)
    text_duration = random.uniform(0.2, 1)
    num_segments = 1


    generate_full_audio(duration, num_segments)


    generate_video(
        duration,
        img_size,
        fps,
        num_generations=num_generations,
        crossfade_duration=crossfade_duration,
        text_duration=text_duration,
    )

