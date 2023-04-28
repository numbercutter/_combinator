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
from audio import apply_high_pass_filter, generate_soothing_sound_bath, generate_drum_pattern, generate_ambient_pattern, generate_bass_pattern, generate_cymbal_pattern, generate_chord, solfeggio_freqs, generate_simple_chord
from visual import generate_visuals, generate_image, generate_random_text, get_random_hashtags, generate_title, save_instagram_caption




def generate_audio(duration, num_segments=1):
    segment_duration = duration / num_segments

    
    # Generate the audio segments
    audio_segments = []
    for _ in range(num_segments):
        
        
        chord = generate_simple_chord(duration=segment_duration)
        audio_segments.append(chord)

    # Concatenate audio segments
    full_audio = audio_segments[0]
    for segment in audio_segments[1:]:
        full_audio = full_audio.append(segment)

    # Create a soothing sound bath for the hook
    hook_audio_file = generate_soothing_sound_bath(3)
    hook_audio = AudioSegment.from_wav(hook_audio_file)  # Read the file back as an AudioSegment

    full_audio = AudioSegment.from_wav(full_audio)
    filtered_audio = apply_high_pass_filter(full_audio)


    drum_loop = generate_drum_pattern(tempo=190, filename="drum_pattern.wav", bars=16)

    # Determine the length of the longest loop
    max_loop_length = len(drum_loop)

    # Repeat the drum loop until it reaches the desired duration
    while len(drum_loop) < duration * 1000:
        drum_loop += drum_loop

    # Trim the drum loop if necessary
    drum_loop = drum_loop[:duration * 1000]

    # Mix the filtered audio with the drum loop
    mixed_audio = drum_loop.overlay(filtered_audio)

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
    "#mysticalliving", "#soundhealingtherapy", "#binauralbeatstherapy", "#personalfinance", "#investing", "#moneytips", "#budgeting", "#wealthmanagement", 
    "#financialfreedom", "#retirementplanning", "#stocks", "#cryptocurrency", "#entrepreneurship", 
    "#taxes", "#creditcards", "#debtfree", "#financialplanning", "#savingmoney", "#investmentstrategies", 
    "#moneymanagement", "#financialadvice", "#stockmarket", "#economics", "#businessgrowth", 
    "#financialliteracy", "#financialindependence", "#investments",  "#fitfam", "#workoutmotivation", "#fitnessgoals", "#fitnessjourney", "#healthyhabits", 
    "#gymlife", "#fitspo", "#personaltrainer", "#fitnesschallenge", "#cardio", 
    "#weightlifting", "#yoga", "#running", "#HIIT", "#crossfit", "#fitnessinspiration", 
    "#bodybuilding", "#fitnessmodel", "#healthydiet", "#nutrition", "#fitnesstips", 
    "#fitnesstransformation", "#fitnessaddict", "#fitnesscommunity", "#fitnessequipment",
    "#funnyvideos", "#jokes", "#humor", "#comedycentral", "#laughoutloud", 
    "#standupcomedy", "#memes", "#comedyclub", "#satire", "#parody", 
    "#roast", "#improv", "#sketchcomedy", "#funnyclips", "#pranks", 
    "#humorous", "#comedian", "#humorist", "#funnybone", "#humorouscontent", 
    "#comedyshow", "#funnythoughts", "#funnymoments", "#comedygold"
    ]
    random_hashtags = get_random_hashtags(hashtags)
    print("Random Hashtags:", random_hashtags)
    
    random_hashtags += ["#getoffinstagram", "#focusonyourgoals"] # add the two new hashtags

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


    generate_audio(duration, num_segments)


    generate_video(
        duration,
        img_size,
        fps,
        num_generations=num_generations,
        crossfade_duration=crossfade_duration,
        text_duration=text_duration,
    )

