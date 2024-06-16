import sys
import os
import cv2
from openai import OpenAI
from google.cloud import texttospeech
from moviepy.editor import *
import numpy as np
import re
import json
import random
import requests

client = OpenAI(api_key=OPEN_API_KEY)
OUTPUT_DIR = 'public/output'

VOICE_LIST = {
    "en-US-Neural2-I": {"gender": "Male", "pitch": "Middle-High", "voice_description": "Young male", "age_range": "20s", "speed": "Fast"},
    "en-US-Neural2-J": {"gender": "Male", "pitch": "Middle-Low", "voice_description": "Slightly husky", "age_range": "40s", "speed": "Medium"},
    "en-US-Neural2-D": {"gender": "Male", "pitch": "Middle", "voice_description": "Calm", "age_range": "20s-30s", "speed": "Slow"},
    "en-US-Neural2-A": {"gender": "Male", "pitch": "High", "voice_description": "Bright", "age_range": "10s-20s", "speed": "Slow"},
    "en-US-Neural2-C": {"gender": "Female", "pitch": "Low", "voice_description": "Benign", "age_range": "40s", "speed": "Medium"},
    "en-US-Neural2-E": {"gender": "Female", "pitch": "Low-High", "voice_description": "Mature", "age_range": "30s", "speed": "Medium"},
    "en-US-Neural2-G": {"gender": "Female", "pitch": "Middle-High", "voice_description": "Ordinary", "age_range": "20s", "speed": "Fast"},
    "en-US-Neural2-H": {"gender": "Female", "pitch": "High-Low", "voice_description": "Happy", "age_range": "10s-20s", "speed": "Fast"},
    "en-US-Neural2-F": {"gender": "Female", "pitch": "High", "voice_description": "Lively", "age_range": "10s", "speed": "Medium"}
}

def convert_to_dict(response):
    cleaned_response = response.strip()
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response[7:].strip()
    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[:-3].strip()
    
    try:
        response_dict = json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        print("Failed to decode JSON. Error:", e)
        response_dict = {}
    
    return response_dict

def remove_punctuation_inside_quotes(text):
    result = []
    toggle = False
    
    for char in text:
        if char == '"':
            toggle = not toggle
            result.append('"')
        elif toggle:
            if char == '.':
                result.append('<DOT>')
            elif char == '!':
                result.append('<EXCL>')
            elif char == '?':
                result.append('<QUES>')
            else:
                result.append(char)
        else:
            result.append(char)
    
    return ''.join(result)

def restore_punctuation_inside_quotes(text):
    return text.replace('<DOT>', '.').replace('<EXCL>', '!').replace('<QUES>', '?')

def split_sentences(text):
    modified_text = remove_punctuation_inside_quotes(text)
    modified_text = modified_text.replace(' "', '\n"')
    sentences = re.split(r'(?<=[.!?:"]) +|\n', modified_text)
    
    final_sentences = []

    for sentence in sentences:
        restored_sentence = restore_punctuation_inside_quotes(sentence.strip())
        if restored_sentence:
            final_sentences.append(restored_sentence)

    return final_sentences

def extract_character(story):
    prompt = f"""
        Given the following story, extract all characters and provide their gender, age, and personality traits. Output the result in JSON format without any additional text or explanation. If any information is not explicitly mentioned, infer based on the context of the story.

        Story:
        {story}

        Output the result in this JSON format:
        {{
            "John": {{"gender": "male", "age": "30", "personality": ["brave", "kind"]}},
            "Mary": {{"gender": "female", "age": "25", "personality": ["intelligent", "curious"]}}
        }}
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    
    characters = response.choices[0].message.content.strip()

    with open(os.path.join(OUTPUT_DIR, 'llm/characters_traits.txt'), mode='w', encoding='utf8') as f:
        f.write(characters)

    return convert_to_dict(characters)

def match_voices(characters, VOICE_LIST):
    prompt = f"""
        Given the following characters and their attributes, match each character to the most appropriate voice from the VOICE_LIST. Consider the gender, age, and personality traits of each character to find the best match. Output the result in JSON format without any additional text or explanation.

        Characters:
        {characters}

        VOICE_LIST:
        {VOICE_LIST}

        Output the result in this JSON format:
        {{
            "John": "en-US-Neural2-I",
            "Mary": "en-US-Neural2-E"
        }}
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    
    characters_TTS = response.choices[0].message.content.strip()

    with open(os.path.join(OUTPUT_DIR, 'llm/match_characters_TTS.txt'), mode='w', encoding='utf8') as f:
        f.write(characters_TTS)
    
    return convert_to_dict(characters_TTS)

def match_sentences(sentence_list, character_list):
    sentence_list_str = "\n".join([f"{i}. {sentence}" for i, sentence in enumerate(sentence_list)])
    sentence_idxs = [idx for idx, sen in enumerate(sentence_list) if sen[0]=='"']
    character_list_str = ", ".join(character_list)
    
    with open(os.path.join(OUTPUT_DIR, 'llm/splitted_sentences.txt'), mode='w', encoding='utf8') as f:
        f.write(sentence_list_str)

    prompt = f"""
        Given the following list of sentences with their indices and a list of characters, identify the index of sentences that correspond to each character's dialogue or monologue. Return the result in a JSON format where each character's name is a key, and the value is a list of indices of sentences they spoke. Output the result in JSON format without any additional text or explanation.
        Additionally, ensure that the given sentence numbers are included in the output.

        Sentence List:
        {sentence_list_str}

        Only consider these sentence numbers: {sentence_idxs}

        Character List:
        {character_list_str}

        Output the result in this JSON format:
        {{
            "John": [0, 1, 3],
            "Mary": [2, 7]
        }}
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )

    dialogue_indices = response.choices[0].message.content.strip()

    with open(os.path.join(OUTPUT_DIR, 'llm/match_character_sentences.txt'), mode='w', encoding='utf8') as f:
        f.write(dialogue_indices)
   
    dialogues = convert_to_dict(dialogue_indices)
    sentence_idxs = [item for sublist in dialogues.values() for item in sublist]

    return dialogues, sentence_idxs

def match_emotions(sentence_list, sentence_idxs):
    sentence_list_str = "\n".join([f"{i}. {sentence}" for i, sentence in enumerate(sentence_list)])
    emotions = "Happiness, Sadness, Anger, Surprise, Fear, Disgust, Neutral"

    prompt = f"""
        Given the following list of sentences with their indices and a list of emotions, identify the sentence indices that correspond to each emotion. For example, if a character is begging, that sentence should correspond to fear. Return the result in a JSON format where each emotion is a key, and the value is a list of indices of sentences. Output the result in JSON format without any additional text or explanation.
        Additionally, ensure that the given sentence numbers are included in the output.
        
        Sentence List:
        {sentence_list_str}

        Only consider these sentence numbers: {sentence_idxs}

        Emotion List:
        {emotions}

        Output the result in this JSON format:
        {{
            "Happiness": [0, 1, 3],
            "Sadness": [2, 7],
            "Anger": [10, 21],
            "Surprise": [8, 9],
            "Fear": [],
            "Disgust": [],
            "Neutral": [5],
        }}
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )

    emotion_indices = response.choices[0].message.content.strip()

    with open(os.path.join(OUTPUT_DIR, 'llm/match_emotion_sentences.txt'), mode='w', encoding='utf8') as f:
        f.write(emotion_indices)

    return convert_to_dict(emotion_indices)

def merge_dict(voice, sentence):
    merged_dict = {}

    for key in sentence:
        merged_dict[key] = {"voice": voice[key], "sentences": sentence[key]}
    
    with open(os.path.join(OUTPUT_DIR, 'llm/merged_dict.txt'), mode='w', encoding='utf8') as f:
        f.write(json.dumps(merged_dict, indent=4))

    return merged_dict

def prosody_emotion(emotion):
    if emotion == 'Fear':
        prosodyRate = "fast"
        pitch = 3
    elif emotion == 'Happiness':
        prosodyRate = "fast"
        pitch = 1
    elif emotion == 'Sadness':
        prosodyRate = "slow"
        pitch = -2
    elif emotion == 'Anger':
        prosodyRate = "fast"
        pitch = 2
    elif emotion == 'Surprise':
        prosodyRate = "fast"
        pitch = 4
    elif emotion == 'Disgust':
        prosodyRate = "slow"
        pitch = -1
    else:
        prosodyRate = "medium"
        pitch = 0
    return prosodyRate, pitch

def create_ssml(sentences, character_info, sentence_emotion, narration, rate, process = 0):
    # sentences : ["...", "...", ...]
    # character_info : {'A': {'voice': '...', 'sentences': [...]}, 'B': {'voice': '...', 'sentences': [...]}} 
    # sentence_emotion : {'Happiness': [], 'Sadness': [], 'Anger': [], 'Surprise': [], 'Fear': [], 'Disgust': [], 'Neutral': []}
    ssml_output = '<speak>\n'

    sentence_to_voice = {}
    sentence_to_emotion = {}

    for emotion, indices in sentence_emotion.items():
        for idx in indices:
            if idx - process < len(sentences) and idx - process >= 0:
                sentence_to_emotion[idx - process] = emotion

    for _, info in character_info.items():
        for idx in info['sentences']:
            if idx - process < len(sentences) and idx - process >= 0:
                sentence_to_voice[idx - process] = info['voice']
    
    for idx, sentence in enumerate(sentences):
        if idx in sentence_to_voice:
            voice_name = sentence_to_voice[idx]
            emotion = sentence_to_emotion.get(idx, 'neutral')

            if isinstance(voice_name, list): 
                voice_name = random.choice(voice_name)  
            r, p = prosody_emotion(emotion)
            ssml_output += f'\t<prosody rate="{r}" pitch="{p}st"><voice name="{voice_name}">{sentence}</voice></prosody><break time="500ms"/>\n'
        else:
            ssml_output += f'\t<prosody rate="{rate}"><voice name="{narration}">{sentence}</voice></prosody><break time="500ms"/>\n'

    ssml_output += '</speak>'
    
    return ssml_output

def scenePartition(content):
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
            {"role": "system", "content": "You are a cartoon storyboard builder."},
            {"role": "user", "content": content + """
            I want to make a cartoon with this story. Divide this story into several cuts and provide with the matched original text. Don't slice one sentence into many scenes. Ensure that every sentence from the original text is included in the scenes.
            Answer in the format like this: "# Scene" + {scene_number} + ": " + {text}.
            """},
            {"role": "assistant", "content": "An example of your answer is '#Scene1: Some part of original text\n#Scene2: Another part of original text'"}
        ]
    )

    scene_text_full = response.choices[0].message.content
   
    with open(os.path.join(OUTPUT_DIR, 'image/original_scene_full.txt'), mode='w', encoding='utf8') as f:
        f.write(scene_text_full)

    chunks = scene_text_full.split("#")
    chunks.pop(0)
    scene_num = 0
    for ch in chunks:
        scene_num += 1
        file_name = os.path.join(OUTPUT_DIR, f'image/original_scene_{scene_num}.txt')
        with open(file_name, mode='w', encoding='utf8') as f_scene:
            scene_text = ch.split(":")
            f_scene.write(scene_text[1])
    return scene_num

def makeTTS(input_ssml, scene):
    voice = "en-US-Neural2-A"
    gender = texttospeech.SsmlVoiceGender.MALE
    client = texttospeech.TextToSpeechClient.from_service_account_file(TTS_API_KEY)

    synthesis_input = texttospeech.SynthesisInput(ssml=input_ssml)

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name=voice,
        ssml_gender=gender,
    )

    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    with open(os.path.join(OUTPUT_DIR, f"audio/tts_{scene}.mp3"), "wb") as out:
        out.write(response.audio_content)
        print(f'Audio content written to file "tts_{scene}.mp3"')

def add_letterbox(image, target_width, target_height):
    image_height, image_width = image.shape[:2]
    aspect_ratio_image = image_width / image_height
    aspect_ratio_target = target_width / target_height

    if aspect_ratio_image > aspect_ratio_target:
        new_width = target_width
        new_height = int(target_width / aspect_ratio_image)
    else:
        new_height = target_height
        new_width = int(target_height * aspect_ratio_image)

    resized_image = cv2.resize(image, (new_width, new_height))

    black_background = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2

    black_background[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

    return black_background

def makeMovie(image_files, audio_files, output_file):
    clips = []
    for image_file, audio_file in zip(image_files, audio_files):
        img_clip = ImageClip(image_file).set_duration(AudioFileClip(audio_file).duration)
        audio_clip = AudioFileClip(audio_file)
        video_clip = img_clip.set_audio(audio_clip)
        clips.append(video_clip)
    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip.write_videofile(os.path.join(OUTPUT_DIR, output_file), fps=10, codec="libx264", bitrate="5000k")

def createBackground(scene_text_full):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a cartoon storyboard builder."},
            {"role": "assistant", "content": "Each scenes are divided as follows.\n" + scene_text_full},
            {"role": "user", "content": """Now make location and time, and atmosphere for each cut.
            Description sentence format for location is "In [adjective] [location] with [scenery],".
            Plus, find an appropriate real location and specify it in each scene's description.
            Description sentence format for time is "the [time] sun makes [atmosphere] atmosphere.".
            Also, if unnecessary, don't change location and time over scenes.
            Answer in the format like below.
            "# Scene" + {scene_number}
            "@ Location: " + {text}
            "@ Time: " + {text}
            """},
            {"role": "assistant", "content": """An example of your answer is
            "# Scene 1
            @ Location: In a serene hillside in the Lake District with rolling green meadows,
            @ Time: the morning sun makes a peaceful atmosphere."
            """}
        ]
    )
    background_full = response.choices[0].message.content
    with open(os.path.join(OUTPUT_DIR, 'image/background_full.txt'), mode='w', encoding='utf8') as f:
        f.write(background_full)

    chunks = background_full.split("#")
    chunks.pop(0)
    scene_num = 0
    for ch in chunks:
        scene_num += 1
        file_name = os.path.join(OUTPUT_DIR, f'image/prompt_scene_{scene_num}.txt')
        with open(file_name, mode='w', encoding='utf8') as f_scene:
            parsed = ch.split("@")
            location = parsed[1].lstrip("@ Location: ").strip('\n')
            time = parsed[2].lstrip("@ Time: ").strip('\n')
            f_scene.write('"' + location + time)
    return scene_num

def analyzeCharacter(content, data):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a cartoon storyboard builder."},
            {"role": "assistant", "content": "The full story is as follows:\n" + content},
            {"role": "assistant", "content": "Each characters' gender and age are already given: " + data},
            {"role": "user", "content": """Right, then, let's analyze the characters. Give descriptions only for main characters who are depicted at least 2 times.
            If a character is a human, fill this format "[Character] is [age] with [hair], [skin], [outfit with color descriptions]."
            Else, fill this format "[Character] is [size], [body color], and [additional features]."
            One sentence per character.
            """},
            {"role": "assistant", "content": """An example of your answer is
            "1. Little Red Riding Hood is 10 years old girl with long and wavy blond hair, fair skin, wearing red hood and cloak, white blouse, blue skirt.
            2. Wolf is large and imposing, gray-furred in color, and has sharp teeth, a long snout, and pointed ears."
            """}
        ]
    )
    character_full = response.choices[0].message.content
    with open(os.path.join(OUTPUT_DIR, 'image/character_full.txt'), mode='w', encoding='utf8') as f:
        f.write(character_full)
    return 0

def createAction(scene_text_full):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a cartoon storyboard builder."},
            {"role": "assistant", "content": "Each scenes are divided as follows.\n" + scene_text_full},
            {"role": "user", "content": """Okay, then let's make scene descriptions for each scene. It is for drawing good illustrations for each scene. And give the list of characters who appear in that scene.
            Answer in the format like below.
            "# Scene" + {scene_number} + ": [Character] [Posture]." + "//[Characters in that scene]"
            """},
            {"role": "assistant", "content": """An example of your answer is
            "# Scene 1: Little Red Riding Hood walking through the forest. The wolf appears and talks to Little Red Riding Hood. // Little Red Riding Hood, Wolf
            # Scene 2: The wolf running through the forest and reaching Granny's house. // Wolf"
            """}
        ]
    )
    action_full = response.choices[0].message.content
    with open(os.path.join(OUTPUT_DIR, 'image/action_full.txt'), mode='w', encoding='utf8') as f:
        f.write(action_full)

    chunks = action_full.split("#")
    chunks.pop(0)
    scene_num = 0
    character_list = ""
    for ch in chunks:
        scene_num += 1
        file_name = os.path.join(OUTPUT_DIR, f'image/prompt_scene_{scene_num}.txt')
        with open(file_name, mode='a', encoding='utf8') as f_scene:
            scene_text = ch.split(":")[1].split("//")
            action = scene_text[0].strip()
            f_scene.write(action)

        character = scene_text[1].strip().lstrip("//")
        character_list += f'# Scene{scene_num}: {character}.\n'

    with open(os.path.join(OUTPUT_DIR, 'image/character_list.txt'), mode='w', encoding='utf8') as f:
        f.write(character_list)
    return 0

def createCharDescription(scene_text_full, character_list, data):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a cartoon storyboard builder."},
            {"role": "assistant", "content": "Each scenes are divided as follows.\n" + scene_text_full},
            {"role": "assistant", "content": "This is list of characters who appear in each scene.\n" + character_list},
            {"role": "assistant", "content": "Each characters' gender and age are already given: " + data},
            {"role": "user", "content": """ Let's write descriptions for characters appearing in each scene.
            If a character is a human, fill this format "[Character] is [geneder] and [age] with [hair with color description], [skin], [body figure], [outfit with color descriptions]."
            Else, fill this format "[Character] is [size], [body color], and [additional features]."
            Also, if unnecessary, don't change the sentences for character description over scenes.
            Answer in this format:
            "# Scene " + {scene_number} + ": " + {text}
            """},
            {"role": "assistant", "content": """ Here is an example answer:
            "# Scene 1: Hare is medium-sized, with sleek brown fur, long ears, and a mischievous expression. Tortoise is small-sized, with a green, bumpy shell, and a determined look."
            "# Scene 2: Hare is medium-sized, with sleek brown fur, long ears, and a mischievous expression."
            "# Scene 3: Hare is medium-sized, with wet sleek brown fur, long ears, and a tedious expression."
            """}
        ]
    )
    char_description_full = response.choices[0].message.content

    with open(os.path.join(OUTPUT_DIR, 'image/char_description_full.txt'), mode='w', encoding='utf8') as f:
        f.write(char_description_full)

    chunks = char_description_full.split("#")
    chunks.pop(0)
    scene_num = 0
    for ch in chunks:
        scene_num += 1
        file_name = os.path.join(OUTPUT_DIR, f'image/prompt_scene_{scene_num}.txt')
        with open(file_name, mode='a', encoding='utf8') as f_scene:
            scene_text = ch.split(":")[1]
            f_scene.write(scene_text.rstrip('\n') + '"')
    return 0

def createImage(style, scene_number):
    for scene_num in range(1, scene_number + 1):
        file_name = os.path.join(OUTPUT_DIR, f'image/prompt_scene_{scene_num}.txt')
        with open(file_name, mode='r', encoding='utf8') as f_scene:
            scene_text = f_scene.read()
        scene_text += ', ' + style
        response = client.images.generate(
            model="dall-e-3",
            prompt=scene_text,
            size="1792x1024",
            quality="standard",
            n=1,
        )

        img_url = response.data[0].url
        img_data = requests.get(img_url).content

        with open(os.path.join(OUTPUT_DIR, f'image/img_scene{scene_num}.png'), 'wb') as handler:
            handler.write(img_data)
            print(f'Image content written to file "img_scene{scene_num}.png"')
    return 0

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python your_python_script.py <story> <gender> <speed> <imageStyle>")
        sys.exit(1)

    STORY = sys.argv[1]
    gender = sys.argv[2]
    SPEED = sys.argv[3]
    IMAGESTYLE = sys.argv[4]

    if gender == "male":
        NARRATION = "en-US-Studio-Q"
    else:
        NARRATION = "en-US-Studio-O"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR + '/llm', exist_ok=True)
    os.makedirs(OUTPUT_DIR + '/image', exist_ok=True)
    os.makedirs(OUTPUT_DIR + '/audio', exist_ok=True)

    scene_number = scenePartition(STORY)
    sentences = split_sentences(STORY)

    characters = extract_character(STORY)
    characters_TTS = match_voices(characters, VOICE_LIST)
    character_sentences, sentence_idxs = match_sentences(sentences, list(characters.keys()))
    emotion_sentences = match_emotions(sentences, sentence_idxs)
    character_info = merge_dict(characters_TTS, character_sentences)
   
    process = 0
    ssml_list = []
    for i in range(1, scene_number + 1):
        with open(os.path.join(OUTPUT_DIR, f'image/original_scene_{i}.txt'), 'r') as file:
            scene_story = file.read()
        sentences = split_sentences(scene_story)
        ssml = create_ssml(sentences, character_info, emotion_sentences, NARRATION, SPEED, process)
        process += len(sentences)
        ssml_list.append(ssml)
    
    for i, ss in enumerate(ssml_list):
        with open(os.path.join(OUTPUT_DIR, f'audio/ssml_{i}.txt'), mode='w', encoding='utf8') as f:
            f.write(ss)

    for i in range(1, scene_number + 1):
        makeTTS(ssml_list[i-1], i)

    with open(os.path.join(OUTPUT_DIR, 'audio/ssml_full.txt'), mode='w', encoding='utf8') as f:
        for s in ssml_list:
            s += "\n"
            f.write(s)

    with open(os.path.join(OUTPUT_DIR, 'image/original_scene_full.txt'), mode='r', encoding='utf8') as file:
        scene_text_full = file.read()
    createBackground(scene_text_full)
    createAction(scene_text_full)
    with open(os.path.join(OUTPUT_DIR, 'image/character_list.txt'), mode='r', encoding='utf8') as file:
        character_list = file.read()
    createCharDescription(scene_text_full, character_list, str(characters))
    createImage(IMAGESTYLE, scene_number)

    image_files = [os.path.join(OUTPUT_DIR, f"image/img_scene{i}.png") for i in range(1, scene_number + 1)]
    audio_files = [os.path.join(OUTPUT_DIR, f"audio/tts_{i}.mp3") for i in range(1, scene_number + 1)]

    frame_width = 1792
    frame_height = 1024

    for image_file in image_files:
        frame = cv2.imread(image_file)
        imagel = add_letterbox(frame, frame_width, frame_height)
        cv2.imwrite(image_file, imagel)

    makeMovie(image_files, audio_files, "output.mp4")