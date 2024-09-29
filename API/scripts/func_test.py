from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from utils.extract_audio import extract_audio
from utils.skilearn_model import *
from model.whisper import Whisper_model
from utils.api_maker import create_dict_data
from vertical_video_maker import *
from utils.lama_make_interval import *
from utils.get_datainfo_llm import *
import os

def func_test(video_path):
    """
    {
        "data": [{desrcibe: "text"
            emoji: "🔍"
            tag: "#test
            source: ML
            number: 1
            transcription: "text"}
            ...
        ],
        "video": [file1, ...],
        "audio": [audio1, ...]

    }
    """
    start_total = time.time()
    if not os.path.exists('short_videos'):
        os.makedirs('short_videos')
    if not os.path.exists('short_audio'):
        os.makedirs('short_audio')
    if not os.path.exists('data_sub'):
        os.makedirs('data_sub')

    path_audio = f"short_audio/{video_path.split('.')[-2].split('/')[-1]}.mp3"
    extract_audio(video_path, path_audio)

    start = time.time()
    # Работа МЛ алгоритма для выделение интервалов
    y_test_df, sr = find_anomalies_audio(path_audio)
    intervals, starting_times, = postproccess_annomalies(y_test_df, sr)
    print(intervals)
    end = time.time()
    print(f'Работа алгоритма МЛ в с {end-start}')

    print('Работа модели виспер с загрузкой модели')
    start = time.time()
    model = Whisper_model()
    transcribe_video = model.transcribe(f"{path_audio}", language='ru', verbose=False, beam_size=5, best_of=5)
    end = time.time()
    print(f'Работа модели виспер с загрузкой модели в с {end - start}')
    Llama_model = lama_model()
    # Работа алгоритма ЛЛМ для выделение интервалов и текстовых описаний
    interval_llm = get_llm_interval(transcribe_video, Llama_model)
    data_full_video = get_llm_meta_info(transcribe_video, Llama_model)
    interval_merge = insert_llm_interval(interval_llm, intervals)
    os.remove(path_audio)

    data_dict = {}
    data_full = []
    video = []
    audio = []
    for i, data_interval in enumerate(interval_merge):
        start, end, source = data_interval
        # Названия видео и аудио отрезка
        video_name = f'short_videos/video{i + 1}.mp4'
        audio_name = f'short_audio/audio{i + 1}.mp3'
        video.append(f'video{i + 1}.mp4')
        audio.append(f'audio{i + 1}.mp3')
        # Cоздание отрезка по видео
        video_interval_file = f'short_videos/origin_video{i + 1}.mp4'
        # #Cохранение видео по интервалам
        ffmpeg_extract_subclip(video_path, start.total_seconds(), end.total_seconds(), targetname=video_interval_file)
        # Выделение аудиодорожки
        extract_audio(video_interval_file, audio_name)
        # Предикт виспера
        print('Работа модели виспер без загрузки модели')
        start = time.time()
        transcribe_video_short = model.transcribe(f"{audio_name}", language='ru', verbose=False, beam_size=5, best_of=5)
        end = time.time()
        print(f'Работа модели виспер без загрузки модели в с {end - start}')
        # Работа алгоритма ЛЛМ для формирования текстовых описаний
        if len(transcribe_video_short['text']) < 200:
            data = create_dict_data(data_full_video)
        else:
            data = get_llm_meta_info(transcribe_video_short, Llama_model)
        # Формирования словаря для АПИ
        data['transcription'] = transcribe_video_short['text']
        data['source'] = source
        data['number'] = i + 1
        data_full.append(data)
        # Cоздание вертикального отрезка по видео
        create_viral_video(video_interval_file, video_name)
        srt_path = f'data_sub/subs{i}.srt'
        generate_subtitles(transcribe_video_short, srt_path)
        ass_path = f'data_sub/subs{i}.ass'
        mp4_path_subs = f'data_sub/video{i}_subs.mp4'
        embed_subtitles(srt_path, ass_path, video_name, mp4_path_subs)
        # Удаление видео по интервалам
        os.remove(video_interval_file)
        os.remove(srt_path)
        os.remove(ass_path)
        # os.remove(video_name)
    data_dict['data'] = data_full
    data_dict['video'] = video
    data_dict['audio'] = audio
    end_total = time.time()
    print(f'Работа алгоритма в с {end_total - start_total}')
    return data_dict