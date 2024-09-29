from app.helpers.file_reader import VideoReader
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from app.config.settings import get_settings
from app.helpers.llama_connector import LlamaConnector
import subprocess
import os
import librosa
import pandas as pd
import datetime
import srt
from srt import Subtitle
import itertools
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import whisper
from datetime import timedelta
import json

settings = get_settings()
model_whisper = whisper.load_model(settings.path_weights_whisper, device=settings.device)


class ManagerViralVideo:

    def __init__(self, params):
        self.params = params
        self.file_reader = VideoReader()

    @staticmethod
    def postproccess_anomalies(y_test_df, sr):

        window = sr // 5

        y_test_df['anomaly'] = (y_test_df['anomaly'] - 1) * -1 / 2

        y_test_df['smoothed'] = y_test_df['anomaly'].rolling(window=window, min_periods=1).max().astype('int')
        y_test_df['smoothed_rev'] = y_test_df['anomaly'].iloc[::-1].rolling(window=window, min_periods=1).max().astype(
            'int').iloc[::-1]
        y_test_df['smoothed'] = y_test_df['smoothed'] * y_test_df['smoothed_rev']

        smoothed_list = y_test_df['smoothed'].tolist()
        intervals = [(x[0], len(list(x[1]))) for x in itertools.groupby(smoothed_list)]

        # get index of intervals >= 0.3s
        min_frames = sr * (len(y_test_df)) * 7.54e-08
        print((len(y_test_df)) * 7.54e-08)

        starting_times = []
        starting_seconds = []
        for idx, interval in enumerate(intervals):
            if interval[0] == 1 and interval[1] >= min_frames:
                starting_frame = 0
                if idx == 0:
                    starting_times.append('00:00')
                else:
                    for i in range(idx):
                        starting_frame += intervals[i][1]
                    num_seconds = int(starting_frame / sr)
                    if not starting_seconds:
                        starting_seconds.append([num_seconds, num_seconds])
                    elif (num_seconds - starting_seconds[-1][1]) <= 4:
                        starting_seconds[-1] = [starting_seconds[-1][0], num_seconds]
                    else:
                        starting_seconds.append([num_seconds, num_seconds])

                    starting_time = str(datetime.timedelta(seconds=num_seconds))[-5:]
                    starting_times.append(starting_time)

        # если ничего не нашли
        if not starting_seconds:
            anomaly_list = y_test_df['anomaly'].astype('int').tolist()
            intervals = [(x[0], len(list(x[1]))) for x in itertools.groupby(anomaly_list)]

            all_lenght = [x[1] for x in intervals]
            longest_three = sorted(all_lenght, reverse=True)[:3]

            starting_times = []
            starting_seconds = []
            for idx, interval in enumerate(intervals):
                if interval[0] == 1 and interval[1] in longest_three:
                    starting_frame = 0
                    if idx == 0:
                        starting_times.append('00:00')
                    else:
                        for i in range(idx):
                            starting_frame += intervals[i][1]
                        num_seconds = int(starting_frame / sr)
                        if not starting_seconds:
                            starting_seconds.append([num_seconds, num_seconds])
                        elif (num_seconds - starting_seconds[-1][1]) <= 4:
                            starting_seconds[-1] = [starting_seconds[-1][0], num_seconds]
                        else:
                            starting_seconds.append([num_seconds, num_seconds])

                        starting_time = str(datetime.timedelta(seconds=num_seconds))[-5:]
                        starting_times.append(starting_time)

        intervals = []
        for pair in starting_seconds:
            intervals.append(
                [(datetime.timedelta(seconds=(pair[0] - 3))), (datetime.timedelta(seconds=(pair[1] + 6))), 'ML'])

        starting_times = list(set(starting_times))
        starting_times.sort()
        return intervals, starting_times

    @staticmethod
    def find_anomalies_audio(path_to_audio):
        y_test, sr = librosa.load(path_to_audio)

        y_test_df = pd.DataFrame.from_dict({'Audio': y_test})

        scaler = StandardScaler()
        np_scaled = scaler.fit_transform(y_test.reshape(-1, 1))
        data = pd.DataFrame(np_scaled)

        outliers_fraction = float(.0007)
        model = IsolationForest(contamination=outliers_fraction)
        model.fit(data)
        y_test_df['anomaly'] = model.predict(data)

        return y_test_df, sr

    def process(self):
        output_data = []
        name_video = self.params.name_video.rsplit(".", 1)[0]
        path_to_audio = f"{settings.path_to_files}/{name_video}_audio.mp3"
        self.file_reader.extract_audio_from_video(f"{settings.path_to_files}/" + self.params.name_video, path_to_audio)

        # ML алгоритм
        y_test_df, sr = self.find_anomalies_audio(path_to_audio)
        intervals, starting_times = self.postproccess_anomalies(y_test_df, sr)

        # Использование модели Whisper
        transcribe_video = model_whisper.transcribe(
            path_to_audio, language='ru', verbose=False, beam_size=5, best_of=5
        )
        llama_connector = LlamaConnector()

        interval_llm = self.get_llm_interval(transcribe_video, llama_connector)
        data_full_video = self.get_llm_meta_info(transcribe_video, llama_connector)
        interval_merge = self.insert_llm_interval(interval_llm, intervals)

        for i, data_interval in enumerate(interval_merge):
            start, end, source = data_interval
            cnt = i + 1

            # make shorts video
            video_interval_file = f'{settings.path_to_files}/{cnt}_{name_video}.mp4'
            ffmpeg_extract_subclip(
                f"{settings.path_to_files}/{self.params.name_video}",
                start.total_seconds(),
                end.total_seconds(),
                targetname=video_interval_file
            )

            # make audio by shorts video
            audio_interval_file = f'{settings.path_to_files}/{cnt}_{name_video}.mp3'
            self.file_reader.extract_audio_from_video(video_interval_file, audio_interval_file)
            transcribe_video_short = model_whisper.transcribe(
                f"{audio_interval_file}", language='ru', verbose=False, beam_size=5, best_of=5
            )

            if len(transcribe_video_short['text']) < 200:
                data = self.create_dict_data(data_full_video)
            else:
                data = self.get_llm_meta_info(transcribe_video_short, llama_connector)

            title = f"{cnt}_vertical_{name_video}"
            output_vid = f'{settings.path_to_files}/{title}.mp4'
            self.create_viral_video(video_interval_file, output_vid)

            # накладываем субтитры на видео
            srt_path = f'{settings.path_to_files}/subs_{cnt}_{name_video}.srt'
            self.generate_subtitles(transcribe_video_short, srt_path)
            ass_path = f'{settings.path_to_files}/subs_{cnt}_{name_video}.ass'
            mp4_path_subs = f'{settings.path_to_files}/result_video_subs_{i}_{name_video}.mp4'
            self.embed_subtitles(srt_path, ass_path, output_vid, mp4_path_subs)

            os.remove(srt_path)
            os.remove(ass_path)
            os.remove(output_vid)
            os.remove(video_interval_file)

            output_data.append({
                "number": cnt, "describe": data['summary'], "emoji": data['emoji'], "tag": data['tags'],
                "source": source, "transcription": transcribe_video_short['text'],
                "title": title, "filename": f"result_video_subs_{i}_{name_video}.mp4", "extname": ".mp4",
                "mimetype": "video/mp4",
                "url": f"/storage/{mp4_path_subs}", "url_audio": f"/storage/{audio_interval_file}"
            })
        return output_data

    @staticmethod
    def embed_subtitles(srt_path, ass_path, mp4_path_origin, mp4_path_subs):
        # Convert SRT file to ASS file
        os.system(f"ffmpeg -i {srt_path} {ass_path}")

        with open(ass_path, 'r', encoding="utf-8") as f:
            lines = f.readlines()

        for i in range(len(lines)):
            if 'Style: Default' in lines[i]:
                # change fontname, fontsize, and boldness
                fontname = "Liberation Sans"  # "Molot" #"Montserrat" #"MS Gothic"
                fontsize = 20
                boldness = -1  # -1(Bold) or 0
                lines[
                    i] = f'Style: Default,{fontname},{fontsize},&Hffffff,&Hffffff,&H0,&H0,{boldness},0,0,0,100,100,0,0,1,1,0,2,10,10,10,0\n'
                break

        with open(ass_path, 'w', encoding="utf-8") as f:
            f.writelines(lines)

        # Embed subtitles in video
        os.system(f"ffmpeg -i {mp4_path_origin} -vf ass={ass_path} {mp4_path_subs}")

    @staticmethod
    def generate_subtitles(result, srt_path="data/test_subs.srt"):
        segments = result["segments"]
        subs = []

        # Iterate over the segments and create SRT subtitles for each segment.
        for data in segments:
            index = data["id"] + 1
            start = data["start"]
            end = data["end"]

            text = data["text"]

            # Create an SRT subtitle object.
            sub = Subtitle(index=1, start=timedelta(seconds=timedelta(seconds=start).seconds,
                                                    microseconds=timedelta(seconds=start).microseconds),
                           end=timedelta(seconds=timedelta(seconds=end).seconds,
                                         microseconds=timedelta(seconds=end).microseconds), content=text,
                           proprietary='')

            # Append the subtitle to the list of subtitles.
            subs.append(sub)

        # Write the SRT file to disk.
        with open(srt_path, mode="w", encoding="utf-8") as f:
            f.write(srt.compose(subs))

    @staticmethod
    def create_dict_data(data_interval_video):
        data = {'summary': data_interval_video['summary'], 'emoji': data_interval_video['emoji'],
                'tags': data_interval_video['tags']}
        return data

    @staticmethod
    def get_timedelta(str_time):
        t = datetime.datetime.strptime(str_time, "%H:%M:%S")
        delta = datetime.timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
        return delta

    def get_llm_interval(self, data_sub, llama_connector):
        result_text = ''
        limit = 4000 if int(os.getenv("PARAM_N_CTX", "2048")) == 2048 else 6000

        for segment in data_sub['segments']:
            start_sec = datetime.timedelta(seconds=int(segment['start']))
            end_sec = datetime.timedelta(seconds=int(segment['end']))
            seg_text = segment['text']
            seg_text = f'''{{'start': {start_sec}, 'end': {end_sec}, 'text': {seg_text}}}, \n'''
            result_text += seg_text
            if len(result_text) > limit:
                break

        query = f'''Ниже представлена текстовая расшифровка видео с таймкодами. Найди самые интересные по смыслу места, они должны идти последовательно.
В списке должно быть не больше десяти элементов.

Дай ответ ввиде списка json
{{
    "start": "hh:mm:ss", 
    "end": "hh:mm:ss", 
    "text": "str"
}}

Расшифровка видео:
{result_text}

Отрезки:
        '''

        messages = [
            {
                'role': 'user',
                'content': query
            }
        ]

        outputs = llama_connector.predict(messages)
        final_interval = []
        try:
            parsed_ans = json.loads(outputs.split('```json')[1][:-3])
            duration = self.get_timedelta(parsed_ans[-1]['end']).seconds - self.get_timedelta(
                parsed_ans[0]['start']).seconds
            if duration > 180:
                end = self.get_timedelta(parsed_ans[0]['start']) + datetime.timedelta(seconds=180)
            else:
                end = self.get_timedelta(parsed_ans[-1]['end'])
            final_interval = [self.get_timedelta(parsed_ans[0]['start']), end, 'LLM']
        except Exception as err:
            print(err)
        return final_interval

    def is_overlap(self, interval_1, interval_2):
        min_len = min([interval_1[1].seconds - interval_1[0].seconds, interval_2[1].seconds - interval_2[0].seconds])

        latest_start = max([interval_1[0].seconds, interval_2[0].seconds])
        earliest_end = min([interval_1[1].seconds, interval_2[1].seconds])

        overlap = max([0, earliest_end - latest_start])

        return overlap > (min_len / 2)

    def get_tags(self, data_sub, llama_connector):
        limit = 4000 if int(os.getenv("PARAM_N_CTX", "2048")) == 2048 else 6000
        query = f'''Ниже представлена текстовая расшифровка видео. Напиши подходящие для него хештеги, отдавай приоритет хештегам из одного слова

Расшифровка видео:
{data_sub['text'][:limit]}

Хештеги:
        '''
        messages = [
            {'role': 'user',
             'content': query}
        ]
        outputs = llama_connector.predict(messages)
        return outputs

    def get_emojies(self, data_sub, llama_connector):
        limit = 4000 if int(os.getenv("PARAM_N_CTX", "2048")) == 2048 else 6000
        query = f'''Ниже представлена текстовая расшифровка видео. Напиши подходящие для него эмодзи не больше трёх

Расшифровка видео:
{data_sub['text'][:limit]}

Эмодзи:
                '''
        messages = [
            {'role': 'user',
             'content': query}
        ]
        outputs = llama_connector.predict(messages)
        return outputs

    def get_summary(self, data_sub, llama_connector):
        limit = 4000 if int(os.getenv("PARAM_N_CTX", "2048")) == 2048 else 6000
        query = f'''Ниже представлена текстовая расшифровка видео. Напиши кратко о чём это видео

Расшифровка видео:
{data_sub['text'][:limit]}
            
                '''
        messages = [
            {'role': 'user',
             'content': query}
        ]
        outputs = llama_connector.predict(messages)
        return outputs

    def get_llm_meta_info(self, data_sub, llama_connector):
        tags = self.get_tags(data_sub, llama_connector)
        emoji = self.get_emojies(data_sub, llama_connector)
        summary = self.get_summary(data_sub, llama_connector)

        return {
            'summary': summary,
            'tags': tags,
            'emoji': emoji
        }

    def insert_llm_interval(self, final_interval, intervals):
        if len(final_interval) == 0:
            return intervals

        combined_intervals = []

        for interval in intervals:
            if not self.is_overlap(final_interval, interval):
                combined_intervals.append(interval)
            else:
                final_interval[2] = 'LLM + ML'

        combined_intervals.append(final_interval)

        return combined_intervals

    def create_viral_video(self, path_to_vid, path_to_out):
        command = f'ffmpeg -i {path_to_vid} -y -lavfi "[0:v]scale=iw:2*trunc(iw*12/17),boxblur=luma_radius=min(h\,w)/20:luma_power=1:chroma_radius=min(cw\,ch)/20:chroma_power=1[bg];[bg][0:v]overlay=(W-w)/2:(H-h)/2,setsar=1" {path_to_out}'
        subprocess.call(command, shell=True)
