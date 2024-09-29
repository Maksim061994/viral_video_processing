import cv2
import subprocess


class VideoReader:
    """
    Чтение видео из файла
    """
    def __init__(self):
        pass

    def extract_audio_from_video(self, path_video, path_to_audio):
        command = f'ffmpeg -i {path_video} -y -ab 160k -ar 44100 -vn {path_to_audio}'
        subprocess.call(command, shell=True)
