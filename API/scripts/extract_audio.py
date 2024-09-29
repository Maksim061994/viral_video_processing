import subprocess


def extract_audio(path_to_vid, path_to_audio):

    command = f'ffmpeg -i {path_to_vid} -y -ab 160k -ar 44100 -vn {path_to_audio}'
    subprocess.call(command, shell=True)