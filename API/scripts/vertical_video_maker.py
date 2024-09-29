import subprocess
from srt import Subtitle
import srt
import os
from datetime import timedelta
import time


def create_viral_video(path_to_vid, path_to_out):
    command = f'ffmpeg -i {path_to_vid} -y -lavfi "[0:v]scale=iw:2*trunc(iw*12/17),boxblur=luma_radius=min(h\,w)/20:luma_power=1:chroma_radius=min(cw\,ch)/20:chroma_power=1[bg];[bg][0:v]overlay=(W-w)/2:(H-h)/2,setsar=1" {path_to_out}'
    subprocess.call(command, shell=True)

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
                                     microseconds=timedelta(seconds=end).microseconds), content=text, proprietary='')

        # Append the subtitle to the list of subtitles.
        subs.append(sub)

    # Write the SRT file to disk.
    with open(srt_path, mode="w", encoding="utf-8") as f:
        f.write(srt.compose(subs))