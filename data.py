import os
import ffmpeg
import moviepy
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip, ffmpeg_extract_audio

path_a = './audio'
path = './video'
path_r = './video_r'

for dir in ['hungry', 'hugging', 'sleepy']:
    folder = os.path.join(path_r, dir)
    files = os.listdir(folder)
    idx = 1

    for file in files:
        if file == '.DS_Store':
            continue
        src = os.path.join(folder, file)
        dst = os.path.join(path, dir, file)
        ffmpeg_extract_subclip(src, 0, 15, targetname=dst)
        idx += 1

# audio extraction
for dir in ['hungry', 'hugging', 'sleepy']:
    folder = os.path.join(path, dir)
    files = os.listdir(folder)

    for file in files:
        if file == '.DS_Store':
            continue
        src = os.path.join(folder, file)
        dst = os.path.join(path_a, dir, file.replace('.mp4', '.wav'))
        ffmpeg_extract_audio(
            inputfile=src, output=dst, bitrate=160000, fps=44100)
