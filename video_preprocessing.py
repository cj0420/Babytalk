import math
import cv2
import os
from tqdm import tqdm
import pandas as pd
import numpy as np

path = './video'
path_f = './frame'

for dir in ['hungry', 'hugging', 'sleepy']:
    folder = os.path.join(path, dir)
    files = os.listdir(folder)
    frame_cnt = 0
    print(dir)

    for file in files:
        print(file)
        cap = cv2.VideoCapture(os.path.join(folder, file))
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                frame_cnt += 1
                frame_path = os.path.join(
                    path_f, dir, str(frame_cnt)+'.jpg')
                print(frame_path)
                cv2.imwrite(frame_path, frame)
            else:
                break
        cap.release()
        cv2.destroyAllWindows()
