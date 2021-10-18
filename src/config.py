import sys
PROJECT_ROOT = "D:/code/python/rppg/myRhythmNet"
sys.path.append(PROJECT_ROOT)
haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
eye_cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml"
VIDEO_ROOT = "../../data/VIPL/"
ST_MAP_ROOT = "/data/video_st_map/"
HR_ROOT = "/data/video_hr/"
DATA_ROOT = "/data/"
FOLD_MAT_ROOT = "/data/fold_split/"
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-3
INPUT_H = 25
INPUT_W = 25
DEVICE = "GPU"
CLIP_SIZE = 300
