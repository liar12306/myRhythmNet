import os
import sys

sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath("../data/"))
import cv2
import os
import glob
import numpy as np
import config
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from joblib import Parallel, delayed, parallel_backend
import time
import scipy.io
from utils.seetaface import api

# download requisite certificates
import ssl;

ssl._create_default_https_context = ssl._create_stdlib_context


# Chunks the ROI into blocks of size 5x5
def chunkify(img, block_width=5, block_height=5):
    shape = img.shape
    x_len = shape[0] // block_width
    y_len = shape[1] // block_height
    # print(x_len, y_len)

    chunks = []
    x_indices = [i for i in range(0, shape[0] + 1, x_len)]
    y_indices = [i for i in range(0, shape[1] + 1, y_len)]

    shapes = list(zip(x_indices, y_indices))

    #  # for plotting purpose
    # implot = plt.imshow(img)
    #
    # end_x_list = []
    # end_y_list = []

    for i in range(len(x_indices) - 1):
        # try:
        start_x = x_indices[i]
        end_x = x_indices[i + 1]
        for j in range(len(y_indices) - 1):
            start_y = y_indices[j]
            end_y = y_indices[j + 1]
            # end_x_list.append(end_x)
            # end_y_list.append(end_y)
            chunks.append(img[start_x:end_x, start_y:end_y])
        # except IndexError:
        #     print('End of Array')

    return chunks


def plot_image(img):
    plt.axis("off")
    plt.imshow(img, origin='upper')
    plt.show()


# Function to read the the video data as an array of frames and additionally return metadata like FPS, Dims etc.
def get_frames_and_video_meta_data(video_path, meta_data_only=False):
    cap = cv2.VideoCapture(video_path)
    frameRate = cap.get(5)  # frame rate

    # Frame dimensions: WxH
    frame_dims = (int(cap.get(3)), int(cap.get(4)))
    # Paper mentions a stride of 0.5 seconds = 15 frames
    sliding_window_stride = int(frameRate / 2)
    num_frames = int(cap.get(7))
    if meta_data_only:
        return {"frame_rate": frameRate, "sliding_window_stride": sliding_window_stride, "num_frames": num_frames}

    # Frames from the video have shape NumFrames x H x W x C
    frames = np.zeros((num_frames, frame_dims[1], frame_dims[0], 3), dtype='uint8')

    frame_counter = 0
    while cap.isOpened():
        # curr_frame_id = int(cap.get(1))  # current frame number
        ret, frame = cap.read()
        if not ret:
            break

        frames[frame_counter, :, :, :] = frame
        frame_counter += 1
        if frame_counter == num_frames:
            break

    cap.release()
    return frames, frameRate, sliding_window_stride


# Threaded function for st_map generation from a single video arg:file in dataset
def generate_st_maps_and_save(video_root):
    # print(f"Generating Maps for file: {file}")
    # maps = np.zeros((10, config.CLIP_SIZE, 25, 3))
    # print(index)
    video_path = video_root + "video.avi"
    file = os.path.join(config.VIDEO_ROOT, video_path)
    st_maps = video_to_maps(file)

    if st_maps is None:
        return 1

    file_name = video_root.replace('/', '_')
    save_path = config.PROJECT_ROOT + config.ST_MAP_ROOT
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    #print(save_path)
    save_path = os.path.join(save_path, "{}.npy".format(file_name))
    np.save(save_path, st_maps)
    return 1


def generate_st_maps():
    data = pd.read_csv(os.path.join(config.VIDEO_ROOT, "data.csv"))
    video_roots = data['root']
    times = 0
    ten_p_data = len(video_roots) // 10
    count = 1
    pre_time = time.time()
    for idx, video_root in enumerate(video_roots):
        if idx == (count * ten_p_data):
            current_time = time.time()
            cost_time = current_time - pre_time
            times += cost_time
            pre_time = current_time
            h = cost_time // 3600
            m = cost_time % 3600 // 60
            s = cost_time % 3600 % 60
            print("\nprocessed data : {}0% | cost time : {}h {}m {}s ".format(count, h, m, s))
            count += 1
        generate_st_maps_and_save(video_root)
    end_time = time.time()

    h = times // 3600
    m = times % 3600 // 60
    s = times % 3600 % 60
    print("\nperprocess data have finish | cost time -- {}h : {}m : {}s".format(h, m, s))


def calculate_target_hr(path="p1/v1/source1/"):
    video_path = config.VIDEO_ROOT + path + "video.avi"
    hr_csv_path = config.VIDEO_ROOT + path + "gt_HR.csv"
    if not os.path.exists(hr_csv_path):
        return 1
    save_path_root = config.PROJECT_ROOT + config.HR_ROOT

    if not os.path.exists(save_path_root):
        os.makedirs(save_path_root)
    save_path = save_path_root + path.replace('/', '_') + "hr.npy"

    if not os.path.exists(save_path):
        cat = cv2.VideoCapture(video_path)
        frame_rate = cat.get(5)
        st_map_path = config.PROJECT_ROOT + config.ST_MAP_ROOT + path.replace('/', '_') + ".npy"
        if not os.path.exists(st_map_path):
            return 1
        hr_csv = list(pd.read_csv(hr_csv_path)["HR"].values)
        map_num = np.load(st_map_path).shape[0]
        hr_list = []
        map_T_size = int(config.CLIP_SIZE / frame_rate)
        end = map_num // 2 + map_T_size
        if (end > len(hr_csv)):
            for i in range(len(hr_csv), end + 1):
                hr_csv.append(hr_csv[-1])
        for idx in tqdm(range(map_num)):
            start_time = idx // 2
            end_time = start_time + map_T_size + 1
            hr = int(np.mean(hr_csv[start_time:end_time]))
            hr_list.append(hr)

        np.save(save_path, hr_list)
    return 1


def generate_rhythmNet_target_hr():
    data = pd.read_csv(os.path.join(config.VIDEO_ROOT, "data.csv"))["root"].values
    for path in tqdm(data):
        calculate_target_hr(path)


def data_list():
    nir_list_path = config.VIDEO_ROOT + "NIR.txt"
    # data_list_path = config.VIDEO_ROOT+"data.csv"
    # data_list = list(pd.read_csv(data_list_path)['root'].values)
    # print(len(data_list))
    nir_list = []
    with open(nir_list_path, "r") as f:
        for line in f.readlines():
            line = (line.strip("\n") + "/").replace('/', '_') + ".npy"
            nir_list.append(line)
    print(nir_list[0])
    # print(len(nir_list))
    # new = list(set(data_list)-set(nir_list))
    #
    # count=0;
    # for i in new:
    #     st_map_name = i.replace('/','_')+".npy"
    #     st_map_path = config.PROJECT_ROOT+config.ST_MAP_ROOT+st_map_name
    #     st_map = np.load(st_map_path)
    #     if(st_map.shape[0]<6):
    #         count+=1
    # print(count)
    count = 0
    st_map_path = config.PROJECT_ROOT + config.ST_MAP_ROOT
    st_map_list = os.listdir(st_map_path)
    print(st_map_list[0])
    print(len(st_map_list))
    normal_video_list = list(set(st_map_list) - set(nir_list))
    print(len(normal_video_list))
    train_list = []
    for st_map_name in normal_video_list:
        st_map = np.load(st_map_path + st_map_name)
        if (st_map.shape[0] >= 6):
            train_list.append(st_map_name)
    print(len(train_list))


def face_split(frame, face, landmarks):
    left_eye = []
    right_eye = []
    noe_face_left = []
    noe_face_right = []
    for landmark in landmarks[36:42]:
        left_eye.append([landmark.x, landmark.y])

    for landmark in landmarks[42:48]:
        right_eye.append([landmark.x, landmark.y])

    noe_face_left.append([face.x, face.y + face.height])
    noe_face_right.append([face.x + face.width, face.y + face.height])
    for landmark in landmarks[0:8]:
        noe_face_left.append([landmark.x, landmark.y])
    for landmark in landmarks[8:16]:
        noe_face_right.append([landmark.x, landmark.y])
    fill_eare = np.array([left_eye, right_eye], dtype="int32")
    cv2.fillPoly(frame, fill_eare, (0, 0, 0))
    cv2.fillPoly(frame, np.array([noe_face_left, noe_face_right], dtype="int32"), (0, 0, 0))

    return frame


def create_maps(faces):
    st_map = np.zeros((len(faces), 25, 3))
    for idx, face in enumerate(faces):
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        roi_blocks = np.array(chunkify(face))
        for block_idx, block in enumerate(roi_blocks):
            avg_pixels = cv2.mean(block)
            st_map[idx, block_idx, 0] = int(avg_pixels[0])
            st_map[idx, block_idx, 1] = int(avg_pixels[1])
            st_map[idx, block_idx, 2] = int(avg_pixels[2])

    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(np.array(st_map, dtype="uint8"))
    # plt.axis("off")

    for block_idx in range(st_map.shape[1]):
        # Not sure about uint8

        r_maxx = np.max(st_map[:, block_idx, 0])
        r_minn = np.min(st_map[:, block_idx, 0])
        g_maxx = np.max(st_map[:, block_idx, 1])
        g_minn = np.min(st_map[:, block_idx, 1])
        b_maxx = np.max(st_map[:, block_idx, 2])
        b_minn = np.min(st_map[:, block_idx, 2])
        st_map[:, block_idx, 0] = (st_map[:, block_idx, 0] - r_minn) / (r_maxx - r_minn) * 255
        st_map[:, block_idx, 1] = (st_map[:, block_idx, 1] - g_minn) / (g_maxx - g_minn) * 255
        st_map[:, block_idx, 2] = (st_map[:, block_idx, 2] - b_minn) / (b_maxx - b_minn) * 255
    # print(st_map)
    # plt.subplot(122)
    # plt.imshow(np.array(st_map, dtype="uint8"))
    # plt.axis("off")
    # plt.show()
    return st_map


def video_to_maps(video="../data/video.avi"):
    #print(video)
    frames, frameRate, sliding_window_stride = get_frames_and_video_meta_data(video)
    num_frame = frames.shape[0]
    clip_size = config.CLIP_SIZE
    seetaFace = api.SeetaFace(api.FACE_DETECT | api.LANDMARKER68)
    faces = []
    for frame in frames:

        detect_result = seetaFace.Detect(frame)

        for i in range(detect_result.size):
            face = detect_result.data[i].pos
            landmarks = seetaFace.mark68(frame, face)

            cv2.rectangle(frame, (face.x, face.y), (face.x + face.width, face.y + face.height), (255, 0, 0), 2)

            for landmark in landmarks[36:48]:
                cv2.circle(frame, (int(landmark.x), int(landmark.y)), 1, (255, 255, 255), -1)
            frame = face_split(frame, face, landmarks)
        faces.append(frame)
    st_maps = []

    for start_frame_index in range(0, num_frame, sliding_window_stride):
        end_frame_index = start_frame_index + clip_size
        if end_frame_index > num_frame:
            break
        st_map = create_maps(faces[start_frame_index:end_frame_index])
        st_maps.append(st_map)
    st_maps = np.array(st_maps)

    return st_maps


if __name__ == "__main__":
    # generate_rhythmNet_target_hr()
    # generate_st_map()
    generate_st_maps()
    #generate_st_maps_and_save("p1/v1/source1/")

