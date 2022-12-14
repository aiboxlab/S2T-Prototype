
import os
import re
import math
import gzip
import json
import pickle5
import FeatureExtractor.utils as u
import pandas as pd
import numpy as np
import torch
import cv2
import datetime as dt
from datetime import datetime
from FeatureExtractor.models.pytorch_i3d import InceptionI3d


# S2T DATA
def save(Data, name):
    with gzip.open(name, 'wb') as handle:
        pickle5.dump(Data, handle, protocol=pickle5.HIGHEST_PROTOCOL)


def read_data_with_subdirectorys(data_path, ext='.mp4'):
    videos_path_list = []
    print("List of all directories in '% s':" % data_path)

    for path, subdirs, files in os.walk(data_path):
        for name in files:
            if name.endswith(ext):
                video = os.path.join(path, name).replace("\\","/")
                videos_path_list.append(video)

    return videos_path_list


# RESCALING
def crop_slices(image, height, width, slice_size):
    return image[0:height, 0+slice_size:width-slice_size]

def resize_square(image, resolution_factor):
    return cv2.resize(image, (resolution_factor, resolution_factor)) 

def make_frame(image, to_resolution):
    height, width, _ = image.shape

    #Difereça entre as dimensões de largura e altura (tamanho dopedaço que será cortado)
    diff = max(height, width) - min(height, width)

    #Vamos dividir o "pedaço" anterior em dois pedaços de igual tamanho (para remover 1 da esquerda e outro da direita)
    slice_size= int(math.floor(diff/2))

    #Vamos cropar o "pedaço" esquedo e o "pedaço" direito
    crop_img = crop_slices(image, height, width, slice_size)#image[0:height, 0+slice_size:width-slice_size]
    #image = cv2.rectangle(image, (0+slice_size, 0), (width-slice_size, height), (255, 0, 0), 2) #Proporção blue

    #Após termos uma imagem quadrada, iremos redimensionar
    resize_img = resize_square(crop_img, to_resolution)

    #resize_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)

    #print(resize_img.shape)

    return resize_img


# FEATURE EXTRACTOR
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def load_all_rgb_frames_from_video_rescaling(video, desired_channel_order='rgb'):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []

    to_resolution = 224

    while(True):

        try:

            ret, frame = cap.read()

            frame = make_frame(frame, to_resolution) # Rescaling

            frame = cv2.resize(frame, dsize=(to_resolution, to_resolution))

            frame_transformed = frame.copy()

            if desired_channel_order == 'bgr':
                frame_transformed = frame_transformed[:, :, [2, 1, 0]]

            frame_transformed = (frame_transformed / 255.) * 2 - 1

            frames.append(frame_transformed)

        except:
            break

    nframes = np.asarray(frames, dtype=np.float32)
    return nframes


def extract_features_fullvideo(model, inp, framespan, stride):
    rv = []

    indices = list(range(len(inp)))
    groups = []
    for ind in indices:

        if ind % stride == 0:
            groups.append(list(range(ind, ind+framespan)))

    for g in groups:
        frames = inp[g[0]: min(g[-1]+1, inp.shape[0])]

        num_pad = 9 - len(frames)
        if num_pad > 0:
            pad = np.tile(np.expand_dims(frames[-1], axis=0), (num_pad, 1, 1, 1))
            frames = np.concatenate([frames, pad], axis=0)

        frames = frames.transpose([3, 0, 1, 2])

        ft = _extract_features(model, frames)

        rv.append(ft)

    return rv


def _extract_features(model, frames):
    inputs = torch.from_numpy(frames)

    inputs = inputs.unsqueeze(0)

    inputs = inputs.cuda()
    with torch.no_grad():
        ft = model.extract_features(inputs)
    ft = ft.squeeze(-1).squeeze(-1)[0].transpose(0, 1)

    ft = ft.cpu()
    return ft


def extract_features_i3d(i3d, video, framespan, stride=2, inp_channels='rgb'):

    frames = load_all_rgb_frames_from_video_rescaling(video, inp_channels)
    
    features = extract_features_fullvideo(i3d, frames, framespan, stride)

    result = []
    for tensor in features:
        result.append(tensor[0])
    sign = torch.stack(result)

    return sign # S2T


def extract_features(video, i3d, span=16):

    sign = extract_features_i3d(i3d, video, framespan=span, stride=2, inp_channels='rgb')

    name = video.split("/")[-1].split(".")[0]
                
    annotation = {"name": name, "signer": "foo", "gloss": "foo", "text": "foo", "sign": sign}

    save([annotation], "data/test"+str(span)+".gzip")
