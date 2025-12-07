import re
import os

import torch
import numpy as np

from PIL import Image
from utils.process_video_audio import LoadVideoAudio
# pdb.set_trace()

from model import DAVE
import pdb
import matplotlib.pyplot as plt
import cv2

# the folder find the videos consisting of video frames and the corredponding audio wav
VIDEO_TEST_FOLDER = './TestVid/'
# where to save the predictions
OUTPUT = 'output'
# where tofind the model weights
MODEL_PATH = 'AVS360_ep25.pkl'
# some config parameters

IMG_WIDTH = 1920
IMG_HIGHT = 3840
TRG_WIDTH = 32
TRG_HIGHT = 40
import time


device = torch.device("cpu")


class PredictSaliency(object):

    def __init__(self):
        super(PredictSaliency, self).__init__()

        self.video_list = [os.path.join(VIDEO_TEST_FOLDER, p) for p in os.listdir(VIDEO_TEST_FOLDER)]
        self.model = DAVE()
        self.model=torch.load(MODEL_PATH, weights_only=False, map_location=device)
        self.output = OUTPUT
        if not os.path.exists(self.output):
                os.mkdir(self.output)
        self.model = self.model
        self.model.eval()

    @staticmethod
    def _load_state_dict_(filepath):
        if os.path.isfile(filepath):
            pdb.set_trace()
            print("=> loading checkpoint '{}'".format(filepath))
            checkpoint = torch.load(filepath, map_location=device)
            
            pattern = re.compile(r'module+\.*')
            state_dict = checkpoint['state_dict']
            
            for key in list(state_dict.keys()):
                if 'video_branch' in key: 
                   state_dict[key[:12] + '_cubic' + key[12:]] = state_dict[key]
            
            for key in list(state_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = re.sub('module.', '', key)
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]

        return state_dict

    def predict(self, stimuli_path, fps, out_path):
        equator_bias = cv2.resize(cv2.imread('ECB.png', 0), (10,8))
        equator_bias = torch.tensor(equator_bias).to(device, dtype=torch.float)
        equator_bias = equator_bias
        equator_bias = equator_bias/equator_bias.max()
        

        if not os.path.exists(out_path):
            os.mkdir(out_path)
        if not os.path.exists(out_path+ '/overlay'):  
            os.mkdir(out_path + '/overlay')
        video_loader = LoadVideoAudio(stimuli_path, fps)
        
        print(f"Vide loader length is {len(video_loader)}")

        vit = iter(video_loader)
        # raise SystemExit

        for idx in range(len(video_loader)):
            startPredTime = time.time()
            video_data_equi, video_data_cube, audio_data, AEM_data = next(vit)

            video_data_equi = video_data_equi.to(device=device, dtype=torch.float)
            video_data_cube = video_data_cube.to(device=device, dtype=torch.float)
            AEM_data = AEM_data.to(device=device, dtype=torch.float)
            audio_data = audio_data.to(device=device, dtype=torch.float)

            prediction = self.model(video_data_equi, video_data_cube, audio_data, AEM_data, equator_bias)
            
            # --- Saliency map ---
            saliency = prediction[0, 0].detach().cpu().numpy()
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
            saliency_img = Image.fromarray((saliency*255).astype(np.uint8))
            
            # Original frame
            frame_array = video_data_equi[0, :, 0].permute(1, 2, 0).cpu().numpy()  # [H,W,C]
            frame_array = (frame_array - frame_array.min()) / (frame_array.max() - frame_array.min())
            frame_img = Image.fromarray((frame_array*255).astype(np.uint8))

            # Resize saliency to match original frame
            saliency_img = saliency_img.resize(frame_img.size, Image.LANCZOS)

            # Apply colormap to saliency
            saliency_color = np.array(saliency_img)
            saliency_color = cv2.applyColorMap(saliency_color, cv2.COLORMAP_JET)
            saliency_color = cv2.cvtColor(saliency_color, cv2.COLOR_BGR2RGB)
            saliency_color = Image.fromarray(saliency_color)

            # Overlay
            overlay = Image.blend(frame_img, saliency_color, alpha=0.5)

            overlay = overlay.resize([IMG_HIGHT, IMG_WIDTH], Image.LANCZOS)
            saliency_img = saliency_img.resize([IMG_HIGHT, IMG_WIDTH], Image.LANCZOS)

            endPredTime = time.time()
            print(f"Current Frame Time: {endPredTime - startPredTime} seconds")

            overlay.save(f'{out_path}/overlay_{idx+1}.jpg')
            saliency_img.save(f'{out_path}/{idx+1}.jpg')  # optional: save raw saliency

    def predict_sequences(self):
        for v in self.video_list[:]:
            # print(v)
            print("gya")
            sample_rate = int(v[-2:])
            bname = os.path.basename(v[:-3])
            output_path = os.path.join(self.output, bname)
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            self.predict(v, sample_rate, output_path)

if __name__ == '__main__':

    p = PredictSaliency()
    # predict all sequences
    # p.predict_sequences()
    # alternatively one can call directy for one video
    p.predict(VIDEO_TEST_FOLDER, 60, OUTPUT) # the second argument is the video FPS.