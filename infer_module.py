#-*- coding: utf-8 -*-

import subprocess
import os, sys
import pathlib
import time
import numpy as np

# from keras.layers import *

'''
1차. 영상으로 부터 프레임 뽑아서 input 벡터 만들고
2차. input 벡터로부터 output 벡터 내는 애 -> 모델에서 얘를 function으로 빼줘야 함(서버 벡엔드에 두기 위해서) (인퍼런스-> output vector)
3차. output 벡터와 music 후보 vector의 거리를 계산해서 가장 가까운 애를 선택하고 나온 음악의 경로
4차. 음악의 경로를 받아서 다운로드(타겟) 폴더에 저장

=>
1) extract : upload 폴더의 영상으로 부터 frame 뽑음
2) getinput : 이미지를 읽어서(cv2/PIL) 전처리(resize, reshape)하여 input vector를 만듦
3) getoutput : 인퍼런스 모듈에 넣음 모듈은 output vector를 떨궈줌
4) findmusic : ouput vector와 기존의 벡터와 거리 계산해서 음악의 경로를 받음
5) getmusic : 해당 음악을 폴더에서 찾아서 download 폴더에 넣어줌

'''
# class Model:
#     def model(self):
#         from keras.applications.inception_v3 import InceptionV3
#         import keras
#         # from keras.layers import *
#         import tensorflow as tf
#         from keras.models import Model,Sequential
#         from keras.layers.wrappers import TimeDistributed
#         from keras import metrics
#         from keras import optimizers 
#         from keras import losses
#         from keras.utils import np_utils

#         print("111111111")
#         v_input_shape = (None, 150, 150, 3)
#         # m_input_shape = (1292,128)
#         v_input = Input(v_input_shape)
#         # m_input = Input(m_input_shape)

#         base_model = InceptionV3(weights=None, include_top=False, pooling='max')
#         inception_layer = TimeDistributed(base_model)(v_input)
#         Drop_layers = keras.layers.Dropout(0.3)(inception_layer)
#         inception_layer2 = TimeDistributed(Dense(128))(Drop_layers)
#         LSTM_layer = LSTM(242)(inception_layer2)
#         Drop_layers2 = keras.layers.Dropout(0.3)(LSTM_layer)
#         output = Dense(128)(Drop_layers2)
#         model = Model(v_input,output)

#         model.load_weights(os.getcwd() + '/media/wi_v_model_weight.h5', os.getcwd() + '/media/wi_m_model_weight.h5')
#         print("22222222")
#         return(model)

class Inference:
    # def __init__(self): #초기화를 이렇게 해주는 게 맞음??? 근데 이 경로는 최초의 영상파일 경로이고 뒤에는 경로가 30초컷, frame, 음악 위치 등으로 바뀌어야 되는데?->그래서 init에서는 변수를 받지 않음
    #     # self.path = path #경로
    #     # self.filename = filename #파일명
    #     # self.start_time = start_time
    #     pass

    #1) extract : upload 폴더의 영상으로 부터 frame 뽑음
    time.sleep(10)
    def extract(self, path, filename, start_time): # 프레임 뽑아 저장
        # 그 사용자의 path, 파일명, 시작시간인지 어떻게 아느냐?->pk를 받아온 후에 찾는다

        # 1.1) 원본 비디오 경로를 받아 30초로 컷
        video_list = [] # 비디오 확인
        video_title = [] 
        file_list = os.listdir(path)
        file_list.sort()
        for i in file_list:
            try: 
                if i == filename: # -> 들어온 이름의 비디오에 대해서만 씀
                    video_list.append(i)
                    i_split = i.split(".")
                    video_title.append(i_split[0])
                    print(video_list)
                    print(video_title)
                    print('Done')
            except Exception as e:
                print("Video check impossible")

        # video_time_list = [] # 길이 확인 
        # for video in video_list:
        #     result = subprocess.Popen('ffprobe -i "{}\\{}.mp4" -show_entries format=duration -v quiet -of csv="p=0"'.format(path, video), stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        #     output = result.communicate()
        #     # print(output[0])
        #     video_time = re.findall("\d+",str(output[0]))[0] #정규식, 정수단위의 초만 남김
        #     # print(video_time)
        #     video_time_list.append(video_time)
        # # file_list = os.listdir(path_dir)

        # 1.2) 원본을 30초 컷/소리 제거
        # make_path_cut = 'home/ubuntu/Notebook/cut/'
        make_path_cut = os.getcwd() + '/media/cut/'
        cut_path = make_path_cut+video_title[0]+'/'
        print(cut_path)
        pathlib.Path(cut_path).mkdir(parents=True, exist_ok=True) # 고유 cut_path 지정
        # path_save_cut = cut_path+filename.split(".")[:-1]+'/'
        # for idx, j in enumerate(video_time_list): # 영상 길이를 받아옴
        # order = int(j)//30 # 30초 단위로 잘라 떨어지지 않는 부분은 버림 - 몫
         # 시작 시간은 클래스 시작할 때 받아옴
        # if ':' in start_time:
        #     split_time = start_time.split(':') # 시:분:초 형식의 시간으로 받아오기 때문에(59초 이상)
        #     start_second = int(split_time[0])*3600 + int(split_time[1])*60 + int(split_time[2])
        # else:
        #     start_second = start_time
        command_v = 'ffmpeg -ss "{}" -t 30 -i "{}{}" -an "{}{}"'.format(start_time, path, video_list[0], cut_path, video_list[0]) # -ss 시작시간 / -t 시간 30초 / -i input 파일 / -an 오디오는 제거 / 마지막에 파일이 저장될 경로 지정해줌 // 영상의 vcodec을 넣으면 30초로 딱 잘리지 않아 부득이하게 삭제했더니 시간이 3배 이상 오래 걸림
        subprocess.call(command_v, shell=True) # shell에서 커맨드 실행해라
        print("command is   =======   ", command_v)

        # 1.3) 프레임 컷
        # 프레임이 저장될 폴더 만듦
        # make_path_frame = 'home/ubuntu/Notebook/frames/'
        make_path_frame = os.getcwd() + '/media/frame/'
        self.frame_path = make_path_frame + video_title[0]+'/'
        pathlib.Path(self.frame_path).mkdir(parents=True, exist_ok=True) # 고유 cut_path 지정
        # os.mkdirs(self.frame_path)

        for i in enumerate(video_list):
            command = 'ffmpeg -ss 00:00 -t 30 -i "{}{}" -r 8 -s 640*360 -qscale:v 2 -f image2  "{}/{}_%3d.jpg"'.format(cut_path, video_list[0], self.frame_path, video_title[0]) # frame 저장되는 경로
            #### frame 번호자동으로 붙는지 test ####
            subprocess.call(command, shell=True)
            print("command is   ===============================   ", command)
    

    #2) getinput : 이미지를 읽어서(cv2) 전처리(resize, reshape)하여 input vector를 만듦
    def read_cv2(self, model):
        import os
        import cv2
        
        
        os.chdir(self.frame_path)
        frame_list = os.listdir()
        frame_list.sort()

        img_info = []
        images = []
        image = []
        for idx,i in enumerate(frame_list):
            if 'jpg' in i:
                if idx < 242: #19360
                    img = cv2.imread('{}'.format(i), cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, dsize=(150, 150), interpolation=cv2.INTER_AREA) # 퍼센트로 줄이자
                    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) #scaling 방식 (255로 나누자)
                    img_info.append(img)
                    print(len(img))
                else:
                    break
        images = np.asarray(img_info)
        [image.append(images[i:i+242]) for i in range(0, len(images), 242)]
        image = np.asarray(image)
        # print(self.image)

        output_vector = model.predict(image)
        print(output_vector)
        return output_vector
        # self.output_vector = model._make_predict_function()
        # print("00000000000000000000000000000000000000")
        # with graph.as_default():
        #     self.output_vector = model.predict(self.image)
        # print("00000000000000000000000000000000000000")
        # return self.output_vector

    # def read_cv2(self, model):
    #     import os
    #     import cv2
    #     import numpy as np
        
    #     os.chdir(self.frame_path)
    #     frame_list = os.listdir()
    #     frame_list.sort()

    #     img_info = []
    #     images = []
    #     self.image = []
    #     for idx,i in enumerate(frame_list):
    #         if 'jpg' in i:
    #             if idx < 242: #19360
    #                 img = cv2.imread('{}'.format(i), cv2.IMREAD_COLOR)
    #                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #                 img = cv2.resize(img, dsize=(150, 150), interpolation=cv2.INTER_AREA) # 퍼센트로 줄이자
    #                 img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) #scaling 방식 (255로 나누자)
    #                 img_info.append(img)
    #                 print(len(img))
    #             else:
    #                 break
    #     images = np.asarray(img_info)
    #     [self.image.append(images[i:i+242]) for i in range(0, len(images), 242)]
    #     self.image = np.asarray(self.image)
    #     print(self.image)

    #     self.output_vector = model.predict(self.image)

     # #3) getoutput : 인퍼런스 모듈에 넣음 모듈은 output vector를 떨궈줌
    # def getoutput(self, image):
    #     # from keras.utils.training_utils import multi_gpu_model
    #     #  from keras.models import load_model
    #     # model = multi_gpu_model(model, gpus=8)
    #     print("11111111111111111111111111111111111111111111")
    #     print(self.image)
    #     self.output_vector = model.predict(self.image) #이 부분에 예측하는 거 넣어
    #     print("2222222222222222222222222222222222222222222")
    #     print(self.image)
    #     return(self.output_vector)
    
    
    
    #4) findmusic : ouput vector와 기존의 음악 벡터와 거리 계산해서 음악의 경로를 받음
    def findmusic(self, output_vector):
#         # 모델에서 거리를 쟀던 함수를 써서....? 음악과 음악의 거리를 잰다 -> mvectors 저장할 때 id화를 해서 번호로 저장하고 그 번호와 음악제목과 파일위치를 저장한 csv를 하나 만들어야 할 듯
        from numpy import genfromtxt
        from scipy.spatial import distance
        # path_mvectors = os.getcwd() + "/media/reference/music/" # 음악 csv 파일 있는 위치
        mvectors = np.genfromtxt('/home/ubuntu/test/eddie2/media/one_vec2.csv', delimiter=",",skip_header=1, usecols=range(1,129)) # 파일명은 음악 벡터 적힌 파일
        distlist = []
        for i in range(128):
            dist = distance.euclidean(output_vector, mvectors[i])
            print(dist)
            distlist.append(dist)

        select = distlist.index(min(distlist))
        print("4444444444")
        return select
        
        # mvectors = genfromtxt(path_mvectors+'m_vector', delimiter = ',')


    #5) getmusic : 해당 음악을 폴더에서 찾아서 download 폴더에 넣어줌
    def getmusic(self, select, video_name_text):
        import pandas as pd
        import shutil
        print(select)
        path_csv = '/home/ubuntu/test/eddie2/media/'
        music_list = pd.read_csv(path_csv + "one_title2.csv", header=None)
        print(music_list.head())
        selected = music_list.iloc[select,1]
        print(selected)

        music_path = '/home/ubuntu/eddie2/media/reference/music/' # 음악파일path
        download_path = '/home/ubuntu/test/eddie2/media/download/'

        newfilename = video_name_text.split('.')
        music_name = newfilename[0] + '.wav'
        print(music_path + selected)
        print(download_path + music_name)
        shutil.copy2(music_path + selected, download_path + music_name) # 이름을 영상 명과 일치하게 
        # shutil.copyfile(music_path + music_name, download_path)
        print("555555555")


# # if __name__ == "__main__":    
