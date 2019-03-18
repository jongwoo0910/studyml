from django.shortcuts import render, get_object_or_404, redirect
from .models import want_time
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse
from eddie.models import Document
from eddie.forms import DocumentForm
from eddie.infer_module import Inference
import os
import time

UPLOAD_DIR = os.getcwd() + '/media/uploads/' # C:\Users\Playdata\py_projects\main\test/uploads/

# 모델 미리 띄워놓기#############################################################################

from keras.applications.inception_v3 import InceptionV3
import keras
from keras.layers import *
import tensorflow as tf
from keras.models import Model,Sequential
from keras.layers.wrappers import TimeDistributed
from keras import metrics
from keras import optimizers 
from keras import losses
from keras.utils import np_utils


print("111111111")
v_input_shape = (None, 150, 150, 3)
# m_input_shape = (1292,128)
v_input = Input(v_input_shape)
# m_input = Input(m_input_shape)

base_model = InceptionV3(weights=None, include_top=False, pooling='max')
inception_layer = TimeDistributed(base_model)(v_input)
Drop_layers = keras.layers.Dropout(0.3)(inception_layer)
inception_layer2 = TimeDistributed(Dense(128))(Drop_layers)
LSTM_layer = LSTM(242)(inception_layer2)
Drop_layers2 = keras.layers.Dropout(0.3)(LSTM_layer)
output = Dense(128)(Drop_layers2)
model = Model(v_input,output)

model.load_weights(os.getcwd() + '/media/wi_v_model_weight.h5', os.getcwd() + '/media/wi_m_model_weight.h5')
model._make_predict_function()
# graph = tf.get_default_graph()
print("22222222")

###############################################################################################

def index(request):
    return render(request, 'index.html')

def update(request):
    starttime = request.POST['starttime']
    endtime = request.POST['endtime']
    video_title = request.POST['file_title']
    video_title_cut = video_title[0:11]
    a = video_title.rfind('\\') + 1
    video_title = video_title[a:]
    if 'docfile' in request.FILES:
        file = request.FILES['docfile']
        filename = video_title_cut + file._name
        fp = open('%s/%s' % (UPLOAD_DIR, filename) , 'wb')
        for chunk in file.chunks():
            fp.write(chunk)
        fp.close()
    b = want_time(video_name_text=video_title, start_time_text=starttime, end_time_text=endtime)
    b.save()
    pk = b.pk
    return redirect('./download/' + str(pk))

def download(request, pk):
    data = want_time.objects.get(pk=pk)
    video_name_text = data.video_name_text
    starttime = data.start_time_text
    end_time_text = data.end_time_text
    video_name_text2 = video_name_text.split('.')
    music_name_text = video_name_text2[0] + '.wav'
    return render(request, 'down.html', {'filename':music_name_text, 'pk':pk})

def ajax_api(request, pk):
    data = want_time.objects.get(pk=pk)
    video_name_text = data.video_name_text
    video_name_text2 = video_name_text.split('.')
    # print("split==============", video_name_text2)
    music_name_text = video_name_text2[0] + '.wav'
    print(music_name_text)
    # print("music_name============", music_name_text)
    # print(os.getcwd()) # C:\Users\Playdata\py_projects\main\test2
    # upfile = os.getcwd() + '/media/download/' + video_name_text
    upfile = '/home/ubuntu/test/eddie2/media/download/' + music_name_text
    # print(upfile)
    ok = 'false'
    if os.path.exists(upfile):
        ok = 'true'
    return HttpResponse(ok)

def ajax_infer(request, pk):
    data = want_time.objects.get(pk=pk)
    video_name_text = data.video_name_text
    starttime = data.start_time_text
    infer = Inference()
    infer.extract(UPLOAD_DIR, video_name_text, starttime)
    global model
    output_vector = infer.read_cv2(model)
    select = infer.findmusic(output_vector)
    infer.getmusic(select, video_name_text)
    # infer.read_cv2()
    return HttpResponse(infer)

def list(request):
    if request.method == 'POST':
        print(request.POST)
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            newdoc = Document(docfile = request.FILES['docfile'])
            newdoc.save()
    else:
        form = DocumentForm()
    documents = Document.objects.all()
    return render(request, 'index.html',{'documents': documents, 'form': form})
