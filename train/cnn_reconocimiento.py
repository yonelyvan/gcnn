from config import *
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import  img_to_array
from keras import backend as K
import numpy as np
import os
from PIL import Image
from sklearn.cross_validation import train_test_split
K.set_image_dim_ordering('th')

def cargar_placa_fragmentada(carpeta):
	os.chdir(path+"proyecto-IA/temp");
	path1=carpeta;#carpeta
	# input image dimensions
	m,n = 20,30
	x=[]
	#imgfiles=os.listdir(path1);
	#imgfiles.sort()
	import natsort
	imgfiles = natsort.natsorted(os.listdir(path1))
	#print imgfiles
	for img in imgfiles:##para cada imagen
	        im=Image.open(path1+'/'+img);
	        im=im.convert(mode='RGB')
	        imrs=im.resize((m,n))
	        imrs=img_to_array(imrs)/255;
	        imrs=imrs.transpose(2,0,1);
	        imrs=imrs.reshape(3,m,n);
	        x.append(imrs)
	x=np.array(x);
	return x

def predecir(x):
	os.chdir(path+"/proyecto-IA/temp");
	# load json and create model
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("model.h5")
	# evaluate loaded model on test data
	loaded_model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

	predictions = loaded_model.predict(x)
	pred = predictions*100
	#print (pred).astype(int)
	clases = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z']
	label_placa = ''
	for i in range(len(pred)):
		m = max(pred[i])
		index =np.where(pred[i] >=m)[0][0]
		if m >50:#si tiene mas de 35% de pertenencia 
			label_placa+=clases[index]
	return label_placa


#x = cargar_placa_fragmentada("p1")
#predecir(x)

