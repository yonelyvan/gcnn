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
print "___________________________________________"
K.set_image_dim_ordering('th')

path = "/home/y/Desktop/gcnn/train";

vector_test =[]
vector_result =[]

def cargar_imagenes_prueba(carpeta):
	path1='/home/y/Desktop/gcnn/train/temp' #path+carpeta;#carpeta
	# input image dimensions
	m,n = 32,32
	x=[]
	import natsort
	imgfiles = natsort.natsorted(os.listdir(path1))
	#print imgfiles
	for img in imgfiles:##para cada imagen
		#print img
		vector_test.append(img)
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
	os.chdir(path+"/temp");
	# load json and create model
	json_file = open('/home/y/Desktop/gcnn/train/model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("/home/y/Desktop/gcnn/train/model.h5")
	# evaluate loaded model on test data
	loaded_model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
	predictions = loaded_model.predict(x)
	pred = predictions*100
	#print (pred).astype(int)
	clases = ['c_0','c_1','c_2','c_3','c_4','c_5','c_6','c_7','c_8','c_9'] ##clases
	label_placa = ''
	for i in range(len(pred)):
		m = max(pred[i])
		index =np.where(pred[i] >=m)[0][0]
		if m >50:#si tiene mas de 35% de pertenencia 
			#print "clase de pertenencia:", clases[index]
			vector_result.append(clases[index])
	return label_placa



x = cargar_imagenes_prueba("/temp")
predecir(x)
for i in range(len(vector_test)):
	print "imagen: ",vector_test[i]," -> clasificacion;", vector_result[i] 