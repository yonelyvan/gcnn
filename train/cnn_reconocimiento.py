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

path = "/home/ynl/Desktop/gcnn/train";

vector_test =[]
vector_result =[]
clases = ['0','1','2','3','4','5','6','7','8','9'] ##clases

def cargar_imagenes_prueba(carpeta):
	path1='/home/ynl/Desktop/gcnn/train/temp' #path+carpeta;#carpeta
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
	json_file = open('/home/ynl/Desktop/gcnn/train/model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("/home/ynl/Desktop/gcnn/train/model.h5")
	# evaluate loaded model on test data
	loaded_model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
	predictions = loaded_model.predict(x)
	pred = predictions*100
	#print (pred).astype(int)
	
	label_placa = ''
	for i in range(len(pred)):
		m = max(pred[i])
		index =np.where(pred[i] >=m)[0][0]
		vector_result.append(clases[index])
		#if m >40:#si tiene mas de 35% de pertenencia 
			#print "clase de pertenencia:", clases[index]
	return label_placa



def test():
	x = cargar_imagenes_prueba("/temp")
	predecir(x)
	#matriz de confusion
	num_clases = 10
	mc = np.zeros((num_clases,num_clases)) #numeor de clases
	total_por_clase = np.zeros((num_clases)) #numero de imagenes por clase
	clases_num = get_clases();
	for i in range(len(vector_test)):
		img_class_label = get_class_from_name(vector_test[i]);
		result_label = vector_result[i];
		mc[ clases_num[img_class_label], clases_num[result_label]] +=1
		total_por_clase[ clases_num[img_class_label] ] +=1
		#print "imagen: ",get_class_from_name(vector_test[i])," -> ", vector_result[i]
		
	#print mc
	print total_por_clase
	#prcentages por clase
	for i in range(total_por_clase.size): 
		value = mc[i,i]
		img_class_label = clases[i]
		print img_class_label, " : ", (100.0*value)/total_por_clase[i], "%";



def get_class_from_name(filename):
	r = "";
	for i in filename:
		if i=="_":
			break;
		r+=i;
	return r;


def get_clases():
	clases = {
		"0" : 0,
		"1" : 1,
		"2" : 2,
		"3" : 3,
		"4" : 4,
		"5" : 5,
		"6" : 6,
		"7" : 7,
		"8" : 8,
		"9" : 9,
	}
	return clases;

test()
