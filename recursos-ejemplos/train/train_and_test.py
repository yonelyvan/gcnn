# entrenamiento y prueba
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import model_from_json
from keras.utils import np_utils
from keras.preprocessing.image import  img_to_array
from keras import backend as K
import keras
import numpy as np
import os
from PIL import Image
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

import time

K.set_image_dim_ordering('th')
print "-------------------------------------------\n"

PATH = os.getcwd();  #/home/y/.../gcnn/train"
folder_datos_prueba = '/temp' 

vector_test =[]
vector_result =[]
clases = ['0','1','2','3','4','5','6','7','8','9'] ##clases

m,n = 64,64 # dimensiones de la imagen de entrada
#DATA ENTRENAMIENTO
def get_data():
	folder_data = "data_num"; #carpeta de dataset
	#m,n = 64,64 # dimensiones de la imagen de entrada
	classes=os.listdir(folder_data) #cada carpeta es una CLASE
	x=[]
	y=[]
	for fol in classes:##para cada clase fol==[class 1,class 2, .., class N]
		imgfiles=os.listdir(folder_data+'/'+fol);
		for img in imgfiles:##para cada imagen
			im=Image.open(folder_data+'/'+fol+'/'+img);
			im=im.convert(mode='RGB') ##  --
			imrs=im.resize((m,n))
			imrs=img_to_array(imrs)/255;
			imrs=imrs.transpose(2,0,1);
			imrs=imrs.reshape(3,m,n);
			x.append(imrs)
			y.append(fol)
	x=np.array(x);
	y=np.array(y);
	return [x,y,classes];

def get_cnn_model(shape, nb_classes): #individuo
	model= Sequential()

	model.add(Convolution2D(16,3,3,border_mode='same',input_shape= shape))
	model.add(Activation('relu'));
	model.add(Convolution2D(16,3,3));
	model.add(Activation('relu'));
	
	model.add(MaxPooling2D(pool_size=(5,5)));
	'''
	model.add(Convolution2D(16,1,1));
	model.add(Activation('relu'));
	model.add(Convolution2D(16,3,3));
	model.add(Activation('relu'));
	model.add(MaxPooling2D(pool_size=(3,3)));
	'''
	model.add(Dropout(0.5));
	model.add(Flatten());
	model.add(Dense(512));
	model.add(Dropout(0.5));
	model.add(Dense(nb_classes));#fullconected?
	model.add(Activation('softmax'));
	model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
	return model


def train():
	[x,y,classes] = get_data()
	nb_classes=len(classes)
	print "Numero de clases",nb_classes
	
	x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2,random_state=4)
	uniques, id_train=np.unique(y_train,return_inverse=True)
	Y_train=np_utils.to_categorical(id_train,nb_classes)
	uniques, id_test=np.unique(y_test,return_inverse=True)
	Y_test=np_utils.to_categorical(id_test,nb_classes)
	model = get_cnn_model(x_train.shape[1:],nb_classes);
	nb_epoch=20; ##iteraciones
	batch_size=50;
	model.fit(x_train,Y_train,batch_size=batch_size,nb_epoch=nb_epoch,verbose=1,validation_data=(x_test, Y_test))
	#guardar entrenamiento
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
	    json_file.write(model_json)
	model.save_weights("model.h5")


############################################################################################

#DATA PRUEBA
def cargar_imagenes_prueba():
	path_data = PATH+folder_datos_prueba
	# input image dimensions
	#m,n = 32,32
	x=[]
	#import natsort #natsort.natsorted(os.listdir(path_data)) #para ordenar por nombre de archivo
	imgfiles = os.listdir(path_data) 
	#print imgfiles
	for img in imgfiles:##para cada imagen
		#print img
		vector_test.append(img)
		im=Image.open(path_data+'/'+img);
		im=im.convert(mode='RGB')## --
		imrs=im.resize((m,n))
		imrs=img_to_array(imrs)/255;
		imrs=imrs.transpose(2,0,1);
		imrs=imrs.reshape(3,m,n);
		x.append(imrs)
	x=np.array(x);
	return x


def predecir(x):
	file_json = 'model.json'
	file_h5 = 'model.h5'
	# load json and create model
	json_file = open(PATH+'/'+file_json, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights(PATH+'/'+file_h5)
	# evaluate loaded model on test data
	loaded_model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
	predictions = loaded_model.predict(x)
	pred = predictions*100
	#print (pred).astype(int)
	label_img = ''
	for i in range(len(pred)):
		m = max(pred[i])
		index =np.where(pred[i] >=m)[0][0]
		vector_result.append(clases[index])
	return label_img



def test():
	x = cargar_imagenes_prueba()
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
		
	print mc
	print total_por_clase
	#prcentages por clase
	sum = 0.0
	for i in range(total_por_clase.size): 
		value = mc[i,i]
		img_class_label = clases[i]
		porcentaje = (100.0*value)/total_por_clase[i]
		print img_class_label, " : ", porcentaje, "%"
		sum += porcentaje
	promedio = sum/num_clases
	print "Promedio: ", promedio
	#return promedio



def get_class_from_name(filename):
	r = ""
	for i in filename:
		if i=="_":
			break
		r+=i
	return r


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
	return clases






train();
print "Entrenamiento terminado"
time.sleep(1.5);

test()









