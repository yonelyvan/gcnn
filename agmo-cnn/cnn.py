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

#vector_test =[]
#vector_result =[]
clases = ['0','1','2','3','4','5','6','7','8','9'] ##clases

m,n = 64,64 # dimensiones de la imagen de entrada

inf = 1e9
EPOCH = 25


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

#config [5,3,..,2] del cromosoma "11010..101" 
def get_cnn_architecture(shape, nb_classes, config): #individuo
	model= Sequential()

	model.add(Convolution2D(16,config[0],config[0],border_mode='same',input_shape= shape))
	model.add(Activation('relu'));
	model.add(Convolution2D(16, config[1],config[1] ));
	model.add(Activation('relu'));
	
	model.add(MaxPooling2D(pool_size=(config[2],config[2])));

	model.add(Dropout(0.5));
	model.add(Flatten());
	model.add(Dense(512));
	model.add(Dropout(0.5));
	model.add(Dense(nb_classes));#fullconected?
	model.add(Activation('softmax'));
	model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
	return model

def get_cnn_architecture2(shape, nb_classes, config): #individuo
	model= Sequential()

	model.add(Convolution2D(16,config[0],config[0],border_mode='same',input_shape= shape))
	model.add(Activation('relu'));
	model.add(Convolution2D(16, config[1],config[1] ));
	model.add(Activation('relu'));
	
	model.add(MaxPooling2D(pool_size=(config[2],config[2])));

	model.add(Convolution2D(16,config[3],config[3]));
	model.add(Activation('relu'));
	model.add(Convolution2D(16,config[4],config[4]));
	model.add(Activation('relu'));
	
	model.add(MaxPooling2D(pool_size=(config[5],config[5])));
	
	model.add(Dropout(0.5));
	model.add(Flatten());
	model.add(Dense(512));
	model.add(Dropout(0.5));
	model.add(Dense(nb_classes));#fullconected?
	model.add(Activation('softmax'));
	model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
	return model

def validar_configuracion(config):
	for i in config:
		if(i<1):
			return False
	return True


def train(config):
	[x,y,classes] = get_data()
	nb_classes=len(classes)
	print "Numero de clases",nb_classes
	
	x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2,random_state=4)
	uniques, id_train=np.unique(y_train,return_inverse=True)
	Y_train=np_utils.to_categorical(id_train,nb_classes)
	uniques, id_test=np.unique(y_test,return_inverse=True)
	Y_test=np_utils.to_categorical(id_test,nb_classes)
	model = get_cnn_architecture2(x_train.shape[1:],nb_classes, config);
	nb_epoch = EPOCH; ##iteraciones
	batch_size = 50;
	#time
	start_time = time.time()
	hist = model.fit(x_train,Y_train,
		batch_size=batch_size,
		nb_epoch=nb_epoch,
		verbose=1) #,
		#validation_data=(x_test, Y_test))
	

	tiempo = time.time() - start_time
	#guardar entrenamiento
	model_json = model.to_json()
	with open("models/model.json", "w") as json_file:
	    json_file.write(model_json)
	model.save_weights("models/model.h5")
	#print "acc: ", np.mean(hist.history["acc"])
	#print "loss: ", np.mean(hist.history["loss"])
	
	return tiempo


############################################################################################

#DATA PRUEBA
def cargar_imagenes_prueba():
	path_data = PATH+folder_datos_prueba
	# input image dimensions
	#m,n = 32,32
	x=[]
	vector_test = []
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
	return [x, vector_test]


def clasificar(x):
	file_json = 'models/model.json'
	file_h5 = 'models/model.h5'
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
	vector_result = []
	for i in range(len(pred)):
		m = max(pred[i])
		index =np.where(pred[i] >=m)[0][0]
		vector_result.append(clases[index])
	return vector_result



def test(num_clases):
	[x, vector_test] = cargar_imagenes_prueba()
	vector_result = clasificar(x)
	#matriz de confusion
	mc = np.zeros((num_clases,num_clases)) #numero de clases
	total_por_clase = np.zeros((num_clases)) #numero de imagenes por clase
	clases_num = get_clases();
	for i in range(len(vector_test)):
		img_class_label = get_class_from_name(vector_test[i])
		result_label = vector_result[i]
		mc[ clases_num[img_class_label], clases_num[result_label]] +=1
		total_por_clase[ clases_num[img_class_label] ] +=1
		#print "imagen: ",get_class_from_name(vector_test[i])," -> ", vector_result[i]
		
	#print mc
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
	print "Promedio: ", promedio, "%"
	error = 100.0-promedio
	return error



def get_class_from_name(filename):
	r = ""
	for i in filename:
		if i=="_":
			break
		r+=i
	return r


def get_clases():
	clases_num = {
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
	return clases_num


def run_cnn( config ):
	num_clases = 10
	time = inf
	if validar_configuracion(config):
		time = train( config )
		return [inf, inf]
	error = test(num_clases)
	return [error, time]



#fitness = run([3,3,5])
#print "FITENESS: error, tiempo"
#print fitness