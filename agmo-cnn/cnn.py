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
from PIL import ImageOps
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

import time


K.set_image_dim_ordering('th')
print "-------------------------------------------\n"

PATH = os.getcwd();  #/home/y/.../gcnn/train"
folder_datos_prueba = '/test' 

#vector_test =[]
#vector_result =[]
#clases = ['n_0','n_1','n_2','n_3','n_4','n_5','n_6','n_7','n_8','n_9']
#clases = ['c_0','c_1','c_2','c_3','c_4','c_5','c_6','c_7','c_8','c_9','c_10','c_11','c_12','c_13','c_14','c_15','c_16','c_17','c_18','c_19','c_20','c_21','c_22','c_23','c_24','c_25','c_26','c_27','c_28','c_29','c_30','c_31','c_32','c_33','c_34','c_35','c_36','c_37']
#clases = ['c_0','c_1','c_2','c_3','c_4','c_5','c_6','c_7','c_8','c_9','c_10','c_11','c_12','c_13','c_14']


m,n = 128, 128 #256,256 # dimensiones de la imagen de entrada
EPOCH = 20



class DATA:
	def __init__(self):
		self.train_x = None
		self.train_y = None
		self.train_clases =  None
		self.test_tx = None
		self.test_ty = None
		self.test_clases = None
		self.num_clases = None

	def get_data(self):
		folder_data = "train"; #carpeta de dataset
		#m,n = 64,64 # dimensiones de la imagen de entrada
		fold_clases=os.listdir(folder_data) #cada carpeta es una CLASE
		fold_clases.sort()
		self.train_x=[]
		self.train_y=[]
		for fol in fold_clases:##para cada clase fol==[class 1,class 2, .., class N]
			imgfiles=os.listdir(folder_data+'/'+fol)
			for img in imgfiles:##para cada imagen
				im=Image.open(folder_data+'/'+fol+'/'+img)
				im=im.convert(mode='RGB') ##  --
				#im=ImageOps.equalize(im)##
				#im=ImageOps.autocontrast(im, cutoff=0.8)
				imrs=im.resize((m,n))
				imrs=img_to_array(imrs)/255
				imrs=imrs.transpose(2,0,1)
				imrs=imrs.reshape(3,m,n)
				self.train_x.append(imrs)
				self.train_y.append(fol)
		self.train_x=np.array(self.train_x)
		self.train_y=np.array(self.train_y)
		self.train_clases = fold_clases
		self.num_clases = len(self.train_clases) 
		print self.train_clases


	def cargar_imagenes_prueba(self):
		folder_data =  PATH+folder_datos_prueba #"data_num"; #carpeta de dataset
		#m,n = 64,64 # dimensiones de la imagen de entrada
		fold_clases=os.listdir(folder_data) #cada carpeta es una CLASE
		fold_clases.sort()
		self.test_tx=[] #image
		self.test_ty=[] #clase
		for fol in fold_clases:##para cada clase fol==[class 1,class 2, .., class N]
			imgfiles=os.listdir(folder_data+'/'+fol);
			for img in imgfiles:##para cada imagen
				im=Image.open(folder_data+'/'+fol+'/'+img);
				im=im.convert(mode='RGB') ##  --
				#im=ImageOps.equalize(im)##
				#im=ImageOps.autocontrast(im, cutoff=0.8)
				imrs=im.resize((m,n))
				imrs=img_to_array(imrs)/255;
				imrs=imrs.transpose(2,0,1);
				imrs=imrs.reshape(3,m,n);
				self.test_tx.append(imrs)
				self.test_ty.append(fol)
				#print "labeli:",fol
		self.test_tx=np.array(self.test_tx);
		self.test_ty=np.array(self.test_ty);
		self.test_clases = fold_clases
		print self.test_clases

def get_cnn_architecture(shape, nb_classes, config): #individuo
	model= Sequential()

	model.add(Convolution2D(16,config[0],config[0],border_mode='same',input_shape= shape))
	model.add(Activation('relu'));
	model.add(Convolution2D(16, config[1],config[1], border_mode='same'));
	model.add(Activation('relu'));
	
	model.add(MaxPooling2D(pool_size=(config[2],config[2])));

	model.add(Convolution2D(16,config[3],config[3], border_mode='same'));
	model.add(Activation('relu'));
	model.add(Convolution2D(16,config[4],config[4], border_mode='same'));
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




def train(config, data):
	print "Numero de clases",data.num_clases
	
	x_train, x_test, y_train, y_test= train_test_split(data.train_x,data.train_y,test_size=0.2,random_state=4)
	uniques, id_train=np.unique(y_train,return_inverse=True)
	Y_train=np_utils.to_categorical(id_train,data.num_clases)
	uniques, id_test=np.unique(y_test,return_inverse=True)
	Y_test=np_utils.to_categorical(id_test,data.num_clases)
	model = get_cnn_architecture(x_train.shape[1:],data.num_clases, config);
	nb_epoch = EPOCH; ##iteraciones
	batch_size = 100;
	#time
	start_time = time.time()
	hist = model.fit(x_train,Y_train,
		batch_size=batch_size,
		nb_epoch=nb_epoch,
		verbose=1,
		validation_data=(x_test, Y_test))
	

	tiempo = time.time() - start_time
	#guardar entrenamiento
	model_json = model.to_json()
	with open("models/model.json", "w") as json_file:
	    json_file.write(model_json)
	model.save_weights("models/model.h5")
	#print "acc: ", np.mean(hist.history["acc"])
	#print "loss: ", np.mean(hist.history["loss"])
	
	return tiempo



def clasificar(data):
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
	predictions = loaded_model.predict(data.test_tx)
	pred = predictions*100
	#print (pred).astype(int)
	vector_result = []
	print "Predicciones"
	for i in range(len(pred)):
		max_pre = max(pred[i])
		index =np.where(pred[i] >=max_pre)[0][0]
		vector_result.append(data.test_clases[index])
	return vector_result



def test(data):
	vector_result = clasificar(data)
	#matriz de confusion
	mc = np.zeros((data.num_clases,data.num_clases)) #numero de clases
	total_por_clase = np.zeros((data.num_clases)) #numero de imagenes por clase
	clases_num = get_clases();
	for i in range(len(data.test_ty)):
		img_class_label = data.test_ty[i]
		result_label = vector_result[i]
		#label class to number class
		#print "NUMEROS: ",img_class_label,result_label,clases_num[img_class_label], clases_num[result_label]
		mc[ clases_num[img_class_label], clases_num[result_label] ] +=1
		total_por_clase[ clases_num[img_class_label] ] +=1
		#print i,")","imagen: ",get_class_from_name(y[i])," -> ", vector_result[i]
		
	print mc
	print total_por_clase
	#prcentages por clase
	sum = 0.0
	for i in range(total_por_clase.size): 
		value = mc[i,i]
		img_class_label = data.test_clases[i]
		porcentaje = (100.0*value)/total_por_clase[i]
		print img_class_label, " : ", porcentaje, "%"
		sum += porcentaje
	promedio = sum/data.num_clases
	#print "Numero de clases(10):",data.num_clases
	print "Promedio: ", promedio, "%"
	error = 100.0-promedio
	return error


def get_clases():
	
	'''
	clases_num = {
		"n_0" : 0,
		"n_1" : 1,
		"n_2" : 2,
		"n_3" : 3,
		"n_4" : 4,
		"n_5" : 5,
		"n_6" : 6,
		"n_7" : 7,
		"n_8" : 8,
		"n_9" : 9,
	}
	clases_num = {
		'c_0' : 0, 'c_1' : 1, 'c_2' : 2, 'c_3' : 3, 'c_4' : 4, 
		'c_5' : 5, 'c_6' : 6, 'c_7' : 7, 'c_8' : 8, 'c_9' : 9, 
		'c_10' : 10,'c_11' : 11,'c_12' : 12,'c_13' : 13,'c_14' : 14,
		'c_15' : 15,'c_16' : 16, 'c_17' : 17, 'c_18' : 18,'c_19' : 19,
		'c_20' : 20,'c_21' : 21,'c_22' : 22,'c_23' : 23,'c_24' : 24,
		'c_25' : 25,'c_26' : 26,'c_27' : 27,'c_28' : 28,'c_29' : 29,
		'c_30' : 30,'c_31' : 31,'c_32' : 32,'c_33' : 33,'c_34' : 34,
		'c_35' : 35,'c_36' : 36,'c_37' : 37,
	}
	'''

	clases_num = {
		'c_0' : 0, 'c_1' : 1, 'c_2' : 2, 'c_3' : 3, 'c_4' : 4, 
		'c_5' : 5, 'c_6' : 6, 'c_7' : 7, 'c_8' : 8, 'c_9' : 9, 
		'c_10' : 10,'c_11' : 11,'c_12' : 12,'c_13' : 13,'c_14' : 14
	}
	
	
	return clases_num


def run_cnn( config, data ):
	#data = DATA()
	#data.get_data()
	#data.cargar_imagenes_prueba()

	print "Training"
	tiempo = train(config, data)
	error = test(data)
	
	print "SLEEP 20s"
	time.sleep(20);

	os.remove("models/model.json")
	os.remove("models/model.h5")
	return [error, tiempo]



def run_cnn_arq(config, data):
	#data = DATA()
	#data.get_data()
	#data.cargar_imagenes_prueba()

	print "Training"
	tiempo = train(config, data)
	error = test(data)

	return [error, tiempo]
