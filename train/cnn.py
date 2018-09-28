from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import  img_to_array
from keras import backend as K
K.set_image_dim_ordering('th')
import keras
import numpy as np
import os
from PIL import Image
from sklearn.cross_validation import train_test_split

PATH = os.getcwd(); #"/home/y/.../gcnn/train"
os.chdir(PATH);

def get_data():
	folder_data = "data_num"; #carpeta de dataset
	m,n = 32,32 # dimensiones de la imagen de entrada
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
	model.add(MaxPooling2D(pool_size=(2,2)));
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
	nb_epoch=50; ##iteraciones
	batch_size=100;
	model.fit(x_train,Y_train,batch_size=batch_size,nb_epoch=nb_epoch,verbose=1,validation_data=(x_test, Y_test))
	#guardar entrenamiento
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
	    json_file.write(model_json)
	model.save_weights("model.h5")


train();




































#################################modelo 2
'''
model = Sequential()
model.add(Convolution2D(16,3,3,border_mode='same',input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Convolution2D(16, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(32, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
'''
#################################
