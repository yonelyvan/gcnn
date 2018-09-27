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
os.chdir("/home/ynl/Desktop/gcnn/train");





def train():
	# input image dimensions
	m,n = 32,32
	path2="data_num";

	classes=os.listdir(path2)
	x=[]
	y=[]
	for fol in classes:##para cada clase fol==[A,B..2]
		imgfiles=os.listdir(path2+'/'+fol);
		for img in imgfiles:##para cada imagen
			im=Image.open(path2+'/'+fol+'/'+img);
			im=im.convert(mode='RGB')
			imrs=im.resize((m,n))
			imrs=img_to_array(imrs)/255;
			imrs=imrs.transpose(2,0,1);
			imrs=imrs.reshape(3,m,n);
			x.append(imrs)
			y.append(fol)
	x=np.array(x);
	y=np.array(y);

	nb_classes=len(classes)
	print "numero de clases",nb_classes
	#nb_filters=32 nb_conv=3
	x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2,random_state=4)

	uniques, id_train=np.unique(y_train,return_inverse=True)
	Y_train=np_utils.to_categorical(id_train,nb_classes)

	uniques, id_test=np.unique(y_test,return_inverse=True)
	Y_test=np_utils.to_categorical(id_test,nb_classes)
	################################# modelo 1
	model= Sequential()
	model.add(Convolution2D(16,3,3,border_mode='same',input_shape=x_train.shape[1:]))
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
	#################################
	nb_epoch=50; ##iteraciones
	batch_size=100;
	model.fit(x_train,Y_train,batch_size=batch_size,nb_epoch=nb_epoch,verbose=1,validation_data=(x_test, Y_test))

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
