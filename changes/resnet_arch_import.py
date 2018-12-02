from __future__ import print_function
import numpy as np
import numpy as np
import pandas as pd
from keras import backend as K
# from keras.applications.resnet50 import ResNet50
import keras.applications.resnet50 as rncnn
#from keras import applications
from keras.applications.resnet50 import identity_block, conv_block
from keras.layers import (Activation, BatchNormalization, Convolution2D, Dense,
                          Flatten, Input, MaxPooling2D, ZeroPadding2D)
from keras.models import Model
from keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical

# get the data
filname = '/content/drive/My Drive/Facial_Expression_Recognition/fer2013.csv'
label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

print(dir(rncnn))

def getData(filname):
    # images are 48x48
    # N = 35887
    Y = []
    X = []
    first = True
    
    with open("/content/drive/My Drive/Facial_Expression_Recognition/badtrainingdata.txt", "r") as text:
        ToBeRemovedTrainingData = []
        for line in text:
            ToBeRemovedTrainingData.append(int(line))
    number = 0
    
    for line in open(filname):
        number+= 1
        if number not in ToBeRemovedTrainingData:
          if first:
              first = False
          else:
              row = line.split(',')
              Y.append(int(row[0]))
              X.append([int(p) for p in row[1].split()])

    X, Y = np.array(X) / 255.0, np.array(Y)
    return X, Y


X, Y = getData(filname)
num_class = len(set(Y))

# To see number of training data point available for each label
def balance_class(Y):
    num_class = set(Y)
    count_class = {}
    for i in range(len(num_class)):
        count_class[i] = sum([1 for y in Y if y == i])
    return count_class

balance = balance_class(Y)

# keras with tensorflow backend
N, D = X.shape
X = X.reshape(N, 48, 48, 1)

# Split in  training set : validation set :  testing set in 80:10:10
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
y_train = (np.arange(num_class) == y_train[:, None]).astype(np.float32)
y_test = (np.arange(num_class) == y_test[:, None]).astype(np.float32)




# define some variables
SHAPE = (48, 48, 1)
bn_axis = 3 if K.image_dim_ordering() == 'tf' else 1


def baseline_model(seed=None):
    # We can't use ResNet50 directly, as it might cause a negative dimension
    # error.
    if seed:
        np.random.seed(seed)

    input_layer = Input(shape=SHAPE)

    x = ZeroPadding2D((3, 3))(input_layer)
    x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    """
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    print(x)
    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    """

    x = Flatten()(x)
    x = Dense(10, activation='softmax', name='fc10')(x)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[categorical_accuracy])
    model = Model(input_layer, x)

    return model

def baseline_model_saved():
  #load json and create model
  json_file = open('/content/drive/My Drive/ECS 289 Deep Learning/model_4layer_2_2_pool.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  model = model_from_json(loaded_model_json)
  #load weights from h5 file
  model.load_weights("/content/drive/My Drive/ECS 289 Deep Learning/model_4layer_2_2_pool.h5")
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[categorical_accuracy])
  return mode

# is_model_saved = True # Indicates that model is saved
is_model_saved = False # Indicates that model is not saved, so you have to train it

# If model is not saved train the CNN model otherwise just load the weights
if(is_model_saved==False ):
  # Train model
    model = baseline_model()
    # Note : 3259 samples is used as validation data &   28,709  as training samples

    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_split=0.1111)
    model_json = model.to_json()
    with open("/content/drive/My Drive/ECS 289 Deep Learning/model_4layer_2_2_pool.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("/content/drive/My Drive/ECS 289 Deep Learning/model_4layer_2_2_pool.h5")
    print("Saved model to disk")
else:
    # Load the trained model
    print("Load model from disk")
    model = baseline_model_saved()

# Model will predict the probability values for 7 labels for a test image
score = model.predict(X_test)
print (model.summary())

new_X = [ np.argmax(item) for item in score ]
y_test2 = [ np.argmax(item) for item in y_test]

# Calculating categorical accuracy taking label having highest probability
accuracy = [ (x==y) for x,y in zip(new_X,y_test2) ]
print(" Accuracy on Test set : " , np.mean(accuracy))
