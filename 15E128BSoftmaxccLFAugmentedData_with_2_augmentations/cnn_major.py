# To Run on CoLab Notebook, this mounts the google drive
# from google.colab import drive
# drive.mount('/content/drive')

from __future__ import print_function
import numpy as np

# get the data
filname = '/content/drive/My Drive/ECS 289 Deep Learning/newfer2013.csv'
label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def getData(filname):
    # images are 48x48
    # N = 35887
    Y = []
    X = []
    first = True
    
#     with open("/content/drive/My Drive/ECS 289 Deep Learning/badtrainingdata.txt", "r") as text:
#         ToBeRemovedTrainingData = []
#         for line in text:
#             ToBeRemovedTrainingData.append(int(line))
#     number = 0
    
    for line in open(filname):
#         number+= 1
#         if number not in ToBeRemovedTrainingData:
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])
        # print("{} {}".format(Y, X))

    X, Y = np.array(X) / 255.0, np.array(Y)
    return X, Y


X, Y = getData(filname)
num_class = len(set(Y))

# To see number of training data point available for each label
# def balance_class(Y):
    # num_class = set(Y)
    # count_class = {}
    # for i in range(len(num_class)):
        # count_class[i] = sum([1 for y in Y if y == i])
    # return count_class

# balance = balance_class(Y) # array that contains number of data points for each class

# keras with tensorflow backend
N, D = X.shape # N = 35887, D = 2304 (48*48)
X = X.reshape(N, 48, 48, 1) # Reshaping X as a list of list of list of list

# Split in  training set : validation set :  testing set in 80:10:10
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0) # Splitting data into training and testing
y_train = (np.arange(num_class) == y_train[:, None]).astype(np.float32) # Each intance in Y_train is converted into a 8*1 vector
y_test = (np.arange(num_class) == y_test[:, None]).astype(np.float32) # Each intance in Y_test is converted into a 8*1 vector


from keras.models import Sequential
from keras.layers import Dense , Activation , Dropout ,Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.metrics import categorical_accuracy
from keras.models import model_from_json

from keras.optimizers import *
from keras.layers.normalization import BatchNormalization

batch_size = 128
epochs = 15

#Shallow CNN model with two Convolution layer & one fully connected layer
def baseline_model():
    # Initialising the CNN
    model = Sequential()

    # 1 - Convolution
    model.add(Conv2D(64,(3,3), border_mode='same', input_shape=(48, 48,1))) # Convo layer has 64, 3*3 kernals and the output dimention is 48*48*64 (since border_mode='same' instead of valid)
    model.add(BatchNormalization()) # Normailizes out of the Conv layer such that mean activation is close to 0 and activation standard deviation is close to 1 (Makes training faster)
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) # Output size is now 47*47*64
    model.add(Dropout(0.25)) # For each training example, it drops 25% of the neurons

    # 2nd Convolution layer
    model.add(Conv2D(128,(5,5), border_mode='same')) # Convo layer has 128, 5*5 kernals and the output dimention is 47*47*128 (since border_mode='same' instead of valid)
    model.add(BatchNormalization()) # Normailizes out of the Conv layer such that mean activation is close to 0 and activation standard deviation is close to 1 (Makes training faster)
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) # Output size is now 46*46*128
    model.add(Dropout(0.25)) # For each training example, it drops 25% of the neurons


    # Flattening
    model.add(Flatten()) #Converts output from 46*46*128 into 294910*1 

    # Fully connected layer 1st layer
    model.add(Dense(256)) # FC has 256 neurons (294910*256 parameters)
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(num_class, activation='softmax')) # Output layer has 8 neurons (one for each class) (256*8 parameters)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[categorical_accuracy])
    return model


def baseline_model_saved(): # Executed if a trained model is saved
    #load json and create model
    json_file = open('/content/drive/My Drive/ECS 289 Deep Learning/model_2layer_2_2_pool.json', 'r') # Opens file
    loaded_model_json = json_file.read() # Read from file
    json_file.close() # Close file
    model = model_from_json(loaded_model_json) # Load saved model
    #load weights from h5 file
    model.load_weights("/content/drive/My Drive/ECS 289 Deep Learning/model_2layer_2_2_pool.h5")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[categorical_accuracy]) # Model is trained with adam optimizer and binary cross entropy loss fuction 
    return model

is_model_saved = False # Indicates that model is saved
# is_model_saved = False # Indicates that model is not saved, so you have to train it

# If model is not saved train the CNN model otherwise just load the weights
if(is_model_saved==False ): 
    # Train model
    model = baseline_model() # Loads model specification
    # Note : 3259 samples is used as validation data &   28,709  as training samples

    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_split=0.1111) # Trains model
    model_json = model.to_json() # Writes model to json file
    with open("/content/drive/My Drive/ECS 289 Deep Learning/model_2layer_2_2_pool.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("/content/drive/My Drive/ECS 289 Deep Learning/model_2layer_2_2_pool.h5") # Saves trained weights
    print("Saved model to disk")
else:
    # Load the trained model
    print("Load model from disk")
    model = baseline_model_saved()


# Model will predict the probability values for 7 labels for a test image
score = model.predict(X_test) # Score stroes the 8*1 dimentional output for each testing example
print (model.summary())

new_X = [ np.argmax(item) for item in score ]
y_test2 = [ np.argmax(item) for item in y_test]

# Calculating categorical accuracy taking label having highest probability
accuracy = [ (x==y) for x,y in zip(new_X,y_test2) ]
print(" Accuracy on Test set : " , np.mean(accuracy))
