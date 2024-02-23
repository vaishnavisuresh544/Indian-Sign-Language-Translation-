from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras import backend as K

import numpy as np
from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os

import matplotlib.pyplot as plt
K.set_image_data_format('channels_first')
# input image dimensions
img_x, img_y = 200, 200
# For grayscale use 1 value and for color images use 3 (R,G,B channels)
img_channels = 1
# number of classes
no_classes = 10
# size of convolutional filter
no_conv = 3
# size of max pooling window
no_pool = 2
no_filters = [2, 4, 8, 16, 32, 64, 128, 256, 512]
dropout_ratio = [0, 0.25, 0.5, 0.75, 1]
input_shape = (img_channels, img_x, img_y)

WeightFileName = ["adaptivethresholdmodeweight.hdf5", "siftmodeweight.hdf5", "nofiltermodeweight.hdf5"]
def trainModel(dataset_type, batch_size, no_epoch):
    global model, path
    # Create CNN model
    model = create_cnn_model()
    # Set path based on dataset type
    if dataset_type == 1:
        path = 'C:\\Users\\VAISHNAVI\\Desktop\\vaishnavi\\capstone_vaishu\\AdaptiveThresholdModeDataSet'
        print(path)
    elif dataset_type == 2:
        path = 'C:\\Users\\VAISHNAVI\\Desktop\\vaishnavi\\capstone_vaishu\\AdaptiveThresholdSiftModeDataSet'
        print(path)
    elif dataset_type == 3:
        path = 'C:\\Users\\VAISHNAVI\\Desktop\\vaishnavi\\capstone_vaishu\\NoFilterModeDataSet'
        print(path)
    else:
        print("Invalid dataset type. Please choose a valid dataset type.")
    #create dataset array   
    listing = os.listdir(path)
    listing.sort()
    dataset = []
    for name in listing:
        dataset.append(name)
        
    image = np.array(Image.open(path +'/' + dataset[0]))
    #fnd size of image
    m, n = image.shape[0:2]
    #find dataset size
    dataset_size = len(dataset)
    # create matrix to store all flattened images
    image_matrix = np.array([np.array(Image.open(path+ '/' + images).convert('L')).flatten() for images in dataset], dtype='f')
    label = np.ones((dataset_size,), dtype=int)
    samples_per_class = dataset_size / no_classes
    s = 0
    r = samples_per_class
    for classIdentifier in range(no_classes):
        label[int(s):int(r)] = classIdentifier
        s = r
        r = s + samples_per_class  
    data, Label = shuffle(image_matrix, label, random_state=2)
    train_data = [data, Label]
    X, y = train_data[0], train_data[1]  # Direct assignment
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)
    X_train = X_train.reshape(X_train.shape[0], img_channels, img_x, img_y).astype('float32') / 255
    X_test = X_test.reshape(X_test.shape[0], img_channels, img_x, img_y).astype('float32') / 255
    Y_train = np.eye(no_classes)[y_train.astype('int')]  # One-hot encoding for Y_train
    Y_test = np.eye(no_classes)[y_test.astype('int')]    # One-hot encoding for Y_test
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=no_epoch, verbose=1, validation_split=0.25)

# Assuming other parts of the code remain the same

    print(history.history.keys())
    print(history.history.keys())
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('acc.png')
    plt.close('all')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss.png')
    plt.close('all')
    print("Model trained successfully")
    
    
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    if dataset_type == 1:
             model.save_weights("./adaptivethresholdmodeweight.hdf5",overwrite=True)
    elif dataset_type == 2:
             model.save_weights("./siftmodeweight.hdf5",overwrite=True)
    elif dataset_type == 3:
             model.save_weights("./nofiltermodeweight.hdf5",overwrite=True)
   
