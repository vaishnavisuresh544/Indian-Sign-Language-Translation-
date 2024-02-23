import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras import backend as K
# Set the image data format to 'channels_first'
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
def create_cnn_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(no_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

model = create_cnn_model()
