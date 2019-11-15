import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential, model_from_yaml, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import tensorflow as tf
import tensorflow.keras as keras


class Basic_Block(keras.Model):
    def __init__(self, filters, downsample=False, stride=1):
        self.expasion = 1
        super(Basic_Block, self).__init__()
        self.downsample = downsample
        self.conv2a = keras.layers.Conv2D(filters=filters,
                                          kernel_size=3,
                                          strides=stride,
                                          kernel_initializer='he_normal',
                                          )
        self.bn2a = keras.layers.BatchNormalization(axis=-1)
        self.conv2b = keras.layers.Conv2D(filters=filters,
                                          kernel_size=3,
                                          padding='same',
                                          kernel_initializer='he_normal'
                                          )
        self.bn2b = keras.layers.BatchNormalization(axis=-1)
        self.relu = keras.layers.ReLU()
        if self.downsample:
            self.conv_shortcut = keras.layers.Conv2D(filters=filters,
                                                     kernel_size=1,
                                                     strides=stride,
                                                     kernel_initializer='he_normal',
                                                     )
            self.bn_shortcut = keras.layers.BatchNormalization(axis=-1)

    def call(self, inputs, **kwargs):
        x = self.conv2a(inputs)
        x = self.bn2a(x)
        x = self.relu(x)
        x = self.conv2b(x)
        x = self.bn2b(x)
        x = self.relu(x)
        if self.downsample:
            shortcut = self.conv_shortcut(inputs)
            shortcut = self.bn_shortcut(shortcut)
        else:
            shortcut = inputs
        x = keras.layers.add([x, shortcut])
        x = self.relu(x)
        return x


class ResNet(keras.Model):
    def __init__(self, block, layers, num_classes=2, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.padding = keras.layers.ZeroPadding2D((3, 3))
        self.conv1 = keras.layers.Conv2D(filters=64,
                                         kernel_size=7,
                                         strides=2,
                                         kernel_initializer='glorot_uniform',
                                         name='conv1')
        self.bn_conv1 = keras.layers.BatchNormalization(axis=3, name='bn_conv1')
        self.max_pool = keras.layers.MaxPooling2D((3, 3),
                                                  strides=2,
                                                  padding='same')
        self.avgpool = keras.layers.GlobalAveragePooling2D(name='avg_pool')
        self.fc = keras.layers.Dense(num_classes, activation='softmax', name='result')
        # layer2
        self.res2 = self.mid_layer(block, 64, layers[0], stride=1, layer_number=2)
        # layer3
        self.res3 = self.mid_layer(block, 128, layers[1], stride=2, layer_number=3)
        # layer4
        self.res4 = self.mid_layer(block, 256, layers[2], stride=2, layer_number=4)
        # layer5
        self.res5 = self.mid_layer(block, 512, layers[3], stride=2, layer_number=5)

    def mid_layer(self, block, filter, block_layers, stride=1, layer_number=1):
        layer = keras.Sequential()
        if stride != 1 or filter * 4 != 64:
            layer.add(block(filters=filter,
                            downsample=True, stride=stride,
                            ))
        for i in range(1, block_layers):
            p = chr(i + ord('a'))
            layer.add(block(filters=filter))
        return layer

    def call(self, inputs, **kwargs):
        x = self.padding((inputs))
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = tf.nn.relu(x)
        x = self.max_pool(x)
        # layer2
        x = self.res2(x)
        # layer3
        x = self.res3(x)
        # layer4
        x = self.res4(x)
        # layer5
        x = self.res5(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x


def resnet18():
    return ResNet(Basic_Block, [2, 2, 2, 2], num_classes=2)


if __name__ == '__main__':
    model = resnet18()
    model.build(input_shape=(None,224, 224, 3))
    model.summary()

def load_data():
    path = 'D:/kaggle/train/'
    files = os.listdir(path)
    images = []
    labels = []
    count = 0
    for i in range(1000):
        if i % 2 == 0:
            img_path = path + 'cat.' + str(i) + '.jpg'
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            images.append(img_array)
            labels.append(0)
        else:
            img_path = path + 'dog.' + str(i) + '.jpg'
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            images.append(img_array)
            labels.append(1)

    data = np.array(images)
    #data = np.expand_dims(data,axis=0)
    labels = np.array(labels)
    print(labels)
    labels = to_categorical(labels, 2)
    return data, labels


'''model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), input_shape=(224, 224, 3), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(Dropout(0.3))
model.add(Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(Dropout(0.3))
model.add(Conv2D(128, kernel_size=(5, 5), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(Dropout(0.5))
model.add(Conv2D(256, kernel_size=(5, 5), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.summary()'''
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0003), metrics=['accuracy'])
images, lables = load_data()
images /= 255
x_train, x_test, y_train, y_test = train_test_split(images, lables, test_size=0.2)
tbCallbacks = callbacks.TensorBoard(log_dir=r'C:\Users\master\cat and dogsProjects\logs', histogram_freq=1,
                                    write_graph=True, write_images=True)
model.fit(x_train, y_train, batch_size=8, epochs=30, verbose=1, validation_data=(x_test, y_test),
          callbacks=[tbCallbacks])
scroe, accuracy = model.evaluate(x_test, y_test, batch_size=8)
print('scroe:', scroe, 'accuracy:', accuracy)
#block_name='{}a'.format(layer_number)