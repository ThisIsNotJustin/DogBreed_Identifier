import os
import numpy as np
import pandas as pd 
import tensorflow as tf

import gradio as gr
from PIL import Image

from tensorflow import keras
from keras import Sequential
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam, SGD
from keras.layers import Dense,Dropout, Lambda, Input, GlobalAveragePooling2D
from keras.utils import to_categorical
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.xception import Xception, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tqdm import tqdm


labels = pd.read_csv('data/labels.csv')
labels.head()
labels['breed'].nunique()

breeds = sorted(list(set(labels['breed'])))
breedLabel = dict(zip(breeds, range(len(breeds))))

def imgs_to_array(directory, label_dataframe, size):
    imgLabel = label_dataframe['breed']
    images = np.zeros([len(label_dataframe), size[0], size[1], 
                       size[2]], dtype=np.uint8)
    y = np.zeros([len(label_dataframe),1], dtype=np.uint8)
    
    for ix, imgName in enumerate(tqdm(label_dataframe['id'].values)):
        imgDir = os.path.join(directory, imgName + '.jpg')
        img = load_img(imgDir)
        img = img.resize((size[0], size[1]))
        img = img_to_array(img)
        images[ix] = img
        
        breed = imgLabel[ix]
        y[ix] = breedLabel[breed]
        
    y = to_categorical(y)
    return images, y

shape = (280, 280, 3)
X,y = imgs_to_array('data/train', labels[:], size = shape)
lra = ReduceLROnPlateau(monitor = 'val_acc', factor = .01, 
                        patience = 3, min_lr = 1e-5, verbose = 1)
earlyStop = EarlyStopping(monitor = 'val_loss', patience = 10, 
                          restore_best_weights = True)

batchSize = 128
epochs = 50
lr = .001
sgd = SGD(learning_rate = lr, momentum = .9, nesterov = False)
adam = Adam(learning_rate = lr, beta_1 = .9, beta_2 = .999, 
            epsilon = None, amsgrad = False)

def get_features(model_name, model_preprocessor, input_size, data):
    input_layer = Input(input_size)
    preprocessor = Lambda(model_preprocessor)(input_layer)
    base_model = model_name(weights='imagenet', include_top=False, 
                            input_shape=input_size)(preprocessor)
    avg = GlobalAveragePooling2D()(base_model)
    feature_extractor = Model(inputs = input_layer, outputs = avg)
    feature_map = feature_extractor.predict(data, verbose=1)
    
    return feature_map

xception_pre = preprocess_input
xception_features = get_features(Xception, xception_pre, shape, X)

resnet_pre = preprocess_input
resnet_features = get_features(InceptionResNetV2, resnet_pre, shape, X)

features = np.concatenate([xception_features,resnet_features,], axis=-1)
print(features.shape)

model = Sequential()
model.add(Dropout(0.7, input_shape=(features.shape[1],)))
model.add(Dense(len(breeds),activation='softmax'))
adam = Adam(learning_rate=lr)

model.compile(optimizer=adam, loss='categorical_crossentropy', 
              metrics=['accuracy'])
history = model.fit(features, y, batch_size = batchSize, 
                    epochs = epochs, validation_split = 0.2, 
                    callbacks = [lra, earlyStop])




## TEST DATA
def imgs_to_arr_test(test_path, img_size = (280, 280, 3)):
    test_filenames = [test_path + fname for fname in os.listdir(test_path)]
    data_size = len(test_filenames)
    imgs = np.zeros([data_size, img_size[0], img_size[1], 3], 
                    dtype=np.uint8)
    
    for ix,img_dir in enumerate(tqdm(test_filenames)):
        img = load_img(img_dir, target_size = shape)
        imgs[ix] = img
    return imgs

test_data = imgs_to_arr_test('data/test/', shape)

def extract_features(data):
    xception_features = get_features(Xception, xception_pre, shape, data)
    resnet_features = get_features(InceptionResNetV2, resnet_pre, shape, data)
    
    features = np.concatenate([xception_features, resnet_features], axis=-1)
    return features

test_features = extract_features(test_data)

## CUSTOM INPUT
#img = load_img('data/input/Yue.jpg', target_size = shape)
#img = np.expand_dims(img, axis=0)
#img.shape

#test_features = extract_features(img)
#pred = model.predict(test_features)
#print(f"predicted label: {breeds[np.argmax(pred[0])]}")
#print(f"probability of prediction: {round(np.max(pred[0])) * 100}%")


# input from gradio web app
def dogbreed_identifier(filepath):
    image = load_img(filepath, target_size = (280, 280, 3))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    print (image.shape)
    test_features = extract_features(image)
    prediction = model.predict(test_features)
        
    return breeds[np.argmax(prediction[0])]

demo = gr.Interface(
    dogbreed_identifier,
    gr.Image(type="filepath"),
    gr.Label(),
    allow_flagging='never',
)

demo.launch(debug='True')
