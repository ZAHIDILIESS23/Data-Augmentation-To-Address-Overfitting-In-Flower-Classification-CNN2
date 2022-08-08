#!/usr/bin/env python
# coding: utf-8

# # Data Augmentation To Address Overfitting In Flower Classification CNN

# In this notebook we will build a CNN to classify flower images. We will also see how our model overfits and how overfitting can be addressed using data augmentation. Data augmentation is a process of generating new training samples from current training dataset using transformations such as zoom, rotations, change in contrast etc
# 
# Credits: I used tensorflow offical tutorial: https://www.tensorflow.org/tutorials/images/classification as a reference and made bunch of changes to make it simpler
# 
# In below image, 4 new training samples are generated from original sample using different transformations

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# We will download flowers dataset from google website and store it locally. In below call it downloads the zip file (.tgz) in cache_dir which is . meaning the current folder

# ## Load flowers dataset

# In[3]:


dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url,  cache_dir='.', untar=True)
# cache_dir indicates where to download data. I specified . which means current directory
# untar true will unzip it


# In[4]:


data_dir


# In[5]:


import pathlib
data_dir = pathlib.Path(data_dir)
data_dir


# In[6]:


list(data_dir.glob('*/*.jpg'))[:5]


# In[7]:


image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)


# In[8]:


roses = list(data_dir.glob('roses/*'))
roses[:5]


# In[9]:


PIL.Image.open(str(roses[1]))


# In[10]:


tulips = list(data_dir.glob('tulips/*'))
PIL.Image.open(str(tulips[0]))


# ## Read flowers images from disk into numpy array using opencv

# In[9]:


flowers_images_dict = {
    'roses': list(data_dir.glob('roses/*')),
    'daisy': list(data_dir.glob('daisy/*')),
    'dandelion': list(data_dir.glob('dandelion/*')),
    'sunflowers': list(data_dir.glob('sunflowers/*')),
    'tulips': list(data_dir.glob('tulips/*')),
}


# In[10]:


flowers_labels_dict = {
    'roses': 0,
    'daisy': 1,
    'dandelion': 2,
    'sunflowers': 3,
    'tulips': 4,
}


# In[13]:



flowers_images_dict['roses'][:5]


# In[14]:


str(flowers_images_dict['roses'][0])


# In[11]:


img = cv2.imread(str(flowers_images_dict['roses'][0])) ## str because of cv2 transform to 3 dimensions


# In[12]:


img.shape


# In[17]:


cv2.resize(img,(180,180)).shape # o have the same size


# In[13]:


X, y = [], []

for flower_name, images in flowers_images_dict.items():
    for image in images:
        img = cv2.imread(str(image))
        resized_img = cv2.resize(img,(180,180))
        X.append(resized_img)
        y.append(flowers_labels_dict[flower_name])


# In[14]:


X = np.array(X)
y = np.array(y)


# ## Train test split

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# ## Preprocessing: scale images

# In[ ]:


X_train_scaled = X_train / 255
X_test_scaled = X_test / 255


# ## Build convolutional neural network and train it

# In[22]:


num_classes = 5
model = Sequential([
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
              
model.fit(X_train_scaled, y_train, epochs=30)   


# In[23]:


model.evaluate(X_test_scaled,y_test)


# Here we see that while train accuracy is very high (99%), the test accuracy is significantly low (66.99%) indicating overfitting. Let's make some predictions before we use data augmentation to address overfitting

# In[24]:


predictions = model.predict(X_test_scaled)
predictions


# In[25]:


score = tf.nn.softmax(predictions[0])


# In[26]:


np.argmax(score)


# In[27]:


y_test[0]


# ## Improve Test Accuracy Using Data Augmentation

# In[ ]:


data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)


# ## Original Image

# In[ ]:


plt.axis('off')
plt.imshow(X[0])


# ## Newly generated training sample using data augmentation

# In[ ]:


plt.axis('off')
plt.imshow(data_augmentation(X)[0].numpy().astype("uint8"))


# ## Train the model using data augmentation and a drop out layer

# In[ ]:


num_classes = 5

model = Sequential([
  data_augmentation,
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
              
model.fit(X_train_scaled, y_train, epochs=30)    


# In[ ]:


model.evaluate(X_test_scaled,y_test)


# You can see that by using data augmentation and drop out layer the accuracy of test set predictions is increased to 73.74%

# In[ ]:




