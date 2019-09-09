#%%
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from sklearn import model_selection
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import cv2
import pickle


#%%
include_none = False

#%%
# get paths of all images
if include_none:
    all_images = [filename for filename in glob('data/categorized/**/*.jpg')]
else:
    all_images = [filename for filename in glob('data/categorized/**/*.jpg')
        if filename.split('/')[-2] != 'none']
image_count = len(all_images)
image_count

#%%
# get shape of each image by checking shape of any image (downsampling 2x)
check_image = cv2.imread(all_images[0])
dims = (check_image.shape[1]//4, check_image.shape[0]//4) # (width, height)
check_image = cv2.resize(check_image, dims)
image_shape = check_image.shape
image_shape

#%%
# images is a vector (4D np array) containing all images, training & test
images = np.empty((image_count, image_shape[0], image_shape[1], 3))
# vector of all labels
labels = np.empty(image_count, dtype=np.int8)
labels_dict = {
    "stop": 0,
    "forward": 1,
    "forward left": 2,
    "forward right": 3,
    "backward": 4,
    "backward left": 5,
    "backward right": 6,
    "left": 7,
    "right": 8
}
if include_none:
    labels_dict["none"] = 9

labels_list = ["stop", "forward", "forward left", "forward right",
    "backward", "backward left", "backward right", "left", "right"]
if include_none:
    labels_list.append("none")

#%%
for i in range(image_count):
    image = all_images[i]
    # assign label for this example
    label = image.split('/')[-2]
    labels[i] = labels_dict[label]
    # convert (RGB) image to numpy array
    # though cozmo's camera gives greyscale images, lime_image requires RGB
    image_arr = cv2.imread(image)
    # resize down
    image_arr = cv2.resize(image_arr, dims)
    # insert image into array of all images
    images[i,:,:,:] = image_arr

#%%
# 60% train, 20% validation, 20% test
train_images, test_images, train_labels, test_labels = model_selection.train_test_split(
    images, labels, test_size=0.2)
train_images, val_images, train_labels, val_labels = model_selection.train_test_split(
    train_images, train_labels, test_size=0.25)

#%%
# visualize an image
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

#%%
# rescale images to work with this particular NN model
train_images = train_images / 255.0
test_images = test_images / 255.0
 
#%%
# show first 25 images
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    # plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.imshow(train_images[i], cmap=plt.cm.gray)
    plt.xlabel(train_labels[i])
plt.show()

#%%
# construct model
model = keras.Sequential([
    keras.layers.Lambda(lambda i: tf.image.rgb_to_grayscale(i)), # convert to greyscale
    keras.layers.Flatten(input_shape=(dims[1], dims[0])),
    keras.layers.Dense(784, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(len(labels_list), activation=tf.nn.softmax)
])

#%%
# compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#%%
# train model
model.fit(train_images, train_labels, epochs=5)

#%%
# save model
model.save("model/cozmo_drive_model.h5")

#%%
# save train/val/test data for use in later analysis
data = [train_images, val_images, test_images, train_labels, val_labels,
    test_labels]
with open("model/pickles/train_val_test.pkl", 'wb') as file:
    pickle.dump(data, file)

#%%
# save labels
label_name_structs = [labels_dict, labels_list]
with open("model/pickles/label_names.pkl", 'wb') as file:
    pickle.dump(label_name_structs, file)

#%%
# evaluate validation accuracy
val_loss, val_acc = model.evaluate(val_images, val_labels)
print('Validation accuracy:', val_acc)

#%%
# evaluate test accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
