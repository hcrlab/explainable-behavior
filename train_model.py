#%%
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from sklearn import datasets, model_selection
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import cv2


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
test_image = cv2.imread(all_images[0], 0) # 0 for greyscale, 1 for color
dims = (test_image.shape[1]//4, test_image.shape[0]//4)
test_image = cv2.resize(test_image, dims)
# test_image = cv2.pyrDown(cv2.pyrDown(test_image))
image_shape = test_image.shape
image_shape

#%%
# images is a vector (3D np array) containing all images, training & test
images = np.empty((image_count, image_shape[0], image_shape[1]))
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
    # convert image to numpy array
    image_arr = cv2.imread(image, 0)
    # resize down
    # image_arr = cv2.pyrDown(cv2.pyrDown(image_arr))
    image_arr = cv2.resize(image_arr, dims)
    # insert image into array of all images
    images[i,:,:] = image_arr

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
# evaluate validation accuracy
val_loss, val_acc = model.evaluate(val_images, val_labels)
print('Validation accuracy:', val_acc)

#%%
# evaluate test accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

#%%
# make predictions
predictions = model.predict(test_images)

#%%
# view a prediction
np.argmax(predictions[0]) # argmax to select label w/ highest prob

#%%
# view true label for this prediction
test_labels[0]

#%%
# plotting functions
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(labels_list[predicted_label],
                                100*np.max(predictions_array),
                                labels_list[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    # selects the array of predicted probabilities & labels for the desired
    # image (i)
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    # plot probability for each of the classes in grey
    thisplot = plt.bar(range(len(labels_list)), predictions_array, color="#777777")
    # sets y-limits to (0, 1) (not sure why this is necessary)
    plt.ylim([0, 1])
    # select the label predicted by model
    predicted_label = np.argmax(predictions_array)

    # color bar for predicted label red
    thisplot[predicted_label].set_color('red')
    # color bar for true label blue
    # this overwrites the color of the predicted label if they're the same label
    thisplot[true_label].set_color('blue')

#%%
# investigate 0th image
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions, test_labels)
plt.show()


#%%
# investigate 12th image
i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions, test_labels)
plt.show()

#%%
# investigate first 13 images
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)
plt.show()

#%%
