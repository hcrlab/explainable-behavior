#%%
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import pickle


#%%
# load model
model = keras.models.load_model("model/cozmo_drive_model.h5")

#%%
# load data
with open("model/pickles/train_val_test.pkl", 'rb') as file:
    train_images, val_images, test_images, train_labels, val_labels, test_labels = pickle.load(file)
    
#%%
# load label names
with open("model/pickles/label_names.pkl", 'rb') as file:
    labels_dict, labels_list = pickle.load(file)

#%%
# make predictions
predictions = model.predict(test_images)

#%%
# get top prediction for calculating precision/recall
top_predictions = np.argmax(predictions, axis=1)

#%%
# save predictions
with open("model/pickles/predictions.pkl", 'wb') as file:
    pickle.dump([predictions, top_predictions], file)

#%%
# classification report
# we only want target names for the labels present in the test set
# e.g. backwards left and right aren't in test set (or training/val)
print(metrics.classification_report(test_labels, top_predictions,
    target_names = [labels_list[i] for i in np.unique(test_labels)]))

#%%
# view a prediction
np.argmax(predictions[0]) # argmax to select label w/ highest prob

#%%
# view true label for this prediction
test_labels[0]

#%%
# plotting functions
def plot_image(i, predictions_array, true_label, img, save=False):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    # plt.imshow(img, cmap=plt.cm.binary)
    plt.imshow(img, cmap=plt.cm.gray)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(labels_list[predicted_label],
                                100*np.max(predictions_array),
                                labels_list[true_label]),
                                color=color)
    
    if save:
        plt.savefig(i)

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
i = 1
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
# investigate first 15 images
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
