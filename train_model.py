#%%
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from sklearn import datasets, metrics, model_selection
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

import cv2
from skimage.segmentation import mark_boundaries
import lime
from lime import lime_image
from skimage.color import label2rgb
from lime.wrappers.scikit_image import SegmentationAlgorithm


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
# get top prediction for calculating precision/recall
top_predictions = np.argmax(predictions, axis=1)

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

#%%
explainer = lime_image.LimeImageExplainer(verbose=False)
segmenter = SegmentationAlgorithm('slic', n_segments=100, compactness=1, sigma=1)

#%%
num_top_labels=4
explanation = explainer.explain_instance(test_images[0], classifier_fn=model.predict,
    top_labels=num_top_labels, hide_color=0, num_samples=10000, segmentation_fn=segmenter)

#%%
temp, mask = explanation.get_image_and_mask(test_labels[0], positive_only=True,
    num_features=5, hide_rest=False)
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8, 4))
ax1.imshow(label2rgb(mask, temp, bg_label=0), interpolation='nearest')
ax1.set_title('Positive Regions for {}'.format(labels_list[test_labels[0]]))
temp, mask = explanation.get_image_and_mask(test_labels[0], positive_only=False,
    num_features=10, hide_rest=False)
ax2.imshow(label2rgb(3-mask, temp, bg_label=0), interpolation = 'nearest')
ax2.set_title('Positive/Negative Regions for {}'.format(labels_list[test_labels[0]]))

#%%
fig, m_axs = plt.subplots(2,num_top_labels, figsize=(12,4))
for i, (c_ax, gt_ax) in zip(explanation.top_labels, m_axs.T):
    temp, mask = explanation.get_image_and_mask(i, positive_only=True, num_features=5,
        hide_rest=False, min_weight=0.01)
    c_ax.imshow(label2rgb(mask,temp, bg_label=0), interpolation='nearest')
    c_ax.set_title('Positive for {}\nScore:{:2.2f}%'.format(labels_list[i], 100*predictions[0, i]))
    c_ax.axis('off')
    action_id = np.random.choice(np.where(train_labels==i)[0])
    gt_ax.imshow(train_images[action_id])
    gt_ax.set_title('Example of {}'.format(labels_list[i]))
    gt_ax.axis('off')

#%%
# generate several explanation summary images
# explanations for correct predictions
for i in np.unique(test_labels):
    print("Generating explanations for correctly classified {} actions...".format(labels_list[i]))
    # create necessary folders
    while True:
        try:
            os.chdir('model/explanations/{}/correct/'.format(labels_list[i]))
            break
        except FileNotFoundError:
            os.makedirs('model/explanations/{}/correct/'.format(labels_list[i]))

    i_locations = np.where(test_labels == i)[0]
    # randomly pick 10 images; if there are fewer than ten to choose from just
    # pick them all
    selection = np.random.choice(i_locations, 10) if len(i_locations) > 10 else i_locations
    i_labels = test_labels[selection]
    i_predictions = top_predictions[selection]
    # generate explanation summary image for each selected image
    for j in range(selection.shape[0]):
        # create explanation
        num_top_labels = 4
        explanation = explainer.explain_instance(test_images[selection[j]], classifier_fn=model.predict,
            top_labels=num_top_labels, hide_color=0, num_samples=1000, segmentation_fn=segmenter)

        # create figure
        fig, m_axs = plt.subplots(2,num_top_labels, figsize=(12,4))
        for k, (c_ax, gt_ax) in zip(explanation.top_labels, m_axs.T):
            temp, mask = explanation.get_image_and_mask(k, positive_only=True, num_features=5,
                hide_rest=False, min_weight=0.01)
            c_ax.imshow(label2rgb(mask,temp, bg_label=0), interpolation='nearest')
            c_ax.set_title('Positive for {}\nScore:{:2.2f}%'.format(labels_list[k], 100*predictions[selection[j], k]))
            c_ax.axis('off')
            action_id = np.random.choice(np.where(train_labels==k)[0])
            gt_ax.imshow(train_images[action_id])
            gt_ax.set_title('Example of {}'.format(labels_list[k]))
            gt_ax.axis('off')
        
        plt.savefig("{}.jpg".format(selection[j]))

os.chdir('../../../..')


#%%
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(test_images[1], model.predict,
    top_labels=3, hide_color=0, num_samples=1000, batch_size=1)

#%%
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0],
    positive_only=False, num_features=5, hide_rest=False, min_weight=0.1)
plt.imshow(mark_boundaries(temp, mask))


#%%
# save lime predictions for first 15 images
explainer = lime_image.LimeImageExplainer()
for i in range(15):
    explanation  = explainer.explain_instance(test_images[i], model.predict, top_labels=3,
    hide_color=0, num_samples=1000, batch_size=1)

