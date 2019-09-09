#%%
from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.segmentation import mark_boundaries
import lime
from lime import lime_image
from skimage.color import label2rgb
from lime.wrappers.scikit_image import SegmentationAlgorithm
import pickle


#%% load model
model = keras.models.load_model("model/cozmo_drive_model.h5")

#%% load data
with open("model/pickles/train_val_test.pkl", 'rb') as file:
    train_images, val_images, test_images, train_labels, val_labels, test_labels = pickle.load(file)

#%%
# load label names
with open("model/pickles/label_names.pkl", 'rb') as file:
    labels_dict, labels_list = pickle.load(file)

#%%
# load top predictions
with open("model/pickles/predictions.pkl", 'rb') as file:
    predictions, top_predictions = pickle.load(file)

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
# generate several explanation summary images for correct predictions
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
    equality = np.where(test_labels == top_predictions)
    # want indices where model predicts correctly AND matches label (i) we're
    # currently working on
    i_locations = np.intersect1d(equality, i_locations)
    # i_locations = np.where(top_predictions[i_locations] == test_labels[i_locations])
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
        plt.close(fig)
    os.chdir('../../../..')
print("done")

#%%
# generate several explanation summary images for incorrect predictions
for i in np.unique(test_labels):
    print("Generating explanations for incorrectly classified {} actions...".format(labels_list[i]))
    # create necessary folders
    while True:
        try:
            os.chdir('model/explanations/{}/incorrect/'.format(labels_list[i]))
            break
        except FileNotFoundError:
            os.makedirs('model/explanations/{}/incorrect/'.format(labels_list[i]))

    i_locations = np.where(test_labels == i)[0]
    equality = np.where(test_labels != top_predictions)
    # want indices where model predicts incorrectly AND matches label (i) we're
    # currently working on
    i_locations = np.intersect1d(equality, i_locations)
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
        plt.close(fig)
    os.chdir('../../../..')
print("done")


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




#%%
