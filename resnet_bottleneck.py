#%%
import numpy as np
import tensorflow as tf
import cv2
import pickle
from glob import glob
from sklearn import model_selection
# from tensorflow.keras.models import Sequential, Model, load_model
# from tensorflow.keras.layers import Dropout, Flatten, Dense
# from tensorflow.keras import applications
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Dense
from keras import applications
import lime
from lime import lime_image
from skimage.color import label2rgb
from lime.wrappers.scikit_image import SegmentationAlgorithm
import matplotlib.pyplot as plt


#%%
def load_data(include_none=False):
    # get paths of all images
    if include_none:
        all_images = [filename for filename in glob('data/categorized/**/*.jpg')]
    else:
        all_images = [filename for filename in glob('data/categorized/**/*.jpg')
            if filename.split('/')[-2] != 'none']
    image_count = len(all_images)
    print("Image count: {}".format(image_count))

    # get shape of each image by checking shape of any image (downsampling 2x)
    check_image = cv2.imread(all_images[0])
    dims = (check_image.shape[1]//4, check_image.shape[0]//4) # (width, height)
    check_image = cv2.resize(check_image, dims)
    image_shape = check_image.shape
    print("Image shape: {}".format(image_shape))
    # images is a 4D np array containing all images, training/val/test
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

    # 60% train, 20% validation, 20% test
    train_images, test_images, train_labels, test_labels = model_selection.train_test_split(
        images, labels, test_size=0.2)
    train_images, val_images, train_labels, val_labels = model_selection.train_test_split(
        train_images, train_labels, test_size=0.25)

    # rescale images to [0, 1]
    train_images = train_images / 255.0
    val_images = val_images / 255.0
    test_images = test_images / 255.0

    return train_images, val_images, test_images, train_labels, val_labels, test_labels


#%%
def save_bottleneck_features(train_images, val_images):
    # sticking with resnet for now, could try resnetv2/resnext
    # TODO: does this need input_shape argument? https://keras.io/applications/#resnet
    model = applications.ResNet50(include_top=False, weights='imagenet')
    bottleneck_features_train = model.predict(train_images)
    bottleneck_features_val = model.predict(val_images)
    np.save(open("pretrained/resnet/bottleneck_features_train.npy", 'wb'),
        bottleneck_features_train)
    np.save(open("pretrained/resnet/bottleneck_features_val.npy", 'wb'),
        bottleneck_features_val)


#%%
def train_top_model(train_labels, val_labels, labels_list,
    top_model_weights_path='pretrained/resnet/bottleneck.h5', epochs=50):
    
    train_data = np.load(
        open("pretrained/resnet/bottleneck_features_train.npy", "rb"))
    val_data = np.load(
        open("pretrained/resnet/bottleneck_features_val.npy", "rb"))

    # model = Sequential()
    # model.add(Flatten(input_shape=train_data.shape[1:]))
    # model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(len(labels_list), activation='softmax'))

    # model.compile(optimizer='rmsprop',
    #     loss='sparse_categorical_crossentropy',
    #     metrics=['accuracy'])
        
    # construct model
    model = Sequential([
        # Lambda(lambda i: tf.image.rgb_to_grayscale(i)), # convert to greyscale
        # Flatten(input_shape=(dims[1], dims[0])),
        Flatten(input_shape=train_data.shape[1:]),
        Dense(784, activation=tf.nn.relu),
        Dense(128, activation=tf.nn.relu),
        Dense(len(labels_list), activation=tf.nn.softmax)
        # Dense(784, activation='relu'),
        # Dense(128, activation='relu'),
        # Dense(len(labels_list), activation='softmax')
    ])

    # compile model
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(train_data, train_labels, epochs=epochs,
        validation_data=(val_data, val_labels))
    
    model.save(top_model_weights_path)
    return model


#%%
def evaluate_top_model(test_images, test_labels,
    top_model_weights_path="pretrained/resnet/bottleneck.h5", save_predictions=False):

    resnet = applications.ResNet50(include_top=False, weights='imagenet')
    bottleneck_features_test = resnet.predict(test_images)
    top_model = load_model(top_model_weights_path)
    predictions = top_model.predict(bottleneck_features_test)
    top_predictions = np.argmax(predictions, axis=1)
    if save_predictions:
        with open("pretrained/resnet/predictions.pkl", 'wb') as file:
            pickle.dump([predictions, top_predictions], file)
    _, test_acc = \
        top_model.evaluate(bottleneck_features_test, test_labels)
    print("Test accuracy:", test_acc)

#%%
def explain_top_model(train_images, train_labels, test_images, test_labels,
    labels_list, top_model_weights_path="pretrained/resnet/bottleneck.h5"):

    # load ResNet
    initial_model = applications.ResNet50(include_top=False, weights='imagenet', input_shape=test_images.shape[1:])
    # get last layer
    last = initial_model.output

    # create top model
    top_model = Sequential()
    top_model.add(load_model(top_model_weights_path))

    preds = top_model(last)
    model = Model(initial_model.input, preds)

    # # copy ResNet to a Sequential model
    # new_model = Sequential()
    # for layer in model.layers:
    #     new_model.add(layer)

    # # concatenate top model to ResNet
    # new_model.add(top_model)

    # compile model (probably don't need)
    # model.compile(optimizer='adam',
    #             loss='sparse_categorical_crossentropy',
    #             metrics=['accuracy'])

    explainer = lime_image.LimeImageExplainer(verbose=False)
    segmenter = SegmentationAlgorithm('slic', n_segments=100, compactness=1, sigma=1)

    num_top_labels=4
    explanation = explainer.explain_instance(test_images[0], classifier_fn=model.predict,
        top_labels=num_top_labels, hide_color=0, num_samples=10000, segmentation_fn=segmenter)
    temp, mask = explanation.get_image_and_mask(test_labels[0], positive_only=True,
        num_features=5, hide_rest=False)
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8, 4))
    ax1.imshow(label2rgb(mask, temp, bg_label=0), interpolation='nearest')
    ax1.set_title('Positive Regions for {}'.format(labels_list[test_labels[0]]))
    temp, mask = explanation.get_image_and_mask(test_labels[0], positive_only=False,
        num_features=10, hide_rest=False)
    ax2.imshow(label2rgb(3-mask, temp, bg_label=0), interpolation = 'nearest')
    ax2.set_title('Positive/Negative Regions for {}'.format(labels_list[test_labels[0]]))

    with open("pretrained/resnet/predictions.pkl", 'rb') as file:
        predictions, top_predictions = pickle.load(file)
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
    plt.show()


#%%
def main(resplit=True, retrain=True):
    # set resplit to true to re-split train/val/test data, false to use existing
    # train/val/test split
    if resplit:
        train_images, val_images, test_images,\
            train_labels, val_labels, test_labels = load_data()
    
        # save data for future use
        data = [train_images, val_images, test_images, train_labels,
            val_labels, test_labels]
        with open("pretrained/resnet/train_val_test.pkl", 'wb') as file:
            pickle.dump(data, file)
    else:
        with open("pretrained/resnet/train_val_test.pkl", 'rb') as file:
            train_images, val_images, test_images, \
                train_labels, val_labels, test_labels = pickle.load(file)
    
    with open("model/pickles/label_names.pkl", 'rb') as file:
        _, labels_list = pickle.load(file)

    if retrain:
        # make sure to delete existing bottleneck feature files...
        save_bottleneck_features(train_images, val_images)
        train_top_model(train_labels, val_labels, labels_list, epochs=15)
    # else:
    #     load_model("pretrained/resnet/bottleneck.h5")

    # evaluate bottleneck model
    evaluate_top_model(test_images, test_labels, save_predictions=True)
    explain_top_model(train_images, train_labels, test_images, test_labels, labels_list)

#%%
# main(False, False)


if __name__ == "__main__":
    # main(True, True)
    main(False, False)


#%%
