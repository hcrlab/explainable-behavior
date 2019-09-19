import numpy as np
import cv2
import pickle
from glob import glob
from sklearn import model_selection
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications


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

    return train_images, val_images, test_images, train_labels, val_labels, test_labels


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


def train_top_model(train_labels, val_labels, labels_list,
    top_model_weights_path='pretrained/resnet/bottleneck.h5', epochs=50):
    
    train_data = np.load(
        open("pretrained/resnet/bottleneck_features_train.npy", "rb"))
    val_data = np.load(
        open("pretrained/resnet/bottleneck_features_val.npy", "rb"))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(labels_list), activation='sigmoid'))

    model.compile(optimizer='rmsprop',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    model.fit(train_data, train_labels, epochs=epochs,
        validation_data=(val_data, val_labels))
    
    model.save_weights(top_model_weights_path)


def main():
    train_images, val_images, test_images,\
        train_labels, val_labels, test_labels = load_data()
    
    with open("model/pickles/label_names.pkl", 'rb') as file:
        _, labels_list = pickle.load(file)

    # evaluate bottleneck
    # save_bottleneck_features(train_images, val_images)
    train_top_model(train_labels, val_labels, labels_list)


if __name__ == "__main__":
    main()
