import os
import numpy as np
import json

try:
    import matplotlib.pyplot as plt
except:
    pass
import random
import pathlib
from PIL import Image
import glob
import tensorflow as tf
from collections import Counter
import matplotlib.patches as patches
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Flatten,
    Activation,
    Dense,
    Dropout,
    Layer,
    Conv2D,
    MaxPool2D,
    BatchNormalization,
)
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array,
)
from tensorflow.keras.applications import vgg16, VGG16
from tensorflow.keras.utils import plot_model

from class_ids import get_class_ids

dog_ids = get_class_ids()

# assigning constanst
WIDHT = 224
HEIGHT = 224
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_VAL = 64

TEST_WIDTH, TEST_HEIGHT = 800, 600

TEST_DIR = os.path.join(os.path.dirname(__file__), "test", "dogs", "*")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "fcnn_vgg16_imagenet.h5")
JSON_PATH = os.path.join(os.path.dirname(__file__), "label_map.json")

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError("please check input directory for trained model")

if not os.path.isfile(JSON_PATH):
    raise FileNotFoundError("please make sure JSON file is located")


with open(JSON_PATH) as f:
    idx_to_label = json.load(f)


class Softmax4D(Layer):
    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super(Softmax4D, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
        s = K.sum(e, axis=self.axis, keepdims=True)

        return e / s

    def get_outout_shape_for(self, input_shape):
        return input_shape


def create_fcnn_from_vgg16(
    width: int, height: int, channels: int, num_class: int
) -> None:
    """
    Function to design the Deep Neural Nettwork using pre-trained Deep Learning Model, inspired from concept of Transfer Learning.
    This function is used to create a fully connected neural network from pre-trained VGG16 model. 
    Arguments
        width : [int]
            required width of image
        height  : [int]
            required height of image
        channel : [int]
            number of channels in image [2 if gray scale 3 else RGB]
        num_class  : [int]
            number of classes of datasets [2 for current, cats and dogs]
    Returns:
        Fully Connected Deep Neural Network
        
    """
    base_model = VGG16(
        weights="imagenet", include_top=False, input_shape=(width, height, channels)
    )
    fcnn_model = Sequential()

    for layer in base_model.layers:
        fcnn_model.add(layer)
    fcnn_model.add(
        Conv2D(filters=4096, kernel_size=(7, 7), name="fc1", activation="relu")
    )
    fcnn_model.add(
        Conv2D(filters=4096, kernel_size=(1, 1), name="fc2", activation="relu")
    )
    fcnn_model.add(Conv2D(filters=1000, kernel_size=(1, 1), name="predictions"))
    fcnn_model.add(Softmax4D(axis=-1, name="softmax"))

    # now reshape dense layers weights to FCNN kernel weights
    vgg_top = VGG16(
        weights="imagenet", include_top=True, input_shape=(width, height, channels)
    )
    for layer in fcnn_model.layers:
        if layer.name.startswith("fc") or layer.name.startswith("pred"):
            orig_layer = vgg_top.get_layer(layer.name)
            W, b = orig_layer.get_weights()
            ax1, ax2, previous_filter, n_filter = layer.get_weights()[0].shape
            new_W = W.reshape(ax1, ax2, -1, n_filter)
            layer.set_weights([new_W, b])
    del base_model
    del vgg_top

    return fcnn_model


def create_localization_fcnn_vgg16(
    width: int, height: int, channels: int, num_class: int
):
    """
    Function to design the Deep Neural Nettwork using pre-trained Deep Learning Model, inspired from concept of Transfer Learning
    Arguments
        width : [int]
            required width of image

        height  : [int]
            required height of image

        channel : [int]
            number of channels in image [2 if gray scale 3 else RGB]

        num_class  : [int]
            number of classes of datasets [2 for current, cats and dogs]

    Returns:
        Fully Connected Deep Neural Network
    """
    base_model = VGG16(
        weights="imagenet", include_top=False, input_shape=(width, height, channels)
    )
    fcnn_model = Sequential()
    for layer in base_model.layers:
        fcnn_model.add(layer)
    fcnn_model.add(
        Conv2D(filters=4096, kernel_size=(7, 7), name="fc1", activation="relu")
    )
    fcnn_model.add(
        Conv2D(filters=4096, kernel_size=(1, 1), name="fc2", activation="relu")
    )
    fcnn_model.add(Conv2D(filters=1000, kernel_size=(1, 1), name="predictions"))
    fcnn_model.add(Softmax4D(axis=-1, name="softmax"))

    del base_model
    return fcnn_model


class MyUtilityFunctions(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        super(MyUtilityFunctions, self).__init__()

    def show_image(self, testfiles) -> None:
        """
        Function to show the image in the console.
        Arguments
            testfiles : [list]
                list of image files
        Returns:
            None
        """
        filename = random.choice(testfiles)
        print(filename)
        img = load_img(filename, target_size=(self.width, self.height))
        img = np.array(img)
        plt.imshow(img)
        plt.show()

    def preprocess(self, testfiles, debug=False) -> np.ndarray:
        """
        Function to preprocess the images for training and testing. This function is used to preprocess the images for training and testing.
        Arguments
            testfiles : [list]
                list of test files
            debug : [bool]
                flag to print the image
        Returns:
            numpy array of images
        """
        filename = random.choice(testfiles)
        img = load_img(filename, target_size=(self.width, self.height))
        img = np.array(img)
        img = vgg16.preprocess_input(img)
        if debug is True:

            plt.imshow(img)
            plt.show()

        return img


class ImageSegmentation(object):
    def __init__(self, image_path: str = None):
        self.image_path = image_path
        super(ImageSegmentation, self).__init__()

    def make_predictions(self, testfiles) -> list:
        """
        Function to make predictions on test files
        Arguments
            testfiles : [list]
                list of test files
                Returns:
                    list of predictions
                    """
        model_localization = create_localization_fcnn_vgg16(
            TEST_HEIGHT, TEST_WIDTH, 3, 2
        )
        model_localization.load_weights(MODEL_PATH)
        # loading, preprocessing and prediction
        testfiles = glob.glob(testfiles)
        filename = random.choice(testfiles)
        img = load_img(filename, target_size=(TEST_HEIGHT, TEST_WIDTH))
        img = np.array(img)
        img = vgg16.preprocess_input(img)
        img = np.array([img])
        predictions = model_localization.predict(img)
        r, c, n = predictions[0].shape

        return predictions, filename, r, c, n

    def build_heatmap(self, preds, class_ids, debug=False):
        """
        Function to build heatmap
        Arguments
            preds : [list]
                list of predictions
            class_ids : [list]
                list of class ids
            debug : [bool]
                debug mode
            Returns:
                heatmap
        """
        preds_ids = (np.argmax(preds, axis=-1))[0]
        hm = np.isin(preds_ids, class_ids)
        if debug is True:
            plt.imshow(hm)
            plt.grid()
        return hm

    def build_heatmap2(preds, class_ids):
        hm2 = preds[0, :, :, class_ids].sum(axis=0)
        plt.imshow(hm2, interpolation="nearest", cmap="viridis")
        plt.grid()

    def extract_region_of_interest(
        self, sw=3, sh=3, neighbours=5, color="red", size=100, show_bounding_box=False
    ) -> list:
        """
        Function to extract region of interest. It is based on the heatmap. It is based on the idea of using a sliding window to extract the region of interest.
        The sliding window is defined by the size of the sliding window.
        Arguments
            sw : [int]
                sliding window size

            sh : [int]
                sliding window size

            neighbours : [int]
                number of neighbours to consider

            color : [str]   
                color of the bounding box

            size : [int]    
                size of the bounding box
                
            show_bounding_box : [bool]
                show bounding box
            Returns:    
                list of bounding boxes
        """
        try:

            if os.path.isfile(self.image_path):
                TEST_DIR = self.image_path

            else:
                TEST_DIR = TEST_DIR

            preds, filename, r, c, n = self.make_predictions(TEST_DIR)
            hm = self.build_heatmap(preds, dog_ids)
            hr, hc = hm.shape

            points = []
            for i in range(hr - sh + 1):
                for j in range(hc - sw + 1):
                    window = hm[i : i + sh, j : j + sw]
                    total = window.sum()
                    if total > neighbours:
                        points.append((i, j))

            rows, cols = zip(*points)

            # get min-max points for rectangle
            ymin, xmin = min(rows), min(cols)
            ymax, xmax = max(rows), max(cols)
            (xmin, ymin), (xmax, ymax)

            bw, bh = abs(xmax - xmin), abs(ymax - ymin)

            x1, y1 = xmin + bw, ymin
            x2, y2 = xmin, ymin + bh

            if show_bounding_box is True:

                scale_w, scale_h = TEST_WIDTH / c, TEST_HEIGHT / r
                img_org = load_img(filename, target_size=(TEST_WIDTH, TEST_HEIGHT))
                img_org = np.array(img_org)
                plt.imshow(img_org)

                rect_scaled = patches.Rectangle(
                    (xmin * scale_w, ymin * scale_h),
                    bw * scale_w,
                    bh * scale_h,
                    linewidth=3,
                    edgecolor=color,
                    facecolor="none",
                )
                plt.gca().add_patch(rect_scaled)
                plt.show()

            return f"image[region of interest] is located at: {[(x1,y1),(x2,y2)]}"

        except Exception as e:
            raise ValueError("could not prediction")


def main():
    img_seg = ImageSegmentation()
    img_seg.extract_region_of_interest()


if __name__ == "__main__":
    main()
