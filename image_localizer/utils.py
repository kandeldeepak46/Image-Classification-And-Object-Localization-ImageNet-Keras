import numpy as np

try:
    import matplotlib.pyplot as plt
except:
    pass

from tensorflow.keras.applications import vgg16, VGG16


class MyUtilityFunctions(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        super(MyUtilityFunctions, self).__init__()

    def show_image(self, testfiles):
        filename = random.choice(testfiles)
        print(filename)
        img = load_img(filename, target_size=(self.width, self.height))
        img = np.array(img)
        plt.imshow(img)
        plt.show()

    def preprocess(self, testfiles, debug=False):
        filename = random.choice(testfiles)
        img = load_img(filename, target_size=(self.width, self.height))
        img = np.array(img)
        img = vgg16.preprocess_input(img)
        if debug is True:

            plt.imshow(img)
            plt.show()

        return img

    def build_heatmap(self, preds, class_ids, debug=False):
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
