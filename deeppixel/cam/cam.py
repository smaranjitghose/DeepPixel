import numpy as np
import tensorflow as tf
from tensorflow import keras

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class CAM:
    """Class activation maps are a simple technique to get the discriminative image regions
    used by a CNN to identify a specific class in the image.

    This is a base class for all CAMs.
    """

    def __init__(self, model, conv_layer, classifier_layers=-1):
        """
        Args:
            model ([type], str): model or path of the model.
            conv_layer ([type]):
                name or index of interested convolutional layer.
                probably last convolutional layer.
            classifier_layers (int, optional):
                list of name or index of final prediction layers. Defaults to -1.
                at default, consecutive layer below :parm:`conv_layer` would be used.
        """

        if isinstance(model, str): # model_path
            self.model = keras.models.load_model(model)
        else:
            self.model = model

        self.conv_layer = self._get_layer(conv_layer)

        if classifier_layers == -1: # default
            conv_layer_index = self._get_layer_index(self.conv_layer)
            # all the layers below conv_layer
            self.classifier_layers = self.model.layers[conv_layer_index + 1 :]
        else:
            self.classifier_layers = list(map(self._get_layer, classifier_layers))

    def _get_layer(self, layer_id):
        """to get layer from :attr:`model`.

        Args:
            layer_id (str, int): name or index of layer.

        Returns:
            A layer instance.
        """

        if isinstance(layer_id, int):
            return self.model.get_layer(index = layer_id)
        else:
            return self.model.get_layer(layer_id)

    def _get_layer_index(self, layer):
        """to get index of :parm:`layer` in :attr:`model.layers`.

        Args:
            layer (layer): a layer instance.

        Returns:
            (int): index of :parm:`layer`.
        """

        return self.model.layers.index(layer)

    def _get_input(self, image):
        """to get input array for :attr:`model` from :parm:`image`.

        Args:
            image (str, Image, array): image path or image instance or image array.

        Returns:
            (array): resized image array.
        """

        size = self.model.input.shape[1:]

        if isinstance(image, str): # image path
            # resizing and changing color mode
            mode = "grayscale" if size[-1] == 1 else "rgb"
            image = keras.preprocessing.image.load_img(image, color_mode=mode, target_size=size)
            img_array = keras.preprocessing.image.img_to_array(image)
            # array to batch 
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255
            return img_array

        elif isinstance(image, Image.Image): # image
            # resizing and changing color mode
            mode = "L" if size[-1] == 1 else "RGB"
            image = image.convert(mode=mode)
            image = image.resize((size[1], size[0]))
            img_array = keras.preprocessing.image.img_to_array(image)
            # array to batch
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255
            return img_array

        else: # image array
            # already a array
            img_array = image
            if img_array.ndim == 2:
                img_array = np.expand_dims(img_array, axis=-1)

            image = keras.preprocessing.image.array_to_img(img_array)
            return self._get_img_array(image)

    def _get_img_array(self, image):
        """convert image to array.

        Args:
            image (str, Image, array): image path or image instance or image array.

        Returns:
            (array): image array.
        """

        if isinstance(image, str): # image path
            image = keras.preprocessing.image.load_img(image)
            img_array = keras.preprocessing.image.img_to_array(image)
            return img_array

        elif isinstance(image, Image.Image): # image
            img_array = keras.preprocessing.image.img_to_array(image)
            return img_array

        else: # image array
            img_array = image
            if img_array.ndim == 2:
                img_array = np.expand_dims(img_array, axis=-1)
            return img_array

    def heat_map(self, input_, class_index=-1):
        """to generate hear map.

        Args:
            input_ (array): image array.
            class_index (int, optional):
                class index. Defaults to -1.
                if class index is set to -1, then top predicted class is used.

        Raises:
            NotImplementedError: Every CAMs must define this.
        """

        raise NotImplementedError

    def _superimpose(self, heat_map, img_array):
        """to superimpose heat map above the image.

        Args:
            heat_map (array): heat map array.
            img_array (array): image array.

        Returns:
            (array): superimposed image array.
        """

        img_size = img_array.shape

        # rescaling heatmap to a range 0-255
        heat_map = np.uint8(255 * heat_map)

        # RGB for colormap
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heat_map = jet_colors[heat_map]

        # resizing heat map
        jet_heat_map = keras.preprocessing.image.array_to_img(jet_heat_map)
        jet_heat_map = jet_heat_map.resize((img_size[1], img_size[0]))
        jet_heat_map = keras.preprocessing.image.img_to_array(jet_heat_map)

        # superimposing
        superimposed_imgarr = jet_heat_map * 0.3 + img_array

        return superimposed_imgarr

    def image(self, image, class_index=-1):
        """to get the superimposed image.
        heat map is generated using :meth:`heat_map` with
        :parm:`image` and :parm:`class_index`.

        Args:
            image (str, Image, array): image path or image instance or image array.
            class_index (int, optional): class index. Defaults to -1.
        """

        input_ = self._get_input(image)
        img_array = self._get_img_array(image)

        heat_map = self.heat_map(input_, class_index)
        superimposed_imgarr = self._superimpose(heat_map, img_array)

        superimposed_img = keras.preprocessing.image.array_to_img(superimposed_imgarr)
        return superimposed_img

    def save(self, image, out_path, class_index=-1):
        """to save the superimposed image to :parm:`out_path`.
        heat map is generated using :meth:`heat_map` with
        :parm:`image` and :parm:`class_index`.

        Args:
            image (str, Image, array): image path or image instance or image array.
            out_path (str): path of superimposed image to save.
            class_index (int, optional): class index. Defaults to -1.
        """

        superimposed_img = self.image(image, class_index)
        superimposed_img.save(out_path)

    def show(self, image, class_index=-1):
        """to show the superimposed image.
        heat map is generated using :meth:`heat_map` with
        :parm:`image` and :parm:`class_index`.

        Args:
            image (str, Image, array): image path or image instance or image array.
            class_index (int, optional): class index. Defaults to -1.
        """

        superimposed_img = self.image(image, class_index)
        superimposed_img.show()

    def plot_heatmap(self, image, class_index=-1):
        """to plot the heat map generated using :meth:`heat_map`
        with :parm:`image` and :parm:`class_index`

        Args:
            image (str, Image, array): image path or image instance or image array.
            class_index (int, optional): class index. Defaults to -1.
        """

        input_ = self._get_input(image)
        heat_map = self.heat_map(input_, class_index)

        plt.matshow(heat_map)
        plt.show()
