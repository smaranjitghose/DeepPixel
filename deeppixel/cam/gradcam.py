import numpy as np
import tensorflow as tf
from tensorflow import keras

from .cam import CAM

class GradCAM(CAM):
    """
    Visual Explanations from Deep Networks via Gradient-based Localization
    """

    def heat_map(self, input_, class_index=-1):
        """to generate GradCAM heatmap for given :param:`input_` and :param:`class_index`
        with respect to :attr:`conv_layer`.

        Args:
            input_ (Tensor, array): Image array
            class_index (int, optional):
                class index. Defaults to -1.
                if class index is set to -1, then top predicted class is used.

        Returns:
            (array): heat map array
        """

        conv_layer_model = keras.Model(self.model.inputs, self.conv_layer.output)

        classifier_input = keras.Input(shape=self.conv_layer.output.shape[1:])
        x = classifier_input
        for layer in self.classifier_layers:
            x = layer(x)
        classifier_model = keras.Model(classifier_input, x)

        # computing the gradient for `class_index` class for `input_`
        # with respect to the activations of the `self.conv_layer`
        with tf.GradientTape() as tape:
            conv_layer_output = conv_layer_model(input_)
            tape.watch(conv_layer_output)
            preds = classifier_model(conv_layer_output)
            ## print("Predicted class:", tf.argmax(preds[0]).numpy())

            if class_index == -1: # default to top pred
                class_index = tf.argmax(preds[0])
            class_channel = preds[:, class_index]

        # The gradient of the `class_index` class with regard
        # to the output feature map of the `self.conv_layer`
        grads = tape.gradient(class_channel, conv_layer_output)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_layer_output = conv_layer_output.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        for i in range(pooled_grads.shape[-1]):
            conv_layer_output[:, :, i] *= pooled_grads[i]

        heat_map = np.mean(conv_layer_output, axis=-1)
        heat_map = np.maximum(heat_map, 0) / max(np.max(heat_map), 1e-10)

        return heat_map
