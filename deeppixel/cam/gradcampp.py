import numpy as np
import tensorflow as tf
from tensorflow import keras

from .cam import CAM

class GradCAMPP(CAM):
    """
    Improved Visual Explanations from Deep Networks
    """

    def heat_map(self, input_, class_index=-1):
        """to generate GradCAM++ heatmap for given :param:`input_` and :param:`class_index`
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
        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape2:
                with tf.GradientTape() as tape3:
                    conv_layer_output = conv_layer_model(input_)
                    tape1.watch(conv_layer_output)
                    tape2.watch(conv_layer_output)
                    tape3.watch(conv_layer_output)
                    preds = classifier_model(conv_layer_output)
                    ## print("Predicted class:", tf.argmax(preds[0]).numpy())

                    if class_index == -1: # default to top pred
                        class_index = tf.argmax(preds[0])
                    class_channel = preds[:, class_index]

                first_grads = tape3.gradient(class_channel, conv_layer_output)
            second_grads = tape2.gradient(first_grads, conv_layer_output)
        third_grads = tape1.gradient(second_grads, conv_layer_output)

        conv_layer_output = conv_layer_output.numpy()[0]

        global_sum = np.sum(conv_layer_output, axis=(0, 1, 2))

        alpha_nume = second_grads.numpy()[0]
        alpha_denomi = second_grads.numpy()[0] * 2.0 + third_grads.numpy()[0] * global_sum
        # to avoid zero division error
        alpha_denomi = np.where(alpha_denomi != 0.0, alpha_denomi, 1e-10)

        alphas = alpha_nume/alpha_denomi
        alphas /= np.sum(alphas, axis=(0,1)) # normalization

        weights = np.maximum(first_grads.numpy()[0], 0)

        deep_linearization_weights = np.sum(weights * alphas, axis=(0, 1))

        heat_map = np.sum(deep_linearization_weights * conv_layer_output, axis=2)
        heat_map = np.maximum(heat_map, 0) / max(np.max(heat_map), 1e-10)

        return heat_map
