import numpy as np
import tensorflow as tf
from tensorflow import keras

from .cam import CAM

class SmoothGrad(CAM):
    """
    Visual Explanations from Deep Networks.
    SmoothGrad: removing noise by adding noise
    """

    def __init__(self, model):
        """
        Args:
            model ([type], str): model or path of the model.
        """

        if isinstance(model, str): # model_path
            self.model = keras.models.load_model(model)
        else:
            self.model = model

    @property
    def conv_layer(self):
        raise NotImplementedError("Not needed for SmoothGrad")

    @property
    def classifier_layers(self):
        raise NotImplementedError("Not needed for SmoothGrad")

    def heat_map(self, input_, class_index=-1, num_samples=5, noise=1.0):
        """to generate SmoothGrad heatmap for given
        :param:`input_` and :param:`class_index`.

        Args:
            input_ (Tensor, array): Image array
            class_index (int, optional):
                class index. Defaults to -1.
                if class index is set to -1, then top predicted class is used.
            num_samples (int): Number of noisy samples to generate for input
            noise (float): Standard deviation for noise normal distribution

        Returns:
            (array): heat map array
        """

        repeated_input = np.repeat(input_, num_samples, axis=0)
        noise = np.random.normal(0.5, noise, repeated_input.shape).astype(np.float32)
        noise /= np.max(noise)

        noisy_input = repeated_input + noise # noisy_image
        noisy_input = np.maximum(noisy_input, 0) / max(np.max(noisy_input), 1e-10)

        with tf.GradientTape() as tape:
            inputs = tf.cast(noisy_input, tf.float32)
            tape.watch(inputs)
            preds = self.model(inputs)

            if class_index == -1: # default to top pred
                class_index = tf.argmax(preds[0])
            num_classes = self.model.output.shape[1]

            expected_output = tf.one_hot([class_index] * noisy_input.shape[0], num_classes)
            loss = keras.losses.categorical_crossentropy(
                expected_output, preds
            )

        grads = tape.gradient(loss, inputs)

        grads_per_image = tf.reshape(grads, (-1, num_samples, *grads.shape[1:]))
        averaged_grads = tf.reduce_mean(grads_per_image, axis=1)

        grayscale_grads = tf.reduce_sum(tf.abs(averaged_grads), axis=-1)

        grayscale_grads = grayscale_grads.numpy()
        heat_map_shape = grayscale_grads.shape[1:]
        heat_map = grayscale_grads.reshape(heat_map_shape)
        heat_map = np.maximum(heat_map, 0) / max(np.max(heat_map), 1e-10)

        return heat_map
