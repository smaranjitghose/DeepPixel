import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

from .cam import CAM

class ScoreCAM(CAM):
    """
    Improved Visual Explanations Via Score-Weighted Class Activation Mapping
    """

    def _softmax(self, x):
        f = np.exp(x)/np.sum(np.exp(x), axis = 1, keepdims = True)
        return f

    def heat_map(self, input_, class_index=-1, max_N=-1):
        """to generate ScoreCAM heatmap for given :param:`input_` and :param:`class_index`
        with respect to :attr:`conv_layer`.

        Args:
            input_ (Tensor, array): Image array
            class_index (int, optional):
                class index. Defaults to -1.
                if class index is set to -1, then top predicted class is used.
            max_N (int):
                Specifying a natural number reduces the number of CNN inferences to that number.
                Defaults to -1 (original Score-CAM).

        Returns:
            (array): heat map array
        """

        conv_layer_model = keras.Model(self.model.inputs, self.conv_layer.output)

        classifier_input = keras.Input(shape=self.conv_layer.output.shape[1:])
        x = classifier_input
        for layer in self.classifier_layers:
            x = layer(x)
        classifier_model = keras.Model(classifier_input, x)

        conv_layer_output = conv_layer_model(input_)

        if class_index == -1: # default to top pred
            preds = classifier_model(conv_layer_output)
            class_index = tf.argmax(preds[0]).numpy()
            ## print("Predicted class:", class_index)

        act_map_array = conv_layer_output.numpy()

        # extract effective maps
        # ref: https://github.com/tabayashi0117/Score-CAM#Faster-Score-CAM
        if max_N != -1:
            act_map_std_list = [np.std(act_map_array[0,:,:,k]) for k in range(act_map_array.shape[3])]
            unsorted_max_indices = np.argpartition(-np.array(act_map_std_list), max_N)[:max_N]
            max_N_indices = unsorted_max_indices[np.argsort(-np.array(act_map_std_list)[unsorted_max_indices])]
            act_map_array = act_map_array[:,:,:,max_N_indices]

        # upsampling to original input size
        input_shape = self.model.input.shape[1:]
        act_map_resized_list = []
        for k in range(act_map_array.shape[3]):
            act_map_resized_list.append(
                cv2.resize(act_map_array[0,:,:,k], tuple(input_shape[:2]), interpolation=cv2.INTER_LINEAR)
            )

        # normalizing
        act_map_normalized_list = []
        for act_map_resized in act_map_resized_list:
            if np.max(act_map_resized) - np.min(act_map_resized) != 0:
                act_map_normalized = act_map_resized
                act_map_normalized /= max(np.max(act_map_resized) - np.min(act_map_resized), 1e-10)
            else:
                act_map_normalized = act_map_resized
            act_map_normalized_list.append(act_map_normalized)

        masked_input_list = []
        for act_map_normalized in act_map_normalized_list:
            masked_input = np.copy(input_)
            for c in range(masked_input.shape[-1]):
                masked_input[0,:,:,c] *= act_map_normalized
            masked_input_list.append(masked_input)
        masked_input_array = np.concatenate(masked_input_list, axis=0)

        pred_from_masked_input_array = self._softmax(self.model.predict(masked_input_array))

        weights = pred_from_masked_input_array[:,class_index]

        heat_map = np.dot(act_map_array[0,:,:,:], weights)
        heat_map = np.maximum(heat_map, 0) / max(np.max(heat_map), 1e-10)

        return heat_map
