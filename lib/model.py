# Copyright 2020 Cloudera, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from tensorflow.keras.applications import EfficientNetB0
import logging
import numpy as np
from tensorflow.keras.models import Model as TFModel

logger = logging.getLogger(__name__)


class Model():
    def __init__(self, model_name="efficientnetb0"):
        """Class to abstract useful methods for extracting representations using a pretrained model.
        Extend this class to use any of the dozen pretrained image analysis models on 
        tensorflow.keras.applications https://www.tensorflow.org/api_docs/python/tf/keras/applications .


        Args:
            model_name (str, optional): [description]. Defaults to "efficientnetb0".
        """
        if (model_name == "efficientnetb0"):
            self.model = EfficientNetB0(include_top=False)
            self.image_size = 224

    def get_features(self, image_array):
        """Get features given a numpy array of images. Accomplished by a forward pass through the pretrained model. 

        Args:
            image_array (np.array): np.array of image features

        Returns:
            np.array: np.array of features
        """
        features = self.model.predict(image_array)
        return features

    def get_layers(self):
        """Get list of layers for the current model

        Returns:
            [dict]: list of dicts containing relevant attributes of each layer in the current model.
        """
        layer_details = [{"name": layer.name, "type": layer.__class__.__name__,
                          "parametercount": layer.count_params(), "layer_index": i,
                          "totallayers": len(self.model.layers)}
                         for (i, layer) in enumerate(self.model.layers)]
        return layer_details

    def get_intermediate_model(self, layer_name):
        """Construct intermediate model from current model

        Args:
            layer_name (str): name of layer to construct model from

        Returns:
            Model: ensorflow.keras.models.Model object.
        """
        intermediate_model = TFModel(
            inputs=self.model.input, outputs=self.model.get_layer(layer_name).output)
        return intermediate_model
