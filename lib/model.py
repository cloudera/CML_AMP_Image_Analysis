# ###########################################################################
#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2020
#  All rights reserved.
#
#  Applicable Open Source License: Apache 2.0
#
#  NOTE: Cloudera open source products are modular software products 
#  made up of hundreds of individual components, each of which was 
#  individually copyrighted.  Each Cloudera open source product is a 
#  collective work under U.S. Copyright Law. Your license to use the 
#  collective work is as provided in your written agreement with  
#  Cloudera.  Used apart from the collective work, this file is 
#  licensed for your use pursuant to the open source license 
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute 
#  this code. If you do not have a written agreement with Cloudera nor 
#  with an authorized and properly licensed third party, you do not 
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED 
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO 
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND 
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU, 
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS 
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE 
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES 
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF 
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# ###########################################################################

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
