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


from lib.model import Model
from lib.utils import image_to_np_array
import numpy as np
from tensorflow.keras.models import Model as TFModel
import os


def test_model():
    sample_image_path = os.getcwd(
    ) + "/app/frontend/build/assets/semsearch/datasets/fashion200/0.jpg"
    efficientnet_model = Model()

    image_array = image_to_np_array(
        sample_image_path, efficientnet_model.image_size)
    features = efficientnet_model.get_features(np.asarray([image_array]))
    assert(features.shape == (1, 7, 7, 1280))


def test_intermediate_model():
    efficientnet_model = Model()
    layer_details = efficientnet_model.get_layers()
    layer_name = layer_details[len(layer_details)-2]["name"]
    intermediate_model = efficientnet_model.get_intermediate_model(layer_name)
    assert (intermediate_model.__class__.__name__ == "Functional")
