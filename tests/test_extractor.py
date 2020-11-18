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


from lib import Extractor
from lib import Model
from lib.utils import image_to_np_array
import numpy as np
import os


images_dir = os.getcwd() + "/app/frontend/build/assets/semsearch/datasets/fashion200"
sample_image_path = images_dir + "/0.jpg"
efficientnet_model = Model(model_name="efficientnetb0")
extractor = Extractor()


def test_extract_dir():
    features, ids = extractor.extract_from_dir(images_dir, efficientnet_model)
    print(features.shape)
    assert len(ids) == 200 and features.shape == (200, 62720)


def test_extract():
    image_array = image_to_np_array(sample_image_path, 224)
    image_array = np.asarray([image_array])
    features = extractor.extract(image_array, efficientnet_model)
    assert features.shape == (1, 62720)
