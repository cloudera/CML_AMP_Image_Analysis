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

import logging
import os
import numpy as np
from lib.utils import image_array_from_dir, image_to_np_array


logger = logging.getLogger(__name__)


class Extractor():
    def __init__(self):
        self.valid_file_types = ["jpg", "png"]

    def extract(self, image_array, model):
        """Extract features, given a numpy array of images.

        Args:
            image_array (numpy array): numpy array of images
            model (Model): lib.Model object

        Returns:
            np.array: numpy array of extracted features (representations)
        """
        features = model.get_features(image_array)
        features = features.reshape(features.shape[0], -1)
        logger.info(">>> feature extraction complete.")
        return features

    def extract_from_dir(self, images_dir, model):
        """Extract images from a directory

        Args:
            images_dir (str): path to a directory
            model (Model): lib.Model

        Returns:
            np.array: numpy.array containing extracted features.
        """
        logger.info(">>> Scanning folder to get files.")
        image_array, image_ids = image_array_from_dir(
            images_dir, model.image_size, self.valid_file_types)
        # print(image_array)
        features = self.extract(image_array, model)
        return features, image_ids
