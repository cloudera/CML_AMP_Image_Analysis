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

from lib import Model
from lib import FaissIndex
from lib import Extractor
import numpy as np
import os

# Specify path to images in the repo
base_path = os.getcwd() + "/app/frontend/build/assets/semsearch/datasets/"
fashion_images_dir = base_path + "fashion200/"
iconic_images_dir = base_path + "iconic200/"
efficientnet_model = Model(model_name="efficientnetb0")
extractor = Extractor()


def create_index(image_dir, index_save_dir):
    """Create an FAISS index and save to disc

    Args:
        image_dir (str): path to directory of images whose features are added to the index
        index_save_dir (str): directory to save index on disc.     

    Returns:
        lib.FaissIndex:  lib.FaissIndex object.
    """
    features, ids = extractor.extract_from_dir(image_dir, efficientnet_model)
    index = FaissIndex(features.shape[1])
    index.add(features, id_strings=ids)
    index.save(index_save_dir)
    return index


def update_index(index, image_dir, index_save_dir):
    """Update an existing lib.FaissIndex 

    Args:
        index (lib.FaissIndex): index to be updated
        image_dir (str): directory of images to be added to index
        index_save_dir (str): directory to save index on disc.     
    """
    features, ids = extractor.extract_from_dir(image_dir, efficientnet_model)
    print(features.shape, len(ids))
    index.add(features, id_strings=ids)
    index.save(index_save_dir)


index = create_index(fashion_images_dir, os.getcwd() + "/faiss")
update_index(index, iconic_images_dir, os.getcwd() + "/faiss")
