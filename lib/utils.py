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

from tensorflow.keras.preprocessing import image as imageprep
import os
import numpy as np
from PIL import Image
import json
import requests
from io import BytesIO


def image_to_np_array(img_path, image_size):
    img = imageprep.load_img(img_path, target_size=(image_size, image_size))
    img = imageprep.img_to_array(img)
    return img


def to_np_array(img, image_size):
    img = img.resize((image_size, image_size))
    img = imageprep.img_to_array(img)
    if (img.shape[2] == 4):
        img = img[..., :3]
    return img


def file_to_np_array(file, image_size):
    img = Image.open(file)
    img = img.resize((image_size, image_size))
    img = imageprep.img_to_array(img)
    if (img.shape[2] == 4):
        img = img[..., :3]
    return img


def url_to_np_array(url, image_size):
    if url.endswith(('.png', '.jpg', '.jpeg')):
        response = requests.get(url)
        img = file_to_np_array(BytesIO(response.content), image_size)
        return img
    else:
        return None


def mkdir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


def image_array_from_dir(dir_path, image_size, valid_file_types):
    image_paths = os.listdir(dir_path)
    image_paths = [os.path.join(dir_path, file_) for file_ in image_paths if file_.split(
        ".")[1] in valid_file_types]

    image_links = [file_.split("build/")[1] for file_ in image_paths]
    image_holder = []
    for img_path in image_paths:
        img_path = os.path.join(dir_path, img_path)
        image_holder.append(image_to_np_array(img_path, image_size))
    return np.asarray(image_holder), image_links


def load_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        return data


def save_json_file(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f)
