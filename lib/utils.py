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
