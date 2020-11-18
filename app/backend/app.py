
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

import argparse
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS, cross_origin
import logging
import os
from lib import Model
from lib import FaissIndex
from lib import Extractor
from lib.utils import file_to_np_array, url_to_np_array
import numpy as np
from PIL import Image
import time

logging.basicConfig(level=logging.INFO)
index = FaissIndex(0, index_dir=os.getcwd() + "/faiss")

efficientnet_model = Model(model_name="efficientnetb0")
extractor = Extractor()
# Point Flask to the front end directory

root_file_path = os.getcwd() + "/app"
print(root_file_path, os.getcwd())
static_folder_root = os.path.join(root_file_path, "frontend/build")
print(static_folder_root)

app = Flask(__name__, static_url_path='',
            static_folder=static_folder_root, template_folder=static_folder_root)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

image_size = 224
k = 20


def get_similar(image_array):
    result = {}
    start_time = time.time()
    image_array = np.asarray([image_array])
    features = extractor.extract(image_array, efficientnet_model)
    extraction_time = time.time() - start_time
    start_time = time.time()
    distances, idx = index.search(features, k)
    search_time = time.time() - start_time

    result["distances"] = distances.tolist()[0]
    result["ids"] = index.decode_ids(idx.tolist()[0])
    result["extractiontime"] = extraction_time
    result["searchtime"] = search_time
    return result


@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/search', methods=["GET", "POST"])
def search():
    result = {"distances": [], "ids": []}
    if request.method == 'POST':
        # check if the post request has the file part
        print(request.form, request.json)
        if 'file' in request.files:
            image_array = file_to_np_array(request.files['file'], image_size)
            result = get_similar(image_array)
        else:
            print("no file found")

        if request.json and 'fileurl' in request.json:
            image_array = url_to_np_array(request.json['fileurl'], image_size)
            result = get_similar(image_array)

    return jsonify(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Application parameters')
    parser.add_argument('-p', '--port', dest='port', type=int,
                        help='port to run model', default=os.environ.get("CDSW_READONLY_PORT"))

    args, unknown = parser.parse_known_args()
    port = args.port
    app.run(port=port)
