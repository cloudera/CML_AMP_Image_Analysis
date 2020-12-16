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
