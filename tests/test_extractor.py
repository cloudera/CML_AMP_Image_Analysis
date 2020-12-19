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
