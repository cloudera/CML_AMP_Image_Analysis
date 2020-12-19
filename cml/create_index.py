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
