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


from lib import FaissIndex
import numpy as np
import os


d = 64                           # dimension
nb = 100000                      # database size
nq = 10000                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

index = FaissIndex(d)


def test_faiss_create():
    index.add(xb, id_strings=list(range(0, xb.shape[0])))
    k = 6
    distances, idx = index.search(xb[:5], k)
    print(distances.shape, idx.shape)
    assert(distances.shape == (5, 6) and idx.shape == (5, 6))


def test_faiss_save():
    index.save(os.getcwd() + "/faiss_test")
    assert(os.path.isfile(os.getcwd() + "/faiss_test/faiss.index")
           and os.path.isfile(os.getcwd() + "/faiss_test/faiss.map"))


def test_faiss_load():
    index = FaissIndex(0)
    index.load(load_dir=os.getcwd() + "/faiss_test")
    assert(len(index.id_map) > 0)
