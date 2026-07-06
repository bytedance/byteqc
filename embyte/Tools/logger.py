# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# This file is part of ByteQC.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https: // www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from logging import handlers
import os
import re
import _pickle as pickle
import json
import h5py
import numpy


class Logger(object):
    '''
    Logger and checkpoint system for embyte.
    '''

    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL,
    }

    def __init__(self, filename, level='info', backCount=10,
                 fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        recover = False
        if os.path.exists(filename):
            recover = True
        filename = re.sub(r'/+', '/', filename)
        self.filename = filename
        self.filepath = self.filename[:self.filename.rfind('/') + 1]
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))

        if not self.logger.handlers:
            sh = logging.StreamHandler()
            sh.setFormatter(format_str)
            self.logger.addHandler(sh)

            fh = handlers.RotatingFileHandler(
                filename=filename)
            fh.setLevel(self.level_relations.get(level))
            fh.setFormatter(format_str)
            self.logger.addHandler(fh)

        self.logger.write = lambda x: self.logger.info(
            x) if (x != '\n') else None

        self.logger.flush = lambda: None

        if recover:
            self.logger.info('-------------------------recover---------------')


class Process_Record:
    def __init__(self, filename, chk_point):
        self.chk_point = chk_point
        filename = re.sub(r'/+', '/', filename)
        self.filename = filename
        self.filepath = self.filename[:self.filename.rfind('/') + 1]
        if os.path.exists(filename):
            with open(filename, 'r') as jsonfile:
                self.recorder = json.load(jsonfile)
        else:
            self.recorder = {
                'HF_chkfile': False,
                'low_level_info_class': False,
                'Cluster': False,
                'energy': False,
                'used_orb_num': False,
                'frag_CE': False,
                'fragment_group': False,

                'subspace_coeff_step': False,

                'eri_step': False,
                'cderi_step': False,
                'cderi_cluster': False,

                'subspace_MP2_step': False,
                'subspace_MP2_cluster': False,

                'cluster_eri_step': False,
                'cluster_cderi_step': False,
                'cluster_cderi_cluster': False,

            }
            with open(filename, 'w') as jsonfile:
                json.dump(self.recorder, jsonfile, indent=4)

    def save(self):
        with open(self.filename, 'w') as jsonfile:
            json.dump(self.recorder, jsonfile, indent=4)

    def save_class(self, class_obj, filename):

        with open(filename, 'wb') as f:
            str_obj = pickle.dumps(class_obj)
            f.write(str_obj)

    def load_class(self, filename):

        with open(filename, 'rb') as f:
            rq = pickle.loads(f.read())
            return rq


class Process_Record_cluster:
    def __init__(self, filename):
        filename = re.sub(r'/+', '/', filename)
        self.filename = filename + '/cluster_recorder'
        self.filepath = self.filename[:self.filename.rfind('/') + 1]

        if os.path.exists(self.filename):
            with open(self.filename, 'r') as jsonfile:
                self.recorder = json.load(jsonfile)
        else:
            self.recorder = {
                'stage': {
                    '0': False,
                    '1': False,
                },
            }
            with open(self.filename, 'w') as jsonfile:
                json.dump(self.recorder, jsonfile, indent=4)

    def save(self):
        with open(self.filename, 'w') as jsonfile:
            json.dump(self.recorder, jsonfile, indent=4)

    def save_class(self, class_obj, filename):

        with open(filename, 'wb') as f:
            str_obj = pickle.dumps(class_obj)
            f.write(str_obj)

    def load_class(self, filename):

        with open(filename, 'rb') as f:
            rq = pickle.loads(f.read())
            return rq

    def save_obj(self, obj, obj_name):
        self.recorder[obj_name] = self.filepath + '/' + obj_name
        self.save()
        if obj_name != 'two_ele':
            self.save_class(obj, self.recorder[obj_name])
        else:
            with h5py.File(self.recorder[obj_name], 'w') as f:
                if obj.shape[0] != obj.shape[1]:
                    f.create_dataset('j3c', data=obj, dtype=numpy.float64)
                else:
                    assert False

    def delet_obj(self, obj_name):
        path = os.path.join(self.filepath, obj_name)
        try:
            os.remove(path)
        except BaseException:
            pass
