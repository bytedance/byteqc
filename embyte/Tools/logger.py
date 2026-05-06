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
import shutil


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

    def close(self):
        for handler in list(self.logger.handlers):
            try:
                handler.flush()
            except Exception:
                pass
            try:
                handler.close()
            except Exception:
                pass
            self.logger.removeHandler(handler)
        logging.Logger.manager.loggerDict.pop(self.logger.name, None)


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
            pickle.dump(class_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_class(self, filename):

        with open(filename, 'rb') as f:
            rq = pickle.load(f)
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
            pickle.dump(class_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_class(self, filename):

        with open(filename, 'rb') as f:
            rq = pickle.load(f)
            return rq

    def _filemp_block_size(self, shape):
        from byteqc import lib

        nproc = max(int(getattr(lib, 'NumFileProcess', 1)), 1)
        return (max(int((shape[0] + nproc - 1) // nproc), 1),)

    def save_array_mp(self, obj, obj_name, dataset_name='array'):
        from byteqc import lib
        from multiprocessing import Pool

        self.recorder[obj_name] = self.filepath + '/' + obj_name
        self.save()
        self.delet_obj(obj_name)

        try:
            import cupy
            is_gpu_array = isinstance(obj, cupy.ndarray)
        except Exception:
            is_gpu_array = False

        if is_gpu_array:
            arr_cpu = obj.get(blocking=True)
        else:
            arr_cpu = numpy.asarray(obj)
        if not arr_cpu.flags.c_contiguous:
            arr_cpu = numpy.ascontiguousarray(arr_cpu)

        with lib.FileMp(self.recorder[obj_name], 'w') as filemp:
            dataset = filemp.create_dataset(
                dataset_name,
                shape=arr_cpu.shape,
                dtype=arr_cpu.dtype,
                blksizes=self._filemp_block_size(arr_cpu.shape),
            )
            pool = Pool(processes=max(int(lib.NumFileProcess), 1))
            try:
                waits = dataset.setitem(numpy.s_[:], arr_cpu, pool=pool)
                for wait in waits:
                    wait.wait()
            finally:
                pool.close()
                pool.join()
        del arr_cpu

    def load_array_mp(self, filename, dataset_name='array'):
        from byteqc import lib
        from multiprocessing import Pool
        import cupyx

        try:
            with lib.FileMp(filename, 'r') as filemp:
                dataset = filemp[dataset_name]
                buf = cupyx.empty_pinned(dataset.shape, dtype=dataset.dtype)
                pool = Pool(processes=max(int(lib.NumFileProcess), 1))
                try:
                    arr = dataset.getitem(numpy.s_[:], pool=pool, buf=buf)
                    arr.wait()
                finally:
                    pool.close()
                    pool.join()
            return buf
        except (OSError, KeyError, ValueError, TypeError):
            return self.load_class(filename)

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
            with h5py.File(path, 'r') as f:
                for value in f.values():
                    if not hasattr(value, 'dtype'):
                        continue
                    if value.dtype.char == 'O' and value.shape == ():
                        marker = value[()]
                        if isinstance(marker, bytes) and marker[:15] == b'#!*DatasetMp*!#':
                            shutil.rmtree(str(marker[15:], 'utf-8'), ignore_errors=True)
        except BaseException:
            pass
        try:
            os.remove(path)
        except BaseException:
            pass
        dirname, basename = os.path.split(path)
        shutil.rmtree(os.path.join(dirname, basename.rsplit('.', 1)[0] + '_Mp'), ignore_errors=True)
