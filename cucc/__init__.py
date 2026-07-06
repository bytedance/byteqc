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

from pyscf import scf
from byteqc.cucc import ccsd
from byteqc.cucc import dfccsd
from pyscf.lib import param


def CCSD(mf, frozen=None, mo_coeff=None, mo_occ=None, gpulim=None,
         cpulim=None, pool=None, path=param.TMPDIR, mem_ratio=0.7):
    if isinstance(mf, scf.uhf.UHF) or isinstance(mf, scf.ghf.GHF):
        AssertionError('Not implement')
    else:
        return RCCSD(mf, frozen, mo_coeff, mo_occ, gpulim, cpulim,
                     pool=pool, path=path, mem_ratio=mem_ratio)


def RCCSD(mf, frozen=None, mo_coeff=None, mo_occ=None, gpulim=None,
          cpulim=None, pool=None, path=param.TMPDIR, mem_ratio=0.7):
    if getattr(mf, 'with_df', None):
        return dfccsd.RCCSD(mf, frozen, mo_coeff, mo_occ, gpulim,
                            cpulim, pool=pool, path=path, mem_ratio=mem_ratio)
    else:
        return ccsd.CCSD(mf, frozen, mo_coeff, mo_occ, gpulim,
                         cpulim, pool=pool, path=path, mem_ratio=mem_ratio)
