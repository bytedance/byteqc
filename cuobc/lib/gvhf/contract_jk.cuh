/*
Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
This file is part of ByteQC.

ByteQC includes code adapted from PySCF (https://github.com/pyscf/pyscf)
and GPU4PySCF (https://github.com/bytedance/gpu4pyscf),
which are licensed under the Apache License 2.0.
The original copyright:
    Copyright 2014-2020 The GPU4PySCF/PySCF Developers. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

template <int NROOTS>
__device__ void GINTkernel_getjk(GINTEnvVars &envs, BasisProdCache &bpcache,
    JKMatrix &jk, double *g, int ish, int jsh, int ksh, int lsh, int igroup);

template <int I, int J, int K, int L>
__device__ void GINTkernel_getjk(GINTEnvVars &envs, BasisProdCache &bpcache,
    JKMatrix &jk, double *g, int ish, int jsh, int ksh, int lsh, int igroup);
