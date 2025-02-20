# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# This file is part of ByteQC.
#
# ByteQC includes code adapted from PySCF (https://github.com/pyscf/pyscf,
# https://github.com/hongzhouye/pyscf/tree/rsdf_direct),
# which is licensed under the Apache License 2.0. The original copyright:
#     Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
#
#     Author: Qiming Sun <osirpt.sun@gmail.com>
#     Author: Hong-Zhou Ye <hzyechem@gmail.com>
#

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler")

add_library(gaft SHARED ft_ao.cu ft_ao_latsum.cu ft_ao_bvk.cu ft_ao_bvk_ins1.cu ft_ao_bvk_ins2.cu ft_ao_bvk_ins3.cu ft_ao_bvk_ins4.cu ft_ao_bvk_ins5.cu ft_ao_bvk_ins6.cu ft_ao_bvk_ins7.cu ft_ao_latsum_ins1.cu ft_ao_latsum_ins2.cu ft_ao_latsum_ins3.cu ft_ao_latsum_ins4.cu ft_ao_latsum_ins5.cu ft_ao_latsum_ins6.cu ft_ao_latsum_ins7.cu ft_ao_GTO.cu)
# add_dependencies(gaft cgto cvhf)

set_target_properties(gaft PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(gaft -lcublas -std=c++11 ${BLAS_LIBRARIES} ${OPENMP_C_PROPERTIES})
