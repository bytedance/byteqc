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

import os

pwd = os.path.abspath(__file__)[:-9]
lib = build = os.path.join(pwd, 'lib')
build = os.path.join(lib, 'build')
print("Begin to build subpackage cupbc")
if not os.path.exists(build):
    os.system('mkdir %s' % build)
if os.system('cmake %s -B %s' % (lib, build)) == 0:
    if os.system('make -j -C %s' % build) == 0:
        print("\033[42mBuilt done for cupbc!\033[0m\n\n")
        exit()
print("\033[41mBuilt failed for cupbc!\033[0m\n\n")
exit(1)
