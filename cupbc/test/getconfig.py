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

import re
import os
_atom = {'LiH': '''
        Li 0 0 0
        H 0.1 0.2 1
        ''', 'O2': '''
        O 0 0 0
        O 1 1 1
        ''', 'Ben': '''
        C  3.2464270000  1.3122060000 3.4519890000
        H  2.9220060000  2.3568840000 3.4887630000
        C  4.2640300000  0.9024360000 4.3073250000
        H  4.7081690000  1.5835020000 5.0121600000
        C  2.6773970000  0.4210740000 2.5469400000
        H  1.8785380000  0.7357020000 1.8788790000
        C  4.7126030000 -0.4210740000 4.2630600000
        H  5.5114620000 -0.7357020000 4.9311210000
        C  3.1259700000 -0.9024360000 2.5026750000
        H  2.6818310000 -1.5835020000 1.7978400000
        C  4.1435730000 -1.3122060000 3.3580110000
        H  4.4679940000 -2.3568840000 3.3212370000
        C  0.4485730000  6.0222060000 3.4519890000
        H  0.7729940000  7.0668840000 3.4887630000
        C -0.5690300000  5.6124360000 4.3073250000
        H -1.0131690000  6.2935020000 5.0121600000
        C  1.0176030000  5.1310740000 2.5469400000
        H  1.8164620000  5.4457020000 1.8788790000
        C -1.0176030000  4.2889260000 4.2630600000
        H -1.8164620000  3.9742980000 4.9311210000
        C  0.5690300000  3.8075640000 2.5026750000
        H  1.0131690000  3.1264980000 1.7978400000
        C -0.4485730000  3.3977940000 3.3580110000
        H -0.7729940000  2.3531160000 3.3212370000
        C  4.1435730000  6.0222060000 6.7630110000
        H  4.4679940000  7.0668840000 6.7262370000
        C  3.1259700000  5.6124360000 5.9076750000
        H  2.6818310000  6.2935020000 5.2028400000
        C  4.7126030000  5.1310740000 7.6680600000
        H  5.5114620000  5.4457020000 8.3361210000
        C  2.6773970000  4.2889260000 5.9519400000
        H  1.8785380000  3.9742980000 5.2838790000
        C  4.2640300000  3.8075640000 7.7123250000
        H  4.7081690000  3.1264980000 8.4171600000
        C  3.2464270000  3.3977940000 6.8569890000
        H  2.9220060000  2.3531160000 6.8937630000
        C  0.5690300000  0.9024360000 5.9076750000
        H  1.0131690000  1.5835020000 5.2028400000
        C -0.4485730000  1.3122060000 6.7630110000
        H -0.7729940000  2.3568840000 6.7262370000
        C  1.0176030000 -0.4210740000 5.9519400000
        H  1.8164620000 -0.7357020000 5.2838790000
        C  0.4485730000 -1.3122060000 6.8569890000
        H  0.7729940000 -2.3568840000 6.8937630000
        C -1.0176030000  0.4210740000 7.6680600000
        H -1.8164620000  0.7357020000 8.3361210000
        C -0.5690300000 -0.9024360000 7.7123250000
        H -1.0131690000 -1.5835020000 8.4171600000
        ''', 'Cu': '''
        Cu  5.13523511  1.48492984     10.01696168
        Cu  2.56761756  1.48241469     10.01695751
        Cu  6.4168657   3.70477914     10.01696168
        Cu  3.85360452  3.70477914     10.01696168
        Cu  0.          0.             12.09812326
        Cu  2.56761756 -2.47289941e-03 1.20987348e+01
        Cu  1.28166718  2.22485848     12.09873482
        Cu  3.85356793  2.22485848     12.09873482
        Cu  1.28437054  0.74153168     14.18192705
        Cu  3.85086457  0.74153168     14.18192705
        Cu  2.56761756  2.9641807      14.18192705
        Cu  5.13523511  2.96482937     14.21182536
        Cu  5.13523511  1.46189307     16.27659119
        Cu  2.56761756  1.48241469     16.25540572
        Cu  6.43681613  3.71629753     16.27659119
        Cu  3.83365409  3.71629753     16.27659119
        Cu  0.          0.             18.46049735
        Cu  2.56761756  2.39718116e-03 1.83102321e+01
        Cu  1.2858848   2.22242344     18.31023213
        Cu  3.84935031  2.22242344     18.31023213
        O   0.          0.             21.47039279
        C   0.          0.             20.31276329''', 'Fe': '''
        C  -3.614273 6.348655  9.944217
        C   0.023424 1.363798  9.946342
        C  -1.136839 6.362540  9.994243
        C   2.426396 1.417096  9.993985
        C   4.956084 0.024295  9.945136
        C   4.904094 1.430753  9.941951
        C   6.148033 2.088834  9.945072
        C   3.650146 2.154619  9.988420
        C   3.676865 3.583126  9.994314
        C  -2.387321 4.196185  9.993883
        C  -2.360388 5.624789  9.988820
        C   2.521750 5.690694  9.946311
        C1  0.660664 3.883298 12.012991
        N  -1.206915 3.455921 10.006894
        N   1.194971 2.069162 10.006962
        N   0.094634 5.710174 10.006366
        N   2.496576 4.323418 10.005332
        Fe  0.645518 3.889471 10.297058
        O   0.680669 3.873948 13.184859''', 'MgO': '''
        C    0.5000000000000000  0.5000000000000000  0.7613750483183692
        O    0.5000000000000000  0.5000000000000000  0.8151282449036122
        Mg   0.0000000000000000  0.7500000000000000  0.5501714244347254
        Mg   0.2500000000000000  0.0000000000000000  0.5501714244347254
        Mg   0.0000000000000000  0.0000000000000000  0.6488613629128637
        Mg   0.2500000000000000  0.7500000000000000  0.6488613629128637
        Mg   0.0000000000000000  0.2500000000000000  0.5501714244347254
        Mg   0.2500000000000000  0.5000000000000000  0.5501714244347254
        Mg   0.0000000000000000  0.5000000000000000  0.6488613629128637
        Mg   0.2500000000000000  0.2500000000000000  0.6488613629128637
        Mg   0.5000000000000000  0.7500000000000000  0.5501714244347254
        Mg   0.7500000000000000  0.0000000000000000  0.5501714244347254
        Mg   0.5000000000000000  0.0000000000000000  0.6488613629128637
        Mg   0.7500000000000000  0.7500000000000000  0.6488613629128637
        Mg   0.5000000000000000  0.2500000000000000  0.5501714244347254
        Mg   0.7500000000000000  0.5000000000000000  0.5501714244347254
        Mg   0.5000000000000000  0.5000000000000000  0.6488613629128637
        Mg   0.7500000000000000  0.2500000000000000  0.6488613629128637
        O    0.0000000000000000  0.7500000000000000  0.6510063231247947
        O    0.2500000000000000  0.0000000000000000  0.6510063231247947
        O    0.0000000000000000  0.0000000000000000  0.5498902444531061
        O    0.2500000000000000  0.7500000000000000  0.5498902444531061
        O    0.0000000000000000  0.2500000000000000  0.6510063231247947
        O    0.2500000000000000  0.5000000000000000  0.6510063231247947
        O    0.0000000000000000  0.5000000000000000  0.5498902444531061
        O    0.2500000000000000  0.2500000000000000  0.5498902444531061
        O    0.5000000000000000  0.7500000000000000  0.6510063231247947
        O    0.7500000000000000  0.0000000000000000  0.6510063231247947
        O    0.5000000000000000  0.0000000000000000  0.5498902444531061
        O    0.7500000000000000  0.7500000000000000  0.5498902444531061
        O    0.5000000000000000  0.2500000000000000  0.6510063231247947
        O    0.7500000000000000  0.5000000000000000  0.6510063231247947
        O    0.5000000000000000  0.5000000000000000  0.5498902444531061
        O    0.7500000000000000  0.2500000000000000  0.5498902444531061
        '''}
_basis = {'dz': 'ccpvdz', 'tz': 'ccpvtz', 'qz': 'ccpvqz'}
_a = {'orth': '''
        7.3899998665 0.0000000000 0.0000000000
        0.0000000000 9.4200000763 0.0000000000
        0.0000000000 0.0000000000 6.8099999428
        ''', 'nonrth': '''
        5.13523511 0.0000000000 0.0000000000
        2.56761756,  4.44724406 0.0000000000
        0.0000000000 0.0000000000 28.38580382
        ''', 'MgO': '''
        8.4409846541513982    0.0000000000000000    0.0000000000000000
        0.0000000000000000    8.4409846541513982    0.0000000000000000
        0.0000000000000000    0.0000000000000000   21.3307384906135482
        '''}
_kmesh = {}
_omega = {}
_pseudo = {}


def getconfig():
    path = os.path.join(os.path.dirname(__file__), "config.txt")
    if not os.path.exists(path):
        print("config.txt not exists, a default one is generated!")
        with open(path, 'w') as f:
            f.write(
                "ATOM: O2\nBASIS: ccpvdz\nA: orth\nKMESH: [2, 2, 2]\nOMEGA: "
                "0.1\nPSEUDO: \nNAME: O2_tz_orth_222_0.1\nOTHERS: ")
    with open(path, 'r') as f:
        lines = f.read()
        match = re.search(
            "^ATOM:\\s*(.*)\\s*$\\n^BASIS:\\s*(.*)\\s*$\\n^A:\\s*(.*)\\s*$"
            "\\n^KMESH:\\s*(.*)\\s*$\\n^OMEGA:\\s*(.*)\\s*$\\n^PSEUDO:\\s*(.*)"
            "\\s*$\\n^NAME:\\s*(.*)\\s*$\\n^OTHERS:\\s*([\\s\\S\\n\\r]*)",
            lines, re.MULTILINE)
        if match is None:
            print("Invalid config.txt:\n%s" % (lines,))
            assert False
        atom = _atom.get(match.group(1), match.group(1))
        basis = _basis.get(match.group(2), match.group(2))
        a = _a.get(match.group(3), match.group(3))
        kmesh = eval(_kmesh.get(match.group(4), match.group(4)))
        omega = eval(_omega.get(match.group(5), match.group(5)))
        pseudo = _pseudo.get(match.group(6), match.group(6))
        name = match.group(7)
        if pseudo == '':
            pseudo = None
        if name != '':
            print(
                '**********Calculating with configure %s**********' %
                (name,))
        return atom, basis, a, kmesh, omega, pseudo, name, match.group(8)
