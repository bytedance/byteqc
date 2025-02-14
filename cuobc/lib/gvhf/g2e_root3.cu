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

__global__ static void GINTint2e_jk_kernel1111(JKMatrix jk,
    BasisProdOffsets offsets, GINTEnvVars envs, BasisProdCache bpcache) {
    int ntasks_ij = offsets.ntasks_ij;
    long ntasks = ntasks_ij * offsets.ntasks_kl;
    long task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int igroup = nprim_ij * nprim_kl;
    ntasks *= igroup;
    if (task_ij >= ntasks)
        return;
    int kl = task_ij % nprim_kl;
    task_ij /= nprim_kl;
    int ij = task_ij % nprim_ij;
    task_ij /= nprim_ij;
    int task_kl = task_ij / ntasks_ij;
    task_ij = task_ij % ntasks_ij;

    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    if (bas_ij < bas_kl) {
        return;
    }
    double norm = envs.fac;
    if (bas_ij == bas_kl) {
        norm *= .5;
    }

    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *ao_loc = bpcache.ao_loc;
    int *bas_pair2bra = bpcache.bas_pair2bra;
    int *bas_pair2ket = bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    int k0 = ao_loc[ksh];
    int l0 = ao_loc[lsh];
    double *__restrict__ a12 = bpcache.a12;
    double *__restrict__ e12 = bpcache.e12;
    double *__restrict__ x12 = bpcache.x12;
    double *__restrict__ y12 = bpcache.y12;
    double *__restrict__ z12 = bpcache.z12;
    int i_dm;
    int nbas = bpcache.nbas;
    double *__restrict__ bas_x = bpcache.bas_coords;
    double *__restrict__ bas_y = bas_x + nbas;
    double *__restrict__ bas_z = bas_y + nbas;

    double gout0 = 0;
    double gout1 = 0;
    double gout2 = 0;
    double gout3 = 0;
    double gout4 = 0;
    double gout5 = 0;
    double gout6 = 0;
    double gout7 = 0;
    double gout8 = 0;
    double gout9 = 0;
    double gout10 = 0;
    double gout11 = 0;
    double gout12 = 0;
    double gout13 = 0;
    double gout14 = 0;
    double gout15 = 0;
    double gout16 = 0;
    double gout17 = 0;
    double gout18 = 0;
    double gout19 = 0;
    double gout20 = 0;
    double gout21 = 0;
    double gout22 = 0;
    double gout23 = 0;
    double gout24 = 0;
    double gout25 = 0;
    double gout26 = 0;
    double gout27 = 0;
    double gout28 = 0;
    double gout29 = 0;
    double gout30 = 0;
    double gout31 = 0;
    double gout32 = 0;
    double gout33 = 0;
    double gout34 = 0;
    double gout35 = 0;
    double gout36 = 0;
    double gout37 = 0;
    double gout38 = 0;
    double gout39 = 0;
    double gout40 = 0;
    double gout41 = 0;
    double gout42 = 0;
    double gout43 = 0;
    double gout44 = 0;
    double gout45 = 0;
    double gout46 = 0;
    double gout47 = 0;
    double gout48 = 0;
    double gout49 = 0;
    double gout50 = 0;
    double gout51 = 0;
    double gout52 = 0;
    double gout53 = 0;
    double gout54 = 0;
    double gout55 = 0;
    double gout56 = 0;
    double gout57 = 0;
    double gout58 = 0;
    double gout59 = 0;
    double gout60 = 0;
    double gout61 = 0;
    double gout62 = 0;
    double gout63 = 0;
    double gout64 = 0;
    double gout65 = 0;
    double gout66 = 0;
    double gout67 = 0;
    double gout68 = 0;
    double gout69 = 0;
    double gout70 = 0;
    double gout71 = 0;
    double gout72 = 0;
    double gout73 = 0;
    double gout74 = 0;
    double gout75 = 0;
    double gout76 = 0;
    double gout77 = 0;
    double gout78 = 0;
    double gout79 = 0;
    double gout80 = 0;
    double xi = bas_x[ish];
    double yi = bas_y[ish];
    double zi = bas_z[ish];
    double xixj = xi - bas_x[jsh];
    double yiyj = yi - bas_y[jsh];
    double zizj = zi - bas_z[jsh];
    double xk = bas_x[ksh];
    double yk = bas_y[ksh];
    double zk = bas_z[ksh];
    double xkxl = xk - bas_x[lsh];
    double ykyl = yk - bas_y[lsh];
    double zkzl = zk - bas_z[lsh];
    auto reduce = SegReduce<double>(igroup);
    ij += prim_ij;
    kl += prim_kl;
    double aij = a12[ij];
    double eij = e12[ij];
    double xij = x12[ij];
    double yij = y12[ij];
    double zij = z12[ij];
    double akl = a12[kl];
    double ekl = e12[kl];
    double xkl = x12[kl];
    double ykl = y12[kl];
    double zkl = z12[kl];
    double xijxkl = xij - xkl;
    double yijykl = yij - ykl;
    double zijzkl = zij - zkl;
    double aijkl = aij + akl;
    double a1 = aij * akl;
    double a0 = a1 / aijkl;
    double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
    double fac = norm * eij * ekl / (sqrt(aijkl) * a1);

    double rw[6];
    double root0, weight0;
    GINTrys_root<3>(x, rw);
    int irys;
    for (irys = 0; irys < 3; ++irys) {
        root0 = rw[irys];
        weight0 = rw[irys + 3];
        double u2 = a0 * root0;
        double tmp4 = .5 / (u2 * aijkl + a1);
        double b00 = u2 * tmp4;
        double tmp1 = 2 * b00;
        double tmp2 = tmp1 * akl;
        double b10 = b00 + tmp4 * akl;
        double c00x = xij - xi - tmp2 * xijxkl;
        double c00y = yij - yi - tmp2 * yijykl;
        double c00z = zij - zi - tmp2 * zijzkl;
        double tmp3 = tmp1 * aij;
        double b01 = b00 + tmp4 * aij;
        double c0px = xkl - xk + tmp3 * xijxkl;
        double c0py = ykl - yk + tmp3 * yijykl;
        double c0pz = zkl - zk + tmp3 * zijzkl;
        double g_0 = 1;
        double g_1 = c00x;
        double g_2 = c00x + xixj;
        double g_3 = c00x * (c00x + xixj) + b10;
        double g_4 = c0px;
        double g_5 = c0px * c00x + b00;
        double g_6 = c0px * (c00x + xixj) + b00;
        double g_7 = b00 * c00x + b10 * c0px + c00x * g_5 + xixj * g_5;
        double g_8 = c0px + xkxl;
        double g_9 = c00x * (c0px + xkxl) + b00;
        double g_10 = xkxl * (xixj + c00x) + xixj * c0px + c0px * c00x + b00;
        double g_11 = xkxl * (xixj * c00x + c00x * c00x + b10) + xixj * g_5 +
                      c00x * g_5 + b10 * c0px + b00 * c00x;
        double g_12 = c0px * (c0px + xkxl) + b01;
        double g_13 = b00 * c0px + b01 * c00x + c0px * g_5 + xkxl * g_5;
        double g_14 = xkxl * (xixj * c0px + c0px * c00x + b00) +
                      xixj * (c0px * c0px + b01) + c0px * g_5 + b01 * c00x +
                      b00 * c0px;
        double g_15 =
            xkxl * (xixj * g_5 + c00x * g_5 + b10 * c0px + b00 * c00x) +
            xixj * (c0px * g_5 + b01 * c00x + b00 * c0px) +
            c00x * (c0px * g_5 + b01 * c00x + b00 * c0px) +
            b10 * (c0px * c0px + b01) + 2 * b00 * g_5;
        double g_16 = 1;
        double g_17 = c00y;
        double g_18 = c00y + yiyj;
        double g_19 = c00y * (c00y + yiyj) + b10;
        double g_20 = c0py;
        double g_21 = c0py * c00y + b00;
        double g_22 = c0py * (c00y + yiyj) + b00;
        double g_23 = b00 * c00y + b10 * c0py + c00y * g_21 + yiyj * g_21;
        double g_24 = c0py + ykyl;
        double g_25 = c00y * (c0py + ykyl) + b00;
        double g_26 = ykyl * (yiyj + c00y) + yiyj * c0py + c0py * c00y + b00;
        double g_27 = ykyl * (yiyj * c00y + c00y * c00y + b10) + yiyj * g_21 +
                      c00y * g_21 + b10 * c0py + b00 * c00y;
        double g_28 = c0py * (c0py + ykyl) + b01;
        double g_29 = b00 * c0py + b01 * c00y + c0py * g_21 + ykyl * g_21;
        double g_30 = ykyl * (yiyj * c0py + c0py * c00y + b00) +
                      yiyj * (c0py * c0py + b01) + c0py * g_21 + b01 * c00y +
                      b00 * c0py;
        double g_31 =
            ykyl * (yiyj * g_21 + c00y * g_21 + b10 * c0py + b00 * c00y) +
            yiyj * (c0py * g_21 + b01 * c00y + b00 * c0py) +
            c00y * (c0py * g_21 + b01 * c00y + b00 * c0py) +
            b10 * (c0py * c0py + b01) + 2 * b00 * g_21;
        double g_32 = weight0 * fac;
        double g_33 = c00z * g_32;
        double g_34 = g_32 * (c00z + zizj);
        double g_35 = b10 * g_32 + c00z * g_33 + zizj * g_33;
        double g_36 = c0pz * g_32;
        double g_37 = b00 * g_32 + c0pz * g_33;
        double g_38 = b00 * g_32 + c0pz * g_33 + zizj * g_36;
        double g_39 = b00 * g_33 + b10 * g_36 + c00z * g_37 + zizj * g_37;
        double g_40 = g_32 * (c0pz + zkzl);
        double g_41 = b00 * g_32 + c0pz * g_33 + zkzl * g_33;
        double g_42 = zkzl * (zizj * g_32 + c00z * g_32) + zizj * g_36 +
                      c0pz * g_33 + b00 * g_32;
        double g_43 = zkzl * (zizj * g_33 + c00z * g_33 + b10 * g_32) +
                      zizj * g_37 + c00z * g_37 + b10 * g_36 + b00 * g_33;
        double g_44 = b01 * g_32 + c0pz * g_36 + zkzl * g_36;
        double g_45 = b00 * g_36 + b01 * g_33 + c0pz * g_37 + zkzl * g_37;
        double g_46 = zkzl * (zizj * g_36 + c0pz * g_33 + b00 * g_32) +
                      zizj * (c0pz * g_36 + b01 * g_32) + c0pz * g_37 +
                      b01 * g_33 + b00 * g_36;
        double g_47 =
            zkzl * (zizj * g_37 + c00z * g_37 + b10 * g_36 + b00 * g_33) +
            zizj * (c0pz * g_37 + b01 * g_33 + b00 * g_36) +
            c00z * (c0pz * g_37 + b01 * g_33 + b00 * g_36) +
            b10 * (c0pz * g_36 + b01 * g_32) + 2 * b00 * g_37;
        gout0 += g_15 * g_16 * g_32;
        gout1 += g_14 * g_17 * g_32;
        gout2 += g_14 * g_16 * g_33;
        gout3 += g_13 * g_18 * g_32;
        gout4 += g_12 * g_19 * g_32;
        gout5 += g_12 * g_18 * g_33;
        gout6 += g_13 * g_16 * g_34;
        gout7 += g_12 * g_17 * g_34;
        gout8 += g_12 * g_16 * g_35;
        gout9 += g_11 * g_20 * g_32;
        gout10 += g_10 * g_21 * g_32;
        gout11 += g_10 * g_20 * g_33;
        gout12 += g_9 * g_22 * g_32;
        gout13 += g_8 * g_23 * g_32;
        gout14 += g_8 * g_22 * g_33;
        gout15 += g_9 * g_20 * g_34;
        gout16 += g_8 * g_21 * g_34;
        gout17 += g_8 * g_20 * g_35;
        gout18 += g_11 * g_16 * g_36;
        gout19 += g_10 * g_17 * g_36;
        gout20 += g_10 * g_16 * g_37;
        gout21 += g_9 * g_18 * g_36;
        gout22 += g_8 * g_19 * g_36;
        gout23 += g_8 * g_18 * g_37;
        gout24 += g_9 * g_16 * g_38;
        gout25 += g_8 * g_17 * g_38;
        gout26 += g_8 * g_16 * g_39;
        gout27 += g_7 * g_24 * g_32;
        gout28 += g_6 * g_25 * g_32;
        gout29 += g_6 * g_24 * g_33;
        gout30 += g_5 * g_26 * g_32;
        gout31 += g_4 * g_27 * g_32;
        gout32 += g_4 * g_26 * g_33;
        gout33 += g_5 * g_24 * g_34;
        gout34 += g_4 * g_25 * g_34;
        gout35 += g_4 * g_24 * g_35;
        gout36 += g_3 * g_28 * g_32;
        gout37 += g_2 * g_29 * g_32;
        gout38 += g_2 * g_28 * g_33;
        gout39 += g_1 * g_30 * g_32;
        gout40 += g_0 * g_31 * g_32;
        gout41 += g_0 * g_30 * g_33;
        gout42 += g_1 * g_28 * g_34;
        gout43 += g_0 * g_29 * g_34;
        gout44 += g_0 * g_28 * g_35;
        gout45 += g_3 * g_24 * g_36;
        gout46 += g_2 * g_25 * g_36;
        gout47 += g_2 * g_24 * g_37;
        gout48 += g_1 * g_26 * g_36;
        gout49 += g_0 * g_27 * g_36;
        gout50 += g_0 * g_26 * g_37;
        gout51 += g_1 * g_24 * g_38;
        gout52 += g_0 * g_25 * g_38;
        gout53 += g_0 * g_24 * g_39;
        gout54 += g_7 * g_16 * g_40;
        gout55 += g_6 * g_17 * g_40;
        gout56 += g_6 * g_16 * g_41;
        gout57 += g_5 * g_18 * g_40;
        gout58 += g_4 * g_19 * g_40;
        gout59 += g_4 * g_18 * g_41;
        gout60 += g_5 * g_16 * g_42;
        gout61 += g_4 * g_17 * g_42;
        gout62 += g_4 * g_16 * g_43;
        gout63 += g_3 * g_20 * g_40;
        gout64 += g_2 * g_21 * g_40;
        gout65 += g_2 * g_20 * g_41;
        gout66 += g_1 * g_22 * g_40;
        gout67 += g_0 * g_23 * g_40;
        gout68 += g_0 * g_22 * g_41;
        gout69 += g_1 * g_20 * g_42;
        gout70 += g_0 * g_21 * g_42;
        gout71 += g_0 * g_20 * g_43;
        gout72 += g_3 * g_16 * g_44;
        gout73 += g_2 * g_17 * g_44;
        gout74 += g_2 * g_16 * g_45;
        gout75 += g_1 * g_18 * g_44;
        gout76 += g_0 * g_19 * g_44;
        gout77 += g_0 * g_18 * g_45;
        gout78 += g_1 * g_16 * g_46;
        gout79 += g_0 * g_17 * g_46;
        gout80 += g_0 * g_16 * g_47;
    }
    double d_0, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8;
    int n_dm = jk.n_dm;
    int nao = jk.nao;
    size_t nao2 = nao * nao;
    double *__restrict__ dm = jk.dm;
    double *vj = jk.vj;
    double *vk = jk.vk;
    for (i_dm = 0; i_dm < n_dm; ++i_dm) {
        if (vj != NULL) {
            // ijkl,ij->kl
            d_0 = dm[(i0 + 0) + nao * (j0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (j0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (j0 + 0)];
            d_3 = dm[(i0 + 0) + nao * (j0 + 1)];
            d_4 = dm[(i0 + 1) + nao * (j0 + 1)];
            d_5 = dm[(i0 + 2) + nao * (j0 + 1)];
            d_6 = dm[(i0 + 0) + nao * (j0 + 2)];
            d_7 = dm[(i0 + 1) + nao * (j0 + 2)];
            d_8 = dm[(i0 + 2) + nao * (j0 + 2)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout3 * d_3 +
                       gout4 * d_4 + gout5 * d_5 + gout6 * d_6 + gout7 * d_7 +
                       gout8 * d_8,
                vj + (k0 + 0) + nao * (l0 + 0));
            reduce(gout9 * d_0 + gout10 * d_1 + gout11 * d_2 + gout12 * d_3 +
                       gout13 * d_4 + gout14 * d_5 + gout15 * d_6 +
                       gout16 * d_7 + gout17 * d_8,
                vj + (k0 + 1) + nao * (l0 + 0));
            reduce(gout18 * d_0 + gout19 * d_1 + gout20 * d_2 + gout21 * d_3 +
                       gout22 * d_4 + gout23 * d_5 + gout24 * d_6 +
                       gout25 * d_7 + gout26 * d_8,
                vj + (k0 + 2) + nao * (l0 + 0));
            reduce(gout27 * d_0 + gout28 * d_1 + gout29 * d_2 + gout30 * d_3 +
                       gout31 * d_4 + gout32 * d_5 + gout33 * d_6 +
                       gout34 * d_7 + gout35 * d_8,
                vj + (k0 + 0) + nao * (l0 + 1));
            reduce(gout36 * d_0 + gout37 * d_1 + gout38 * d_2 + gout39 * d_3 +
                       gout40 * d_4 + gout41 * d_5 + gout42 * d_6 +
                       gout43 * d_7 + gout44 * d_8,
                vj + (k0 + 1) + nao * (l0 + 1));
            reduce(gout45 * d_0 + gout46 * d_1 + gout47 * d_2 + gout48 * d_3 +
                       gout49 * d_4 + gout50 * d_5 + gout51 * d_6 +
                       gout52 * d_7 + gout53 * d_8,
                vj + (k0 + 2) + nao * (l0 + 1));
            reduce(gout54 * d_0 + gout55 * d_1 + gout56 * d_2 + gout57 * d_3 +
                       gout58 * d_4 + gout59 * d_5 + gout60 * d_6 +
                       gout61 * d_7 + gout62 * d_8,
                vj + (k0 + 0) + nao * (l0 + 2));
            reduce(gout63 * d_0 + gout64 * d_1 + gout65 * d_2 + gout66 * d_3 +
                       gout67 * d_4 + gout68 * d_5 + gout69 * d_6 +
                       gout70 * d_7 + gout71 * d_8,
                vj + (k0 + 1) + nao * (l0 + 2));
            reduce(gout72 * d_0 + gout73 * d_1 + gout74 * d_2 + gout75 * d_3 +
                       gout76 * d_4 + gout77 * d_5 + gout78 * d_6 +
                       gout79 * d_7 + gout80 * d_8,
                vj + (k0 + 2) + nao * (l0 + 2));
            // ijkl,kl->ij
            d_0 = dm[(k0 + 0) + nao * (l0 + 0)];
            d_1 = dm[(k0 + 1) + nao * (l0 + 0)];
            d_2 = dm[(k0 + 2) + nao * (l0 + 0)];
            d_3 = dm[(k0 + 0) + nao * (l0 + 1)];
            d_4 = dm[(k0 + 1) + nao * (l0 + 1)];
            d_5 = dm[(k0 + 2) + nao * (l0 + 1)];
            d_6 = dm[(k0 + 0) + nao * (l0 + 2)];
            d_7 = dm[(k0 + 1) + nao * (l0 + 2)];
            d_8 = dm[(k0 + 2) + nao * (l0 + 2)];
            reduce(gout0 * d_0 + gout9 * d_1 + gout18 * d_2 + gout27 * d_3 +
                       gout36 * d_4 + gout45 * d_5 + gout54 * d_6 +
                       gout63 * d_7 + gout72 * d_8,
                vj + (i0 + 0) + nao * (j0 + 0));
            reduce(gout1 * d_0 + gout10 * d_1 + gout19 * d_2 + gout28 * d_3 +
                       gout37 * d_4 + gout46 * d_5 + gout55 * d_6 +
                       gout64 * d_7 + gout73 * d_8,
                vj + (i0 + 1) + nao * (j0 + 0));
            reduce(gout2 * d_0 + gout11 * d_1 + gout20 * d_2 + gout29 * d_3 +
                       gout38 * d_4 + gout47 * d_5 + gout56 * d_6 +
                       gout65 * d_7 + gout74 * d_8,
                vj + (i0 + 2) + nao * (j0 + 0));
            reduce(gout3 * d_0 + gout12 * d_1 + gout21 * d_2 + gout30 * d_3 +
                       gout39 * d_4 + gout48 * d_5 + gout57 * d_6 +
                       gout66 * d_7 + gout75 * d_8,
                vj + (i0 + 0) + nao * (j0 + 1));
            reduce(gout4 * d_0 + gout13 * d_1 + gout22 * d_2 + gout31 * d_3 +
                       gout40 * d_4 + gout49 * d_5 + gout58 * d_6 +
                       gout67 * d_7 + gout76 * d_8,
                vj + (i0 + 1) + nao * (j0 + 1));
            reduce(gout5 * d_0 + gout14 * d_1 + gout23 * d_2 + gout32 * d_3 +
                       gout41 * d_4 + gout50 * d_5 + gout59 * d_6 +
                       gout68 * d_7 + gout77 * d_8,
                vj + (i0 + 2) + nao * (j0 + 1));
            reduce(gout6 * d_0 + gout15 * d_1 + gout24 * d_2 + gout33 * d_3 +
                       gout42 * d_4 + gout51 * d_5 + gout60 * d_6 +
                       gout69 * d_7 + gout78 * d_8,
                vj + (i0 + 0) + nao * (j0 + 2));
            reduce(gout7 * d_0 + gout16 * d_1 + gout25 * d_2 + gout34 * d_3 +
                       gout43 * d_4 + gout52 * d_5 + gout61 * d_6 +
                       gout70 * d_7 + gout79 * d_8,
                vj + (i0 + 1) + nao * (j0 + 2));
            reduce(gout8 * d_0 + gout17 * d_1 + gout26 * d_2 + gout35 * d_3 +
                       gout44 * d_4 + gout53 * d_5 + gout62 * d_6 +
                       gout71 * d_7 + gout80 * d_8,
                vj + (i0 + 2) + nao * (j0 + 2));
            vj += nao2;
        }
        if (vk != NULL) {
            // ijkl,jl->ik
            d_0 = dm[(j0 + 0) + nao * (l0 + 0)];
            d_1 = dm[(j0 + 1) + nao * (l0 + 0)];
            d_2 = dm[(j0 + 2) + nao * (l0 + 0)];
            d_3 = dm[(j0 + 0) + nao * (l0 + 1)];
            d_4 = dm[(j0 + 1) + nao * (l0 + 1)];
            d_5 = dm[(j0 + 2) + nao * (l0 + 1)];
            d_6 = dm[(j0 + 0) + nao * (l0 + 2)];
            d_7 = dm[(j0 + 1) + nao * (l0 + 2)];
            d_8 = dm[(j0 + 2) + nao * (l0 + 2)];
            reduce(gout0 * d_0 + gout3 * d_1 + gout6 * d_2 + gout27 * d_3 +
                       gout30 * d_4 + gout33 * d_5 + gout54 * d_6 +
                       gout57 * d_7 + gout60 * d_8,
                vk + (i0 + 0) + nao * (k0 + 0));
            reduce(gout1 * d_0 + gout4 * d_1 + gout7 * d_2 + gout28 * d_3 +
                       gout31 * d_4 + gout34 * d_5 + gout55 * d_6 +
                       gout58 * d_7 + gout61 * d_8,
                vk + (i0 + 1) + nao * (k0 + 0));
            reduce(gout2 * d_0 + gout5 * d_1 + gout8 * d_2 + gout29 * d_3 +
                       gout32 * d_4 + gout35 * d_5 + gout56 * d_6 +
                       gout59 * d_7 + gout62 * d_8,
                vk + (i0 + 2) + nao * (k0 + 0));
            reduce(gout9 * d_0 + gout12 * d_1 + gout15 * d_2 + gout36 * d_3 +
                       gout39 * d_4 + gout42 * d_5 + gout63 * d_6 +
                       gout66 * d_7 + gout69 * d_8,
                vk + (i0 + 0) + nao * (k0 + 1));
            reduce(gout10 * d_0 + gout13 * d_1 + gout16 * d_2 + gout37 * d_3 +
                       gout40 * d_4 + gout43 * d_5 + gout64 * d_6 +
                       gout67 * d_7 + gout70 * d_8,
                vk + (i0 + 1) + nao * (k0 + 1));
            reduce(gout11 * d_0 + gout14 * d_1 + gout17 * d_2 + gout38 * d_3 +
                       gout41 * d_4 + gout44 * d_5 + gout65 * d_6 +
                       gout68 * d_7 + gout71 * d_8,
                vk + (i0 + 2) + nao * (k0 + 1));
            reduce(gout18 * d_0 + gout21 * d_1 + gout24 * d_2 + gout45 * d_3 +
                       gout48 * d_4 + gout51 * d_5 + gout72 * d_6 +
                       gout75 * d_7 + gout78 * d_8,
                vk + (i0 + 0) + nao * (k0 + 2));
            reduce(gout19 * d_0 + gout22 * d_1 + gout25 * d_2 + gout46 * d_3 +
                       gout49 * d_4 + gout52 * d_5 + gout73 * d_6 +
                       gout76 * d_7 + gout79 * d_8,
                vk + (i0 + 1) + nao * (k0 + 2));
            reduce(gout20 * d_0 + gout23 * d_1 + gout26 * d_2 + gout47 * d_3 +
                       gout50 * d_4 + gout53 * d_5 + gout74 * d_6 +
                       gout77 * d_7 + gout80 * d_8,
                vk + (i0 + 2) + nao * (k0 + 2));
            // ijkl,jk->il
            d_0 = dm[(j0 + 0) + nao * (k0 + 0)];
            d_1 = dm[(j0 + 1) + nao * (k0 + 0)];
            d_2 = dm[(j0 + 2) + nao * (k0 + 0)];
            d_3 = dm[(j0 + 0) + nao * (k0 + 1)];
            d_4 = dm[(j0 + 1) + nao * (k0 + 1)];
            d_5 = dm[(j0 + 2) + nao * (k0 + 1)];
            d_6 = dm[(j0 + 0) + nao * (k0 + 2)];
            d_7 = dm[(j0 + 1) + nao * (k0 + 2)];
            d_8 = dm[(j0 + 2) + nao * (k0 + 2)];
            reduce(gout0 * d_0 + gout3 * d_1 + gout6 * d_2 + gout9 * d_3 +
                       gout12 * d_4 + gout15 * d_5 + gout18 * d_6 +
                       gout21 * d_7 + gout24 * d_8,
                vk + (i0 + 0) + nao * (l0 + 0));
            reduce(gout1 * d_0 + gout4 * d_1 + gout7 * d_2 + gout10 * d_3 +
                       gout13 * d_4 + gout16 * d_5 + gout19 * d_6 +
                       gout22 * d_7 + gout25 * d_8,
                vk + (i0 + 1) + nao * (l0 + 0));
            reduce(gout2 * d_0 + gout5 * d_1 + gout8 * d_2 + gout11 * d_3 +
                       gout14 * d_4 + gout17 * d_5 + gout20 * d_6 +
                       gout23 * d_7 + gout26 * d_8,
                vk + (i0 + 2) + nao * (l0 + 0));
            reduce(gout27 * d_0 + gout30 * d_1 + gout33 * d_2 + gout36 * d_3 +
                       gout39 * d_4 + gout42 * d_5 + gout45 * d_6 +
                       gout48 * d_7 + gout51 * d_8,
                vk + (i0 + 0) + nao * (l0 + 1));
            reduce(gout28 * d_0 + gout31 * d_1 + gout34 * d_2 + gout37 * d_3 +
                       gout40 * d_4 + gout43 * d_5 + gout46 * d_6 +
                       gout49 * d_7 + gout52 * d_8,
                vk + (i0 + 1) + nao * (l0 + 1));
            reduce(gout29 * d_0 + gout32 * d_1 + gout35 * d_2 + gout38 * d_3 +
                       gout41 * d_4 + gout44 * d_5 + gout47 * d_6 +
                       gout50 * d_7 + gout53 * d_8,
                vk + (i0 + 2) + nao * (l0 + 1));
            reduce(gout54 * d_0 + gout57 * d_1 + gout60 * d_2 + gout63 * d_3 +
                       gout66 * d_4 + gout69 * d_5 + gout72 * d_6 +
                       gout75 * d_7 + gout78 * d_8,
                vk + (i0 + 0) + nao * (l0 + 2));
            reduce(gout55 * d_0 + gout58 * d_1 + gout61 * d_2 + gout64 * d_3 +
                       gout67 * d_4 + gout70 * d_5 + gout73 * d_6 +
                       gout76 * d_7 + gout79 * d_8,
                vk + (i0 + 1) + nao * (l0 + 2));
            reduce(gout56 * d_0 + gout59 * d_1 + gout62 * d_2 + gout65 * d_3 +
                       gout68 * d_4 + gout71 * d_5 + gout74 * d_6 +
                       gout77 * d_7 + gout80 * d_8,
                vk + (i0 + 2) + nao * (l0 + 2));
            // ijkl,il->jk
            d_0 = dm[(i0 + 0) + nao * (l0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (l0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (l0 + 0)];
            d_3 = dm[(i0 + 0) + nao * (l0 + 1)];
            d_4 = dm[(i0 + 1) + nao * (l0 + 1)];
            d_5 = dm[(i0 + 2) + nao * (l0 + 1)];
            d_6 = dm[(i0 + 0) + nao * (l0 + 2)];
            d_7 = dm[(i0 + 1) + nao * (l0 + 2)];
            d_8 = dm[(i0 + 2) + nao * (l0 + 2)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout27 * d_3 +
                       gout28 * d_4 + gout29 * d_5 + gout54 * d_6 +
                       gout55 * d_7 + gout56 * d_8,
                vk + (j0 + 0) + nao * (k0 + 0));
            reduce(gout3 * d_0 + gout4 * d_1 + gout5 * d_2 + gout30 * d_3 +
                       gout31 * d_4 + gout32 * d_5 + gout57 * d_6 +
                       gout58 * d_7 + gout59 * d_8,
                vk + (j0 + 1) + nao * (k0 + 0));
            reduce(gout6 * d_0 + gout7 * d_1 + gout8 * d_2 + gout33 * d_3 +
                       gout34 * d_4 + gout35 * d_5 + gout60 * d_6 +
                       gout61 * d_7 + gout62 * d_8,
                vk + (j0 + 2) + nao * (k0 + 0));
            reduce(gout9 * d_0 + gout10 * d_1 + gout11 * d_2 + gout36 * d_3 +
                       gout37 * d_4 + gout38 * d_5 + gout63 * d_6 +
                       gout64 * d_7 + gout65 * d_8,
                vk + (j0 + 0) + nao * (k0 + 1));
            reduce(gout12 * d_0 + gout13 * d_1 + gout14 * d_2 + gout39 * d_3 +
                       gout40 * d_4 + gout41 * d_5 + gout66 * d_6 +
                       gout67 * d_7 + gout68 * d_8,
                vk + (j0 + 1) + nao * (k0 + 1));
            reduce(gout15 * d_0 + gout16 * d_1 + gout17 * d_2 + gout42 * d_3 +
                       gout43 * d_4 + gout44 * d_5 + gout69 * d_6 +
                       gout70 * d_7 + gout71 * d_8,
                vk + (j0 + 2) + nao * (k0 + 1));
            reduce(gout18 * d_0 + gout19 * d_1 + gout20 * d_2 + gout45 * d_3 +
                       gout46 * d_4 + gout47 * d_5 + gout72 * d_6 +
                       gout73 * d_7 + gout74 * d_8,
                vk + (j0 + 0) + nao * (k0 + 2));
            reduce(gout21 * d_0 + gout22 * d_1 + gout23 * d_2 + gout48 * d_3 +
                       gout49 * d_4 + gout50 * d_5 + gout75 * d_6 +
                       gout76 * d_7 + gout77 * d_8,
                vk + (j0 + 1) + nao * (k0 + 2));
            reduce(gout24 * d_0 + gout25 * d_1 + gout26 * d_2 + gout51 * d_3 +
                       gout52 * d_4 + gout53 * d_5 + gout78 * d_6 +
                       gout79 * d_7 + gout80 * d_8,
                vk + (j0 + 2) + nao * (k0 + 2));
            // ijkl,ik->jl
            d_0 = dm[(i0 + 0) + nao * (k0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (k0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (k0 + 0)];
            d_3 = dm[(i0 + 0) + nao * (k0 + 1)];
            d_4 = dm[(i0 + 1) + nao * (k0 + 1)];
            d_5 = dm[(i0 + 2) + nao * (k0 + 1)];
            d_6 = dm[(i0 + 0) + nao * (k0 + 2)];
            d_7 = dm[(i0 + 1) + nao * (k0 + 2)];
            d_8 = dm[(i0 + 2) + nao * (k0 + 2)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout9 * d_3 +
                       gout10 * d_4 + gout11 * d_5 + gout18 * d_6 +
                       gout19 * d_7 + gout20 * d_8,
                vk + (j0 + 0) + nao * (l0 + 0));
            reduce(gout3 * d_0 + gout4 * d_1 + gout5 * d_2 + gout12 * d_3 +
                       gout13 * d_4 + gout14 * d_5 + gout21 * d_6 +
                       gout22 * d_7 + gout23 * d_8,
                vk + (j0 + 1) + nao * (l0 + 0));
            reduce(gout6 * d_0 + gout7 * d_1 + gout8 * d_2 + gout15 * d_3 +
                       gout16 * d_4 + gout17 * d_5 + gout24 * d_6 +
                       gout25 * d_7 + gout26 * d_8,
                vk + (j0 + 2) + nao * (l0 + 0));
            reduce(gout27 * d_0 + gout28 * d_1 + gout29 * d_2 + gout36 * d_3 +
                       gout37 * d_4 + gout38 * d_5 + gout45 * d_6 +
                       gout46 * d_7 + gout47 * d_8,
                vk + (j0 + 0) + nao * (l0 + 1));
            reduce(gout30 * d_0 + gout31 * d_1 + gout32 * d_2 + gout39 * d_3 +
                       gout40 * d_4 + gout41 * d_5 + gout48 * d_6 +
                       gout49 * d_7 + gout50 * d_8,
                vk + (j0 + 1) + nao * (l0 + 1));
            reduce(gout33 * d_0 + gout34 * d_1 + gout35 * d_2 + gout42 * d_3 +
                       gout43 * d_4 + gout44 * d_5 + gout51 * d_6 +
                       gout52 * d_7 + gout53 * d_8,
                vk + (j0 + 2) + nao * (l0 + 1));
            reduce(gout54 * d_0 + gout55 * d_1 + gout56 * d_2 + gout63 * d_3 +
                       gout64 * d_4 + gout65 * d_5 + gout72 * d_6 +
                       gout73 * d_7 + gout74 * d_8,
                vk + (j0 + 0) + nao * (l0 + 2));
            reduce(gout57 * d_0 + gout58 * d_1 + gout59 * d_2 + gout66 * d_3 +
                       gout67 * d_4 + gout68 * d_5 + gout75 * d_6 +
                       gout76 * d_7 + gout77 * d_8,
                vk + (j0 + 1) + nao * (l0 + 2));
            reduce(gout60 * d_0 + gout61 * d_1 + gout62 * d_2 + gout69 * d_3 +
                       gout70 * d_4 + gout71 * d_5 + gout78 * d_6 +
                       gout79 * d_7 + gout80 * d_8,
                vk + (j0 + 2) + nao * (l0 + 2));
            vk += nao2;
        }
        dm += nao2;
    }
}

__global__ static void GINTint2e_jk_kernel2011(JKMatrix jk,
    BasisProdOffsets offsets, GINTEnvVars envs, BasisProdCache bpcache) {
    int ntasks_ij = offsets.ntasks_ij;
    long ntasks = ntasks_ij * offsets.ntasks_kl;
    long task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int igroup = nprim_ij * nprim_kl;
    ntasks *= igroup;
    if (task_ij >= ntasks)
        return;
    int kl = task_ij % nprim_kl;
    task_ij /= nprim_kl;
    int ij = task_ij % nprim_ij;
    task_ij /= nprim_ij;
    int task_kl = task_ij / ntasks_ij;
    task_ij = task_ij % ntasks_ij;

    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    if (bas_ij < bas_kl) {
        return;
    }
    double norm = envs.fac;
    if (bas_ij == bas_kl) {
        norm *= .5;
    }

    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *ao_loc = bpcache.ao_loc;
    int *bas_pair2bra = bpcache.bas_pair2bra;
    int *bas_pair2ket = bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    int k0 = ao_loc[ksh];
    int l0 = ao_loc[lsh];
    double *__restrict__ a12 = bpcache.a12;
    double *__restrict__ e12 = bpcache.e12;
    double *__restrict__ x12 = bpcache.x12;
    double *__restrict__ y12 = bpcache.y12;
    double *__restrict__ z12 = bpcache.z12;
    int i_dm;
    int nbas = bpcache.nbas;
    double *__restrict__ bas_x = bpcache.bas_coords;
    double *__restrict__ bas_y = bas_x + nbas;
    double *__restrict__ bas_z = bas_y + nbas;

    double gout0 = 0;
    double gout1 = 0;
    double gout2 = 0;
    double gout3 = 0;
    double gout4 = 0;
    double gout5 = 0;
    double gout6 = 0;
    double gout7 = 0;
    double gout8 = 0;
    double gout9 = 0;
    double gout10 = 0;
    double gout11 = 0;
    double gout12 = 0;
    double gout13 = 0;
    double gout14 = 0;
    double gout15 = 0;
    double gout16 = 0;
    double gout17 = 0;
    double gout18 = 0;
    double gout19 = 0;
    double gout20 = 0;
    double gout21 = 0;
    double gout22 = 0;
    double gout23 = 0;
    double gout24 = 0;
    double gout25 = 0;
    double gout26 = 0;
    double gout27 = 0;
    double gout28 = 0;
    double gout29 = 0;
    double gout30 = 0;
    double gout31 = 0;
    double gout32 = 0;
    double gout33 = 0;
    double gout34 = 0;
    double gout35 = 0;
    double gout36 = 0;
    double gout37 = 0;
    double gout38 = 0;
    double gout39 = 0;
    double gout40 = 0;
    double gout41 = 0;
    double gout42 = 0;
    double gout43 = 0;
    double gout44 = 0;
    double gout45 = 0;
    double gout46 = 0;
    double gout47 = 0;
    double gout48 = 0;
    double gout49 = 0;
    double gout50 = 0;
    double gout51 = 0;
    double gout52 = 0;
    double gout53 = 0;
    double xi = bas_x[ish];
    double yi = bas_y[ish];
    double zi = bas_z[ish];
    double xk = bas_x[ksh];
    double yk = bas_y[ksh];
    double zk = bas_z[ksh];
    double xkxl = xk - bas_x[lsh];
    double ykyl = yk - bas_y[lsh];
    double zkzl = zk - bas_z[lsh];
    auto reduce = SegReduce<double>(igroup);
    ij += prim_ij;
    kl += prim_kl;
    double aij = a12[ij];
    double eij = e12[ij];
    double xij = x12[ij];
    double yij = y12[ij];
    double zij = z12[ij];
    double akl = a12[kl];
    double ekl = e12[kl];
    double xkl = x12[kl];
    double ykl = y12[kl];
    double zkl = z12[kl];
    double xijxkl = xij - xkl;
    double yijykl = yij - ykl;
    double zijzkl = zij - zkl;
    double aijkl = aij + akl;
    double a1 = aij * akl;
    double a0 = a1 / aijkl;
    double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
    double fac = norm * eij * ekl / (sqrt(aijkl) * a1);

    double rw[6];
    double root0, weight0;
    GINTrys_root<3>(x, rw);
    int irys;
    for (irys = 0; irys < 3; ++irys) {
        root0 = rw[irys];
        weight0 = rw[irys + 3];
        double u2 = a0 * root0;
        double tmp4 = .5 / (u2 * aijkl + a1);
        double b00 = u2 * tmp4;
        double tmp1 = 2 * b00;
        double tmp2 = tmp1 * akl;
        double b10 = b00 + tmp4 * akl;
        double c00x = xij - xi - tmp2 * xijxkl;
        double c00y = yij - yi - tmp2 * yijykl;
        double c00z = zij - zi - tmp2 * zijzkl;
        double tmp3 = tmp1 * aij;
        double b01 = b00 + tmp4 * aij;
        double c0px = xkl - xk + tmp3 * xijxkl;
        double c0py = ykl - yk + tmp3 * yijykl;
        double c0pz = zkl - zk + tmp3 * zijzkl;
        double g_0 = 1;
        double g_1 = c00x;
        double g_2 = c00x * c00x + b10;
        double g_3 = c0px;
        double g_4 = c0px * c00x + b00;
        double g_5 = b00 * c00x + b10 * c0px + c00x * g_4;
        double g_6 = c0px + xkxl;
        double g_7 = c00x * (c0px + xkxl) + b00;
        double g_8 = b00 * c00x + b10 * c0px + c00x * g_4 + xkxl * g_2;
        double g_9 = c0px * (c0px + xkxl) + b01;
        double g_10 = b00 * c0px + b01 * c00x + c0px * g_4 + xkxl * g_4;
        double g_11 = xkxl * g_5 +
                      c00x * (c0px * g_4 + b01 * c00x + b00 * c0px) +
                      b10 * (c0px * c0px + b01) + 2 * b00 * g_4;
        double g_12 = 1;
        double g_13 = c00y;
        double g_14 = c00y * c00y + b10;
        double g_15 = c0py;
        double g_16 = c0py * c00y + b00;
        double g_17 = b00 * c00y + b10 * c0py + c00y * g_16;
        double g_18 = c0py + ykyl;
        double g_19 = c00y * (c0py + ykyl) + b00;
        double g_20 = b00 * c00y + b10 * c0py + c00y * g_16 + ykyl * g_14;
        double g_21 = c0py * (c0py + ykyl) + b01;
        double g_22 = b00 * c0py + b01 * c00y + c0py * g_16 + ykyl * g_16;
        double g_23 = ykyl * g_17 +
                      c00y * (c0py * g_16 + b01 * c00y + b00 * c0py) +
                      b10 * (c0py * c0py + b01) + 2 * b00 * g_16;
        double g_24 = weight0 * fac;
        double g_25 = c00z * g_24;
        double g_26 = b10 * g_24 + c00z * g_25;
        double g_27 = c0pz * g_24;
        double g_28 = b00 * g_24 + c0pz * g_25;
        double g_29 = b00 * g_25 + b10 * g_27 + c00z * g_28;
        double g_30 = g_24 * (c0pz + zkzl);
        double g_31 = b00 * g_24 + c0pz * g_25 + zkzl * g_25;
        double g_32 = b00 * g_25 + b10 * g_27 + c00z * g_28 + zkzl * g_26;
        double g_33 = b01 * g_24 + c0pz * g_27 + zkzl * g_27;
        double g_34 = b00 * g_27 + b01 * g_25 + c0pz * g_28 + zkzl * g_28;
        double g_35 = zkzl * g_29 +
                      c00z * (c0pz * g_28 + b01 * g_25 + b00 * g_27) +
                      b10 * (c0pz * g_27 + b01 * g_24) + 2 * b00 * g_28;
        gout0 += g_11 * g_12 * g_24;
        gout1 += g_10 * g_13 * g_24;
        gout2 += g_10 * g_12 * g_25;
        gout3 += g_9 * g_14 * g_24;
        gout4 += g_9 * g_13 * g_25;
        gout5 += g_9 * g_12 * g_26;
        gout6 += g_8 * g_15 * g_24;
        gout7 += g_7 * g_16 * g_24;
        gout8 += g_7 * g_15 * g_25;
        gout9 += g_6 * g_17 * g_24;
        gout10 += g_6 * g_16 * g_25;
        gout11 += g_6 * g_15 * g_26;
        gout12 += g_8 * g_12 * g_27;
        gout13 += g_7 * g_13 * g_27;
        gout14 += g_7 * g_12 * g_28;
        gout15 += g_6 * g_14 * g_27;
        gout16 += g_6 * g_13 * g_28;
        gout17 += g_6 * g_12 * g_29;
        gout18 += g_5 * g_18 * g_24;
        gout19 += g_4 * g_19 * g_24;
        gout20 += g_4 * g_18 * g_25;
        gout21 += g_3 * g_20 * g_24;
        gout22 += g_3 * g_19 * g_25;
        gout23 += g_3 * g_18 * g_26;
        gout24 += g_2 * g_21 * g_24;
        gout25 += g_1 * g_22 * g_24;
        gout26 += g_1 * g_21 * g_25;
        gout27 += g_0 * g_23 * g_24;
        gout28 += g_0 * g_22 * g_25;
        gout29 += g_0 * g_21 * g_26;
        gout30 += g_2 * g_18 * g_27;
        gout31 += g_1 * g_19 * g_27;
        gout32 += g_1 * g_18 * g_28;
        gout33 += g_0 * g_20 * g_27;
        gout34 += g_0 * g_19 * g_28;
        gout35 += g_0 * g_18 * g_29;
        gout36 += g_5 * g_12 * g_30;
        gout37 += g_4 * g_13 * g_30;
        gout38 += g_4 * g_12 * g_31;
        gout39 += g_3 * g_14 * g_30;
        gout40 += g_3 * g_13 * g_31;
        gout41 += g_3 * g_12 * g_32;
        gout42 += g_2 * g_15 * g_30;
        gout43 += g_1 * g_16 * g_30;
        gout44 += g_1 * g_15 * g_31;
        gout45 += g_0 * g_17 * g_30;
        gout46 += g_0 * g_16 * g_31;
        gout47 += g_0 * g_15 * g_32;
        gout48 += g_2 * g_12 * g_33;
        gout49 += g_1 * g_13 * g_33;
        gout50 += g_1 * g_12 * g_34;
        gout51 += g_0 * g_14 * g_33;
        gout52 += g_0 * g_13 * g_34;
        gout53 += g_0 * g_12 * g_35;
    }
    double d_0, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, d_9;
    double d_10, d_11, d_12, d_13, d_14, d_15, d_16, d_17;
    int n_dm = jk.n_dm;
    int nao = jk.nao;
    size_t nao2 = nao * nao;
    double *__restrict__ dm = jk.dm;
    double *vj = jk.vj;
    double *vk = jk.vk;
    for (i_dm = 0; i_dm < n_dm; ++i_dm) {
        if (vj != NULL) {
            // ijkl,ij->kl
            d_0 = dm[(i0 + 0) + nao * (j0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (j0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (j0 + 0)];
            d_3 = dm[(i0 + 3) + nao * (j0 + 0)];
            d_4 = dm[(i0 + 4) + nao * (j0 + 0)];
            d_5 = dm[(i0 + 5) + nao * (j0 + 0)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout3 * d_3 +
                       gout4 * d_4 + gout5 * d_5,
                vj + (k0 + 0) + nao * (l0 + 0));
            reduce(gout6 * d_0 + gout7 * d_1 + gout8 * d_2 + gout9 * d_3 +
                       gout10 * d_4 + gout11 * d_5,
                vj + (k0 + 1) + nao * (l0 + 0));
            reduce(gout12 * d_0 + gout13 * d_1 + gout14 * d_2 + gout15 * d_3 +
                       gout16 * d_4 + gout17 * d_5,
                vj + (k0 + 2) + nao * (l0 + 0));
            reduce(gout18 * d_0 + gout19 * d_1 + gout20 * d_2 + gout21 * d_3 +
                       gout22 * d_4 + gout23 * d_5,
                vj + (k0 + 0) + nao * (l0 + 1));
            reduce(gout24 * d_0 + gout25 * d_1 + gout26 * d_2 + gout27 * d_3 +
                       gout28 * d_4 + gout29 * d_5,
                vj + (k0 + 1) + nao * (l0 + 1));
            reduce(gout30 * d_0 + gout31 * d_1 + gout32 * d_2 + gout33 * d_3 +
                       gout34 * d_4 + gout35 * d_5,
                vj + (k0 + 2) + nao * (l0 + 1));
            reduce(gout36 * d_0 + gout37 * d_1 + gout38 * d_2 + gout39 * d_3 +
                       gout40 * d_4 + gout41 * d_5,
                vj + (k0 + 0) + nao * (l0 + 2));
            reduce(gout42 * d_0 + gout43 * d_1 + gout44 * d_2 + gout45 * d_3 +
                       gout46 * d_4 + gout47 * d_5,
                vj + (k0 + 1) + nao * (l0 + 2));
            reduce(gout48 * d_0 + gout49 * d_1 + gout50 * d_2 + gout51 * d_3 +
                       gout52 * d_4 + gout53 * d_5,
                vj + (k0 + 2) + nao * (l0 + 2));
            // ijkl,kl->ij
            d_0 = dm[(k0 + 0) + nao * (l0 + 0)];
            d_1 = dm[(k0 + 1) + nao * (l0 + 0)];
            d_2 = dm[(k0 + 2) + nao * (l0 + 0)];
            d_3 = dm[(k0 + 0) + nao * (l0 + 1)];
            d_4 = dm[(k0 + 1) + nao * (l0 + 1)];
            d_5 = dm[(k0 + 2) + nao * (l0 + 1)];
            d_6 = dm[(k0 + 0) + nao * (l0 + 2)];
            d_7 = dm[(k0 + 1) + nao * (l0 + 2)];
            d_8 = dm[(k0 + 2) + nao * (l0 + 2)];
            reduce(gout0 * d_0 + gout6 * d_1 + gout12 * d_2 + gout18 * d_3 +
                       gout24 * d_4 + gout30 * d_5 + gout36 * d_6 +
                       gout42 * d_7 + gout48 * d_8,
                vj + (i0 + 0) + nao * (j0 + 0));
            reduce(gout1 * d_0 + gout7 * d_1 + gout13 * d_2 + gout19 * d_3 +
                       gout25 * d_4 + gout31 * d_5 + gout37 * d_6 +
                       gout43 * d_7 + gout49 * d_8,
                vj + (i0 + 1) + nao * (j0 + 0));
            reduce(gout2 * d_0 + gout8 * d_1 + gout14 * d_2 + gout20 * d_3 +
                       gout26 * d_4 + gout32 * d_5 + gout38 * d_6 +
                       gout44 * d_7 + gout50 * d_8,
                vj + (i0 + 2) + nao * (j0 + 0));
            reduce(gout3 * d_0 + gout9 * d_1 + gout15 * d_2 + gout21 * d_3 +
                       gout27 * d_4 + gout33 * d_5 + gout39 * d_6 +
                       gout45 * d_7 + gout51 * d_8,
                vj + (i0 + 3) + nao * (j0 + 0));
            reduce(gout4 * d_0 + gout10 * d_1 + gout16 * d_2 + gout22 * d_3 +
                       gout28 * d_4 + gout34 * d_5 + gout40 * d_6 +
                       gout46 * d_7 + gout52 * d_8,
                vj + (i0 + 4) + nao * (j0 + 0));
            reduce(gout5 * d_0 + gout11 * d_1 + gout17 * d_2 + gout23 * d_3 +
                       gout29 * d_4 + gout35 * d_5 + gout41 * d_6 +
                       gout47 * d_7 + gout53 * d_8,
                vj + (i0 + 5) + nao * (j0 + 0));
            vj += nao2;
        }
        if (vk != NULL) {
            // ijkl,jl->ik
            d_0 = dm[(j0 + 0) + nao * (l0 + 0)];
            d_1 = dm[(j0 + 0) + nao * (l0 + 1)];
            d_2 = dm[(j0 + 0) + nao * (l0 + 2)];
            reduce(gout0 * d_0 + gout18 * d_1 + gout36 * d_2,
                vk + (i0 + 0) + nao * (k0 + 0));
            reduce(gout1 * d_0 + gout19 * d_1 + gout37 * d_2,
                vk + (i0 + 1) + nao * (k0 + 0));
            reduce(gout2 * d_0 + gout20 * d_1 + gout38 * d_2,
                vk + (i0 + 2) + nao * (k0 + 0));
            reduce(gout3 * d_0 + gout21 * d_1 + gout39 * d_2,
                vk + (i0 + 3) + nao * (k0 + 0));
            reduce(gout4 * d_0 + gout22 * d_1 + gout40 * d_2,
                vk + (i0 + 4) + nao * (k0 + 0));
            reduce(gout5 * d_0 + gout23 * d_1 + gout41 * d_2,
                vk + (i0 + 5) + nao * (k0 + 0));
            reduce(gout6 * d_0 + gout24 * d_1 + gout42 * d_2,
                vk + (i0 + 0) + nao * (k0 + 1));
            reduce(gout7 * d_0 + gout25 * d_1 + gout43 * d_2,
                vk + (i0 + 1) + nao * (k0 + 1));
            reduce(gout8 * d_0 + gout26 * d_1 + gout44 * d_2,
                vk + (i0 + 2) + nao * (k0 + 1));
            reduce(gout9 * d_0 + gout27 * d_1 + gout45 * d_2,
                vk + (i0 + 3) + nao * (k0 + 1));
            reduce(gout10 * d_0 + gout28 * d_1 + gout46 * d_2,
                vk + (i0 + 4) + nao * (k0 + 1));
            reduce(gout11 * d_0 + gout29 * d_1 + gout47 * d_2,
                vk + (i0 + 5) + nao * (k0 + 1));
            reduce(gout12 * d_0 + gout30 * d_1 + gout48 * d_2,
                vk + (i0 + 0) + nao * (k0 + 2));
            reduce(gout13 * d_0 + gout31 * d_1 + gout49 * d_2,
                vk + (i0 + 1) + nao * (k0 + 2));
            reduce(gout14 * d_0 + gout32 * d_1 + gout50 * d_2,
                vk + (i0 + 2) + nao * (k0 + 2));
            reduce(gout15 * d_0 + gout33 * d_1 + gout51 * d_2,
                vk + (i0 + 3) + nao * (k0 + 2));
            reduce(gout16 * d_0 + gout34 * d_1 + gout52 * d_2,
                vk + (i0 + 4) + nao * (k0 + 2));
            reduce(gout17 * d_0 + gout35 * d_1 + gout53 * d_2,
                vk + (i0 + 5) + nao * (k0 + 2));
            // ijkl,jk->il
            d_0 = dm[(j0 + 0) + nao * (k0 + 0)];
            d_1 = dm[(j0 + 0) + nao * (k0 + 1)];
            d_2 = dm[(j0 + 0) + nao * (k0 + 2)];
            reduce(gout0 * d_0 + gout6 * d_1 + gout12 * d_2,
                vk + (i0 + 0) + nao * (l0 + 0));
            reduce(gout1 * d_0 + gout7 * d_1 + gout13 * d_2,
                vk + (i0 + 1) + nao * (l0 + 0));
            reduce(gout2 * d_0 + gout8 * d_1 + gout14 * d_2,
                vk + (i0 + 2) + nao * (l0 + 0));
            reduce(gout3 * d_0 + gout9 * d_1 + gout15 * d_2,
                vk + (i0 + 3) + nao * (l0 + 0));
            reduce(gout4 * d_0 + gout10 * d_1 + gout16 * d_2,
                vk + (i0 + 4) + nao * (l0 + 0));
            reduce(gout5 * d_0 + gout11 * d_1 + gout17 * d_2,
                vk + (i0 + 5) + nao * (l0 + 0));
            reduce(gout18 * d_0 + gout24 * d_1 + gout30 * d_2,
                vk + (i0 + 0) + nao * (l0 + 1));
            reduce(gout19 * d_0 + gout25 * d_1 + gout31 * d_2,
                vk + (i0 + 1) + nao * (l0 + 1));
            reduce(gout20 * d_0 + gout26 * d_1 + gout32 * d_2,
                vk + (i0 + 2) + nao * (l0 + 1));
            reduce(gout21 * d_0 + gout27 * d_1 + gout33 * d_2,
                vk + (i0 + 3) + nao * (l0 + 1));
            reduce(gout22 * d_0 + gout28 * d_1 + gout34 * d_2,
                vk + (i0 + 4) + nao * (l0 + 1));
            reduce(gout23 * d_0 + gout29 * d_1 + gout35 * d_2,
                vk + (i0 + 5) + nao * (l0 + 1));
            reduce(gout36 * d_0 + gout42 * d_1 + gout48 * d_2,
                vk + (i0 + 0) + nao * (l0 + 2));
            reduce(gout37 * d_0 + gout43 * d_1 + gout49 * d_2,
                vk + (i0 + 1) + nao * (l0 + 2));
            reduce(gout38 * d_0 + gout44 * d_1 + gout50 * d_2,
                vk + (i0 + 2) + nao * (l0 + 2));
            reduce(gout39 * d_0 + gout45 * d_1 + gout51 * d_2,
                vk + (i0 + 3) + nao * (l0 + 2));
            reduce(gout40 * d_0 + gout46 * d_1 + gout52 * d_2,
                vk + (i0 + 4) + nao * (l0 + 2));
            reduce(gout41 * d_0 + gout47 * d_1 + gout53 * d_2,
                vk + (i0 + 5) + nao * (l0 + 2));
            // ijkl,il->jk
            d_0 = dm[(i0 + 0) + nao * (l0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (l0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (l0 + 0)];
            d_3 = dm[(i0 + 3) + nao * (l0 + 0)];
            d_4 = dm[(i0 + 4) + nao * (l0 + 0)];
            d_5 = dm[(i0 + 5) + nao * (l0 + 0)];
            d_6 = dm[(i0 + 0) + nao * (l0 + 1)];
            d_7 = dm[(i0 + 1) + nao * (l0 + 1)];
            d_8 = dm[(i0 + 2) + nao * (l0 + 1)];
            d_9 = dm[(i0 + 3) + nao * (l0 + 1)];
            d_10 = dm[(i0 + 4) + nao * (l0 + 1)];
            d_11 = dm[(i0 + 5) + nao * (l0 + 1)];
            d_12 = dm[(i0 + 0) + nao * (l0 + 2)];
            d_13 = dm[(i0 + 1) + nao * (l0 + 2)];
            d_14 = dm[(i0 + 2) + nao * (l0 + 2)];
            d_15 = dm[(i0 + 3) + nao * (l0 + 2)];
            d_16 = dm[(i0 + 4) + nao * (l0 + 2)];
            d_17 = dm[(i0 + 5) + nao * (l0 + 2)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout3 * d_3 +
                       gout4 * d_4 + gout5 * d_5 + gout18 * d_6 +
                       gout19 * d_7 + gout20 * d_8 + gout21 * d_9 +
                       gout22 * d_10 + gout23 * d_11 + gout36 * d_12 +
                       gout37 * d_13 + gout38 * d_14 + gout39 * d_15 +
                       gout40 * d_16 + gout41 * d_17,
                vk + (j0 + 0) + nao * (k0 + 0));
            reduce(gout6 * d_0 + gout7 * d_1 + gout8 * d_2 + gout9 * d_3 +
                       gout10 * d_4 + gout11 * d_5 + gout24 * d_6 +
                       gout25 * d_7 + gout26 * d_8 + gout27 * d_9 +
                       gout28 * d_10 + gout29 * d_11 + gout42 * d_12 +
                       gout43 * d_13 + gout44 * d_14 + gout45 * d_15 +
                       gout46 * d_16 + gout47 * d_17,
                vk + (j0 + 0) + nao * (k0 + 1));
            reduce(gout12 * d_0 + gout13 * d_1 + gout14 * d_2 + gout15 * d_3 +
                       gout16 * d_4 + gout17 * d_5 + gout30 * d_6 +
                       gout31 * d_7 + gout32 * d_8 + gout33 * d_9 +
                       gout34 * d_10 + gout35 * d_11 + gout48 * d_12 +
                       gout49 * d_13 + gout50 * d_14 + gout51 * d_15 +
                       gout52 * d_16 + gout53 * d_17,
                vk + (j0 + 0) + nao * (k0 + 2));
            // ijkl,ik->jl
            d_0 = dm[(i0 + 0) + nao * (k0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (k0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (k0 + 0)];
            d_3 = dm[(i0 + 3) + nao * (k0 + 0)];
            d_4 = dm[(i0 + 4) + nao * (k0 + 0)];
            d_5 = dm[(i0 + 5) + nao * (k0 + 0)];
            d_6 = dm[(i0 + 0) + nao * (k0 + 1)];
            d_7 = dm[(i0 + 1) + nao * (k0 + 1)];
            d_8 = dm[(i0 + 2) + nao * (k0 + 1)];
            d_9 = dm[(i0 + 3) + nao * (k0 + 1)];
            d_10 = dm[(i0 + 4) + nao * (k0 + 1)];
            d_11 = dm[(i0 + 5) + nao * (k0 + 1)];
            d_12 = dm[(i0 + 0) + nao * (k0 + 2)];
            d_13 = dm[(i0 + 1) + nao * (k0 + 2)];
            d_14 = dm[(i0 + 2) + nao * (k0 + 2)];
            d_15 = dm[(i0 + 3) + nao * (k0 + 2)];
            d_16 = dm[(i0 + 4) + nao * (k0 + 2)];
            d_17 = dm[(i0 + 5) + nao * (k0 + 2)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout3 * d_3 +
                       gout4 * d_4 + gout5 * d_5 + gout6 * d_6 + gout7 * d_7 +
                       gout8 * d_8 + gout9 * d_9 + gout10 * d_10 +
                       gout11 * d_11 + gout12 * d_12 + gout13 * d_13 +
                       gout14 * d_14 + gout15 * d_15 + gout16 * d_16 +
                       gout17 * d_17,
                vk + (j0 + 0) + nao * (l0 + 0));
            reduce(gout18 * d_0 + gout19 * d_1 + gout20 * d_2 + gout21 * d_3 +
                       gout22 * d_4 + gout23 * d_5 + gout24 * d_6 +
                       gout25 * d_7 + gout26 * d_8 + gout27 * d_9 +
                       gout28 * d_10 + gout29 * d_11 + gout30 * d_12 +
                       gout31 * d_13 + gout32 * d_14 + gout33 * d_15 +
                       gout34 * d_16 + gout35 * d_17,
                vk + (j0 + 0) + nao * (l0 + 1));
            reduce(gout36 * d_0 + gout37 * d_1 + gout38 * d_2 + gout39 * d_3 +
                       gout40 * d_4 + gout41 * d_5 + gout42 * d_6 +
                       gout43 * d_7 + gout44 * d_8 + gout45 * d_9 +
                       gout46 * d_10 + gout47 * d_11 + gout48 * d_12 +
                       gout49 * d_13 + gout50 * d_14 + gout51 * d_15 +
                       gout52 * d_16 + gout53 * d_17,
                vk + (j0 + 0) + nao * (l0 + 2));
            vk += nao2;
        }
        dm += nao2;
    }
}

__global__ static void GINTint2e_jk_kernel2020(JKMatrix jk,
    BasisProdOffsets offsets, GINTEnvVars envs, BasisProdCache bpcache) {
    int ntasks_ij = offsets.ntasks_ij;
    long ntasks = ntasks_ij * offsets.ntasks_kl;
    long task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int igroup = nprim_ij * nprim_kl;
    ntasks *= igroup;
    if (task_ij >= ntasks)
        return;
    int kl = task_ij % nprim_kl;
    task_ij /= nprim_kl;
    int ij = task_ij % nprim_ij;
    task_ij /= nprim_ij;
    int task_kl = task_ij / ntasks_ij;
    task_ij = task_ij % ntasks_ij;

    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    if (bas_ij < bas_kl) {
        return;
    }
    double norm = envs.fac;
    if (bas_ij == bas_kl) {
        norm *= .5;
    }

    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *ao_loc = bpcache.ao_loc;
    int *bas_pair2bra = bpcache.bas_pair2bra;
    int *bas_pair2ket = bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    int k0 = ao_loc[ksh];
    int l0 = ao_loc[lsh];
    double *__restrict__ a12 = bpcache.a12;
    double *__restrict__ e12 = bpcache.e12;
    double *__restrict__ x12 = bpcache.x12;
    double *__restrict__ y12 = bpcache.y12;
    double *__restrict__ z12 = bpcache.z12;
    int i_dm;
    int nbas = bpcache.nbas;
    double *__restrict__ bas_x = bpcache.bas_coords;
    double *__restrict__ bas_y = bas_x + nbas;
    double *__restrict__ bas_z = bas_y + nbas;

    double gout0 = 0;
    double gout1 = 0;
    double gout2 = 0;
    double gout3 = 0;
    double gout4 = 0;
    double gout5 = 0;
    double gout6 = 0;
    double gout7 = 0;
    double gout8 = 0;
    double gout9 = 0;
    double gout10 = 0;
    double gout11 = 0;
    double gout12 = 0;
    double gout13 = 0;
    double gout14 = 0;
    double gout15 = 0;
    double gout16 = 0;
    double gout17 = 0;
    double gout18 = 0;
    double gout19 = 0;
    double gout20 = 0;
    double gout21 = 0;
    double gout22 = 0;
    double gout23 = 0;
    double gout24 = 0;
    double gout25 = 0;
    double gout26 = 0;
    double gout27 = 0;
    double gout28 = 0;
    double gout29 = 0;
    double gout30 = 0;
    double gout31 = 0;
    double gout32 = 0;
    double gout33 = 0;
    double gout34 = 0;
    double gout35 = 0;
    double xi = bas_x[ish];
    double yi = bas_y[ish];
    double zi = bas_z[ish];
    double xk = bas_x[ksh];
    double yk = bas_y[ksh];
    double zk = bas_z[ksh];
    auto reduce = SegReduce<double>(igroup);
    ij += prim_ij;
    kl += prim_kl;
    double aij = a12[ij];
    double eij = e12[ij];
    double xij = x12[ij];
    double yij = y12[ij];
    double zij = z12[ij];
    double akl = a12[kl];
    double ekl = e12[kl];
    double xkl = x12[kl];
    double ykl = y12[kl];
    double zkl = z12[kl];
    double xijxkl = xij - xkl;
    double yijykl = yij - ykl;
    double zijzkl = zij - zkl;
    double aijkl = aij + akl;
    double a1 = aij * akl;
    double a0 = a1 / aijkl;
    double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
    double fac = norm * eij * ekl / (sqrt(aijkl) * a1);

    double rw[6];
    double root0, weight0;
    GINTrys_root<3>(x, rw);
    int irys;
    for (irys = 0; irys < 3; ++irys) {
        root0 = rw[irys];
        weight0 = rw[irys + 3];
        double u2 = a0 * root0;
        double tmp4 = .5 / (u2 * aijkl + a1);
        double b00 = u2 * tmp4;
        double tmp1 = 2 * b00;
        double tmp2 = tmp1 * akl;
        double b10 = b00 + tmp4 * akl;
        double c00x = xij - xi - tmp2 * xijxkl;
        double c00y = yij - yi - tmp2 * yijykl;
        double c00z = zij - zi - tmp2 * zijzkl;
        double tmp3 = tmp1 * aij;
        double b01 = b00 + tmp4 * aij;
        double c0px = xkl - xk + tmp3 * xijxkl;
        double c0py = ykl - yk + tmp3 * yijykl;
        double c0pz = zkl - zk + tmp3 * zijzkl;
        double g_0 = 1;
        double g_1 = c00x;
        double g_2 = c00x * c00x + b10;
        double g_3 = c0px;
        double g_4 = c0px * c00x + b00;
        double g_5 = b00 * c00x + b10 * c0px + c00x * g_4;
        double g_6 = c0px * c0px + b01;
        double g_7 = b00 * c0px + b01 * c00x + c0px * g_4;
        double g_8 = 2 * b00 * g_4 + b10 * g_6 + c00x * g_7;
        double g_9 = 1;
        double g_10 = c00y;
        double g_11 = c00y * c00y + b10;
        double g_12 = c0py;
        double g_13 = c0py * c00y + b00;
        double g_14 = b00 * c00y + b10 * c0py + c00y * g_13;
        double g_15 = c0py * c0py + b01;
        double g_16 = b00 * c0py + b01 * c00y + c0py * g_13;
        double g_17 = 2 * b00 * g_13 + b10 * g_15 + c00y * g_16;
        double g_18 = weight0 * fac;
        double g_19 = c00z * g_18;
        double g_20 = b10 * g_18 + c00z * g_19;
        double g_21 = c0pz * g_18;
        double g_22 = b00 * g_18 + c0pz * g_19;
        double g_23 = b00 * g_19 + b10 * g_21 + c00z * g_22;
        double g_24 = b01 * g_18 + c0pz * g_21;
        double g_25 = b00 * g_21 + b01 * g_19 + c0pz * g_22;
        double g_26 = 2 * b00 * g_22 + b10 * g_24 + c00z * g_25;
        gout0 += g_8 * g_9 * g_18;
        gout1 += g_7 * g_10 * g_18;
        gout2 += g_7 * g_9 * g_19;
        gout3 += g_6 * g_11 * g_18;
        gout4 += g_6 * g_10 * g_19;
        gout5 += g_6 * g_9 * g_20;
        gout6 += g_5 * g_12 * g_18;
        gout7 += g_4 * g_13 * g_18;
        gout8 += g_4 * g_12 * g_19;
        gout9 += g_3 * g_14 * g_18;
        gout10 += g_3 * g_13 * g_19;
        gout11 += g_3 * g_12 * g_20;
        gout12 += g_5 * g_9 * g_21;
        gout13 += g_4 * g_10 * g_21;
        gout14 += g_4 * g_9 * g_22;
        gout15 += g_3 * g_11 * g_21;
        gout16 += g_3 * g_10 * g_22;
        gout17 += g_3 * g_9 * g_23;
        gout18 += g_2 * g_15 * g_18;
        gout19 += g_1 * g_16 * g_18;
        gout20 += g_1 * g_15 * g_19;
        gout21 += g_0 * g_17 * g_18;
        gout22 += g_0 * g_16 * g_19;
        gout23 += g_0 * g_15 * g_20;
        gout24 += g_2 * g_12 * g_21;
        gout25 += g_1 * g_13 * g_21;
        gout26 += g_1 * g_12 * g_22;
        gout27 += g_0 * g_14 * g_21;
        gout28 += g_0 * g_13 * g_22;
        gout29 += g_0 * g_12 * g_23;
        gout30 += g_2 * g_9 * g_24;
        gout31 += g_1 * g_10 * g_24;
        gout32 += g_1 * g_9 * g_25;
        gout33 += g_0 * g_11 * g_24;
        gout34 += g_0 * g_10 * g_25;
        gout35 += g_0 * g_9 * g_26;
    }
    double d_0, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, d_9;
    double d_10, d_11, d_12, d_13, d_14, d_15, d_16, d_17, d_18, d_19;
    double d_20, d_21, d_22, d_23, d_24, d_25, d_26, d_27, d_28, d_29;
    double d_30, d_31, d_32, d_33, d_34, d_35;
    int n_dm = jk.n_dm;
    int nao = jk.nao;
    size_t nao2 = nao * nao;
    double *__restrict__ dm = jk.dm;
    double *vj = jk.vj;
    double *vk = jk.vk;
    for (i_dm = 0; i_dm < n_dm; ++i_dm) {
        if (vj != NULL) {
            // ijkl,ij->kl
            d_0 = dm[(i0 + 0) + nao * (j0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (j0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (j0 + 0)];
            d_3 = dm[(i0 + 3) + nao * (j0 + 0)];
            d_4 = dm[(i0 + 4) + nao * (j0 + 0)];
            d_5 = dm[(i0 + 5) + nao * (j0 + 0)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout3 * d_3 +
                       gout4 * d_4 + gout5 * d_5,
                vj + (k0 + 0) + nao * (l0 + 0));
            reduce(gout6 * d_0 + gout7 * d_1 + gout8 * d_2 + gout9 * d_3 +
                       gout10 * d_4 + gout11 * d_5,
                vj + (k0 + 1) + nao * (l0 + 0));
            reduce(gout12 * d_0 + gout13 * d_1 + gout14 * d_2 + gout15 * d_3 +
                       gout16 * d_4 + gout17 * d_5,
                vj + (k0 + 2) + nao * (l0 + 0));
            reduce(gout18 * d_0 + gout19 * d_1 + gout20 * d_2 + gout21 * d_3 +
                       gout22 * d_4 + gout23 * d_5,
                vj + (k0 + 3) + nao * (l0 + 0));
            reduce(gout24 * d_0 + gout25 * d_1 + gout26 * d_2 + gout27 * d_3 +
                       gout28 * d_4 + gout29 * d_5,
                vj + (k0 + 4) + nao * (l0 + 0));
            reduce(gout30 * d_0 + gout31 * d_1 + gout32 * d_2 + gout33 * d_3 +
                       gout34 * d_4 + gout35 * d_5,
                vj + (k0 + 5) + nao * (l0 + 0));
            // ijkl,kl->ij
            d_0 = dm[(k0 + 0) + nao * (l0 + 0)];
            d_1 = dm[(k0 + 1) + nao * (l0 + 0)];
            d_2 = dm[(k0 + 2) + nao * (l0 + 0)];
            d_3 = dm[(k0 + 3) + nao * (l0 + 0)];
            d_4 = dm[(k0 + 4) + nao * (l0 + 0)];
            d_5 = dm[(k0 + 5) + nao * (l0 + 0)];
            reduce(gout0 * d_0 + gout6 * d_1 + gout12 * d_2 + gout18 * d_3 +
                       gout24 * d_4 + gout30 * d_5,
                vj + (i0 + 0) + nao * (j0 + 0));
            reduce(gout1 * d_0 + gout7 * d_1 + gout13 * d_2 + gout19 * d_3 +
                       gout25 * d_4 + gout31 * d_5,
                vj + (i0 + 1) + nao * (j0 + 0));
            reduce(gout2 * d_0 + gout8 * d_1 + gout14 * d_2 + gout20 * d_3 +
                       gout26 * d_4 + gout32 * d_5,
                vj + (i0 + 2) + nao * (j0 + 0));
            reduce(gout3 * d_0 + gout9 * d_1 + gout15 * d_2 + gout21 * d_3 +
                       gout27 * d_4 + gout33 * d_5,
                vj + (i0 + 3) + nao * (j0 + 0));
            reduce(gout4 * d_0 + gout10 * d_1 + gout16 * d_2 + gout22 * d_3 +
                       gout28 * d_4 + gout34 * d_5,
                vj + (i0 + 4) + nao * (j0 + 0));
            reduce(gout5 * d_0 + gout11 * d_1 + gout17 * d_2 + gout23 * d_3 +
                       gout29 * d_4 + gout35 * d_5,
                vj + (i0 + 5) + nao * (j0 + 0));
            vj += nao2;
        }
        if (vk != NULL) {
            // ijkl,jl->ik
            d_0 = dm[(j0 + 0) + nao * (l0 + 0)];
            reduce(gout0 * d_0, vk + (i0 + 0) + nao * (k0 + 0));
            reduce(gout1 * d_0, vk + (i0 + 1) + nao * (k0 + 0));
            reduce(gout2 * d_0, vk + (i0 + 2) + nao * (k0 + 0));
            reduce(gout3 * d_0, vk + (i0 + 3) + nao * (k0 + 0));
            reduce(gout4 * d_0, vk + (i0 + 4) + nao * (k0 + 0));
            reduce(gout5 * d_0, vk + (i0 + 5) + nao * (k0 + 0));
            reduce(gout6 * d_0, vk + (i0 + 0) + nao * (k0 + 1));
            reduce(gout7 * d_0, vk + (i0 + 1) + nao * (k0 + 1));
            reduce(gout8 * d_0, vk + (i0 + 2) + nao * (k0 + 1));
            reduce(gout9 * d_0, vk + (i0 + 3) + nao * (k0 + 1));
            reduce(gout10 * d_0, vk + (i0 + 4) + nao * (k0 + 1));
            reduce(gout11 * d_0, vk + (i0 + 5) + nao * (k0 + 1));
            reduce(gout12 * d_0, vk + (i0 + 0) + nao * (k0 + 2));
            reduce(gout13 * d_0, vk + (i0 + 1) + nao * (k0 + 2));
            reduce(gout14 * d_0, vk + (i0 + 2) + nao * (k0 + 2));
            reduce(gout15 * d_0, vk + (i0 + 3) + nao * (k0 + 2));
            reduce(gout16 * d_0, vk + (i0 + 4) + nao * (k0 + 2));
            reduce(gout17 * d_0, vk + (i0 + 5) + nao * (k0 + 2));
            reduce(gout18 * d_0, vk + (i0 + 0) + nao * (k0 + 3));
            reduce(gout19 * d_0, vk + (i0 + 1) + nao * (k0 + 3));
            reduce(gout20 * d_0, vk + (i0 + 2) + nao * (k0 + 3));
            reduce(gout21 * d_0, vk + (i0 + 3) + nao * (k0 + 3));
            reduce(gout22 * d_0, vk + (i0 + 4) + nao * (k0 + 3));
            reduce(gout23 * d_0, vk + (i0 + 5) + nao * (k0 + 3));
            reduce(gout24 * d_0, vk + (i0 + 0) + nao * (k0 + 4));
            reduce(gout25 * d_0, vk + (i0 + 1) + nao * (k0 + 4));
            reduce(gout26 * d_0, vk + (i0 + 2) + nao * (k0 + 4));
            reduce(gout27 * d_0, vk + (i0 + 3) + nao * (k0 + 4));
            reduce(gout28 * d_0, vk + (i0 + 4) + nao * (k0 + 4));
            reduce(gout29 * d_0, vk + (i0 + 5) + nao * (k0 + 4));
            reduce(gout30 * d_0, vk + (i0 + 0) + nao * (k0 + 5));
            reduce(gout31 * d_0, vk + (i0 + 1) + nao * (k0 + 5));
            reduce(gout32 * d_0, vk + (i0 + 2) + nao * (k0 + 5));
            reduce(gout33 * d_0, vk + (i0 + 3) + nao * (k0 + 5));
            reduce(gout34 * d_0, vk + (i0 + 4) + nao * (k0 + 5));
            reduce(gout35 * d_0, vk + (i0 + 5) + nao * (k0 + 5));
            // ijkl,jk->il
            d_0 = dm[(j0 + 0) + nao * (k0 + 0)];
            d_1 = dm[(j0 + 0) + nao * (k0 + 1)];
            d_2 = dm[(j0 + 0) + nao * (k0 + 2)];
            d_3 = dm[(j0 + 0) + nao * (k0 + 3)];
            d_4 = dm[(j0 + 0) + nao * (k0 + 4)];
            d_5 = dm[(j0 + 0) + nao * (k0 + 5)];
            reduce(gout0 * d_0 + gout6 * d_1 + gout12 * d_2 + gout18 * d_3 +
                       gout24 * d_4 + gout30 * d_5,
                vk + (i0 + 0) + nao * (l0 + 0));
            reduce(gout1 * d_0 + gout7 * d_1 + gout13 * d_2 + gout19 * d_3 +
                       gout25 * d_4 + gout31 * d_5,
                vk + (i0 + 1) + nao * (l0 + 0));
            reduce(gout2 * d_0 + gout8 * d_1 + gout14 * d_2 + gout20 * d_3 +
                       gout26 * d_4 + gout32 * d_5,
                vk + (i0 + 2) + nao * (l0 + 0));
            reduce(gout3 * d_0 + gout9 * d_1 + gout15 * d_2 + gout21 * d_3 +
                       gout27 * d_4 + gout33 * d_5,
                vk + (i0 + 3) + nao * (l0 + 0));
            reduce(gout4 * d_0 + gout10 * d_1 + gout16 * d_2 + gout22 * d_3 +
                       gout28 * d_4 + gout34 * d_5,
                vk + (i0 + 4) + nao * (l0 + 0));
            reduce(gout5 * d_0 + gout11 * d_1 + gout17 * d_2 + gout23 * d_3 +
                       gout29 * d_4 + gout35 * d_5,
                vk + (i0 + 5) + nao * (l0 + 0));
            // ijkl,il->jk
            d_0 = dm[(i0 + 0) + nao * (l0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (l0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (l0 + 0)];
            d_3 = dm[(i0 + 3) + nao * (l0 + 0)];
            d_4 = dm[(i0 + 4) + nao * (l0 + 0)];
            d_5 = dm[(i0 + 5) + nao * (l0 + 0)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout3 * d_3 +
                       gout4 * d_4 + gout5 * d_5,
                vk + (j0 + 0) + nao * (k0 + 0));
            reduce(gout6 * d_0 + gout7 * d_1 + gout8 * d_2 + gout9 * d_3 +
                       gout10 * d_4 + gout11 * d_5,
                vk + (j0 + 0) + nao * (k0 + 1));
            reduce(gout12 * d_0 + gout13 * d_1 + gout14 * d_2 + gout15 * d_3 +
                       gout16 * d_4 + gout17 * d_5,
                vk + (j0 + 0) + nao * (k0 + 2));
            reduce(gout18 * d_0 + gout19 * d_1 + gout20 * d_2 + gout21 * d_3 +
                       gout22 * d_4 + gout23 * d_5,
                vk + (j0 + 0) + nao * (k0 + 3));
            reduce(gout24 * d_0 + gout25 * d_1 + gout26 * d_2 + gout27 * d_3 +
                       gout28 * d_4 + gout29 * d_5,
                vk + (j0 + 0) + nao * (k0 + 4));
            reduce(gout30 * d_0 + gout31 * d_1 + gout32 * d_2 + gout33 * d_3 +
                       gout34 * d_4 + gout35 * d_5,
                vk + (j0 + 0) + nao * (k0 + 5));
            // ijkl,ik->jl
            d_0 = dm[(i0 + 0) + nao * (k0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (k0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (k0 + 0)];
            d_3 = dm[(i0 + 3) + nao * (k0 + 0)];
            d_4 = dm[(i0 + 4) + nao * (k0 + 0)];
            d_5 = dm[(i0 + 5) + nao * (k0 + 0)];
            d_6 = dm[(i0 + 0) + nao * (k0 + 1)];
            d_7 = dm[(i0 + 1) + nao * (k0 + 1)];
            d_8 = dm[(i0 + 2) + nao * (k0 + 1)];
            d_9 = dm[(i0 + 3) + nao * (k0 + 1)];
            d_10 = dm[(i0 + 4) + nao * (k0 + 1)];
            d_11 = dm[(i0 + 5) + nao * (k0 + 1)];
            d_12 = dm[(i0 + 0) + nao * (k0 + 2)];
            d_13 = dm[(i0 + 1) + nao * (k0 + 2)];
            d_14 = dm[(i0 + 2) + nao * (k0 + 2)];
            d_15 = dm[(i0 + 3) + nao * (k0 + 2)];
            d_16 = dm[(i0 + 4) + nao * (k0 + 2)];
            d_17 = dm[(i0 + 5) + nao * (k0 + 2)];
            d_18 = dm[(i0 + 0) + nao * (k0 + 3)];
            d_19 = dm[(i0 + 1) + nao * (k0 + 3)];
            d_20 = dm[(i0 + 2) + nao * (k0 + 3)];
            d_21 = dm[(i0 + 3) + nao * (k0 + 3)];
            d_22 = dm[(i0 + 4) + nao * (k0 + 3)];
            d_23 = dm[(i0 + 5) + nao * (k0 + 3)];
            d_24 = dm[(i0 + 0) + nao * (k0 + 4)];
            d_25 = dm[(i0 + 1) + nao * (k0 + 4)];
            d_26 = dm[(i0 + 2) + nao * (k0 + 4)];
            d_27 = dm[(i0 + 3) + nao * (k0 + 4)];
            d_28 = dm[(i0 + 4) + nao * (k0 + 4)];
            d_29 = dm[(i0 + 5) + nao * (k0 + 4)];
            d_30 = dm[(i0 + 0) + nao * (k0 + 5)];
            d_31 = dm[(i0 + 1) + nao * (k0 + 5)];
            d_32 = dm[(i0 + 2) + nao * (k0 + 5)];
            d_33 = dm[(i0 + 3) + nao * (k0 + 5)];
            d_34 = dm[(i0 + 4) + nao * (k0 + 5)];
            d_35 = dm[(i0 + 5) + nao * (k0 + 5)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout3 * d_3 +
                       gout4 * d_4 + gout5 * d_5 + gout6 * d_6 + gout7 * d_7 +
                       gout8 * d_8 + gout9 * d_9 + gout10 * d_10 +
                       gout11 * d_11 + gout12 * d_12 + gout13 * d_13 +
                       gout14 * d_14 + gout15 * d_15 + gout16 * d_16 +
                       gout17 * d_17 + gout18 * d_18 + gout19 * d_19 +
                       gout20 * d_20 + gout21 * d_21 + gout22 * d_22 +
                       gout23 * d_23 + gout24 * d_24 + gout25 * d_25 +
                       gout26 * d_26 + gout27 * d_27 + gout28 * d_28 +
                       gout29 * d_29 + gout30 * d_30 + gout31 * d_31 +
                       gout32 * d_32 + gout33 * d_33 + gout34 * d_34 +
                       gout35 * d_35,
                vk + (j0 + 0) + nao * (l0 + 0));
            vk += nao2;
        }
        dm += nao2;
    }
}

__global__ static void GINTint2e_jk_kernel2021(JKMatrix jk,
    BasisProdOffsets offsets, GINTEnvVars envs, BasisProdCache bpcache) {
    int ntasks_ij = offsets.ntasks_ij;
    long ntasks = ntasks_ij * offsets.ntasks_kl;
    long task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int igroup = nprim_ij * nprim_kl;
    ntasks *= igroup;
    if (task_ij >= ntasks)
        return;
    int kl = task_ij % nprim_kl;
    task_ij /= nprim_kl;
    int ij = task_ij % nprim_ij;
    task_ij /= nprim_ij;
    int task_kl = task_ij / ntasks_ij;
    task_ij = task_ij % ntasks_ij;

    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    if (bas_ij < bas_kl) {
        return;
    }
    double norm = envs.fac;
    if (bas_ij == bas_kl) {
        norm *= .5;
    }

    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *ao_loc = bpcache.ao_loc;
    int *bas_pair2bra = bpcache.bas_pair2bra;
    int *bas_pair2ket = bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    int k0 = ao_loc[ksh];
    int l0 = ao_loc[lsh];
    double *__restrict__ a12 = bpcache.a12;
    double *__restrict__ e12 = bpcache.e12;
    double *__restrict__ x12 = bpcache.x12;
    double *__restrict__ y12 = bpcache.y12;
    double *__restrict__ z12 = bpcache.z12;
    int i_dm;
    int nbas = bpcache.nbas;
    double *__restrict__ bas_x = bpcache.bas_coords;
    double *__restrict__ bas_y = bas_x + nbas;
    double *__restrict__ bas_z = bas_y + nbas;

    double gout0 = 0;
    double gout1 = 0;
    double gout2 = 0;
    double gout3 = 0;
    double gout4 = 0;
    double gout5 = 0;
    double gout6 = 0;
    double gout7 = 0;
    double gout8 = 0;
    double gout9 = 0;
    double gout10 = 0;
    double gout11 = 0;
    double gout12 = 0;
    double gout13 = 0;
    double gout14 = 0;
    double gout15 = 0;
    double gout16 = 0;
    double gout17 = 0;
    double gout18 = 0;
    double gout19 = 0;
    double gout20 = 0;
    double gout21 = 0;
    double gout22 = 0;
    double gout23 = 0;
    double gout24 = 0;
    double gout25 = 0;
    double gout26 = 0;
    double gout27 = 0;
    double gout28 = 0;
    double gout29 = 0;
    double gout30 = 0;
    double gout31 = 0;
    double gout32 = 0;
    double gout33 = 0;
    double gout34 = 0;
    double gout35 = 0;
    double gout36 = 0;
    double gout37 = 0;
    double gout38 = 0;
    double gout39 = 0;
    double gout40 = 0;
    double gout41 = 0;
    double gout42 = 0;
    double gout43 = 0;
    double gout44 = 0;
    double gout45 = 0;
    double gout46 = 0;
    double gout47 = 0;
    double gout48 = 0;
    double gout49 = 0;
    double gout50 = 0;
    double gout51 = 0;
    double gout52 = 0;
    double gout53 = 0;
    double gout54 = 0;
    double gout55 = 0;
    double gout56 = 0;
    double gout57 = 0;
    double gout58 = 0;
    double gout59 = 0;
    double gout60 = 0;
    double gout61 = 0;
    double gout62 = 0;
    double gout63 = 0;
    double gout64 = 0;
    double gout65 = 0;
    double gout66 = 0;
    double gout67 = 0;
    double gout68 = 0;
    double gout69 = 0;
    double gout70 = 0;
    double gout71 = 0;
    double gout72 = 0;
    double gout73 = 0;
    double gout74 = 0;
    double gout75 = 0;
    double gout76 = 0;
    double gout77 = 0;
    double gout78 = 0;
    double gout79 = 0;
    double gout80 = 0;
    double gout81 = 0;
    double gout82 = 0;
    double gout83 = 0;
    double gout84 = 0;
    double gout85 = 0;
    double gout86 = 0;
    double gout87 = 0;
    double gout88 = 0;
    double gout89 = 0;
    double gout90 = 0;
    double gout91 = 0;
    double gout92 = 0;
    double gout93 = 0;
    double gout94 = 0;
    double gout95 = 0;
    double gout96 = 0;
    double gout97 = 0;
    double gout98 = 0;
    double gout99 = 0;
    double gout100 = 0;
    double gout101 = 0;
    double gout102 = 0;
    double gout103 = 0;
    double gout104 = 0;
    double gout105 = 0;
    double gout106 = 0;
    double gout107 = 0;
    double xi = bas_x[ish];
    double yi = bas_y[ish];
    double zi = bas_z[ish];
    double xk = bas_x[ksh];
    double yk = bas_y[ksh];
    double zk = bas_z[ksh];
    double xkxl = xk - bas_x[lsh];
    double ykyl = yk - bas_y[lsh];
    double zkzl = zk - bas_z[lsh];
    auto reduce = SegReduce<double>(igroup);
    ij += prim_ij;
    kl += prim_kl;
    double aij = a12[ij];
    double eij = e12[ij];
    double xij = x12[ij];
    double yij = y12[ij];
    double zij = z12[ij];
    double akl = a12[kl];
    double ekl = e12[kl];
    double xkl = x12[kl];
    double ykl = y12[kl];
    double zkl = z12[kl];
    double xijxkl = xij - xkl;
    double yijykl = yij - ykl;
    double zijzkl = zij - zkl;
    double aijkl = aij + akl;
    double a1 = aij * akl;
    double a0 = a1 / aijkl;
    double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
    double fac = norm * eij * ekl / (sqrt(aijkl) * a1);

    double rw[6];
    double root0, weight0;
    GINTrys_root<3>(x, rw);
    int irys;
    for (irys = 0; irys < 3; ++irys) {
        root0 = rw[irys];
        weight0 = rw[irys + 3];
        double u2 = a0 * root0;
        double tmp4 = .5 / (u2 * aijkl + a1);
        double b00 = u2 * tmp4;
        double tmp1 = 2 * b00;
        double tmp2 = tmp1 * akl;
        double b10 = b00 + tmp4 * akl;
        double c00x = xij - xi - tmp2 * xijxkl;
        double c00y = yij - yi - tmp2 * yijykl;
        double c00z = zij - zi - tmp2 * zijzkl;
        double tmp3 = tmp1 * aij;
        double b01 = b00 + tmp4 * aij;
        double c0px = xkl - xk + tmp3 * xijxkl;
        double c0py = ykl - yk + tmp3 * yijykl;
        double c0pz = zkl - zk + tmp3 * zijzkl;
        double g_0 = 1;
        double g_1 = c00x;
        double g_2 = c00x * c00x + b10;
        double g_3 = c0px;
        double g_4 = c0px * c00x + b00;
        double g_5 = b00 * c00x + b10 * c0px + c00x * g_4;
        double g_6 = c0px * c0px + b01;
        double g_7 = b00 * c0px + b01 * c00x + c0px * g_4;
        double g_8 = 2 * b00 * g_4 + b10 * g_6 + c00x * g_7;
        double g_9 = c0px + xkxl;
        double g_10 = c00x * (c0px + xkxl) + b00;
        double g_11 = b00 * c00x + b10 * c0px + c00x * g_4 + xkxl * g_2;
        double g_12 = c0px * (c0px + xkxl) + b01;
        double g_13 = b00 * c0px + b01 * c00x + c0px * g_4 + xkxl * g_4;
        double g_14 = 2 * b00 * g_4 + b10 * g_6 + c00x * g_7 + xkxl * g_5;
        double g_15 = c0px * (2 * b01 + g_6) + xkxl * g_6;
        double g_16 = 2 * b01 * g_4 + b00 * g_6 + c0px * g_7 + xkxl * g_7;
        double g_17 = xkxl * g_8 +
                      c00x * (c0px * g_7 + 2 * b01 * g_4 + b00 * g_6) +
                      b10 * (c0px * g_6 + 2 * b01 * c0px) + 3 * b00 * g_7;
        double g_18 = 1;
        double g_19 = c00y;
        double g_20 = c00y * c00y + b10;
        double g_21 = c0py;
        double g_22 = c0py * c00y + b00;
        double g_23 = b00 * c00y + b10 * c0py + c00y * g_22;
        double g_24 = c0py * c0py + b01;
        double g_25 = b00 * c0py + b01 * c00y + c0py * g_22;
        double g_26 = 2 * b00 * g_22 + b10 * g_24 + c00y * g_25;
        double g_27 = c0py + ykyl;
        double g_28 = c00y * (c0py + ykyl) + b00;
        double g_29 = b00 * c00y + b10 * c0py + c00y * g_22 + ykyl * g_20;
        double g_30 = c0py * (c0py + ykyl) + b01;
        double g_31 = b00 * c0py + b01 * c00y + c0py * g_22 + ykyl * g_22;
        double g_32 = 2 * b00 * g_22 + b10 * g_24 + c00y * g_25 + ykyl * g_23;
        double g_33 = c0py * (2 * b01 + g_24) + ykyl * g_24;
        double g_34 = 2 * b01 * g_22 + b00 * g_24 + c0py * g_25 + ykyl * g_25;
        double g_35 = ykyl * g_26 +
                      c00y * (c0py * g_25 + 2 * b01 * g_22 + b00 * g_24) +
                      b10 * (c0py * g_24 + 2 * b01 * c0py) + 3 * b00 * g_25;
        double g_36 = weight0 * fac;
        double g_37 = c00z * g_36;
        double g_38 = b10 * g_36 + c00z * g_37;
        double g_39 = c0pz * g_36;
        double g_40 = b00 * g_36 + c0pz * g_37;
        double g_41 = b00 * g_37 + b10 * g_39 + c00z * g_40;
        double g_42 = b01 * g_36 + c0pz * g_39;
        double g_43 = b00 * g_39 + b01 * g_37 + c0pz * g_40;
        double g_44 = 2 * b00 * g_40 + b10 * g_42 + c00z * g_43;
        double g_45 = g_36 * (c0pz + zkzl);
        double g_46 = b00 * g_36 + c0pz * g_37 + zkzl * g_37;
        double g_47 = b00 * g_37 + b10 * g_39 + c00z * g_40 + zkzl * g_38;
        double g_48 = b01 * g_36 + c0pz * g_39 + zkzl * g_39;
        double g_49 = b00 * g_39 + b01 * g_37 + c0pz * g_40 + zkzl * g_40;
        double g_50 = 2 * b00 * g_40 + b10 * g_42 + c00z * g_43 + zkzl * g_41;
        double g_51 = 2 * b01 * g_39 + c0pz * g_42 + zkzl * g_42;
        double g_52 = 2 * b01 * g_40 + b00 * g_42 + c0pz * g_43 + zkzl * g_43;
        double g_53 = zkzl * g_44 +
                      c00z * (c0pz * g_43 + 2 * b01 * g_40 + b00 * g_42) +
                      b10 * (c0pz * g_42 + 2 * b01 * g_39) + 3 * b00 * g_43;
        gout0 += g_17 * g_18 * g_36;
        gout1 += g_16 * g_19 * g_36;
        gout2 += g_16 * g_18 * g_37;
        gout3 += g_15 * g_20 * g_36;
        gout4 += g_15 * g_19 * g_37;
        gout5 += g_15 * g_18 * g_38;
        gout6 += g_14 * g_21 * g_36;
        gout7 += g_13 * g_22 * g_36;
        gout8 += g_13 * g_21 * g_37;
        gout9 += g_12 * g_23 * g_36;
        gout10 += g_12 * g_22 * g_37;
        gout11 += g_12 * g_21 * g_38;
        gout12 += g_14 * g_18 * g_39;
        gout13 += g_13 * g_19 * g_39;
        gout14 += g_13 * g_18 * g_40;
        gout15 += g_12 * g_20 * g_39;
        gout16 += g_12 * g_19 * g_40;
        gout17 += g_12 * g_18 * g_41;
        gout18 += g_11 * g_24 * g_36;
        gout19 += g_10 * g_25 * g_36;
        gout20 += g_10 * g_24 * g_37;
        gout21 += g_9 * g_26 * g_36;
        gout22 += g_9 * g_25 * g_37;
        gout23 += g_9 * g_24 * g_38;
        gout24 += g_11 * g_21 * g_39;
        gout25 += g_10 * g_22 * g_39;
        gout26 += g_10 * g_21 * g_40;
        gout27 += g_9 * g_23 * g_39;
        gout28 += g_9 * g_22 * g_40;
        gout29 += g_9 * g_21 * g_41;
        gout30 += g_11 * g_18 * g_42;
        gout31 += g_10 * g_19 * g_42;
        gout32 += g_10 * g_18 * g_43;
        gout33 += g_9 * g_20 * g_42;
        gout34 += g_9 * g_19 * g_43;
        gout35 += g_9 * g_18 * g_44;
        gout36 += g_8 * g_27 * g_36;
        gout37 += g_7 * g_28 * g_36;
        gout38 += g_7 * g_27 * g_37;
        gout39 += g_6 * g_29 * g_36;
        gout40 += g_6 * g_28 * g_37;
        gout41 += g_6 * g_27 * g_38;
        gout42 += g_5 * g_30 * g_36;
        gout43 += g_4 * g_31 * g_36;
        gout44 += g_4 * g_30 * g_37;
        gout45 += g_3 * g_32 * g_36;
        gout46 += g_3 * g_31 * g_37;
        gout47 += g_3 * g_30 * g_38;
        gout48 += g_5 * g_27 * g_39;
        gout49 += g_4 * g_28 * g_39;
        gout50 += g_4 * g_27 * g_40;
        gout51 += g_3 * g_29 * g_39;
        gout52 += g_3 * g_28 * g_40;
        gout53 += g_3 * g_27 * g_41;
        gout54 += g_2 * g_33 * g_36;
        gout55 += g_1 * g_34 * g_36;
        gout56 += g_1 * g_33 * g_37;
        gout57 += g_0 * g_35 * g_36;
        gout58 += g_0 * g_34 * g_37;
        gout59 += g_0 * g_33 * g_38;
        gout60 += g_2 * g_30 * g_39;
        gout61 += g_1 * g_31 * g_39;
        gout62 += g_1 * g_30 * g_40;
        gout63 += g_0 * g_32 * g_39;
        gout64 += g_0 * g_31 * g_40;
        gout65 += g_0 * g_30 * g_41;
        gout66 += g_2 * g_27 * g_42;
        gout67 += g_1 * g_28 * g_42;
        gout68 += g_1 * g_27 * g_43;
        gout69 += g_0 * g_29 * g_42;
        gout70 += g_0 * g_28 * g_43;
        gout71 += g_0 * g_27 * g_44;
        gout72 += g_8 * g_18 * g_45;
        gout73 += g_7 * g_19 * g_45;
        gout74 += g_7 * g_18 * g_46;
        gout75 += g_6 * g_20 * g_45;
        gout76 += g_6 * g_19 * g_46;
        gout77 += g_6 * g_18 * g_47;
        gout78 += g_5 * g_21 * g_45;
        gout79 += g_4 * g_22 * g_45;
        gout80 += g_4 * g_21 * g_46;
        gout81 += g_3 * g_23 * g_45;
        gout82 += g_3 * g_22 * g_46;
        gout83 += g_3 * g_21 * g_47;
        gout84 += g_5 * g_18 * g_48;
        gout85 += g_4 * g_19 * g_48;
        gout86 += g_4 * g_18 * g_49;
        gout87 += g_3 * g_20 * g_48;
        gout88 += g_3 * g_19 * g_49;
        gout89 += g_3 * g_18 * g_50;
        gout90 += g_2 * g_24 * g_45;
        gout91 += g_1 * g_25 * g_45;
        gout92 += g_1 * g_24 * g_46;
        gout93 += g_0 * g_26 * g_45;
        gout94 += g_0 * g_25 * g_46;
        gout95 += g_0 * g_24 * g_47;
        gout96 += g_2 * g_21 * g_48;
        gout97 += g_1 * g_22 * g_48;
        gout98 += g_1 * g_21 * g_49;
        gout99 += g_0 * g_23 * g_48;
        gout100 += g_0 * g_22 * g_49;
        gout101 += g_0 * g_21 * g_50;
        gout102 += g_2 * g_18 * g_51;
        gout103 += g_1 * g_19 * g_51;
        gout104 += g_1 * g_18 * g_52;
        gout105 += g_0 * g_20 * g_51;
        gout106 += g_0 * g_19 * g_52;
        gout107 += g_0 * g_18 * g_53;
    }
    double d_0, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, d_9;
    double d_10, d_11, d_12, d_13, d_14, d_15, d_16, d_17, d_18, d_19;
    double d_20, d_21, d_22, d_23, d_24, d_25, d_26, d_27, d_28, d_29;
    double d_30, d_31, d_32, d_33, d_34, d_35;
    int n_dm = jk.n_dm;
    int nao = jk.nao;
    size_t nao2 = nao * nao;
    double *__restrict__ dm = jk.dm;
    double *vj = jk.vj;
    double *vk = jk.vk;
    for (i_dm = 0; i_dm < n_dm; ++i_dm) {
        if (vj != NULL) {
            // ijkl,ij->kl
            d_0 = dm[(i0 + 0) + nao * (j0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (j0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (j0 + 0)];
            d_3 = dm[(i0 + 3) + nao * (j0 + 0)];
            d_4 = dm[(i0 + 4) + nao * (j0 + 0)];
            d_5 = dm[(i0 + 5) + nao * (j0 + 0)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout3 * d_3 +
                       gout4 * d_4 + gout5 * d_5,
                vj + (k0 + 0) + nao * (l0 + 0));
            reduce(gout6 * d_0 + gout7 * d_1 + gout8 * d_2 + gout9 * d_3 +
                       gout10 * d_4 + gout11 * d_5,
                vj + (k0 + 1) + nao * (l0 + 0));
            reduce(gout12 * d_0 + gout13 * d_1 + gout14 * d_2 + gout15 * d_3 +
                       gout16 * d_4 + gout17 * d_5,
                vj + (k0 + 2) + nao * (l0 + 0));
            reduce(gout18 * d_0 + gout19 * d_1 + gout20 * d_2 + gout21 * d_3 +
                       gout22 * d_4 + gout23 * d_5,
                vj + (k0 + 3) + nao * (l0 + 0));
            reduce(gout24 * d_0 + gout25 * d_1 + gout26 * d_2 + gout27 * d_3 +
                       gout28 * d_4 + gout29 * d_5,
                vj + (k0 + 4) + nao * (l0 + 0));
            reduce(gout30 * d_0 + gout31 * d_1 + gout32 * d_2 + gout33 * d_3 +
                       gout34 * d_4 + gout35 * d_5,
                vj + (k0 + 5) + nao * (l0 + 0));
            reduce(gout36 * d_0 + gout37 * d_1 + gout38 * d_2 + gout39 * d_3 +
                       gout40 * d_4 + gout41 * d_5,
                vj + (k0 + 0) + nao * (l0 + 1));
            reduce(gout42 * d_0 + gout43 * d_1 + gout44 * d_2 + gout45 * d_3 +
                       gout46 * d_4 + gout47 * d_5,
                vj + (k0 + 1) + nao * (l0 + 1));
            reduce(gout48 * d_0 + gout49 * d_1 + gout50 * d_2 + gout51 * d_3 +
                       gout52 * d_4 + gout53 * d_5,
                vj + (k0 + 2) + nao * (l0 + 1));
            reduce(gout54 * d_0 + gout55 * d_1 + gout56 * d_2 + gout57 * d_3 +
                       gout58 * d_4 + gout59 * d_5,
                vj + (k0 + 3) + nao * (l0 + 1));
            reduce(gout60 * d_0 + gout61 * d_1 + gout62 * d_2 + gout63 * d_3 +
                       gout64 * d_4 + gout65 * d_5,
                vj + (k0 + 4) + nao * (l0 + 1));
            reduce(gout66 * d_0 + gout67 * d_1 + gout68 * d_2 + gout69 * d_3 +
                       gout70 * d_4 + gout71 * d_5,
                vj + (k0 + 5) + nao * (l0 + 1));
            reduce(gout72 * d_0 + gout73 * d_1 + gout74 * d_2 + gout75 * d_3 +
                       gout76 * d_4 + gout77 * d_5,
                vj + (k0 + 0) + nao * (l0 + 2));
            reduce(gout78 * d_0 + gout79 * d_1 + gout80 * d_2 + gout81 * d_3 +
                       gout82 * d_4 + gout83 * d_5,
                vj + (k0 + 1) + nao * (l0 + 2));
            reduce(gout84 * d_0 + gout85 * d_1 + gout86 * d_2 + gout87 * d_3 +
                       gout88 * d_4 + gout89 * d_5,
                vj + (k0 + 2) + nao * (l0 + 2));
            reduce(gout90 * d_0 + gout91 * d_1 + gout92 * d_2 + gout93 * d_3 +
                       gout94 * d_4 + gout95 * d_5,
                vj + (k0 + 3) + nao * (l0 + 2));
            reduce(gout96 * d_0 + gout97 * d_1 + gout98 * d_2 + gout99 * d_3 +
                       gout100 * d_4 + gout101 * d_5,
                vj + (k0 + 4) + nao * (l0 + 2));
            reduce(gout102 * d_0 + gout103 * d_1 + gout104 * d_2 +
                       gout105 * d_3 + gout106 * d_4 + gout107 * d_5,
                vj + (k0 + 5) + nao * (l0 + 2));
            // ijkl,kl->ij
            d_0 = dm[(k0 + 0) + nao * (l0 + 0)];
            d_1 = dm[(k0 + 1) + nao * (l0 + 0)];
            d_2 = dm[(k0 + 2) + nao * (l0 + 0)];
            d_3 = dm[(k0 + 3) + nao * (l0 + 0)];
            d_4 = dm[(k0 + 4) + nao * (l0 + 0)];
            d_5 = dm[(k0 + 5) + nao * (l0 + 0)];
            d_6 = dm[(k0 + 0) + nao * (l0 + 1)];
            d_7 = dm[(k0 + 1) + nao * (l0 + 1)];
            d_8 = dm[(k0 + 2) + nao * (l0 + 1)];
            d_9 = dm[(k0 + 3) + nao * (l0 + 1)];
            d_10 = dm[(k0 + 4) + nao * (l0 + 1)];
            d_11 = dm[(k0 + 5) + nao * (l0 + 1)];
            d_12 = dm[(k0 + 0) + nao * (l0 + 2)];
            d_13 = dm[(k0 + 1) + nao * (l0 + 2)];
            d_14 = dm[(k0 + 2) + nao * (l0 + 2)];
            d_15 = dm[(k0 + 3) + nao * (l0 + 2)];
            d_16 = dm[(k0 + 4) + nao * (l0 + 2)];
            d_17 = dm[(k0 + 5) + nao * (l0 + 2)];
            reduce(gout0 * d_0 + gout6 * d_1 + gout12 * d_2 + gout18 * d_3 +
                       gout24 * d_4 + gout30 * d_5 + gout36 * d_6 +
                       gout42 * d_7 + gout48 * d_8 + gout54 * d_9 +
                       gout60 * d_10 + gout66 * d_11 + gout72 * d_12 +
                       gout78 * d_13 + gout84 * d_14 + gout90 * d_15 +
                       gout96 * d_16 + gout102 * d_17,
                vj + (i0 + 0) + nao * (j0 + 0));
            reduce(gout1 * d_0 + gout7 * d_1 + gout13 * d_2 + gout19 * d_3 +
                       gout25 * d_4 + gout31 * d_5 + gout37 * d_6 +
                       gout43 * d_7 + gout49 * d_8 + gout55 * d_9 +
                       gout61 * d_10 + gout67 * d_11 + gout73 * d_12 +
                       gout79 * d_13 + gout85 * d_14 + gout91 * d_15 +
                       gout97 * d_16 + gout103 * d_17,
                vj + (i0 + 1) + nao * (j0 + 0));
            reduce(gout2 * d_0 + gout8 * d_1 + gout14 * d_2 + gout20 * d_3 +
                       gout26 * d_4 + gout32 * d_5 + gout38 * d_6 +
                       gout44 * d_7 + gout50 * d_8 + gout56 * d_9 +
                       gout62 * d_10 + gout68 * d_11 + gout74 * d_12 +
                       gout80 * d_13 + gout86 * d_14 + gout92 * d_15 +
                       gout98 * d_16 + gout104 * d_17,
                vj + (i0 + 2) + nao * (j0 + 0));
            reduce(gout3 * d_0 + gout9 * d_1 + gout15 * d_2 + gout21 * d_3 +
                       gout27 * d_4 + gout33 * d_5 + gout39 * d_6 +
                       gout45 * d_7 + gout51 * d_8 + gout57 * d_9 +
                       gout63 * d_10 + gout69 * d_11 + gout75 * d_12 +
                       gout81 * d_13 + gout87 * d_14 + gout93 * d_15 +
                       gout99 * d_16 + gout105 * d_17,
                vj + (i0 + 3) + nao * (j0 + 0));
            reduce(gout4 * d_0 + gout10 * d_1 + gout16 * d_2 + gout22 * d_3 +
                       gout28 * d_4 + gout34 * d_5 + gout40 * d_6 +
                       gout46 * d_7 + gout52 * d_8 + gout58 * d_9 +
                       gout64 * d_10 + gout70 * d_11 + gout76 * d_12 +
                       gout82 * d_13 + gout88 * d_14 + gout94 * d_15 +
                       gout100 * d_16 + gout106 * d_17,
                vj + (i0 + 4) + nao * (j0 + 0));
            reduce(gout5 * d_0 + gout11 * d_1 + gout17 * d_2 + gout23 * d_3 +
                       gout29 * d_4 + gout35 * d_5 + gout41 * d_6 +
                       gout47 * d_7 + gout53 * d_8 + gout59 * d_9 +
                       gout65 * d_10 + gout71 * d_11 + gout77 * d_12 +
                       gout83 * d_13 + gout89 * d_14 + gout95 * d_15 +
                       gout101 * d_16 + gout107 * d_17,
                vj + (i0 + 5) + nao * (j0 + 0));
            vj += nao2;
        }
        if (vk != NULL) {
            // ijkl,jl->ik
            d_0 = dm[(j0 + 0) + nao * (l0 + 0)];
            d_1 = dm[(j0 + 0) + nao * (l0 + 1)];
            d_2 = dm[(j0 + 0) + nao * (l0 + 2)];
            reduce(gout0 * d_0 + gout36 * d_1 + gout72 * d_2,
                vk + (i0 + 0) + nao * (k0 + 0));
            reduce(gout1 * d_0 + gout37 * d_1 + gout73 * d_2,
                vk + (i0 + 1) + nao * (k0 + 0));
            reduce(gout2 * d_0 + gout38 * d_1 + gout74 * d_2,
                vk + (i0 + 2) + nao * (k0 + 0));
            reduce(gout3 * d_0 + gout39 * d_1 + gout75 * d_2,
                vk + (i0 + 3) + nao * (k0 + 0));
            reduce(gout4 * d_0 + gout40 * d_1 + gout76 * d_2,
                vk + (i0 + 4) + nao * (k0 + 0));
            reduce(gout5 * d_0 + gout41 * d_1 + gout77 * d_2,
                vk + (i0 + 5) + nao * (k0 + 0));
            reduce(gout6 * d_0 + gout42 * d_1 + gout78 * d_2,
                vk + (i0 + 0) + nao * (k0 + 1));
            reduce(gout7 * d_0 + gout43 * d_1 + gout79 * d_2,
                vk + (i0 + 1) + nao * (k0 + 1));
            reduce(gout8 * d_0 + gout44 * d_1 + gout80 * d_2,
                vk + (i0 + 2) + nao * (k0 + 1));
            reduce(gout9 * d_0 + gout45 * d_1 + gout81 * d_2,
                vk + (i0 + 3) + nao * (k0 + 1));
            reduce(gout10 * d_0 + gout46 * d_1 + gout82 * d_2,
                vk + (i0 + 4) + nao * (k0 + 1));
            reduce(gout11 * d_0 + gout47 * d_1 + gout83 * d_2,
                vk + (i0 + 5) + nao * (k0 + 1));
            reduce(gout12 * d_0 + gout48 * d_1 + gout84 * d_2,
                vk + (i0 + 0) + nao * (k0 + 2));
            reduce(gout13 * d_0 + gout49 * d_1 + gout85 * d_2,
                vk + (i0 + 1) + nao * (k0 + 2));
            reduce(gout14 * d_0 + gout50 * d_1 + gout86 * d_2,
                vk + (i0 + 2) + nao * (k0 + 2));
            reduce(gout15 * d_0 + gout51 * d_1 + gout87 * d_2,
                vk + (i0 + 3) + nao * (k0 + 2));
            reduce(gout16 * d_0 + gout52 * d_1 + gout88 * d_2,
                vk + (i0 + 4) + nao * (k0 + 2));
            reduce(gout17 * d_0 + gout53 * d_1 + gout89 * d_2,
                vk + (i0 + 5) + nao * (k0 + 2));
            reduce(gout18 * d_0 + gout54 * d_1 + gout90 * d_2,
                vk + (i0 + 0) + nao * (k0 + 3));
            reduce(gout19 * d_0 + gout55 * d_1 + gout91 * d_2,
                vk + (i0 + 1) + nao * (k0 + 3));
            reduce(gout20 * d_0 + gout56 * d_1 + gout92 * d_2,
                vk + (i0 + 2) + nao * (k0 + 3));
            reduce(gout21 * d_0 + gout57 * d_1 + gout93 * d_2,
                vk + (i0 + 3) + nao * (k0 + 3));
            reduce(gout22 * d_0 + gout58 * d_1 + gout94 * d_2,
                vk + (i0 + 4) + nao * (k0 + 3));
            reduce(gout23 * d_0 + gout59 * d_1 + gout95 * d_2,
                vk + (i0 + 5) + nao * (k0 + 3));
            reduce(gout24 * d_0 + gout60 * d_1 + gout96 * d_2,
                vk + (i0 + 0) + nao * (k0 + 4));
            reduce(gout25 * d_0 + gout61 * d_1 + gout97 * d_2,
                vk + (i0 + 1) + nao * (k0 + 4));
            reduce(gout26 * d_0 + gout62 * d_1 + gout98 * d_2,
                vk + (i0 + 2) + nao * (k0 + 4));
            reduce(gout27 * d_0 + gout63 * d_1 + gout99 * d_2,
                vk + (i0 + 3) + nao * (k0 + 4));
            reduce(gout28 * d_0 + gout64 * d_1 + gout100 * d_2,
                vk + (i0 + 4) + nao * (k0 + 4));
            reduce(gout29 * d_0 + gout65 * d_1 + gout101 * d_2,
                vk + (i0 + 5) + nao * (k0 + 4));
            reduce(gout30 * d_0 + gout66 * d_1 + gout102 * d_2,
                vk + (i0 + 0) + nao * (k0 + 5));
            reduce(gout31 * d_0 + gout67 * d_1 + gout103 * d_2,
                vk + (i0 + 1) + nao * (k0 + 5));
            reduce(gout32 * d_0 + gout68 * d_1 + gout104 * d_2,
                vk + (i0 + 2) + nao * (k0 + 5));
            reduce(gout33 * d_0 + gout69 * d_1 + gout105 * d_2,
                vk + (i0 + 3) + nao * (k0 + 5));
            reduce(gout34 * d_0 + gout70 * d_1 + gout106 * d_2,
                vk + (i0 + 4) + nao * (k0 + 5));
            reduce(gout35 * d_0 + gout71 * d_1 + gout107 * d_2,
                vk + (i0 + 5) + nao * (k0 + 5));
            // ijkl,jk->il
            d_0 = dm[(j0 + 0) + nao * (k0 + 0)];
            d_1 = dm[(j0 + 0) + nao * (k0 + 1)];
            d_2 = dm[(j0 + 0) + nao * (k0 + 2)];
            d_3 = dm[(j0 + 0) + nao * (k0 + 3)];
            d_4 = dm[(j0 + 0) + nao * (k0 + 4)];
            d_5 = dm[(j0 + 0) + nao * (k0 + 5)];
            reduce(gout0 * d_0 + gout6 * d_1 + gout12 * d_2 + gout18 * d_3 +
                       gout24 * d_4 + gout30 * d_5,
                vk + (i0 + 0) + nao * (l0 + 0));
            reduce(gout1 * d_0 + gout7 * d_1 + gout13 * d_2 + gout19 * d_3 +
                       gout25 * d_4 + gout31 * d_5,
                vk + (i0 + 1) + nao * (l0 + 0));
            reduce(gout2 * d_0 + gout8 * d_1 + gout14 * d_2 + gout20 * d_3 +
                       gout26 * d_4 + gout32 * d_5,
                vk + (i0 + 2) + nao * (l0 + 0));
            reduce(gout3 * d_0 + gout9 * d_1 + gout15 * d_2 + gout21 * d_3 +
                       gout27 * d_4 + gout33 * d_5,
                vk + (i0 + 3) + nao * (l0 + 0));
            reduce(gout4 * d_0 + gout10 * d_1 + gout16 * d_2 + gout22 * d_3 +
                       gout28 * d_4 + gout34 * d_5,
                vk + (i0 + 4) + nao * (l0 + 0));
            reduce(gout5 * d_0 + gout11 * d_1 + gout17 * d_2 + gout23 * d_3 +
                       gout29 * d_4 + gout35 * d_5,
                vk + (i0 + 5) + nao * (l0 + 0));
            reduce(gout36 * d_0 + gout42 * d_1 + gout48 * d_2 + gout54 * d_3 +
                       gout60 * d_4 + gout66 * d_5,
                vk + (i0 + 0) + nao * (l0 + 1));
            reduce(gout37 * d_0 + gout43 * d_1 + gout49 * d_2 + gout55 * d_3 +
                       gout61 * d_4 + gout67 * d_5,
                vk + (i0 + 1) + nao * (l0 + 1));
            reduce(gout38 * d_0 + gout44 * d_1 + gout50 * d_2 + gout56 * d_3 +
                       gout62 * d_4 + gout68 * d_5,
                vk + (i0 + 2) + nao * (l0 + 1));
            reduce(gout39 * d_0 + gout45 * d_1 + gout51 * d_2 + gout57 * d_3 +
                       gout63 * d_4 + gout69 * d_5,
                vk + (i0 + 3) + nao * (l0 + 1));
            reduce(gout40 * d_0 + gout46 * d_1 + gout52 * d_2 + gout58 * d_3 +
                       gout64 * d_4 + gout70 * d_5,
                vk + (i0 + 4) + nao * (l0 + 1));
            reduce(gout41 * d_0 + gout47 * d_1 + gout53 * d_2 + gout59 * d_3 +
                       gout65 * d_4 + gout71 * d_5,
                vk + (i0 + 5) + nao * (l0 + 1));
            reduce(gout72 * d_0 + gout78 * d_1 + gout84 * d_2 + gout90 * d_3 +
                       gout96 * d_4 + gout102 * d_5,
                vk + (i0 + 0) + nao * (l0 + 2));
            reduce(gout73 * d_0 + gout79 * d_1 + gout85 * d_2 + gout91 * d_3 +
                       gout97 * d_4 + gout103 * d_5,
                vk + (i0 + 1) + nao * (l0 + 2));
            reduce(gout74 * d_0 + gout80 * d_1 + gout86 * d_2 + gout92 * d_3 +
                       gout98 * d_4 + gout104 * d_5,
                vk + (i0 + 2) + nao * (l0 + 2));
            reduce(gout75 * d_0 + gout81 * d_1 + gout87 * d_2 + gout93 * d_3 +
                       gout99 * d_4 + gout105 * d_5,
                vk + (i0 + 3) + nao * (l0 + 2));
            reduce(gout76 * d_0 + gout82 * d_1 + gout88 * d_2 + gout94 * d_3 +
                       gout100 * d_4 + gout106 * d_5,
                vk + (i0 + 4) + nao * (l0 + 2));
            reduce(gout77 * d_0 + gout83 * d_1 + gout89 * d_2 + gout95 * d_3 +
                       gout101 * d_4 + gout107 * d_5,
                vk + (i0 + 5) + nao * (l0 + 2));
            // ijkl,il->jk
            d_0 = dm[(i0 + 0) + nao * (l0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (l0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (l0 + 0)];
            d_3 = dm[(i0 + 3) + nao * (l0 + 0)];
            d_4 = dm[(i0 + 4) + nao * (l0 + 0)];
            d_5 = dm[(i0 + 5) + nao * (l0 + 0)];
            d_6 = dm[(i0 + 0) + nao * (l0 + 1)];
            d_7 = dm[(i0 + 1) + nao * (l0 + 1)];
            d_8 = dm[(i0 + 2) + nao * (l0 + 1)];
            d_9 = dm[(i0 + 3) + nao * (l0 + 1)];
            d_10 = dm[(i0 + 4) + nao * (l0 + 1)];
            d_11 = dm[(i0 + 5) + nao * (l0 + 1)];
            d_12 = dm[(i0 + 0) + nao * (l0 + 2)];
            d_13 = dm[(i0 + 1) + nao * (l0 + 2)];
            d_14 = dm[(i0 + 2) + nao * (l0 + 2)];
            d_15 = dm[(i0 + 3) + nao * (l0 + 2)];
            d_16 = dm[(i0 + 4) + nao * (l0 + 2)];
            d_17 = dm[(i0 + 5) + nao * (l0 + 2)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout3 * d_3 +
                       gout4 * d_4 + gout5 * d_5 + gout36 * d_6 +
                       gout37 * d_7 + gout38 * d_8 + gout39 * d_9 +
                       gout40 * d_10 + gout41 * d_11 + gout72 * d_12 +
                       gout73 * d_13 + gout74 * d_14 + gout75 * d_15 +
                       gout76 * d_16 + gout77 * d_17,
                vk + (j0 + 0) + nao * (k0 + 0));
            reduce(gout6 * d_0 + gout7 * d_1 + gout8 * d_2 + gout9 * d_3 +
                       gout10 * d_4 + gout11 * d_5 + gout42 * d_6 +
                       gout43 * d_7 + gout44 * d_8 + gout45 * d_9 +
                       gout46 * d_10 + gout47 * d_11 + gout78 * d_12 +
                       gout79 * d_13 + gout80 * d_14 + gout81 * d_15 +
                       gout82 * d_16 + gout83 * d_17,
                vk + (j0 + 0) + nao * (k0 + 1));
            reduce(gout12 * d_0 + gout13 * d_1 + gout14 * d_2 + gout15 * d_3 +
                       gout16 * d_4 + gout17 * d_5 + gout48 * d_6 +
                       gout49 * d_7 + gout50 * d_8 + gout51 * d_9 +
                       gout52 * d_10 + gout53 * d_11 + gout84 * d_12 +
                       gout85 * d_13 + gout86 * d_14 + gout87 * d_15 +
                       gout88 * d_16 + gout89 * d_17,
                vk + (j0 + 0) + nao * (k0 + 2));
            reduce(gout18 * d_0 + gout19 * d_1 + gout20 * d_2 + gout21 * d_3 +
                       gout22 * d_4 + gout23 * d_5 + gout54 * d_6 +
                       gout55 * d_7 + gout56 * d_8 + gout57 * d_9 +
                       gout58 * d_10 + gout59 * d_11 + gout90 * d_12 +
                       gout91 * d_13 + gout92 * d_14 + gout93 * d_15 +
                       gout94 * d_16 + gout95 * d_17,
                vk + (j0 + 0) + nao * (k0 + 3));
            reduce(gout24 * d_0 + gout25 * d_1 + gout26 * d_2 + gout27 * d_3 +
                       gout28 * d_4 + gout29 * d_5 + gout60 * d_6 +
                       gout61 * d_7 + gout62 * d_8 + gout63 * d_9 +
                       gout64 * d_10 + gout65 * d_11 + gout96 * d_12 +
                       gout97 * d_13 + gout98 * d_14 + gout99 * d_15 +
                       gout100 * d_16 + gout101 * d_17,
                vk + (j0 + 0) + nao * (k0 + 4));
            reduce(gout30 * d_0 + gout31 * d_1 + gout32 * d_2 + gout33 * d_3 +
                       gout34 * d_4 + gout35 * d_5 + gout66 * d_6 +
                       gout67 * d_7 + gout68 * d_8 + gout69 * d_9 +
                       gout70 * d_10 + gout71 * d_11 + gout102 * d_12 +
                       gout103 * d_13 + gout104 * d_14 + gout105 * d_15 +
                       gout106 * d_16 + gout107 * d_17,
                vk + (j0 + 0) + nao * (k0 + 5));
            // ijkl,ik->jl
            d_0 = dm[(i0 + 0) + nao * (k0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (k0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (k0 + 0)];
            d_3 = dm[(i0 + 3) + nao * (k0 + 0)];
            d_4 = dm[(i0 + 4) + nao * (k0 + 0)];
            d_5 = dm[(i0 + 5) + nao * (k0 + 0)];
            d_6 = dm[(i0 + 0) + nao * (k0 + 1)];
            d_7 = dm[(i0 + 1) + nao * (k0 + 1)];
            d_8 = dm[(i0 + 2) + nao * (k0 + 1)];
            d_9 = dm[(i0 + 3) + nao * (k0 + 1)];
            d_10 = dm[(i0 + 4) + nao * (k0 + 1)];
            d_11 = dm[(i0 + 5) + nao * (k0 + 1)];
            d_12 = dm[(i0 + 0) + nao * (k0 + 2)];
            d_13 = dm[(i0 + 1) + nao * (k0 + 2)];
            d_14 = dm[(i0 + 2) + nao * (k0 + 2)];
            d_15 = dm[(i0 + 3) + nao * (k0 + 2)];
            d_16 = dm[(i0 + 4) + nao * (k0 + 2)];
            d_17 = dm[(i0 + 5) + nao * (k0 + 2)];
            d_18 = dm[(i0 + 0) + nao * (k0 + 3)];
            d_19 = dm[(i0 + 1) + nao * (k0 + 3)];
            d_20 = dm[(i0 + 2) + nao * (k0 + 3)];
            d_21 = dm[(i0 + 3) + nao * (k0 + 3)];
            d_22 = dm[(i0 + 4) + nao * (k0 + 3)];
            d_23 = dm[(i0 + 5) + nao * (k0 + 3)];
            d_24 = dm[(i0 + 0) + nao * (k0 + 4)];
            d_25 = dm[(i0 + 1) + nao * (k0 + 4)];
            d_26 = dm[(i0 + 2) + nao * (k0 + 4)];
            d_27 = dm[(i0 + 3) + nao * (k0 + 4)];
            d_28 = dm[(i0 + 4) + nao * (k0 + 4)];
            d_29 = dm[(i0 + 5) + nao * (k0 + 4)];
            d_30 = dm[(i0 + 0) + nao * (k0 + 5)];
            d_31 = dm[(i0 + 1) + nao * (k0 + 5)];
            d_32 = dm[(i0 + 2) + nao * (k0 + 5)];
            d_33 = dm[(i0 + 3) + nao * (k0 + 5)];
            d_34 = dm[(i0 + 4) + nao * (k0 + 5)];
            d_35 = dm[(i0 + 5) + nao * (k0 + 5)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout3 * d_3 +
                       gout4 * d_4 + gout5 * d_5 + gout6 * d_6 + gout7 * d_7 +
                       gout8 * d_8 + gout9 * d_9 + gout10 * d_10 +
                       gout11 * d_11 + gout12 * d_12 + gout13 * d_13 +
                       gout14 * d_14 + gout15 * d_15 + gout16 * d_16 +
                       gout17 * d_17 + gout18 * d_18 + gout19 * d_19 +
                       gout20 * d_20 + gout21 * d_21 + gout22 * d_22 +
                       gout23 * d_23 + gout24 * d_24 + gout25 * d_25 +
                       gout26 * d_26 + gout27 * d_27 + gout28 * d_28 +
                       gout29 * d_29 + gout30 * d_30 + gout31 * d_31 +
                       gout32 * d_32 + gout33 * d_33 + gout34 * d_34 +
                       gout35 * d_35,
                vk + (j0 + 0) + nao * (l0 + 0));
            reduce(gout36 * d_0 + gout37 * d_1 + gout38 * d_2 + gout39 * d_3 +
                       gout40 * d_4 + gout41 * d_5 + gout42 * d_6 +
                       gout43 * d_7 + gout44 * d_8 + gout45 * d_9 +
                       gout46 * d_10 + gout47 * d_11 + gout48 * d_12 +
                       gout49 * d_13 + gout50 * d_14 + gout51 * d_15 +
                       gout52 * d_16 + gout53 * d_17 + gout54 * d_18 +
                       gout55 * d_19 + gout56 * d_20 + gout57 * d_21 +
                       gout58 * d_22 + gout59 * d_23 + gout60 * d_24 +
                       gout61 * d_25 + gout62 * d_26 + gout63 * d_27 +
                       gout64 * d_28 + gout65 * d_29 + gout66 * d_30 +
                       gout67 * d_31 + gout68 * d_32 + gout69 * d_33 +
                       gout70 * d_34 + gout71 * d_35,
                vk + (j0 + 0) + nao * (l0 + 1));
            reduce(gout72 * d_0 + gout73 * d_1 + gout74 * d_2 + gout75 * d_3 +
                       gout76 * d_4 + gout77 * d_5 + gout78 * d_6 +
                       gout79 * d_7 + gout80 * d_8 + gout81 * d_9 +
                       gout82 * d_10 + gout83 * d_11 + gout84 * d_12 +
                       gout85 * d_13 + gout86 * d_14 + gout87 * d_15 +
                       gout88 * d_16 + gout89 * d_17 + gout90 * d_18 +
                       gout91 * d_19 + gout92 * d_20 + gout93 * d_21 +
                       gout94 * d_22 + gout95 * d_23 + gout96 * d_24 +
                       gout97 * d_25 + gout98 * d_26 + gout99 * d_27 +
                       gout100 * d_28 + gout101 * d_29 + gout102 * d_30 +
                       gout103 * d_31 + gout104 * d_32 + gout105 * d_33 +
                       gout106 * d_34 + gout107 * d_35,
                vk + (j0 + 0) + nao * (l0 + 2));
            vk += nao2;
        }
        dm += nao2;
    }
}

__global__ static void GINTint2e_jk_kernel2110(JKMatrix jk,
    BasisProdOffsets offsets, GINTEnvVars envs, BasisProdCache bpcache) {
    int ntasks_ij = offsets.ntasks_ij;
    long ntasks = ntasks_ij * offsets.ntasks_kl;
    long task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int igroup = nprim_ij * nprim_kl;
    ntasks *= igroup;
    if (task_ij >= ntasks)
        return;
    int kl = task_ij % nprim_kl;
    task_ij /= nprim_kl;
    int ij = task_ij % nprim_ij;
    task_ij /= nprim_ij;
    int task_kl = task_ij / ntasks_ij;
    task_ij = task_ij % ntasks_ij;

    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    if (bas_ij < bas_kl) {
        return;
    }
    double norm = envs.fac;
    if (bas_ij == bas_kl) {
        norm *= .5;
    }

    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *ao_loc = bpcache.ao_loc;
    int *bas_pair2bra = bpcache.bas_pair2bra;
    int *bas_pair2ket = bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    int k0 = ao_loc[ksh];
    int l0 = ao_loc[lsh];
    double *__restrict__ a12 = bpcache.a12;
    double *__restrict__ e12 = bpcache.e12;
    double *__restrict__ x12 = bpcache.x12;
    double *__restrict__ y12 = bpcache.y12;
    double *__restrict__ z12 = bpcache.z12;
    int i_dm;
    int nbas = bpcache.nbas;
    double *__restrict__ bas_x = bpcache.bas_coords;
    double *__restrict__ bas_y = bas_x + nbas;
    double *__restrict__ bas_z = bas_y + nbas;

    double gout0 = 0;
    double gout1 = 0;
    double gout2 = 0;
    double gout3 = 0;
    double gout4 = 0;
    double gout5 = 0;
    double gout6 = 0;
    double gout7 = 0;
    double gout8 = 0;
    double gout9 = 0;
    double gout10 = 0;
    double gout11 = 0;
    double gout12 = 0;
    double gout13 = 0;
    double gout14 = 0;
    double gout15 = 0;
    double gout16 = 0;
    double gout17 = 0;
    double gout18 = 0;
    double gout19 = 0;
    double gout20 = 0;
    double gout21 = 0;
    double gout22 = 0;
    double gout23 = 0;
    double gout24 = 0;
    double gout25 = 0;
    double gout26 = 0;
    double gout27 = 0;
    double gout28 = 0;
    double gout29 = 0;
    double gout30 = 0;
    double gout31 = 0;
    double gout32 = 0;
    double gout33 = 0;
    double gout34 = 0;
    double gout35 = 0;
    double gout36 = 0;
    double gout37 = 0;
    double gout38 = 0;
    double gout39 = 0;
    double gout40 = 0;
    double gout41 = 0;
    double gout42 = 0;
    double gout43 = 0;
    double gout44 = 0;
    double gout45 = 0;
    double gout46 = 0;
    double gout47 = 0;
    double gout48 = 0;
    double gout49 = 0;
    double gout50 = 0;
    double gout51 = 0;
    double gout52 = 0;
    double gout53 = 0;
    double xi = bas_x[ish];
    double yi = bas_y[ish];
    double zi = bas_z[ish];
    double xixj = xi - bas_x[jsh];
    double yiyj = yi - bas_y[jsh];
    double zizj = zi - bas_z[jsh];
    double xk = bas_x[ksh];
    double yk = bas_y[ksh];
    double zk = bas_z[ksh];
    auto reduce = SegReduce<double>(igroup);
    ij += prim_ij;
    kl += prim_kl;
    double aij = a12[ij];
    double eij = e12[ij];
    double xij = x12[ij];
    double yij = y12[ij];
    double zij = z12[ij];
    double akl = a12[kl];
    double ekl = e12[kl];
    double xkl = x12[kl];
    double ykl = y12[kl];
    double zkl = z12[kl];
    double xijxkl = xij - xkl;
    double yijykl = yij - ykl;
    double zijzkl = zij - zkl;
    double aijkl = aij + akl;
    double a1 = aij * akl;
    double a0 = a1 / aijkl;
    double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
    double fac = norm * eij * ekl / (sqrt(aijkl) * a1);

    double rw[6];
    double root0, weight0;
    GINTrys_root<3>(x, rw);
    int irys;
    for (irys = 0; irys < 3; ++irys) {
        root0 = rw[irys];
        weight0 = rw[irys + 3];
        double u2 = a0 * root0;
        double tmp4 = .5 / (u2 * aijkl + a1);
        double b00 = u2 * tmp4;
        double tmp1 = 2 * b00;
        double tmp2 = tmp1 * akl;
        double b10 = b00 + tmp4 * akl;
        double c00x = xij - xi - tmp2 * xijxkl;
        double c00y = yij - yi - tmp2 * yijykl;
        double c00z = zij - zi - tmp2 * zijzkl;
        double tmp3 = tmp1 * aij;
        double c0px = xkl - xk + tmp3 * xijxkl;
        double c0py = ykl - yk + tmp3 * yijykl;
        double c0pz = zkl - zk + tmp3 * zijzkl;
        double g_0 = 1;
        double g_1 = c00x;
        double g_2 = c00x * c00x + b10;
        double g_3 = c00x + xixj;
        double g_4 = c00x * (c00x + xixj) + b10;
        double g_5 = c00x * (2 * b10 + g_2) + xixj * g_2;
        double g_6 = c0px;
        double g_7 = c0px * c00x + b00;
        double g_8 = b00 * c00x + b10 * c0px + c00x * g_7;
        double g_9 = c0px * (c00x + xixj) + b00;
        double g_10 = b00 * c00x + b10 * c0px + c00x * g_7 + xixj * g_7;
        double g_11 = 2 * b10 * g_7 + b00 * g_2 + c00x * g_8 + xixj * g_8;
        double g_12 = 1;
        double g_13 = c00y;
        double g_14 = c00y * c00y + b10;
        double g_15 = c00y + yiyj;
        double g_16 = c00y * (c00y + yiyj) + b10;
        double g_17 = c00y * (2 * b10 + g_14) + yiyj * g_14;
        double g_18 = c0py;
        double g_19 = c0py * c00y + b00;
        double g_20 = b00 * c00y + b10 * c0py + c00y * g_19;
        double g_21 = c0py * (c00y + yiyj) + b00;
        double g_22 = b00 * c00y + b10 * c0py + c00y * g_19 + yiyj * g_19;
        double g_23 = 2 * b10 * g_19 + b00 * g_14 + c00y * g_20 + yiyj * g_20;
        double g_24 = weight0 * fac;
        double g_25 = c00z * g_24;
        double g_26 = b10 * g_24 + c00z * g_25;
        double g_27 = g_24 * (c00z + zizj);
        double g_28 = b10 * g_24 + c00z * g_25 + zizj * g_25;
        double g_29 = 2 * b10 * g_25 + c00z * g_26 + zizj * g_26;
        double g_30 = c0pz * g_24;
        double g_31 = b00 * g_24 + c0pz * g_25;
        double g_32 = b00 * g_25 + b10 * g_30 + c00z * g_31;
        double g_33 = b00 * g_24 + c0pz * g_25 + zizj * g_30;
        double g_34 = b00 * g_25 + b10 * g_30 + c00z * g_31 + zizj * g_31;
        double g_35 = 2 * b10 * g_31 + b00 * g_26 + c00z * g_32 + zizj * g_32;
        gout0 += g_11 * g_12 * g_24;
        gout1 += g_10 * g_13 * g_24;
        gout2 += g_10 * g_12 * g_25;
        gout3 += g_9 * g_14 * g_24;
        gout4 += g_9 * g_13 * g_25;
        gout5 += g_9 * g_12 * g_26;
        gout6 += g_8 * g_15 * g_24;
        gout7 += g_7 * g_16 * g_24;
        gout8 += g_7 * g_15 * g_25;
        gout9 += g_6 * g_17 * g_24;
        gout10 += g_6 * g_16 * g_25;
        gout11 += g_6 * g_15 * g_26;
        gout12 += g_8 * g_12 * g_27;
        gout13 += g_7 * g_13 * g_27;
        gout14 += g_7 * g_12 * g_28;
        gout15 += g_6 * g_14 * g_27;
        gout16 += g_6 * g_13 * g_28;
        gout17 += g_6 * g_12 * g_29;
        gout18 += g_5 * g_18 * g_24;
        gout19 += g_4 * g_19 * g_24;
        gout20 += g_4 * g_18 * g_25;
        gout21 += g_3 * g_20 * g_24;
        gout22 += g_3 * g_19 * g_25;
        gout23 += g_3 * g_18 * g_26;
        gout24 += g_2 * g_21 * g_24;
        gout25 += g_1 * g_22 * g_24;
        gout26 += g_1 * g_21 * g_25;
        gout27 += g_0 * g_23 * g_24;
        gout28 += g_0 * g_22 * g_25;
        gout29 += g_0 * g_21 * g_26;
        gout30 += g_2 * g_18 * g_27;
        gout31 += g_1 * g_19 * g_27;
        gout32 += g_1 * g_18 * g_28;
        gout33 += g_0 * g_20 * g_27;
        gout34 += g_0 * g_19 * g_28;
        gout35 += g_0 * g_18 * g_29;
        gout36 += g_5 * g_12 * g_30;
        gout37 += g_4 * g_13 * g_30;
        gout38 += g_4 * g_12 * g_31;
        gout39 += g_3 * g_14 * g_30;
        gout40 += g_3 * g_13 * g_31;
        gout41 += g_3 * g_12 * g_32;
        gout42 += g_2 * g_15 * g_30;
        gout43 += g_1 * g_16 * g_30;
        gout44 += g_1 * g_15 * g_31;
        gout45 += g_0 * g_17 * g_30;
        gout46 += g_0 * g_16 * g_31;
        gout47 += g_0 * g_15 * g_32;
        gout48 += g_2 * g_12 * g_33;
        gout49 += g_1 * g_13 * g_33;
        gout50 += g_1 * g_12 * g_34;
        gout51 += g_0 * g_14 * g_33;
        gout52 += g_0 * g_13 * g_34;
        gout53 += g_0 * g_12 * g_35;
    }
    double d_0, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, d_9;
    double d_10, d_11, d_12, d_13, d_14, d_15, d_16, d_17;
    int n_dm = jk.n_dm;
    int nao = jk.nao;
    size_t nao2 = nao * nao;
    double *__restrict__ dm = jk.dm;
    double *vj = jk.vj;
    double *vk = jk.vk;
    for (i_dm = 0; i_dm < n_dm; ++i_dm) {
        if (vj != NULL) {
            // ijkl,ij->kl
            d_0 = dm[(i0 + 0) + nao * (j0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (j0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (j0 + 0)];
            d_3 = dm[(i0 + 3) + nao * (j0 + 0)];
            d_4 = dm[(i0 + 4) + nao * (j0 + 0)];
            d_5 = dm[(i0 + 5) + nao * (j0 + 0)];
            d_6 = dm[(i0 + 0) + nao * (j0 + 1)];
            d_7 = dm[(i0 + 1) + nao * (j0 + 1)];
            d_8 = dm[(i0 + 2) + nao * (j0 + 1)];
            d_9 = dm[(i0 + 3) + nao * (j0 + 1)];
            d_10 = dm[(i0 + 4) + nao * (j0 + 1)];
            d_11 = dm[(i0 + 5) + nao * (j0 + 1)];
            d_12 = dm[(i0 + 0) + nao * (j0 + 2)];
            d_13 = dm[(i0 + 1) + nao * (j0 + 2)];
            d_14 = dm[(i0 + 2) + nao * (j0 + 2)];
            d_15 = dm[(i0 + 3) + nao * (j0 + 2)];
            d_16 = dm[(i0 + 4) + nao * (j0 + 2)];
            d_17 = dm[(i0 + 5) + nao * (j0 + 2)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout3 * d_3 +
                       gout4 * d_4 + gout5 * d_5 + gout6 * d_6 + gout7 * d_7 +
                       gout8 * d_8 + gout9 * d_9 + gout10 * d_10 +
                       gout11 * d_11 + gout12 * d_12 + gout13 * d_13 +
                       gout14 * d_14 + gout15 * d_15 + gout16 * d_16 +
                       gout17 * d_17,
                vj + (k0 + 0) + nao * (l0 + 0));
            reduce(gout18 * d_0 + gout19 * d_1 + gout20 * d_2 + gout21 * d_3 +
                       gout22 * d_4 + gout23 * d_5 + gout24 * d_6 +
                       gout25 * d_7 + gout26 * d_8 + gout27 * d_9 +
                       gout28 * d_10 + gout29 * d_11 + gout30 * d_12 +
                       gout31 * d_13 + gout32 * d_14 + gout33 * d_15 +
                       gout34 * d_16 + gout35 * d_17,
                vj + (k0 + 1) + nao * (l0 + 0));
            reduce(gout36 * d_0 + gout37 * d_1 + gout38 * d_2 + gout39 * d_3 +
                       gout40 * d_4 + gout41 * d_5 + gout42 * d_6 +
                       gout43 * d_7 + gout44 * d_8 + gout45 * d_9 +
                       gout46 * d_10 + gout47 * d_11 + gout48 * d_12 +
                       gout49 * d_13 + gout50 * d_14 + gout51 * d_15 +
                       gout52 * d_16 + gout53 * d_17,
                vj + (k0 + 2) + nao * (l0 + 0));
            // ijkl,kl->ij
            d_0 = dm[(k0 + 0) + nao * (l0 + 0)];
            d_1 = dm[(k0 + 1) + nao * (l0 + 0)];
            d_2 = dm[(k0 + 2) + nao * (l0 + 0)];
            reduce(gout0 * d_0 + gout18 * d_1 + gout36 * d_2,
                vj + (i0 + 0) + nao * (j0 + 0));
            reduce(gout1 * d_0 + gout19 * d_1 + gout37 * d_2,
                vj + (i0 + 1) + nao * (j0 + 0));
            reduce(gout2 * d_0 + gout20 * d_1 + gout38 * d_2,
                vj + (i0 + 2) + nao * (j0 + 0));
            reduce(gout3 * d_0 + gout21 * d_1 + gout39 * d_2,
                vj + (i0 + 3) + nao * (j0 + 0));
            reduce(gout4 * d_0 + gout22 * d_1 + gout40 * d_2,
                vj + (i0 + 4) + nao * (j0 + 0));
            reduce(gout5 * d_0 + gout23 * d_1 + gout41 * d_2,
                vj + (i0 + 5) + nao * (j0 + 0));
            reduce(gout6 * d_0 + gout24 * d_1 + gout42 * d_2,
                vj + (i0 + 0) + nao * (j0 + 1));
            reduce(gout7 * d_0 + gout25 * d_1 + gout43 * d_2,
                vj + (i0 + 1) + nao * (j0 + 1));
            reduce(gout8 * d_0 + gout26 * d_1 + gout44 * d_2,
                vj + (i0 + 2) + nao * (j0 + 1));
            reduce(gout9 * d_0 + gout27 * d_1 + gout45 * d_2,
                vj + (i0 + 3) + nao * (j0 + 1));
            reduce(gout10 * d_0 + gout28 * d_1 + gout46 * d_2,
                vj + (i0 + 4) + nao * (j0 + 1));
            reduce(gout11 * d_0 + gout29 * d_1 + gout47 * d_2,
                vj + (i0 + 5) + nao * (j0 + 1));
            reduce(gout12 * d_0 + gout30 * d_1 + gout48 * d_2,
                vj + (i0 + 0) + nao * (j0 + 2));
            reduce(gout13 * d_0 + gout31 * d_1 + gout49 * d_2,
                vj + (i0 + 1) + nao * (j0 + 2));
            reduce(gout14 * d_0 + gout32 * d_1 + gout50 * d_2,
                vj + (i0 + 2) + nao * (j0 + 2));
            reduce(gout15 * d_0 + gout33 * d_1 + gout51 * d_2,
                vj + (i0 + 3) + nao * (j0 + 2));
            reduce(gout16 * d_0 + gout34 * d_1 + gout52 * d_2,
                vj + (i0 + 4) + nao * (j0 + 2));
            reduce(gout17 * d_0 + gout35 * d_1 + gout53 * d_2,
                vj + (i0 + 5) + nao * (j0 + 2));
            vj += nao2;
        }
        if (vk != NULL) {
            // ijkl,jl->ik
            d_0 = dm[(j0 + 0) + nao * (l0 + 0)];
            d_1 = dm[(j0 + 1) + nao * (l0 + 0)];
            d_2 = dm[(j0 + 2) + nao * (l0 + 0)];
            reduce(gout0 * d_0 + gout6 * d_1 + gout12 * d_2,
                vk + (i0 + 0) + nao * (k0 + 0));
            reduce(gout1 * d_0 + gout7 * d_1 + gout13 * d_2,
                vk + (i0 + 1) + nao * (k0 + 0));
            reduce(gout2 * d_0 + gout8 * d_1 + gout14 * d_2,
                vk + (i0 + 2) + nao * (k0 + 0));
            reduce(gout3 * d_0 + gout9 * d_1 + gout15 * d_2,
                vk + (i0 + 3) + nao * (k0 + 0));
            reduce(gout4 * d_0 + gout10 * d_1 + gout16 * d_2,
                vk + (i0 + 4) + nao * (k0 + 0));
            reduce(gout5 * d_0 + gout11 * d_1 + gout17 * d_2,
                vk + (i0 + 5) + nao * (k0 + 0));
            reduce(gout18 * d_0 + gout24 * d_1 + gout30 * d_2,
                vk + (i0 + 0) + nao * (k0 + 1));
            reduce(gout19 * d_0 + gout25 * d_1 + gout31 * d_2,
                vk + (i0 + 1) + nao * (k0 + 1));
            reduce(gout20 * d_0 + gout26 * d_1 + gout32 * d_2,
                vk + (i0 + 2) + nao * (k0 + 1));
            reduce(gout21 * d_0 + gout27 * d_1 + gout33 * d_2,
                vk + (i0 + 3) + nao * (k0 + 1));
            reduce(gout22 * d_0 + gout28 * d_1 + gout34 * d_2,
                vk + (i0 + 4) + nao * (k0 + 1));
            reduce(gout23 * d_0 + gout29 * d_1 + gout35 * d_2,
                vk + (i0 + 5) + nao * (k0 + 1));
            reduce(gout36 * d_0 + gout42 * d_1 + gout48 * d_2,
                vk + (i0 + 0) + nao * (k0 + 2));
            reduce(gout37 * d_0 + gout43 * d_1 + gout49 * d_2,
                vk + (i0 + 1) + nao * (k0 + 2));
            reduce(gout38 * d_0 + gout44 * d_1 + gout50 * d_2,
                vk + (i0 + 2) + nao * (k0 + 2));
            reduce(gout39 * d_0 + gout45 * d_1 + gout51 * d_2,
                vk + (i0 + 3) + nao * (k0 + 2));
            reduce(gout40 * d_0 + gout46 * d_1 + gout52 * d_2,
                vk + (i0 + 4) + nao * (k0 + 2));
            reduce(gout41 * d_0 + gout47 * d_1 + gout53 * d_2,
                vk + (i0 + 5) + nao * (k0 + 2));
            // ijkl,jk->il
            d_0 = dm[(j0 + 0) + nao * (k0 + 0)];
            d_1 = dm[(j0 + 1) + nao * (k0 + 0)];
            d_2 = dm[(j0 + 2) + nao * (k0 + 0)];
            d_3 = dm[(j0 + 0) + nao * (k0 + 1)];
            d_4 = dm[(j0 + 1) + nao * (k0 + 1)];
            d_5 = dm[(j0 + 2) + nao * (k0 + 1)];
            d_6 = dm[(j0 + 0) + nao * (k0 + 2)];
            d_7 = dm[(j0 + 1) + nao * (k0 + 2)];
            d_8 = dm[(j0 + 2) + nao * (k0 + 2)];
            reduce(gout0 * d_0 + gout6 * d_1 + gout12 * d_2 + gout18 * d_3 +
                       gout24 * d_4 + gout30 * d_5 + gout36 * d_6 +
                       gout42 * d_7 + gout48 * d_8,
                vk + (i0 + 0) + nao * (l0 + 0));
            reduce(gout1 * d_0 + gout7 * d_1 + gout13 * d_2 + gout19 * d_3 +
                       gout25 * d_4 + gout31 * d_5 + gout37 * d_6 +
                       gout43 * d_7 + gout49 * d_8,
                vk + (i0 + 1) + nao * (l0 + 0));
            reduce(gout2 * d_0 + gout8 * d_1 + gout14 * d_2 + gout20 * d_3 +
                       gout26 * d_4 + gout32 * d_5 + gout38 * d_6 +
                       gout44 * d_7 + gout50 * d_8,
                vk + (i0 + 2) + nao * (l0 + 0));
            reduce(gout3 * d_0 + gout9 * d_1 + gout15 * d_2 + gout21 * d_3 +
                       gout27 * d_4 + gout33 * d_5 + gout39 * d_6 +
                       gout45 * d_7 + gout51 * d_8,
                vk + (i0 + 3) + nao * (l0 + 0));
            reduce(gout4 * d_0 + gout10 * d_1 + gout16 * d_2 + gout22 * d_3 +
                       gout28 * d_4 + gout34 * d_5 + gout40 * d_6 +
                       gout46 * d_7 + gout52 * d_8,
                vk + (i0 + 4) + nao * (l0 + 0));
            reduce(gout5 * d_0 + gout11 * d_1 + gout17 * d_2 + gout23 * d_3 +
                       gout29 * d_4 + gout35 * d_5 + gout41 * d_6 +
                       gout47 * d_7 + gout53 * d_8,
                vk + (i0 + 5) + nao * (l0 + 0));
            // ijkl,il->jk
            d_0 = dm[(i0 + 0) + nao * (l0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (l0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (l0 + 0)];
            d_3 = dm[(i0 + 3) + nao * (l0 + 0)];
            d_4 = dm[(i0 + 4) + nao * (l0 + 0)];
            d_5 = dm[(i0 + 5) + nao * (l0 + 0)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout3 * d_3 +
                       gout4 * d_4 + gout5 * d_5,
                vk + (j0 + 0) + nao * (k0 + 0));
            reduce(gout6 * d_0 + gout7 * d_1 + gout8 * d_2 + gout9 * d_3 +
                       gout10 * d_4 + gout11 * d_5,
                vk + (j0 + 1) + nao * (k0 + 0));
            reduce(gout12 * d_0 + gout13 * d_1 + gout14 * d_2 + gout15 * d_3 +
                       gout16 * d_4 + gout17 * d_5,
                vk + (j0 + 2) + nao * (k0 + 0));
            reduce(gout18 * d_0 + gout19 * d_1 + gout20 * d_2 + gout21 * d_3 +
                       gout22 * d_4 + gout23 * d_5,
                vk + (j0 + 0) + nao * (k0 + 1));
            reduce(gout24 * d_0 + gout25 * d_1 + gout26 * d_2 + gout27 * d_3 +
                       gout28 * d_4 + gout29 * d_5,
                vk + (j0 + 1) + nao * (k0 + 1));
            reduce(gout30 * d_0 + gout31 * d_1 + gout32 * d_2 + gout33 * d_3 +
                       gout34 * d_4 + gout35 * d_5,
                vk + (j0 + 2) + nao * (k0 + 1));
            reduce(gout36 * d_0 + gout37 * d_1 + gout38 * d_2 + gout39 * d_3 +
                       gout40 * d_4 + gout41 * d_5,
                vk + (j0 + 0) + nao * (k0 + 2));
            reduce(gout42 * d_0 + gout43 * d_1 + gout44 * d_2 + gout45 * d_3 +
                       gout46 * d_4 + gout47 * d_5,
                vk + (j0 + 1) + nao * (k0 + 2));
            reduce(gout48 * d_0 + gout49 * d_1 + gout50 * d_2 + gout51 * d_3 +
                       gout52 * d_4 + gout53 * d_5,
                vk + (j0 + 2) + nao * (k0 + 2));
            // ijkl,ik->jl
            d_0 = dm[(i0 + 0) + nao * (k0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (k0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (k0 + 0)];
            d_3 = dm[(i0 + 3) + nao * (k0 + 0)];
            d_4 = dm[(i0 + 4) + nao * (k0 + 0)];
            d_5 = dm[(i0 + 5) + nao * (k0 + 0)];
            d_6 = dm[(i0 + 0) + nao * (k0 + 1)];
            d_7 = dm[(i0 + 1) + nao * (k0 + 1)];
            d_8 = dm[(i0 + 2) + nao * (k0 + 1)];
            d_9 = dm[(i0 + 3) + nao * (k0 + 1)];
            d_10 = dm[(i0 + 4) + nao * (k0 + 1)];
            d_11 = dm[(i0 + 5) + nao * (k0 + 1)];
            d_12 = dm[(i0 + 0) + nao * (k0 + 2)];
            d_13 = dm[(i0 + 1) + nao * (k0 + 2)];
            d_14 = dm[(i0 + 2) + nao * (k0 + 2)];
            d_15 = dm[(i0 + 3) + nao * (k0 + 2)];
            d_16 = dm[(i0 + 4) + nao * (k0 + 2)];
            d_17 = dm[(i0 + 5) + nao * (k0 + 2)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout3 * d_3 +
                       gout4 * d_4 + gout5 * d_5 + gout18 * d_6 +
                       gout19 * d_7 + gout20 * d_8 + gout21 * d_9 +
                       gout22 * d_10 + gout23 * d_11 + gout36 * d_12 +
                       gout37 * d_13 + gout38 * d_14 + gout39 * d_15 +
                       gout40 * d_16 + gout41 * d_17,
                vk + (j0 + 0) + nao * (l0 + 0));
            reduce(gout6 * d_0 + gout7 * d_1 + gout8 * d_2 + gout9 * d_3 +
                       gout10 * d_4 + gout11 * d_5 + gout24 * d_6 +
                       gout25 * d_7 + gout26 * d_8 + gout27 * d_9 +
                       gout28 * d_10 + gout29 * d_11 + gout42 * d_12 +
                       gout43 * d_13 + gout44 * d_14 + gout45 * d_15 +
                       gout46 * d_16 + gout47 * d_17,
                vk + (j0 + 1) + nao * (l0 + 0));
            reduce(gout12 * d_0 + gout13 * d_1 + gout14 * d_2 + gout15 * d_3 +
                       gout16 * d_4 + gout17 * d_5 + gout30 * d_6 +
                       gout31 * d_7 + gout32 * d_8 + gout33 * d_9 +
                       gout34 * d_10 + gout35 * d_11 + gout48 * d_12 +
                       gout49 * d_13 + gout50 * d_14 + gout51 * d_15 +
                       gout52 * d_16 + gout53 * d_17,
                vk + (j0 + 2) + nao * (l0 + 0));
            vk += nao2;
        }
        dm += nao2;
    }
}

__global__ static void GINTint2e_jk_kernel2111(JKMatrix jk,
    BasisProdOffsets offsets, GINTEnvVars envs, BasisProdCache bpcache) {
    int ntasks_ij = offsets.ntasks_ij;
    long ntasks = ntasks_ij * offsets.ntasks_kl;
    long task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int igroup = nprim_ij * nprim_kl;
    ntasks *= igroup;
    if (task_ij >= ntasks)
        return;
    int kl = task_ij % nprim_kl;
    task_ij /= nprim_kl;
    int ij = task_ij % nprim_ij;
    task_ij /= nprim_ij;
    int task_kl = task_ij / ntasks_ij;
    task_ij = task_ij % ntasks_ij;

    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    if (bas_ij < bas_kl) {
        return;
    }
    double norm = envs.fac;
    if (bas_ij == bas_kl) {
        norm *= .5;
    }

    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *ao_loc = bpcache.ao_loc;
    int *bas_pair2bra = bpcache.bas_pair2bra;
    int *bas_pair2ket = bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    int k0 = ao_loc[ksh];
    int l0 = ao_loc[lsh];
    double *__restrict__ a12 = bpcache.a12;
    double *__restrict__ e12 = bpcache.e12;
    double *__restrict__ x12 = bpcache.x12;
    double *__restrict__ y12 = bpcache.y12;
    double *__restrict__ z12 = bpcache.z12;
    int i_dm;
    int nbas = bpcache.nbas;
    double *__restrict__ bas_x = bpcache.bas_coords;
    double *__restrict__ bas_y = bas_x + nbas;
    double *__restrict__ bas_z = bas_y + nbas;

    double gout0 = 0;
    double gout1 = 0;
    double gout2 = 0;
    double gout3 = 0;
    double gout4 = 0;
    double gout5 = 0;
    double gout6 = 0;
    double gout7 = 0;
    double gout8 = 0;
    double gout9 = 0;
    double gout10 = 0;
    double gout11 = 0;
    double gout12 = 0;
    double gout13 = 0;
    double gout14 = 0;
    double gout15 = 0;
    double gout16 = 0;
    double gout17 = 0;
    double gout18 = 0;
    double gout19 = 0;
    double gout20 = 0;
    double gout21 = 0;
    double gout22 = 0;
    double gout23 = 0;
    double gout24 = 0;
    double gout25 = 0;
    double gout26 = 0;
    double gout27 = 0;
    double gout28 = 0;
    double gout29 = 0;
    double gout30 = 0;
    double gout31 = 0;
    double gout32 = 0;
    double gout33 = 0;
    double gout34 = 0;
    double gout35 = 0;
    double gout36 = 0;
    double gout37 = 0;
    double gout38 = 0;
    double gout39 = 0;
    double gout40 = 0;
    double gout41 = 0;
    double gout42 = 0;
    double gout43 = 0;
    double gout44 = 0;
    double gout45 = 0;
    double gout46 = 0;
    double gout47 = 0;
    double gout48 = 0;
    double gout49 = 0;
    double gout50 = 0;
    double gout51 = 0;
    double gout52 = 0;
    double gout53 = 0;
    double gout54 = 0;
    double gout55 = 0;
    double gout56 = 0;
    double gout57 = 0;
    double gout58 = 0;
    double gout59 = 0;
    double gout60 = 0;
    double gout61 = 0;
    double gout62 = 0;
    double gout63 = 0;
    double gout64 = 0;
    double gout65 = 0;
    double gout66 = 0;
    double gout67 = 0;
    double gout68 = 0;
    double gout69 = 0;
    double gout70 = 0;
    double gout71 = 0;
    double gout72 = 0;
    double gout73 = 0;
    double gout74 = 0;
    double gout75 = 0;
    double gout76 = 0;
    double gout77 = 0;
    double gout78 = 0;
    double gout79 = 0;
    double gout80 = 0;
    double gout81 = 0;
    double gout82 = 0;
    double gout83 = 0;
    double gout84 = 0;
    double gout85 = 0;
    double gout86 = 0;
    double gout87 = 0;
    double gout88 = 0;
    double gout89 = 0;
    double gout90 = 0;
    double gout91 = 0;
    double gout92 = 0;
    double gout93 = 0;
    double gout94 = 0;
    double gout95 = 0;
    double gout96 = 0;
    double gout97 = 0;
    double gout98 = 0;
    double gout99 = 0;
    double gout100 = 0;
    double gout101 = 0;
    double gout102 = 0;
    double gout103 = 0;
    double gout104 = 0;
    double gout105 = 0;
    double gout106 = 0;
    double gout107 = 0;
    double gout108 = 0;
    double gout109 = 0;
    double gout110 = 0;
    double gout111 = 0;
    double gout112 = 0;
    double gout113 = 0;
    double gout114 = 0;
    double gout115 = 0;
    double gout116 = 0;
    double gout117 = 0;
    double gout118 = 0;
    double gout119 = 0;
    double gout120 = 0;
    double gout121 = 0;
    double gout122 = 0;
    double gout123 = 0;
    double gout124 = 0;
    double gout125 = 0;
    double gout126 = 0;
    double gout127 = 0;
    double gout128 = 0;
    double gout129 = 0;
    double gout130 = 0;
    double gout131 = 0;
    double gout132 = 0;
    double gout133 = 0;
    double gout134 = 0;
    double gout135 = 0;
    double gout136 = 0;
    double gout137 = 0;
    double gout138 = 0;
    double gout139 = 0;
    double gout140 = 0;
    double gout141 = 0;
    double gout142 = 0;
    double gout143 = 0;
    double gout144 = 0;
    double gout145 = 0;
    double gout146 = 0;
    double gout147 = 0;
    double gout148 = 0;
    double gout149 = 0;
    double gout150 = 0;
    double gout151 = 0;
    double gout152 = 0;
    double gout153 = 0;
    double gout154 = 0;
    double gout155 = 0;
    double gout156 = 0;
    double gout157 = 0;
    double gout158 = 0;
    double gout159 = 0;
    double gout160 = 0;
    double gout161 = 0;
    double xi = bas_x[ish];
    double yi = bas_y[ish];
    double zi = bas_z[ish];
    double xixj = xi - bas_x[jsh];
    double yiyj = yi - bas_y[jsh];
    double zizj = zi - bas_z[jsh];
    double xk = bas_x[ksh];
    double yk = bas_y[ksh];
    double zk = bas_z[ksh];
    double xkxl = xk - bas_x[lsh];
    double ykyl = yk - bas_y[lsh];
    double zkzl = zk - bas_z[lsh];
    auto reduce = SegReduce<double>(igroup);
    ij += prim_ij;
    kl += prim_kl;
    double aij = a12[ij];
    double eij = e12[ij];
    double xij = x12[ij];
    double yij = y12[ij];
    double zij = z12[ij];
    double akl = a12[kl];
    double ekl = e12[kl];
    double xkl = x12[kl];
    double ykl = y12[kl];
    double zkl = z12[kl];
    double xijxkl = xij - xkl;
    double yijykl = yij - ykl;
    double zijzkl = zij - zkl;
    double aijkl = aij + akl;
    double a1 = aij * akl;
    double a0 = a1 / aijkl;
    double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
    double fac = norm * eij * ekl / (sqrt(aijkl) * a1);

    double rw[6];
    double root0, weight0;
    GINTrys_root<3>(x, rw);
    int irys;
    for (irys = 0; irys < 3; ++irys) {
        root0 = rw[irys];
        weight0 = rw[irys + 3];
        double u2 = a0 * root0;
        double tmp4 = .5 / (u2 * aijkl + a1);
        double b00 = u2 * tmp4;
        double tmp1 = 2 * b00;
        double tmp2 = tmp1 * akl;
        double b10 = b00 + tmp4 * akl;
        double c00x = xij - xi - tmp2 * xijxkl;
        double c00y = yij - yi - tmp2 * yijykl;
        double c00z = zij - zi - tmp2 * zijzkl;
        double tmp3 = tmp1 * aij;
        double b01 = b00 + tmp4 * aij;
        double c0px = xkl - xk + tmp3 * xijxkl;
        double c0py = ykl - yk + tmp3 * yijykl;
        double c0pz = zkl - zk + tmp3 * zijzkl;
        double g_0 = 1;
        double g_1 = c00x;
        double g_2 = c00x * c00x + b10;
        double g_3 = c00x + xixj;
        double g_4 = c00x * (c00x + xixj) + b10;
        double g_5 = c00x * (2 * b10 + g_2) + xixj * g_2;
        double g_6 = c0px;
        double g_7 = c0px * c00x + b00;
        double g_8 = b00 * c00x + b10 * c0px + c00x * g_7;
        double g_9 = c0px * (c00x + xixj) + b00;
        double g_10 = b00 * c00x + b10 * c0px + c00x * g_7 + xixj * g_7;
        double g_11 = 2 * b10 * g_7 + b00 * g_2 + c00x * g_8 + xixj * g_8;
        double g_12 = c0px + xkxl;
        double g_13 = c00x * (c0px + xkxl) + b00;
        double g_14 = b00 * c00x + b10 * c0px + c00x * g_7 + xkxl * g_2;
        double g_15 = xkxl * (xixj + c00x) + xixj * c0px + c0px * c00x + b00;
        double g_16 = xkxl * (xixj * c00x + c00x * c00x + b10) + xixj * g_7 +
                      c00x * g_7 + b10 * c0px + b00 * c00x;
        double g_17 = xkxl * (xixj * g_2 + c00x * g_2 + 2 * b10 * c00x) +
                      xixj * g_8 + c00x * g_8 + 2 * b10 * g_7 + b00 * g_2;
        double g_18 = c0px * (c0px + xkxl) + b01;
        double g_19 = b00 * c0px + b01 * c00x + c0px * g_7 + xkxl * g_7;
        double g_20 = xkxl * g_8 +
                      c00x * (c0px * g_7 + b01 * c00x + b00 * c0px) +
                      b10 * (c0px * c0px + b01) + 2 * b00 * g_7;
        double g_21 = xkxl * (xixj * c0px + c0px * c00x + b00) +
                      xixj * (c0px * c0px + b01) + c0px * g_7 + b01 * c00x +
                      b00 * c0px;
        double g_22 =
            xkxl * (xixj * g_7 + c00x * g_7 + b10 * c0px + b00 * c00x) +
            xixj * (c0px * g_7 + b01 * c00x + b00 * c0px) +
            c00x * (c0px * g_7 + b01 * c00x + b00 * c0px) +
            b10 * (c0px * c0px + b01) + 2 * b00 * g_7;
        double g_23 =
            xkxl * (xixj * g_8 + c00x * g_8 + 2 * b10 * g_7 + b00 * g_2) +
            xixj * (c00x * (c0px * g_7 + b01 * c00x + b00 * c0px) +
                       b10 * (c0px * c0px + b01) + 2 * b00 * g_7) +
            c00x * (c00x * (c0px * g_7 + b01 * c00x + b00 * c0px) +
                       b10 * (c0px * c0px + b01) + 2 * b00 * g_7) +
            2 * b10 * (c0px * g_7 + b01 * c00x + b00 * c0px) + 2 * b00 * g_8;
        double g_24 = 1;
        double g_25 = c00y;
        double g_26 = c00y * c00y + b10;
        double g_27 = c00y + yiyj;
        double g_28 = c00y * (c00y + yiyj) + b10;
        double g_29 = c00y * (2 * b10 + g_26) + yiyj * g_26;
        double g_30 = c0py;
        double g_31 = c0py * c00y + b00;
        double g_32 = b00 * c00y + b10 * c0py + c00y * g_31;
        double g_33 = c0py * (c00y + yiyj) + b00;
        double g_34 = b00 * c00y + b10 * c0py + c00y * g_31 + yiyj * g_31;
        double g_35 = 2 * b10 * g_31 + b00 * g_26 + c00y * g_32 + yiyj * g_32;
        double g_36 = c0py + ykyl;
        double g_37 = c00y * (c0py + ykyl) + b00;
        double g_38 = b00 * c00y + b10 * c0py + c00y * g_31 + ykyl * g_26;
        double g_39 = ykyl * (yiyj + c00y) + yiyj * c0py + c0py * c00y + b00;
        double g_40 = ykyl * (yiyj * c00y + c00y * c00y + b10) + yiyj * g_31 +
                      c00y * g_31 + b10 * c0py + b00 * c00y;
        double g_41 = ykyl * (yiyj * g_26 + c00y * g_26 + 2 * b10 * c00y) +
                      yiyj * g_32 + c00y * g_32 + 2 * b10 * g_31 + b00 * g_26;
        double g_42 = c0py * (c0py + ykyl) + b01;
        double g_43 = b00 * c0py + b01 * c00y + c0py * g_31 + ykyl * g_31;
        double g_44 = ykyl * g_32 +
                      c00y * (c0py * g_31 + b01 * c00y + b00 * c0py) +
                      b10 * (c0py * c0py + b01) + 2 * b00 * g_31;
        double g_45 = ykyl * (yiyj * c0py + c0py * c00y + b00) +
                      yiyj * (c0py * c0py + b01) + c0py * g_31 + b01 * c00y +
                      b00 * c0py;
        double g_46 =
            ykyl * (yiyj * g_31 + c00y * g_31 + b10 * c0py + b00 * c00y) +
            yiyj * (c0py * g_31 + b01 * c00y + b00 * c0py) +
            c00y * (c0py * g_31 + b01 * c00y + b00 * c0py) +
            b10 * (c0py * c0py + b01) + 2 * b00 * g_31;
        double g_47 =
            ykyl * (yiyj * g_32 + c00y * g_32 + 2 * b10 * g_31 + b00 * g_26) +
            yiyj * (c00y * (c0py * g_31 + b01 * c00y + b00 * c0py) +
                       b10 * (c0py * c0py + b01) + 2 * b00 * g_31) +
            c00y * (c00y * (c0py * g_31 + b01 * c00y + b00 * c0py) +
                       b10 * (c0py * c0py + b01) + 2 * b00 * g_31) +
            2 * b10 * (c0py * g_31 + b01 * c00y + b00 * c0py) + 2 * b00 * g_32;
        double g_48 = weight0 * fac;
        double g_49 = c00z * g_48;
        double g_50 = b10 * g_48 + c00z * g_49;
        double g_51 = g_48 * (c00z + zizj);
        double g_52 = b10 * g_48 + c00z * g_49 + zizj * g_49;
        double g_53 = 2 * b10 * g_49 + c00z * g_50 + zizj * g_50;
        double g_54 = c0pz * g_48;
        double g_55 = b00 * g_48 + c0pz * g_49;
        double g_56 = b00 * g_49 + b10 * g_54 + c00z * g_55;
        double g_57 = b00 * g_48 + c0pz * g_49 + zizj * g_54;
        double g_58 = b00 * g_49 + b10 * g_54 + c00z * g_55 + zizj * g_55;
        double g_59 = 2 * b10 * g_55 + b00 * g_50 + c00z * g_56 + zizj * g_56;
        double g_60 = g_48 * (c0pz + zkzl);
        double g_61 = b00 * g_48 + c0pz * g_49 + zkzl * g_49;
        double g_62 = b00 * g_49 + b10 * g_54 + c00z * g_55 + zkzl * g_50;
        double g_63 = zkzl * (zizj * g_48 + c00z * g_48) + zizj * g_54 +
                      c0pz * g_49 + b00 * g_48;
        double g_64 = zkzl * (zizj * g_49 + c00z * g_49 + b10 * g_48) +
                      zizj * g_55 + c00z * g_55 + b10 * g_54 + b00 * g_49;
        double g_65 = zkzl * (zizj * g_50 + c00z * g_50 + 2 * b10 * g_49) +
                      zizj * g_56 + c00z * g_56 + 2 * b10 * g_55 + b00 * g_50;
        double g_66 = b01 * g_48 + c0pz * g_54 + zkzl * g_54;
        double g_67 = b00 * g_54 + b01 * g_49 + c0pz * g_55 + zkzl * g_55;
        double g_68 = zkzl * g_56 +
                      c00z * (c0pz * g_55 + b01 * g_49 + b00 * g_54) +
                      b10 * (c0pz * g_54 + b01 * g_48) + 2 * b00 * g_55;
        double g_69 = zkzl * (zizj * g_54 + c0pz * g_49 + b00 * g_48) +
                      zizj * (c0pz * g_54 + b01 * g_48) + c0pz * g_55 +
                      b01 * g_49 + b00 * g_54;
        double g_70 =
            zkzl * (zizj * g_55 + c00z * g_55 + b10 * g_54 + b00 * g_49) +
            zizj * (c0pz * g_55 + b01 * g_49 + b00 * g_54) +
            c00z * (c0pz * g_55 + b01 * g_49 + b00 * g_54) +
            b10 * (c0pz * g_54 + b01 * g_48) + 2 * b00 * g_55;
        double g_71 =
            zkzl * (zizj * g_56 + c00z * g_56 + 2 * b10 * g_55 + b00 * g_50) +
            zizj * (c00z * (c0pz * g_55 + b01 * g_49 + b00 * g_54) +
                       b10 * (c0pz * g_54 + b01 * g_48) + 2 * b00 * g_55) +
            c00z * (c00z * (c0pz * g_55 + b01 * g_49 + b00 * g_54) +
                       b10 * (c0pz * g_54 + b01 * g_48) + 2 * b00 * g_55) +
            2 * b10 * (c0pz * g_55 + b01 * g_49 + b00 * g_54) + 2 * b00 * g_56;
        gout0 += g_23 * g_24 * g_48;
        gout1 += g_22 * g_25 * g_48;
        gout2 += g_22 * g_24 * g_49;
        gout3 += g_21 * g_26 * g_48;
        gout4 += g_21 * g_25 * g_49;
        gout5 += g_21 * g_24 * g_50;
        gout6 += g_20 * g_27 * g_48;
        gout7 += g_19 * g_28 * g_48;
        gout8 += g_19 * g_27 * g_49;
        gout9 += g_18 * g_29 * g_48;
        gout10 += g_18 * g_28 * g_49;
        gout11 += g_18 * g_27 * g_50;
        gout12 += g_20 * g_24 * g_51;
        gout13 += g_19 * g_25 * g_51;
        gout14 += g_19 * g_24 * g_52;
        gout15 += g_18 * g_26 * g_51;
        gout16 += g_18 * g_25 * g_52;
        gout17 += g_18 * g_24 * g_53;
        gout18 += g_17 * g_30 * g_48;
        gout19 += g_16 * g_31 * g_48;
        gout20 += g_16 * g_30 * g_49;
        gout21 += g_15 * g_32 * g_48;
        gout22 += g_15 * g_31 * g_49;
        gout23 += g_15 * g_30 * g_50;
        gout24 += g_14 * g_33 * g_48;
        gout25 += g_13 * g_34 * g_48;
        gout26 += g_13 * g_33 * g_49;
        gout27 += g_12 * g_35 * g_48;
        gout28 += g_12 * g_34 * g_49;
        gout29 += g_12 * g_33 * g_50;
        gout30 += g_14 * g_30 * g_51;
        gout31 += g_13 * g_31 * g_51;
        gout32 += g_13 * g_30 * g_52;
        gout33 += g_12 * g_32 * g_51;
        gout34 += g_12 * g_31 * g_52;
        gout35 += g_12 * g_30 * g_53;
        gout36 += g_17 * g_24 * g_54;
        gout37 += g_16 * g_25 * g_54;
        gout38 += g_16 * g_24 * g_55;
        gout39 += g_15 * g_26 * g_54;
        gout40 += g_15 * g_25 * g_55;
        gout41 += g_15 * g_24 * g_56;
        gout42 += g_14 * g_27 * g_54;
        gout43 += g_13 * g_28 * g_54;
        gout44 += g_13 * g_27 * g_55;
        gout45 += g_12 * g_29 * g_54;
        gout46 += g_12 * g_28 * g_55;
        gout47 += g_12 * g_27 * g_56;
        gout48 += g_14 * g_24 * g_57;
        gout49 += g_13 * g_25 * g_57;
        gout50 += g_13 * g_24 * g_58;
        gout51 += g_12 * g_26 * g_57;
        gout52 += g_12 * g_25 * g_58;
        gout53 += g_12 * g_24 * g_59;
        gout54 += g_11 * g_36 * g_48;
        gout55 += g_10 * g_37 * g_48;
        gout56 += g_10 * g_36 * g_49;
        gout57 += g_9 * g_38 * g_48;
        gout58 += g_9 * g_37 * g_49;
        gout59 += g_9 * g_36 * g_50;
        gout60 += g_8 * g_39 * g_48;
        gout61 += g_7 * g_40 * g_48;
        gout62 += g_7 * g_39 * g_49;
        gout63 += g_6 * g_41 * g_48;
        gout64 += g_6 * g_40 * g_49;
        gout65 += g_6 * g_39 * g_50;
        gout66 += g_8 * g_36 * g_51;
        gout67 += g_7 * g_37 * g_51;
        gout68 += g_7 * g_36 * g_52;
        gout69 += g_6 * g_38 * g_51;
        gout70 += g_6 * g_37 * g_52;
        gout71 += g_6 * g_36 * g_53;
        gout72 += g_5 * g_42 * g_48;
        gout73 += g_4 * g_43 * g_48;
        gout74 += g_4 * g_42 * g_49;
        gout75 += g_3 * g_44 * g_48;
        gout76 += g_3 * g_43 * g_49;
        gout77 += g_3 * g_42 * g_50;
        gout78 += g_2 * g_45 * g_48;
        gout79 += g_1 * g_46 * g_48;
        gout80 += g_1 * g_45 * g_49;
        gout81 += g_0 * g_47 * g_48;
        gout82 += g_0 * g_46 * g_49;
        gout83 += g_0 * g_45 * g_50;
        gout84 += g_2 * g_42 * g_51;
        gout85 += g_1 * g_43 * g_51;
        gout86 += g_1 * g_42 * g_52;
        gout87 += g_0 * g_44 * g_51;
        gout88 += g_0 * g_43 * g_52;
        gout89 += g_0 * g_42 * g_53;
        gout90 += g_5 * g_36 * g_54;
        gout91 += g_4 * g_37 * g_54;
        gout92 += g_4 * g_36 * g_55;
        gout93 += g_3 * g_38 * g_54;
        gout94 += g_3 * g_37 * g_55;
        gout95 += g_3 * g_36 * g_56;
        gout96 += g_2 * g_39 * g_54;
        gout97 += g_1 * g_40 * g_54;
        gout98 += g_1 * g_39 * g_55;
        gout99 += g_0 * g_41 * g_54;
        gout100 += g_0 * g_40 * g_55;
        gout101 += g_0 * g_39 * g_56;
        gout102 += g_2 * g_36 * g_57;
        gout103 += g_1 * g_37 * g_57;
        gout104 += g_1 * g_36 * g_58;
        gout105 += g_0 * g_38 * g_57;
        gout106 += g_0 * g_37 * g_58;
        gout107 += g_0 * g_36 * g_59;
        gout108 += g_11 * g_24 * g_60;
        gout109 += g_10 * g_25 * g_60;
        gout110 += g_10 * g_24 * g_61;
        gout111 += g_9 * g_26 * g_60;
        gout112 += g_9 * g_25 * g_61;
        gout113 += g_9 * g_24 * g_62;
        gout114 += g_8 * g_27 * g_60;
        gout115 += g_7 * g_28 * g_60;
        gout116 += g_7 * g_27 * g_61;
        gout117 += g_6 * g_29 * g_60;
        gout118 += g_6 * g_28 * g_61;
        gout119 += g_6 * g_27 * g_62;
        gout120 += g_8 * g_24 * g_63;
        gout121 += g_7 * g_25 * g_63;
        gout122 += g_7 * g_24 * g_64;
        gout123 += g_6 * g_26 * g_63;
        gout124 += g_6 * g_25 * g_64;
        gout125 += g_6 * g_24 * g_65;
        gout126 += g_5 * g_30 * g_60;
        gout127 += g_4 * g_31 * g_60;
        gout128 += g_4 * g_30 * g_61;
        gout129 += g_3 * g_32 * g_60;
        gout130 += g_3 * g_31 * g_61;
        gout131 += g_3 * g_30 * g_62;
        gout132 += g_2 * g_33 * g_60;
        gout133 += g_1 * g_34 * g_60;
        gout134 += g_1 * g_33 * g_61;
        gout135 += g_0 * g_35 * g_60;
        gout136 += g_0 * g_34 * g_61;
        gout137 += g_0 * g_33 * g_62;
        gout138 += g_2 * g_30 * g_63;
        gout139 += g_1 * g_31 * g_63;
        gout140 += g_1 * g_30 * g_64;
        gout141 += g_0 * g_32 * g_63;
        gout142 += g_0 * g_31 * g_64;
        gout143 += g_0 * g_30 * g_65;
        gout144 += g_5 * g_24 * g_66;
        gout145 += g_4 * g_25 * g_66;
        gout146 += g_4 * g_24 * g_67;
        gout147 += g_3 * g_26 * g_66;
        gout148 += g_3 * g_25 * g_67;
        gout149 += g_3 * g_24 * g_68;
        gout150 += g_2 * g_27 * g_66;
        gout151 += g_1 * g_28 * g_66;
        gout152 += g_1 * g_27 * g_67;
        gout153 += g_0 * g_29 * g_66;
        gout154 += g_0 * g_28 * g_67;
        gout155 += g_0 * g_27 * g_68;
        gout156 += g_2 * g_24 * g_69;
        gout157 += g_1 * g_25 * g_69;
        gout158 += g_1 * g_24 * g_70;
        gout159 += g_0 * g_26 * g_69;
        gout160 += g_0 * g_25 * g_70;
        gout161 += g_0 * g_24 * g_71;
    }
    double d_0, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, d_9;
    double d_10, d_11, d_12, d_13, d_14, d_15, d_16, d_17;
    int n_dm = jk.n_dm;
    int nao = jk.nao;
    size_t nao2 = nao * nao;
    double *__restrict__ dm = jk.dm;
    double *vj = jk.vj;
    double *vk = jk.vk;
    for (i_dm = 0; i_dm < n_dm; ++i_dm) {
        if (vj != NULL) {
            // ijkl,ij->kl
            d_0 = dm[(i0 + 0) + nao * (j0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (j0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (j0 + 0)];
            d_3 = dm[(i0 + 3) + nao * (j0 + 0)];
            d_4 = dm[(i0 + 4) + nao * (j0 + 0)];
            d_5 = dm[(i0 + 5) + nao * (j0 + 0)];
            d_6 = dm[(i0 + 0) + nao * (j0 + 1)];
            d_7 = dm[(i0 + 1) + nao * (j0 + 1)];
            d_8 = dm[(i0 + 2) + nao * (j0 + 1)];
            d_9 = dm[(i0 + 3) + nao * (j0 + 1)];
            d_10 = dm[(i0 + 4) + nao * (j0 + 1)];
            d_11 = dm[(i0 + 5) + nao * (j0 + 1)];
            d_12 = dm[(i0 + 0) + nao * (j0 + 2)];
            d_13 = dm[(i0 + 1) + nao * (j0 + 2)];
            d_14 = dm[(i0 + 2) + nao * (j0 + 2)];
            d_15 = dm[(i0 + 3) + nao * (j0 + 2)];
            d_16 = dm[(i0 + 4) + nao * (j0 + 2)];
            d_17 = dm[(i0 + 5) + nao * (j0 + 2)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout3 * d_3 +
                       gout4 * d_4 + gout5 * d_5 + gout6 * d_6 + gout7 * d_7 +
                       gout8 * d_8 + gout9 * d_9 + gout10 * d_10 +
                       gout11 * d_11 + gout12 * d_12 + gout13 * d_13 +
                       gout14 * d_14 + gout15 * d_15 + gout16 * d_16 +
                       gout17 * d_17,
                vj + (k0 + 0) + nao * (l0 + 0));
            reduce(gout18 * d_0 + gout19 * d_1 + gout20 * d_2 + gout21 * d_3 +
                       gout22 * d_4 + gout23 * d_5 + gout24 * d_6 +
                       gout25 * d_7 + gout26 * d_8 + gout27 * d_9 +
                       gout28 * d_10 + gout29 * d_11 + gout30 * d_12 +
                       gout31 * d_13 + gout32 * d_14 + gout33 * d_15 +
                       gout34 * d_16 + gout35 * d_17,
                vj + (k0 + 1) + nao * (l0 + 0));
            reduce(gout36 * d_0 + gout37 * d_1 + gout38 * d_2 + gout39 * d_3 +
                       gout40 * d_4 + gout41 * d_5 + gout42 * d_6 +
                       gout43 * d_7 + gout44 * d_8 + gout45 * d_9 +
                       gout46 * d_10 + gout47 * d_11 + gout48 * d_12 +
                       gout49 * d_13 + gout50 * d_14 + gout51 * d_15 +
                       gout52 * d_16 + gout53 * d_17,
                vj + (k0 + 2) + nao * (l0 + 0));
            reduce(gout54 * d_0 + gout55 * d_1 + gout56 * d_2 + gout57 * d_3 +
                       gout58 * d_4 + gout59 * d_5 + gout60 * d_6 +
                       gout61 * d_7 + gout62 * d_8 + gout63 * d_9 +
                       gout64 * d_10 + gout65 * d_11 + gout66 * d_12 +
                       gout67 * d_13 + gout68 * d_14 + gout69 * d_15 +
                       gout70 * d_16 + gout71 * d_17,
                vj + (k0 + 0) + nao * (l0 + 1));
            reduce(gout72 * d_0 + gout73 * d_1 + gout74 * d_2 + gout75 * d_3 +
                       gout76 * d_4 + gout77 * d_5 + gout78 * d_6 +
                       gout79 * d_7 + gout80 * d_8 + gout81 * d_9 +
                       gout82 * d_10 + gout83 * d_11 + gout84 * d_12 +
                       gout85 * d_13 + gout86 * d_14 + gout87 * d_15 +
                       gout88 * d_16 + gout89 * d_17,
                vj + (k0 + 1) + nao * (l0 + 1));
            reduce(gout90 * d_0 + gout91 * d_1 + gout92 * d_2 + gout93 * d_3 +
                       gout94 * d_4 + gout95 * d_5 + gout96 * d_6 +
                       gout97 * d_7 + gout98 * d_8 + gout99 * d_9 +
                       gout100 * d_10 + gout101 * d_11 + gout102 * d_12 +
                       gout103 * d_13 + gout104 * d_14 + gout105 * d_15 +
                       gout106 * d_16 + gout107 * d_17,
                vj + (k0 + 2) + nao * (l0 + 1));
            reduce(gout108 * d_0 + gout109 * d_1 + gout110 * d_2 +
                       gout111 * d_3 + gout112 * d_4 + gout113 * d_5 +
                       gout114 * d_6 + gout115 * d_7 + gout116 * d_8 +
                       gout117 * d_9 + gout118 * d_10 + gout119 * d_11 +
                       gout120 * d_12 + gout121 * d_13 + gout122 * d_14 +
                       gout123 * d_15 + gout124 * d_16 + gout125 * d_17,
                vj + (k0 + 0) + nao * (l0 + 2));
            reduce(gout126 * d_0 + gout127 * d_1 + gout128 * d_2 +
                       gout129 * d_3 + gout130 * d_4 + gout131 * d_5 +
                       gout132 * d_6 + gout133 * d_7 + gout134 * d_8 +
                       gout135 * d_9 + gout136 * d_10 + gout137 * d_11 +
                       gout138 * d_12 + gout139 * d_13 + gout140 * d_14 +
                       gout141 * d_15 + gout142 * d_16 + gout143 * d_17,
                vj + (k0 + 1) + nao * (l0 + 2));
            reduce(gout144 * d_0 + gout145 * d_1 + gout146 * d_2 +
                       gout147 * d_3 + gout148 * d_4 + gout149 * d_5 +
                       gout150 * d_6 + gout151 * d_7 + gout152 * d_8 +
                       gout153 * d_9 + gout154 * d_10 + gout155 * d_11 +
                       gout156 * d_12 + gout157 * d_13 + gout158 * d_14 +
                       gout159 * d_15 + gout160 * d_16 + gout161 * d_17,
                vj + (k0 + 2) + nao * (l0 + 2));
            // ijkl,kl->ij
            d_0 = dm[(k0 + 0) + nao * (l0 + 0)];
            d_1 = dm[(k0 + 1) + nao * (l0 + 0)];
            d_2 = dm[(k0 + 2) + nao * (l0 + 0)];
            d_3 = dm[(k0 + 0) + nao * (l0 + 1)];
            d_4 = dm[(k0 + 1) + nao * (l0 + 1)];
            d_5 = dm[(k0 + 2) + nao * (l0 + 1)];
            d_6 = dm[(k0 + 0) + nao * (l0 + 2)];
            d_7 = dm[(k0 + 1) + nao * (l0 + 2)];
            d_8 = dm[(k0 + 2) + nao * (l0 + 2)];
            reduce(gout0 * d_0 + gout18 * d_1 + gout36 * d_2 + gout54 * d_3 +
                       gout72 * d_4 + gout90 * d_5 + gout108 * d_6 +
                       gout126 * d_7 + gout144 * d_8,
                vj + (i0 + 0) + nao * (j0 + 0));
            reduce(gout1 * d_0 + gout19 * d_1 + gout37 * d_2 + gout55 * d_3 +
                       gout73 * d_4 + gout91 * d_5 + gout109 * d_6 +
                       gout127 * d_7 + gout145 * d_8,
                vj + (i0 + 1) + nao * (j0 + 0));
            reduce(gout2 * d_0 + gout20 * d_1 + gout38 * d_2 + gout56 * d_3 +
                       gout74 * d_4 + gout92 * d_5 + gout110 * d_6 +
                       gout128 * d_7 + gout146 * d_8,
                vj + (i0 + 2) + nao * (j0 + 0));
            reduce(gout3 * d_0 + gout21 * d_1 + gout39 * d_2 + gout57 * d_3 +
                       gout75 * d_4 + gout93 * d_5 + gout111 * d_6 +
                       gout129 * d_7 + gout147 * d_8,
                vj + (i0 + 3) + nao * (j0 + 0));
            reduce(gout4 * d_0 + gout22 * d_1 + gout40 * d_2 + gout58 * d_3 +
                       gout76 * d_4 + gout94 * d_5 + gout112 * d_6 +
                       gout130 * d_7 + gout148 * d_8,
                vj + (i0 + 4) + nao * (j0 + 0));
            reduce(gout5 * d_0 + gout23 * d_1 + gout41 * d_2 + gout59 * d_3 +
                       gout77 * d_4 + gout95 * d_5 + gout113 * d_6 +
                       gout131 * d_7 + gout149 * d_8,
                vj + (i0 + 5) + nao * (j0 + 0));
            reduce(gout6 * d_0 + gout24 * d_1 + gout42 * d_2 + gout60 * d_3 +
                       gout78 * d_4 + gout96 * d_5 + gout114 * d_6 +
                       gout132 * d_7 + gout150 * d_8,
                vj + (i0 + 0) + nao * (j0 + 1));
            reduce(gout7 * d_0 + gout25 * d_1 + gout43 * d_2 + gout61 * d_3 +
                       gout79 * d_4 + gout97 * d_5 + gout115 * d_6 +
                       gout133 * d_7 + gout151 * d_8,
                vj + (i0 + 1) + nao * (j0 + 1));
            reduce(gout8 * d_0 + gout26 * d_1 + gout44 * d_2 + gout62 * d_3 +
                       gout80 * d_4 + gout98 * d_5 + gout116 * d_6 +
                       gout134 * d_7 + gout152 * d_8,
                vj + (i0 + 2) + nao * (j0 + 1));
            reduce(gout9 * d_0 + gout27 * d_1 + gout45 * d_2 + gout63 * d_3 +
                       gout81 * d_4 + gout99 * d_5 + gout117 * d_6 +
                       gout135 * d_7 + gout153 * d_8,
                vj + (i0 + 3) + nao * (j0 + 1));
            reduce(gout10 * d_0 + gout28 * d_1 + gout46 * d_2 + gout64 * d_3 +
                       gout82 * d_4 + gout100 * d_5 + gout118 * d_6 +
                       gout136 * d_7 + gout154 * d_8,
                vj + (i0 + 4) + nao * (j0 + 1));
            reduce(gout11 * d_0 + gout29 * d_1 + gout47 * d_2 + gout65 * d_3 +
                       gout83 * d_4 + gout101 * d_5 + gout119 * d_6 +
                       gout137 * d_7 + gout155 * d_8,
                vj + (i0 + 5) + nao * (j0 + 1));
            reduce(gout12 * d_0 + gout30 * d_1 + gout48 * d_2 + gout66 * d_3 +
                       gout84 * d_4 + gout102 * d_5 + gout120 * d_6 +
                       gout138 * d_7 + gout156 * d_8,
                vj + (i0 + 0) + nao * (j0 + 2));
            reduce(gout13 * d_0 + gout31 * d_1 + gout49 * d_2 + gout67 * d_3 +
                       gout85 * d_4 + gout103 * d_5 + gout121 * d_6 +
                       gout139 * d_7 + gout157 * d_8,
                vj + (i0 + 1) + nao * (j0 + 2));
            reduce(gout14 * d_0 + gout32 * d_1 + gout50 * d_2 + gout68 * d_3 +
                       gout86 * d_4 + gout104 * d_5 + gout122 * d_6 +
                       gout140 * d_7 + gout158 * d_8,
                vj + (i0 + 2) + nao * (j0 + 2));
            reduce(gout15 * d_0 + gout33 * d_1 + gout51 * d_2 + gout69 * d_3 +
                       gout87 * d_4 + gout105 * d_5 + gout123 * d_6 +
                       gout141 * d_7 + gout159 * d_8,
                vj + (i0 + 3) + nao * (j0 + 2));
            reduce(gout16 * d_0 + gout34 * d_1 + gout52 * d_2 + gout70 * d_3 +
                       gout88 * d_4 + gout106 * d_5 + gout124 * d_6 +
                       gout142 * d_7 + gout160 * d_8,
                vj + (i0 + 4) + nao * (j0 + 2));
            reduce(gout17 * d_0 + gout35 * d_1 + gout53 * d_2 + gout71 * d_3 +
                       gout89 * d_4 + gout107 * d_5 + gout125 * d_6 +
                       gout143 * d_7 + gout161 * d_8,
                vj + (i0 + 5) + nao * (j0 + 2));
            vj += nao2;
        }
        if (vk != NULL) {
            // ijkl,jl->ik
            d_0 = dm[(j0 + 0) + nao * (l0 + 0)];
            d_1 = dm[(j0 + 1) + nao * (l0 + 0)];
            d_2 = dm[(j0 + 2) + nao * (l0 + 0)];
            d_3 = dm[(j0 + 0) + nao * (l0 + 1)];
            d_4 = dm[(j0 + 1) + nao * (l0 + 1)];
            d_5 = dm[(j0 + 2) + nao * (l0 + 1)];
            d_6 = dm[(j0 + 0) + nao * (l0 + 2)];
            d_7 = dm[(j0 + 1) + nao * (l0 + 2)];
            d_8 = dm[(j0 + 2) + nao * (l0 + 2)];
            reduce(gout0 * d_0 + gout6 * d_1 + gout12 * d_2 + gout54 * d_3 +
                       gout60 * d_4 + gout66 * d_5 + gout108 * d_6 +
                       gout114 * d_7 + gout120 * d_8,
                vk + (i0 + 0) + nao * (k0 + 0));
            reduce(gout1 * d_0 + gout7 * d_1 + gout13 * d_2 + gout55 * d_3 +
                       gout61 * d_4 + gout67 * d_5 + gout109 * d_6 +
                       gout115 * d_7 + gout121 * d_8,
                vk + (i0 + 1) + nao * (k0 + 0));
            reduce(gout2 * d_0 + gout8 * d_1 + gout14 * d_2 + gout56 * d_3 +
                       gout62 * d_4 + gout68 * d_5 + gout110 * d_6 +
                       gout116 * d_7 + gout122 * d_8,
                vk + (i0 + 2) + nao * (k0 + 0));
            reduce(gout3 * d_0 + gout9 * d_1 + gout15 * d_2 + gout57 * d_3 +
                       gout63 * d_4 + gout69 * d_5 + gout111 * d_6 +
                       gout117 * d_7 + gout123 * d_8,
                vk + (i0 + 3) + nao * (k0 + 0));
            reduce(gout4 * d_0 + gout10 * d_1 + gout16 * d_2 + gout58 * d_3 +
                       gout64 * d_4 + gout70 * d_5 + gout112 * d_6 +
                       gout118 * d_7 + gout124 * d_8,
                vk + (i0 + 4) + nao * (k0 + 0));
            reduce(gout5 * d_0 + gout11 * d_1 + gout17 * d_2 + gout59 * d_3 +
                       gout65 * d_4 + gout71 * d_5 + gout113 * d_6 +
                       gout119 * d_7 + gout125 * d_8,
                vk + (i0 + 5) + nao * (k0 + 0));
            reduce(gout18 * d_0 + gout24 * d_1 + gout30 * d_2 + gout72 * d_3 +
                       gout78 * d_4 + gout84 * d_5 + gout126 * d_6 +
                       gout132 * d_7 + gout138 * d_8,
                vk + (i0 + 0) + nao * (k0 + 1));
            reduce(gout19 * d_0 + gout25 * d_1 + gout31 * d_2 + gout73 * d_3 +
                       gout79 * d_4 + gout85 * d_5 + gout127 * d_6 +
                       gout133 * d_7 + gout139 * d_8,
                vk + (i0 + 1) + nao * (k0 + 1));
            reduce(gout20 * d_0 + gout26 * d_1 + gout32 * d_2 + gout74 * d_3 +
                       gout80 * d_4 + gout86 * d_5 + gout128 * d_6 +
                       gout134 * d_7 + gout140 * d_8,
                vk + (i0 + 2) + nao * (k0 + 1));
            reduce(gout21 * d_0 + gout27 * d_1 + gout33 * d_2 + gout75 * d_3 +
                       gout81 * d_4 + gout87 * d_5 + gout129 * d_6 +
                       gout135 * d_7 + gout141 * d_8,
                vk + (i0 + 3) + nao * (k0 + 1));
            reduce(gout22 * d_0 + gout28 * d_1 + gout34 * d_2 + gout76 * d_3 +
                       gout82 * d_4 + gout88 * d_5 + gout130 * d_6 +
                       gout136 * d_7 + gout142 * d_8,
                vk + (i0 + 4) + nao * (k0 + 1));
            reduce(gout23 * d_0 + gout29 * d_1 + gout35 * d_2 + gout77 * d_3 +
                       gout83 * d_4 + gout89 * d_5 + gout131 * d_6 +
                       gout137 * d_7 + gout143 * d_8,
                vk + (i0 + 5) + nao * (k0 + 1));
            reduce(gout36 * d_0 + gout42 * d_1 + gout48 * d_2 + gout90 * d_3 +
                       gout96 * d_4 + gout102 * d_5 + gout144 * d_6 +
                       gout150 * d_7 + gout156 * d_8,
                vk + (i0 + 0) + nao * (k0 + 2));
            reduce(gout37 * d_0 + gout43 * d_1 + gout49 * d_2 + gout91 * d_3 +
                       gout97 * d_4 + gout103 * d_5 + gout145 * d_6 +
                       gout151 * d_7 + gout157 * d_8,
                vk + (i0 + 1) + nao * (k0 + 2));
            reduce(gout38 * d_0 + gout44 * d_1 + gout50 * d_2 + gout92 * d_3 +
                       gout98 * d_4 + gout104 * d_5 + gout146 * d_6 +
                       gout152 * d_7 + gout158 * d_8,
                vk + (i0 + 2) + nao * (k0 + 2));
            reduce(gout39 * d_0 + gout45 * d_1 + gout51 * d_2 + gout93 * d_3 +
                       gout99 * d_4 + gout105 * d_5 + gout147 * d_6 +
                       gout153 * d_7 + gout159 * d_8,
                vk + (i0 + 3) + nao * (k0 + 2));
            reduce(gout40 * d_0 + gout46 * d_1 + gout52 * d_2 + gout94 * d_3 +
                       gout100 * d_4 + gout106 * d_5 + gout148 * d_6 +
                       gout154 * d_7 + gout160 * d_8,
                vk + (i0 + 4) + nao * (k0 + 2));
            reduce(gout41 * d_0 + gout47 * d_1 + gout53 * d_2 + gout95 * d_3 +
                       gout101 * d_4 + gout107 * d_5 + gout149 * d_6 +
                       gout155 * d_7 + gout161 * d_8,
                vk + (i0 + 5) + nao * (k0 + 2));
            // ijkl,jk->il
            d_0 = dm[(j0 + 0) + nao * (k0 + 0)];
            d_1 = dm[(j0 + 1) + nao * (k0 + 0)];
            d_2 = dm[(j0 + 2) + nao * (k0 + 0)];
            d_3 = dm[(j0 + 0) + nao * (k0 + 1)];
            d_4 = dm[(j0 + 1) + nao * (k0 + 1)];
            d_5 = dm[(j0 + 2) + nao * (k0 + 1)];
            d_6 = dm[(j0 + 0) + nao * (k0 + 2)];
            d_7 = dm[(j0 + 1) + nao * (k0 + 2)];
            d_8 = dm[(j0 + 2) + nao * (k0 + 2)];
            reduce(gout0 * d_0 + gout6 * d_1 + gout12 * d_2 + gout18 * d_3 +
                       gout24 * d_4 + gout30 * d_5 + gout36 * d_6 +
                       gout42 * d_7 + gout48 * d_8,
                vk + (i0 + 0) + nao * (l0 + 0));
            reduce(gout1 * d_0 + gout7 * d_1 + gout13 * d_2 + gout19 * d_3 +
                       gout25 * d_4 + gout31 * d_5 + gout37 * d_6 +
                       gout43 * d_7 + gout49 * d_8,
                vk + (i0 + 1) + nao * (l0 + 0));
            reduce(gout2 * d_0 + gout8 * d_1 + gout14 * d_2 + gout20 * d_3 +
                       gout26 * d_4 + gout32 * d_5 + gout38 * d_6 +
                       gout44 * d_7 + gout50 * d_8,
                vk + (i0 + 2) + nao * (l0 + 0));
            reduce(gout3 * d_0 + gout9 * d_1 + gout15 * d_2 + gout21 * d_3 +
                       gout27 * d_4 + gout33 * d_5 + gout39 * d_6 +
                       gout45 * d_7 + gout51 * d_8,
                vk + (i0 + 3) + nao * (l0 + 0));
            reduce(gout4 * d_0 + gout10 * d_1 + gout16 * d_2 + gout22 * d_3 +
                       gout28 * d_4 + gout34 * d_5 + gout40 * d_6 +
                       gout46 * d_7 + gout52 * d_8,
                vk + (i0 + 4) + nao * (l0 + 0));
            reduce(gout5 * d_0 + gout11 * d_1 + gout17 * d_2 + gout23 * d_3 +
                       gout29 * d_4 + gout35 * d_5 + gout41 * d_6 +
                       gout47 * d_7 + gout53 * d_8,
                vk + (i0 + 5) + nao * (l0 + 0));
            reduce(gout54 * d_0 + gout60 * d_1 + gout66 * d_2 + gout72 * d_3 +
                       gout78 * d_4 + gout84 * d_5 + gout90 * d_6 +
                       gout96 * d_7 + gout102 * d_8,
                vk + (i0 + 0) + nao * (l0 + 1));
            reduce(gout55 * d_0 + gout61 * d_1 + gout67 * d_2 + gout73 * d_3 +
                       gout79 * d_4 + gout85 * d_5 + gout91 * d_6 +
                       gout97 * d_7 + gout103 * d_8,
                vk + (i0 + 1) + nao * (l0 + 1));
            reduce(gout56 * d_0 + gout62 * d_1 + gout68 * d_2 + gout74 * d_3 +
                       gout80 * d_4 + gout86 * d_5 + gout92 * d_6 +
                       gout98 * d_7 + gout104 * d_8,
                vk + (i0 + 2) + nao * (l0 + 1));
            reduce(gout57 * d_0 + gout63 * d_1 + gout69 * d_2 + gout75 * d_3 +
                       gout81 * d_4 + gout87 * d_5 + gout93 * d_6 +
                       gout99 * d_7 + gout105 * d_8,
                vk + (i0 + 3) + nao * (l0 + 1));
            reduce(gout58 * d_0 + gout64 * d_1 + gout70 * d_2 + gout76 * d_3 +
                       gout82 * d_4 + gout88 * d_5 + gout94 * d_6 +
                       gout100 * d_7 + gout106 * d_8,
                vk + (i0 + 4) + nao * (l0 + 1));
            reduce(gout59 * d_0 + gout65 * d_1 + gout71 * d_2 + gout77 * d_3 +
                       gout83 * d_4 + gout89 * d_5 + gout95 * d_6 +
                       gout101 * d_7 + gout107 * d_8,
                vk + (i0 + 5) + nao * (l0 + 1));
            reduce(gout108 * d_0 + gout114 * d_1 + gout120 * d_2 +
                       gout126 * d_3 + gout132 * d_4 + gout138 * d_5 +
                       gout144 * d_6 + gout150 * d_7 + gout156 * d_8,
                vk + (i0 + 0) + nao * (l0 + 2));
            reduce(gout109 * d_0 + gout115 * d_1 + gout121 * d_2 +
                       gout127 * d_3 + gout133 * d_4 + gout139 * d_5 +
                       gout145 * d_6 + gout151 * d_7 + gout157 * d_8,
                vk + (i0 + 1) + nao * (l0 + 2));
            reduce(gout110 * d_0 + gout116 * d_1 + gout122 * d_2 +
                       gout128 * d_3 + gout134 * d_4 + gout140 * d_5 +
                       gout146 * d_6 + gout152 * d_7 + gout158 * d_8,
                vk + (i0 + 2) + nao * (l0 + 2));
            reduce(gout111 * d_0 + gout117 * d_1 + gout123 * d_2 +
                       gout129 * d_3 + gout135 * d_4 + gout141 * d_5 +
                       gout147 * d_6 + gout153 * d_7 + gout159 * d_8,
                vk + (i0 + 3) + nao * (l0 + 2));
            reduce(gout112 * d_0 + gout118 * d_1 + gout124 * d_2 +
                       gout130 * d_3 + gout136 * d_4 + gout142 * d_5 +
                       gout148 * d_6 + gout154 * d_7 + gout160 * d_8,
                vk + (i0 + 4) + nao * (l0 + 2));
            reduce(gout113 * d_0 + gout119 * d_1 + gout125 * d_2 +
                       gout131 * d_3 + gout137 * d_4 + gout143 * d_5 +
                       gout149 * d_6 + gout155 * d_7 + gout161 * d_8,
                vk + (i0 + 5) + nao * (l0 + 2));
            // ijkl,il->jk
            d_0 = dm[(i0 + 0) + nao * (l0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (l0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (l0 + 0)];
            d_3 = dm[(i0 + 3) + nao * (l0 + 0)];
            d_4 = dm[(i0 + 4) + nao * (l0 + 0)];
            d_5 = dm[(i0 + 5) + nao * (l0 + 0)];
            d_6 = dm[(i0 + 0) + nao * (l0 + 1)];
            d_7 = dm[(i0 + 1) + nao * (l0 + 1)];
            d_8 = dm[(i0 + 2) + nao * (l0 + 1)];
            d_9 = dm[(i0 + 3) + nao * (l0 + 1)];
            d_10 = dm[(i0 + 4) + nao * (l0 + 1)];
            d_11 = dm[(i0 + 5) + nao * (l0 + 1)];
            d_12 = dm[(i0 + 0) + nao * (l0 + 2)];
            d_13 = dm[(i0 + 1) + nao * (l0 + 2)];
            d_14 = dm[(i0 + 2) + nao * (l0 + 2)];
            d_15 = dm[(i0 + 3) + nao * (l0 + 2)];
            d_16 = dm[(i0 + 4) + nao * (l0 + 2)];
            d_17 = dm[(i0 + 5) + nao * (l0 + 2)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout3 * d_3 +
                       gout4 * d_4 + gout5 * d_5 + gout54 * d_6 +
                       gout55 * d_7 + gout56 * d_8 + gout57 * d_9 +
                       gout58 * d_10 + gout59 * d_11 + gout108 * d_12 +
                       gout109 * d_13 + gout110 * d_14 + gout111 * d_15 +
                       gout112 * d_16 + gout113 * d_17,
                vk + (j0 + 0) + nao * (k0 + 0));
            reduce(gout6 * d_0 + gout7 * d_1 + gout8 * d_2 + gout9 * d_3 +
                       gout10 * d_4 + gout11 * d_5 + gout60 * d_6 +
                       gout61 * d_7 + gout62 * d_8 + gout63 * d_9 +
                       gout64 * d_10 + gout65 * d_11 + gout114 * d_12 +
                       gout115 * d_13 + gout116 * d_14 + gout117 * d_15 +
                       gout118 * d_16 + gout119 * d_17,
                vk + (j0 + 1) + nao * (k0 + 0));
            reduce(gout12 * d_0 + gout13 * d_1 + gout14 * d_2 + gout15 * d_3 +
                       gout16 * d_4 + gout17 * d_5 + gout66 * d_6 +
                       gout67 * d_7 + gout68 * d_8 + gout69 * d_9 +
                       gout70 * d_10 + gout71 * d_11 + gout120 * d_12 +
                       gout121 * d_13 + gout122 * d_14 + gout123 * d_15 +
                       gout124 * d_16 + gout125 * d_17,
                vk + (j0 + 2) + nao * (k0 + 0));
            reduce(gout18 * d_0 + gout19 * d_1 + gout20 * d_2 + gout21 * d_3 +
                       gout22 * d_4 + gout23 * d_5 + gout72 * d_6 +
                       gout73 * d_7 + gout74 * d_8 + gout75 * d_9 +
                       gout76 * d_10 + gout77 * d_11 + gout126 * d_12 +
                       gout127 * d_13 + gout128 * d_14 + gout129 * d_15 +
                       gout130 * d_16 + gout131 * d_17,
                vk + (j0 + 0) + nao * (k0 + 1));
            reduce(gout24 * d_0 + gout25 * d_1 + gout26 * d_2 + gout27 * d_3 +
                       gout28 * d_4 + gout29 * d_5 + gout78 * d_6 +
                       gout79 * d_7 + gout80 * d_8 + gout81 * d_9 +
                       gout82 * d_10 + gout83 * d_11 + gout132 * d_12 +
                       gout133 * d_13 + gout134 * d_14 + gout135 * d_15 +
                       gout136 * d_16 + gout137 * d_17,
                vk + (j0 + 1) + nao * (k0 + 1));
            reduce(gout30 * d_0 + gout31 * d_1 + gout32 * d_2 + gout33 * d_3 +
                       gout34 * d_4 + gout35 * d_5 + gout84 * d_6 +
                       gout85 * d_7 + gout86 * d_8 + gout87 * d_9 +
                       gout88 * d_10 + gout89 * d_11 + gout138 * d_12 +
                       gout139 * d_13 + gout140 * d_14 + gout141 * d_15 +
                       gout142 * d_16 + gout143 * d_17,
                vk + (j0 + 2) + nao * (k0 + 1));
            reduce(gout36 * d_0 + gout37 * d_1 + gout38 * d_2 + gout39 * d_3 +
                       gout40 * d_4 + gout41 * d_5 + gout90 * d_6 +
                       gout91 * d_7 + gout92 * d_8 + gout93 * d_9 +
                       gout94 * d_10 + gout95 * d_11 + gout144 * d_12 +
                       gout145 * d_13 + gout146 * d_14 + gout147 * d_15 +
                       gout148 * d_16 + gout149 * d_17,
                vk + (j0 + 0) + nao * (k0 + 2));
            reduce(gout42 * d_0 + gout43 * d_1 + gout44 * d_2 + gout45 * d_3 +
                       gout46 * d_4 + gout47 * d_5 + gout96 * d_6 +
                       gout97 * d_7 + gout98 * d_8 + gout99 * d_9 +
                       gout100 * d_10 + gout101 * d_11 + gout150 * d_12 +
                       gout151 * d_13 + gout152 * d_14 + gout153 * d_15 +
                       gout154 * d_16 + gout155 * d_17,
                vk + (j0 + 1) + nao * (k0 + 2));
            reduce(gout48 * d_0 + gout49 * d_1 + gout50 * d_2 + gout51 * d_3 +
                       gout52 * d_4 + gout53 * d_5 + gout102 * d_6 +
                       gout103 * d_7 + gout104 * d_8 + gout105 * d_9 +
                       gout106 * d_10 + gout107 * d_11 + gout156 * d_12 +
                       gout157 * d_13 + gout158 * d_14 + gout159 * d_15 +
                       gout160 * d_16 + gout161 * d_17,
                vk + (j0 + 2) + nao * (k0 + 2));
            // ijkl,ik->jl
            d_0 = dm[(i0 + 0) + nao * (k0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (k0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (k0 + 0)];
            d_3 = dm[(i0 + 3) + nao * (k0 + 0)];
            d_4 = dm[(i0 + 4) + nao * (k0 + 0)];
            d_5 = dm[(i0 + 5) + nao * (k0 + 0)];
            d_6 = dm[(i0 + 0) + nao * (k0 + 1)];
            d_7 = dm[(i0 + 1) + nao * (k0 + 1)];
            d_8 = dm[(i0 + 2) + nao * (k0 + 1)];
            d_9 = dm[(i0 + 3) + nao * (k0 + 1)];
            d_10 = dm[(i0 + 4) + nao * (k0 + 1)];
            d_11 = dm[(i0 + 5) + nao * (k0 + 1)];
            d_12 = dm[(i0 + 0) + nao * (k0 + 2)];
            d_13 = dm[(i0 + 1) + nao * (k0 + 2)];
            d_14 = dm[(i0 + 2) + nao * (k0 + 2)];
            d_15 = dm[(i0 + 3) + nao * (k0 + 2)];
            d_16 = dm[(i0 + 4) + nao * (k0 + 2)];
            d_17 = dm[(i0 + 5) + nao * (k0 + 2)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout3 * d_3 +
                       gout4 * d_4 + gout5 * d_5 + gout18 * d_6 +
                       gout19 * d_7 + gout20 * d_8 + gout21 * d_9 +
                       gout22 * d_10 + gout23 * d_11 + gout36 * d_12 +
                       gout37 * d_13 + gout38 * d_14 + gout39 * d_15 +
                       gout40 * d_16 + gout41 * d_17,
                vk + (j0 + 0) + nao * (l0 + 0));
            reduce(gout6 * d_0 + gout7 * d_1 + gout8 * d_2 + gout9 * d_3 +
                       gout10 * d_4 + gout11 * d_5 + gout24 * d_6 +
                       gout25 * d_7 + gout26 * d_8 + gout27 * d_9 +
                       gout28 * d_10 + gout29 * d_11 + gout42 * d_12 +
                       gout43 * d_13 + gout44 * d_14 + gout45 * d_15 +
                       gout46 * d_16 + gout47 * d_17,
                vk + (j0 + 1) + nao * (l0 + 0));
            reduce(gout12 * d_0 + gout13 * d_1 + gout14 * d_2 + gout15 * d_3 +
                       gout16 * d_4 + gout17 * d_5 + gout30 * d_6 +
                       gout31 * d_7 + gout32 * d_8 + gout33 * d_9 +
                       gout34 * d_10 + gout35 * d_11 + gout48 * d_12 +
                       gout49 * d_13 + gout50 * d_14 + gout51 * d_15 +
                       gout52 * d_16 + gout53 * d_17,
                vk + (j0 + 2) + nao * (l0 + 0));
            reduce(gout54 * d_0 + gout55 * d_1 + gout56 * d_2 + gout57 * d_3 +
                       gout58 * d_4 + gout59 * d_5 + gout72 * d_6 +
                       gout73 * d_7 + gout74 * d_8 + gout75 * d_9 +
                       gout76 * d_10 + gout77 * d_11 + gout90 * d_12 +
                       gout91 * d_13 + gout92 * d_14 + gout93 * d_15 +
                       gout94 * d_16 + gout95 * d_17,
                vk + (j0 + 0) + nao * (l0 + 1));
            reduce(gout60 * d_0 + gout61 * d_1 + gout62 * d_2 + gout63 * d_3 +
                       gout64 * d_4 + gout65 * d_5 + gout78 * d_6 +
                       gout79 * d_7 + gout80 * d_8 + gout81 * d_9 +
                       gout82 * d_10 + gout83 * d_11 + gout96 * d_12 +
                       gout97 * d_13 + gout98 * d_14 + gout99 * d_15 +
                       gout100 * d_16 + gout101 * d_17,
                vk + (j0 + 1) + nao * (l0 + 1));
            reduce(gout66 * d_0 + gout67 * d_1 + gout68 * d_2 + gout69 * d_3 +
                       gout70 * d_4 + gout71 * d_5 + gout84 * d_6 +
                       gout85 * d_7 + gout86 * d_8 + gout87 * d_9 +
                       gout88 * d_10 + gout89 * d_11 + gout102 * d_12 +
                       gout103 * d_13 + gout104 * d_14 + gout105 * d_15 +
                       gout106 * d_16 + gout107 * d_17,
                vk + (j0 + 2) + nao * (l0 + 1));
            reduce(gout108 * d_0 + gout109 * d_1 + gout110 * d_2 +
                       gout111 * d_3 + gout112 * d_4 + gout113 * d_5 +
                       gout126 * d_6 + gout127 * d_7 + gout128 * d_8 +
                       gout129 * d_9 + gout130 * d_10 + gout131 * d_11 +
                       gout144 * d_12 + gout145 * d_13 + gout146 * d_14 +
                       gout147 * d_15 + gout148 * d_16 + gout149 * d_17,
                vk + (j0 + 0) + nao * (l0 + 2));
            reduce(gout114 * d_0 + gout115 * d_1 + gout116 * d_2 +
                       gout117 * d_3 + gout118 * d_4 + gout119 * d_5 +
                       gout132 * d_6 + gout133 * d_7 + gout134 * d_8 +
                       gout135 * d_9 + gout136 * d_10 + gout137 * d_11 +
                       gout150 * d_12 + gout151 * d_13 + gout152 * d_14 +
                       gout153 * d_15 + gout154 * d_16 + gout155 * d_17,
                vk + (j0 + 1) + nao * (l0 + 2));
            reduce(gout120 * d_0 + gout121 * d_1 + gout122 * d_2 +
                       gout123 * d_3 + gout124 * d_4 + gout125 * d_5 +
                       gout138 * d_6 + gout139 * d_7 + gout140 * d_8 +
                       gout141 * d_9 + gout142 * d_10 + gout143 * d_11 +
                       gout156 * d_12 + gout157 * d_13 + gout158 * d_14 +
                       gout159 * d_15 + gout160 * d_16 + gout161 * d_17,
                vk + (j0 + 2) + nao * (l0 + 2));
            vk += nao2;
        }
        dm += nao2;
    }
}

__global__ static void GINTint2e_jk_kernel2120(JKMatrix jk,
    BasisProdOffsets offsets, GINTEnvVars envs, BasisProdCache bpcache) {
    int ntasks_ij = offsets.ntasks_ij;
    long ntasks = ntasks_ij * offsets.ntasks_kl;
    long task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int igroup = nprim_ij * nprim_kl;
    ntasks *= igroup;
    if (task_ij >= ntasks)
        return;
    int kl = task_ij % nprim_kl;
    task_ij /= nprim_kl;
    int ij = task_ij % nprim_ij;
    task_ij /= nprim_ij;
    int task_kl = task_ij / ntasks_ij;
    task_ij = task_ij % ntasks_ij;

    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    if (bas_ij < bas_kl) {
        return;
    }
    double norm = envs.fac;
    if (bas_ij == bas_kl) {
        norm *= .5;
    }

    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *ao_loc = bpcache.ao_loc;
    int *bas_pair2bra = bpcache.bas_pair2bra;
    int *bas_pair2ket = bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    int k0 = ao_loc[ksh];
    int l0 = ao_loc[lsh];
    double *__restrict__ a12 = bpcache.a12;
    double *__restrict__ e12 = bpcache.e12;
    double *__restrict__ x12 = bpcache.x12;
    double *__restrict__ y12 = bpcache.y12;
    double *__restrict__ z12 = bpcache.z12;
    int i_dm;
    int nbas = bpcache.nbas;
    double *__restrict__ bas_x = bpcache.bas_coords;
    double *__restrict__ bas_y = bas_x + nbas;
    double *__restrict__ bas_z = bas_y + nbas;

    double gout0 = 0;
    double gout1 = 0;
    double gout2 = 0;
    double gout3 = 0;
    double gout4 = 0;
    double gout5 = 0;
    double gout6 = 0;
    double gout7 = 0;
    double gout8 = 0;
    double gout9 = 0;
    double gout10 = 0;
    double gout11 = 0;
    double gout12 = 0;
    double gout13 = 0;
    double gout14 = 0;
    double gout15 = 0;
    double gout16 = 0;
    double gout17 = 0;
    double gout18 = 0;
    double gout19 = 0;
    double gout20 = 0;
    double gout21 = 0;
    double gout22 = 0;
    double gout23 = 0;
    double gout24 = 0;
    double gout25 = 0;
    double gout26 = 0;
    double gout27 = 0;
    double gout28 = 0;
    double gout29 = 0;
    double gout30 = 0;
    double gout31 = 0;
    double gout32 = 0;
    double gout33 = 0;
    double gout34 = 0;
    double gout35 = 0;
    double gout36 = 0;
    double gout37 = 0;
    double gout38 = 0;
    double gout39 = 0;
    double gout40 = 0;
    double gout41 = 0;
    double gout42 = 0;
    double gout43 = 0;
    double gout44 = 0;
    double gout45 = 0;
    double gout46 = 0;
    double gout47 = 0;
    double gout48 = 0;
    double gout49 = 0;
    double gout50 = 0;
    double gout51 = 0;
    double gout52 = 0;
    double gout53 = 0;
    double gout54 = 0;
    double gout55 = 0;
    double gout56 = 0;
    double gout57 = 0;
    double gout58 = 0;
    double gout59 = 0;
    double gout60 = 0;
    double gout61 = 0;
    double gout62 = 0;
    double gout63 = 0;
    double gout64 = 0;
    double gout65 = 0;
    double gout66 = 0;
    double gout67 = 0;
    double gout68 = 0;
    double gout69 = 0;
    double gout70 = 0;
    double gout71 = 0;
    double gout72 = 0;
    double gout73 = 0;
    double gout74 = 0;
    double gout75 = 0;
    double gout76 = 0;
    double gout77 = 0;
    double gout78 = 0;
    double gout79 = 0;
    double gout80 = 0;
    double gout81 = 0;
    double gout82 = 0;
    double gout83 = 0;
    double gout84 = 0;
    double gout85 = 0;
    double gout86 = 0;
    double gout87 = 0;
    double gout88 = 0;
    double gout89 = 0;
    double gout90 = 0;
    double gout91 = 0;
    double gout92 = 0;
    double gout93 = 0;
    double gout94 = 0;
    double gout95 = 0;
    double gout96 = 0;
    double gout97 = 0;
    double gout98 = 0;
    double gout99 = 0;
    double gout100 = 0;
    double gout101 = 0;
    double gout102 = 0;
    double gout103 = 0;
    double gout104 = 0;
    double gout105 = 0;
    double gout106 = 0;
    double gout107 = 0;
    double xi = bas_x[ish];
    double yi = bas_y[ish];
    double zi = bas_z[ish];
    double xixj = xi - bas_x[jsh];
    double yiyj = yi - bas_y[jsh];
    double zizj = zi - bas_z[jsh];
    double xk = bas_x[ksh];
    double yk = bas_y[ksh];
    double zk = bas_z[ksh];
    auto reduce = SegReduce<double>(igroup);
    ij += prim_ij;
    kl += prim_kl;
    double aij = a12[ij];
    double eij = e12[ij];
    double xij = x12[ij];
    double yij = y12[ij];
    double zij = z12[ij];
    double akl = a12[kl];
    double ekl = e12[kl];
    double xkl = x12[kl];
    double ykl = y12[kl];
    double zkl = z12[kl];
    double xijxkl = xij - xkl;
    double yijykl = yij - ykl;
    double zijzkl = zij - zkl;
    double aijkl = aij + akl;
    double a1 = aij * akl;
    double a0 = a1 / aijkl;
    double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
    double fac = norm * eij * ekl / (sqrt(aijkl) * a1);

    double rw[6];
    double root0, weight0;
    GINTrys_root<3>(x, rw);
    int irys;
    for (irys = 0; irys < 3; ++irys) {
        root0 = rw[irys];
        weight0 = rw[irys + 3];
        double u2 = a0 * root0;
        double tmp4 = .5 / (u2 * aijkl + a1);
        double b00 = u2 * tmp4;
        double tmp1 = 2 * b00;
        double tmp2 = tmp1 * akl;
        double b10 = b00 + tmp4 * akl;
        double c00x = xij - xi - tmp2 * xijxkl;
        double c00y = yij - yi - tmp2 * yijykl;
        double c00z = zij - zi - tmp2 * zijzkl;
        double tmp3 = tmp1 * aij;
        double b01 = b00 + tmp4 * aij;
        double c0px = xkl - xk + tmp3 * xijxkl;
        double c0py = ykl - yk + tmp3 * yijykl;
        double c0pz = zkl - zk + tmp3 * zijzkl;
        double g_0 = 1;
        double g_1 = c00x;
        double g_2 = c00x * c00x + b10;
        double g_3 = c00x + xixj;
        double g_4 = c00x * (c00x + xixj) + b10;
        double g_5 = c00x * (2 * b10 + g_2) + xixj * g_2;
        double g_6 = c0px;
        double g_7 = c0px * c00x + b00;
        double g_8 = b00 * c00x + b10 * c0px + c00x * g_7;
        double g_9 = c0px * (c00x + xixj) + b00;
        double g_10 = b00 * c00x + b10 * c0px + c00x * g_7 + xixj * g_7;
        double g_11 = 2 * b10 * g_7 + b00 * g_2 + c00x * g_8 + xixj * g_8;
        double g_12 = c0px * c0px + b01;
        double g_13 = b00 * c0px + b01 * c00x + c0px * g_7;
        double g_14 = 2 * b00 * g_7 + b10 * g_12 + c00x * g_13;
        double g_15 = b00 * c0px + b01 * c00x + c0px * g_7 + xixj * g_12;
        double g_16 = 2 * b00 * g_7 + b10 * g_12 + c00x * g_13 + xixj * g_13;
        double g_17 = 2 * (b00 * g_8 + b10 * g_13) + c00x * g_14 + xixj * g_14;
        double g_18 = 1;
        double g_19 = c00y;
        double g_20 = c00y * c00y + b10;
        double g_21 = c00y + yiyj;
        double g_22 = c00y * (c00y + yiyj) + b10;
        double g_23 = c00y * (2 * b10 + g_20) + yiyj * g_20;
        double g_24 = c0py;
        double g_25 = c0py * c00y + b00;
        double g_26 = b00 * c00y + b10 * c0py + c00y * g_25;
        double g_27 = c0py * (c00y + yiyj) + b00;
        double g_28 = b00 * c00y + b10 * c0py + c00y * g_25 + yiyj * g_25;
        double g_29 = 2 * b10 * g_25 + b00 * g_20 + c00y * g_26 + yiyj * g_26;
        double g_30 = c0py * c0py + b01;
        double g_31 = b00 * c0py + b01 * c00y + c0py * g_25;
        double g_32 = 2 * b00 * g_25 + b10 * g_30 + c00y * g_31;
        double g_33 = b00 * c0py + b01 * c00y + c0py * g_25 + yiyj * g_30;
        double g_34 = 2 * b00 * g_25 + b10 * g_30 + c00y * g_31 + yiyj * g_31;
        double g_35 =
            2 * (b00 * g_26 + b10 * g_31) + c00y * g_32 + yiyj * g_32;
        double g_36 = weight0 * fac;
        double g_37 = c00z * g_36;
        double g_38 = b10 * g_36 + c00z * g_37;
        double g_39 = g_36 * (c00z + zizj);
        double g_40 = b10 * g_36 + c00z * g_37 + zizj * g_37;
        double g_41 = 2 * b10 * g_37 + c00z * g_38 + zizj * g_38;
        double g_42 = c0pz * g_36;
        double g_43 = b00 * g_36 + c0pz * g_37;
        double g_44 = b00 * g_37 + b10 * g_42 + c00z * g_43;
        double g_45 = b00 * g_36 + c0pz * g_37 + zizj * g_42;
        double g_46 = b00 * g_37 + b10 * g_42 + c00z * g_43 + zizj * g_43;
        double g_47 = 2 * b10 * g_43 + b00 * g_38 + c00z * g_44 + zizj * g_44;
        double g_48 = b01 * g_36 + c0pz * g_42;
        double g_49 = b00 * g_42 + b01 * g_37 + c0pz * g_43;
        double g_50 = 2 * b00 * g_43 + b10 * g_48 + c00z * g_49;
        double g_51 = b00 * g_42 + b01 * g_37 + c0pz * g_43 + zizj * g_48;
        double g_52 = 2 * b00 * g_43 + b10 * g_48 + c00z * g_49 + zizj * g_49;
        double g_53 =
            2 * (b00 * g_44 + b10 * g_49) + c00z * g_50 + zizj * g_50;
        gout0 += g_17 * g_18 * g_36;
        gout1 += g_16 * g_19 * g_36;
        gout2 += g_16 * g_18 * g_37;
        gout3 += g_15 * g_20 * g_36;
        gout4 += g_15 * g_19 * g_37;
        gout5 += g_15 * g_18 * g_38;
        gout6 += g_14 * g_21 * g_36;
        gout7 += g_13 * g_22 * g_36;
        gout8 += g_13 * g_21 * g_37;
        gout9 += g_12 * g_23 * g_36;
        gout10 += g_12 * g_22 * g_37;
        gout11 += g_12 * g_21 * g_38;
        gout12 += g_14 * g_18 * g_39;
        gout13 += g_13 * g_19 * g_39;
        gout14 += g_13 * g_18 * g_40;
        gout15 += g_12 * g_20 * g_39;
        gout16 += g_12 * g_19 * g_40;
        gout17 += g_12 * g_18 * g_41;
        gout18 += g_11 * g_24 * g_36;
        gout19 += g_10 * g_25 * g_36;
        gout20 += g_10 * g_24 * g_37;
        gout21 += g_9 * g_26 * g_36;
        gout22 += g_9 * g_25 * g_37;
        gout23 += g_9 * g_24 * g_38;
        gout24 += g_8 * g_27 * g_36;
        gout25 += g_7 * g_28 * g_36;
        gout26 += g_7 * g_27 * g_37;
        gout27 += g_6 * g_29 * g_36;
        gout28 += g_6 * g_28 * g_37;
        gout29 += g_6 * g_27 * g_38;
        gout30 += g_8 * g_24 * g_39;
        gout31 += g_7 * g_25 * g_39;
        gout32 += g_7 * g_24 * g_40;
        gout33 += g_6 * g_26 * g_39;
        gout34 += g_6 * g_25 * g_40;
        gout35 += g_6 * g_24 * g_41;
        gout36 += g_11 * g_18 * g_42;
        gout37 += g_10 * g_19 * g_42;
        gout38 += g_10 * g_18 * g_43;
        gout39 += g_9 * g_20 * g_42;
        gout40 += g_9 * g_19 * g_43;
        gout41 += g_9 * g_18 * g_44;
        gout42 += g_8 * g_21 * g_42;
        gout43 += g_7 * g_22 * g_42;
        gout44 += g_7 * g_21 * g_43;
        gout45 += g_6 * g_23 * g_42;
        gout46 += g_6 * g_22 * g_43;
        gout47 += g_6 * g_21 * g_44;
        gout48 += g_8 * g_18 * g_45;
        gout49 += g_7 * g_19 * g_45;
        gout50 += g_7 * g_18 * g_46;
        gout51 += g_6 * g_20 * g_45;
        gout52 += g_6 * g_19 * g_46;
        gout53 += g_6 * g_18 * g_47;
        gout54 += g_5 * g_30 * g_36;
        gout55 += g_4 * g_31 * g_36;
        gout56 += g_4 * g_30 * g_37;
        gout57 += g_3 * g_32 * g_36;
        gout58 += g_3 * g_31 * g_37;
        gout59 += g_3 * g_30 * g_38;
        gout60 += g_2 * g_33 * g_36;
        gout61 += g_1 * g_34 * g_36;
        gout62 += g_1 * g_33 * g_37;
        gout63 += g_0 * g_35 * g_36;
        gout64 += g_0 * g_34 * g_37;
        gout65 += g_0 * g_33 * g_38;
        gout66 += g_2 * g_30 * g_39;
        gout67 += g_1 * g_31 * g_39;
        gout68 += g_1 * g_30 * g_40;
        gout69 += g_0 * g_32 * g_39;
        gout70 += g_0 * g_31 * g_40;
        gout71 += g_0 * g_30 * g_41;
        gout72 += g_5 * g_24 * g_42;
        gout73 += g_4 * g_25 * g_42;
        gout74 += g_4 * g_24 * g_43;
        gout75 += g_3 * g_26 * g_42;
        gout76 += g_3 * g_25 * g_43;
        gout77 += g_3 * g_24 * g_44;
        gout78 += g_2 * g_27 * g_42;
        gout79 += g_1 * g_28 * g_42;
        gout80 += g_1 * g_27 * g_43;
        gout81 += g_0 * g_29 * g_42;
        gout82 += g_0 * g_28 * g_43;
        gout83 += g_0 * g_27 * g_44;
        gout84 += g_2 * g_24 * g_45;
        gout85 += g_1 * g_25 * g_45;
        gout86 += g_1 * g_24 * g_46;
        gout87 += g_0 * g_26 * g_45;
        gout88 += g_0 * g_25 * g_46;
        gout89 += g_0 * g_24 * g_47;
        gout90 += g_5 * g_18 * g_48;
        gout91 += g_4 * g_19 * g_48;
        gout92 += g_4 * g_18 * g_49;
        gout93 += g_3 * g_20 * g_48;
        gout94 += g_3 * g_19 * g_49;
        gout95 += g_3 * g_18 * g_50;
        gout96 += g_2 * g_21 * g_48;
        gout97 += g_1 * g_22 * g_48;
        gout98 += g_1 * g_21 * g_49;
        gout99 += g_0 * g_23 * g_48;
        gout100 += g_0 * g_22 * g_49;
        gout101 += g_0 * g_21 * g_50;
        gout102 += g_2 * g_18 * g_51;
        gout103 += g_1 * g_19 * g_51;
        gout104 += g_1 * g_18 * g_52;
        gout105 += g_0 * g_20 * g_51;
        gout106 += g_0 * g_19 * g_52;
        gout107 += g_0 * g_18 * g_53;
    }
    double d_0, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, d_9;
    double d_10, d_11, d_12, d_13, d_14, d_15, d_16, d_17, d_18, d_19;
    double d_20, d_21, d_22, d_23, d_24, d_25, d_26, d_27, d_28, d_29;
    double d_30, d_31, d_32, d_33, d_34, d_35;
    int n_dm = jk.n_dm;
    int nao = jk.nao;
    size_t nao2 = nao * nao;
    double *__restrict__ dm = jk.dm;
    double *vj = jk.vj;
    double *vk = jk.vk;
    for (i_dm = 0; i_dm < n_dm; ++i_dm) {
        if (vj != NULL) {
            // ijkl,ij->kl
            d_0 = dm[(i0 + 0) + nao * (j0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (j0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (j0 + 0)];
            d_3 = dm[(i0 + 3) + nao * (j0 + 0)];
            d_4 = dm[(i0 + 4) + nao * (j0 + 0)];
            d_5 = dm[(i0 + 5) + nao * (j0 + 0)];
            d_6 = dm[(i0 + 0) + nao * (j0 + 1)];
            d_7 = dm[(i0 + 1) + nao * (j0 + 1)];
            d_8 = dm[(i0 + 2) + nao * (j0 + 1)];
            d_9 = dm[(i0 + 3) + nao * (j0 + 1)];
            d_10 = dm[(i0 + 4) + nao * (j0 + 1)];
            d_11 = dm[(i0 + 5) + nao * (j0 + 1)];
            d_12 = dm[(i0 + 0) + nao * (j0 + 2)];
            d_13 = dm[(i0 + 1) + nao * (j0 + 2)];
            d_14 = dm[(i0 + 2) + nao * (j0 + 2)];
            d_15 = dm[(i0 + 3) + nao * (j0 + 2)];
            d_16 = dm[(i0 + 4) + nao * (j0 + 2)];
            d_17 = dm[(i0 + 5) + nao * (j0 + 2)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout3 * d_3 +
                       gout4 * d_4 + gout5 * d_5 + gout6 * d_6 + gout7 * d_7 +
                       gout8 * d_8 + gout9 * d_9 + gout10 * d_10 +
                       gout11 * d_11 + gout12 * d_12 + gout13 * d_13 +
                       gout14 * d_14 + gout15 * d_15 + gout16 * d_16 +
                       gout17 * d_17,
                vj + (k0 + 0) + nao * (l0 + 0));
            reduce(gout18 * d_0 + gout19 * d_1 + gout20 * d_2 + gout21 * d_3 +
                       gout22 * d_4 + gout23 * d_5 + gout24 * d_6 +
                       gout25 * d_7 + gout26 * d_8 + gout27 * d_9 +
                       gout28 * d_10 + gout29 * d_11 + gout30 * d_12 +
                       gout31 * d_13 + gout32 * d_14 + gout33 * d_15 +
                       gout34 * d_16 + gout35 * d_17,
                vj + (k0 + 1) + nao * (l0 + 0));
            reduce(gout36 * d_0 + gout37 * d_1 + gout38 * d_2 + gout39 * d_3 +
                       gout40 * d_4 + gout41 * d_5 + gout42 * d_6 +
                       gout43 * d_7 + gout44 * d_8 + gout45 * d_9 +
                       gout46 * d_10 + gout47 * d_11 + gout48 * d_12 +
                       gout49 * d_13 + gout50 * d_14 + gout51 * d_15 +
                       gout52 * d_16 + gout53 * d_17,
                vj + (k0 + 2) + nao * (l0 + 0));
            reduce(gout54 * d_0 + gout55 * d_1 + gout56 * d_2 + gout57 * d_3 +
                       gout58 * d_4 + gout59 * d_5 + gout60 * d_6 +
                       gout61 * d_7 + gout62 * d_8 + gout63 * d_9 +
                       gout64 * d_10 + gout65 * d_11 + gout66 * d_12 +
                       gout67 * d_13 + gout68 * d_14 + gout69 * d_15 +
                       gout70 * d_16 + gout71 * d_17,
                vj + (k0 + 3) + nao * (l0 + 0));
            reduce(gout72 * d_0 + gout73 * d_1 + gout74 * d_2 + gout75 * d_3 +
                       gout76 * d_4 + gout77 * d_5 + gout78 * d_6 +
                       gout79 * d_7 + gout80 * d_8 + gout81 * d_9 +
                       gout82 * d_10 + gout83 * d_11 + gout84 * d_12 +
                       gout85 * d_13 + gout86 * d_14 + gout87 * d_15 +
                       gout88 * d_16 + gout89 * d_17,
                vj + (k0 + 4) + nao * (l0 + 0));
            reduce(gout90 * d_0 + gout91 * d_1 + gout92 * d_2 + gout93 * d_3 +
                       gout94 * d_4 + gout95 * d_5 + gout96 * d_6 +
                       gout97 * d_7 + gout98 * d_8 + gout99 * d_9 +
                       gout100 * d_10 + gout101 * d_11 + gout102 * d_12 +
                       gout103 * d_13 + gout104 * d_14 + gout105 * d_15 +
                       gout106 * d_16 + gout107 * d_17,
                vj + (k0 + 5) + nao * (l0 + 0));
            // ijkl,kl->ij
            d_0 = dm[(k0 + 0) + nao * (l0 + 0)];
            d_1 = dm[(k0 + 1) + nao * (l0 + 0)];
            d_2 = dm[(k0 + 2) + nao * (l0 + 0)];
            d_3 = dm[(k0 + 3) + nao * (l0 + 0)];
            d_4 = dm[(k0 + 4) + nao * (l0 + 0)];
            d_5 = dm[(k0 + 5) + nao * (l0 + 0)];
            reduce(gout0 * d_0 + gout18 * d_1 + gout36 * d_2 + gout54 * d_3 +
                       gout72 * d_4 + gout90 * d_5,
                vj + (i0 + 0) + nao * (j0 + 0));
            reduce(gout1 * d_0 + gout19 * d_1 + gout37 * d_2 + gout55 * d_3 +
                       gout73 * d_4 + gout91 * d_5,
                vj + (i0 + 1) + nao * (j0 + 0));
            reduce(gout2 * d_0 + gout20 * d_1 + gout38 * d_2 + gout56 * d_3 +
                       gout74 * d_4 + gout92 * d_5,
                vj + (i0 + 2) + nao * (j0 + 0));
            reduce(gout3 * d_0 + gout21 * d_1 + gout39 * d_2 + gout57 * d_3 +
                       gout75 * d_4 + gout93 * d_5,
                vj + (i0 + 3) + nao * (j0 + 0));
            reduce(gout4 * d_0 + gout22 * d_1 + gout40 * d_2 + gout58 * d_3 +
                       gout76 * d_4 + gout94 * d_5,
                vj + (i0 + 4) + nao * (j0 + 0));
            reduce(gout5 * d_0 + gout23 * d_1 + gout41 * d_2 + gout59 * d_3 +
                       gout77 * d_4 + gout95 * d_5,
                vj + (i0 + 5) + nao * (j0 + 0));
            reduce(gout6 * d_0 + gout24 * d_1 + gout42 * d_2 + gout60 * d_3 +
                       gout78 * d_4 + gout96 * d_5,
                vj + (i0 + 0) + nao * (j0 + 1));
            reduce(gout7 * d_0 + gout25 * d_1 + gout43 * d_2 + gout61 * d_3 +
                       gout79 * d_4 + gout97 * d_5,
                vj + (i0 + 1) + nao * (j0 + 1));
            reduce(gout8 * d_0 + gout26 * d_1 + gout44 * d_2 + gout62 * d_3 +
                       gout80 * d_4 + gout98 * d_5,
                vj + (i0 + 2) + nao * (j0 + 1));
            reduce(gout9 * d_0 + gout27 * d_1 + gout45 * d_2 + gout63 * d_3 +
                       gout81 * d_4 + gout99 * d_5,
                vj + (i0 + 3) + nao * (j0 + 1));
            reduce(gout10 * d_0 + gout28 * d_1 + gout46 * d_2 + gout64 * d_3 +
                       gout82 * d_4 + gout100 * d_5,
                vj + (i0 + 4) + nao * (j0 + 1));
            reduce(gout11 * d_0 + gout29 * d_1 + gout47 * d_2 + gout65 * d_3 +
                       gout83 * d_4 + gout101 * d_5,
                vj + (i0 + 5) + nao * (j0 + 1));
            reduce(gout12 * d_0 + gout30 * d_1 + gout48 * d_2 + gout66 * d_3 +
                       gout84 * d_4 + gout102 * d_5,
                vj + (i0 + 0) + nao * (j0 + 2));
            reduce(gout13 * d_0 + gout31 * d_1 + gout49 * d_2 + gout67 * d_3 +
                       gout85 * d_4 + gout103 * d_5,
                vj + (i0 + 1) + nao * (j0 + 2));
            reduce(gout14 * d_0 + gout32 * d_1 + gout50 * d_2 + gout68 * d_3 +
                       gout86 * d_4 + gout104 * d_5,
                vj + (i0 + 2) + nao * (j0 + 2));
            reduce(gout15 * d_0 + gout33 * d_1 + gout51 * d_2 + gout69 * d_3 +
                       gout87 * d_4 + gout105 * d_5,
                vj + (i0 + 3) + nao * (j0 + 2));
            reduce(gout16 * d_0 + gout34 * d_1 + gout52 * d_2 + gout70 * d_3 +
                       gout88 * d_4 + gout106 * d_5,
                vj + (i0 + 4) + nao * (j0 + 2));
            reduce(gout17 * d_0 + gout35 * d_1 + gout53 * d_2 + gout71 * d_3 +
                       gout89 * d_4 + gout107 * d_5,
                vj + (i0 + 5) + nao * (j0 + 2));
            vj += nao2;
        }
        if (vk != NULL) {
            // ijkl,jl->ik
            d_0 = dm[(j0 + 0) + nao * (l0 + 0)];
            d_1 = dm[(j0 + 1) + nao * (l0 + 0)];
            d_2 = dm[(j0 + 2) + nao * (l0 + 0)];
            reduce(gout0 * d_0 + gout6 * d_1 + gout12 * d_2,
                vk + (i0 + 0) + nao * (k0 + 0));
            reduce(gout1 * d_0 + gout7 * d_1 + gout13 * d_2,
                vk + (i0 + 1) + nao * (k0 + 0));
            reduce(gout2 * d_0 + gout8 * d_1 + gout14 * d_2,
                vk + (i0 + 2) + nao * (k0 + 0));
            reduce(gout3 * d_0 + gout9 * d_1 + gout15 * d_2,
                vk + (i0 + 3) + nao * (k0 + 0));
            reduce(gout4 * d_0 + gout10 * d_1 + gout16 * d_2,
                vk + (i0 + 4) + nao * (k0 + 0));
            reduce(gout5 * d_0 + gout11 * d_1 + gout17 * d_2,
                vk + (i0 + 5) + nao * (k0 + 0));
            reduce(gout18 * d_0 + gout24 * d_1 + gout30 * d_2,
                vk + (i0 + 0) + nao * (k0 + 1));
            reduce(gout19 * d_0 + gout25 * d_1 + gout31 * d_2,
                vk + (i0 + 1) + nao * (k0 + 1));
            reduce(gout20 * d_0 + gout26 * d_1 + gout32 * d_2,
                vk + (i0 + 2) + nao * (k0 + 1));
            reduce(gout21 * d_0 + gout27 * d_1 + gout33 * d_2,
                vk + (i0 + 3) + nao * (k0 + 1));
            reduce(gout22 * d_0 + gout28 * d_1 + gout34 * d_2,
                vk + (i0 + 4) + nao * (k0 + 1));
            reduce(gout23 * d_0 + gout29 * d_1 + gout35 * d_2,
                vk + (i0 + 5) + nao * (k0 + 1));
            reduce(gout36 * d_0 + gout42 * d_1 + gout48 * d_2,
                vk + (i0 + 0) + nao * (k0 + 2));
            reduce(gout37 * d_0 + gout43 * d_1 + gout49 * d_2,
                vk + (i0 + 1) + nao * (k0 + 2));
            reduce(gout38 * d_0 + gout44 * d_1 + gout50 * d_2,
                vk + (i0 + 2) + nao * (k0 + 2));
            reduce(gout39 * d_0 + gout45 * d_1 + gout51 * d_2,
                vk + (i0 + 3) + nao * (k0 + 2));
            reduce(gout40 * d_0 + gout46 * d_1 + gout52 * d_2,
                vk + (i0 + 4) + nao * (k0 + 2));
            reduce(gout41 * d_0 + gout47 * d_1 + gout53 * d_2,
                vk + (i0 + 5) + nao * (k0 + 2));
            reduce(gout54 * d_0 + gout60 * d_1 + gout66 * d_2,
                vk + (i0 + 0) + nao * (k0 + 3));
            reduce(gout55 * d_0 + gout61 * d_1 + gout67 * d_2,
                vk + (i0 + 1) + nao * (k0 + 3));
            reduce(gout56 * d_0 + gout62 * d_1 + gout68 * d_2,
                vk + (i0 + 2) + nao * (k0 + 3));
            reduce(gout57 * d_0 + gout63 * d_1 + gout69 * d_2,
                vk + (i0 + 3) + nao * (k0 + 3));
            reduce(gout58 * d_0 + gout64 * d_1 + gout70 * d_2,
                vk + (i0 + 4) + nao * (k0 + 3));
            reduce(gout59 * d_0 + gout65 * d_1 + gout71 * d_2,
                vk + (i0 + 5) + nao * (k0 + 3));
            reduce(gout72 * d_0 + gout78 * d_1 + gout84 * d_2,
                vk + (i0 + 0) + nao * (k0 + 4));
            reduce(gout73 * d_0 + gout79 * d_1 + gout85 * d_2,
                vk + (i0 + 1) + nao * (k0 + 4));
            reduce(gout74 * d_0 + gout80 * d_1 + gout86 * d_2,
                vk + (i0 + 2) + nao * (k0 + 4));
            reduce(gout75 * d_0 + gout81 * d_1 + gout87 * d_2,
                vk + (i0 + 3) + nao * (k0 + 4));
            reduce(gout76 * d_0 + gout82 * d_1 + gout88 * d_2,
                vk + (i0 + 4) + nao * (k0 + 4));
            reduce(gout77 * d_0 + gout83 * d_1 + gout89 * d_2,
                vk + (i0 + 5) + nao * (k0 + 4));
            reduce(gout90 * d_0 + gout96 * d_1 + gout102 * d_2,
                vk + (i0 + 0) + nao * (k0 + 5));
            reduce(gout91 * d_0 + gout97 * d_1 + gout103 * d_2,
                vk + (i0 + 1) + nao * (k0 + 5));
            reduce(gout92 * d_0 + gout98 * d_1 + gout104 * d_2,
                vk + (i0 + 2) + nao * (k0 + 5));
            reduce(gout93 * d_0 + gout99 * d_1 + gout105 * d_2,
                vk + (i0 + 3) + nao * (k0 + 5));
            reduce(gout94 * d_0 + gout100 * d_1 + gout106 * d_2,
                vk + (i0 + 4) + nao * (k0 + 5));
            reduce(gout95 * d_0 + gout101 * d_1 + gout107 * d_2,
                vk + (i0 + 5) + nao * (k0 + 5));
            // ijkl,jk->il
            d_0 = dm[(j0 + 0) + nao * (k0 + 0)];
            d_1 = dm[(j0 + 1) + nao * (k0 + 0)];
            d_2 = dm[(j0 + 2) + nao * (k0 + 0)];
            d_3 = dm[(j0 + 0) + nao * (k0 + 1)];
            d_4 = dm[(j0 + 1) + nao * (k0 + 1)];
            d_5 = dm[(j0 + 2) + nao * (k0 + 1)];
            d_6 = dm[(j0 + 0) + nao * (k0 + 2)];
            d_7 = dm[(j0 + 1) + nao * (k0 + 2)];
            d_8 = dm[(j0 + 2) + nao * (k0 + 2)];
            d_9 = dm[(j0 + 0) + nao * (k0 + 3)];
            d_10 = dm[(j0 + 1) + nao * (k0 + 3)];
            d_11 = dm[(j0 + 2) + nao * (k0 + 3)];
            d_12 = dm[(j0 + 0) + nao * (k0 + 4)];
            d_13 = dm[(j0 + 1) + nao * (k0 + 4)];
            d_14 = dm[(j0 + 2) + nao * (k0 + 4)];
            d_15 = dm[(j0 + 0) + nao * (k0 + 5)];
            d_16 = dm[(j0 + 1) + nao * (k0 + 5)];
            d_17 = dm[(j0 + 2) + nao * (k0 + 5)];
            reduce(gout0 * d_0 + gout6 * d_1 + gout12 * d_2 + gout18 * d_3 +
                       gout24 * d_4 + gout30 * d_5 + gout36 * d_6 +
                       gout42 * d_7 + gout48 * d_8 + gout54 * d_9 +
                       gout60 * d_10 + gout66 * d_11 + gout72 * d_12 +
                       gout78 * d_13 + gout84 * d_14 + gout90 * d_15 +
                       gout96 * d_16 + gout102 * d_17,
                vk + (i0 + 0) + nao * (l0 + 0));
            reduce(gout1 * d_0 + gout7 * d_1 + gout13 * d_2 + gout19 * d_3 +
                       gout25 * d_4 + gout31 * d_5 + gout37 * d_6 +
                       gout43 * d_7 + gout49 * d_8 + gout55 * d_9 +
                       gout61 * d_10 + gout67 * d_11 + gout73 * d_12 +
                       gout79 * d_13 + gout85 * d_14 + gout91 * d_15 +
                       gout97 * d_16 + gout103 * d_17,
                vk + (i0 + 1) + nao * (l0 + 0));
            reduce(gout2 * d_0 + gout8 * d_1 + gout14 * d_2 + gout20 * d_3 +
                       gout26 * d_4 + gout32 * d_5 + gout38 * d_6 +
                       gout44 * d_7 + gout50 * d_8 + gout56 * d_9 +
                       gout62 * d_10 + gout68 * d_11 + gout74 * d_12 +
                       gout80 * d_13 + gout86 * d_14 + gout92 * d_15 +
                       gout98 * d_16 + gout104 * d_17,
                vk + (i0 + 2) + nao * (l0 + 0));
            reduce(gout3 * d_0 + gout9 * d_1 + gout15 * d_2 + gout21 * d_3 +
                       gout27 * d_4 + gout33 * d_5 + gout39 * d_6 +
                       gout45 * d_7 + gout51 * d_8 + gout57 * d_9 +
                       gout63 * d_10 + gout69 * d_11 + gout75 * d_12 +
                       gout81 * d_13 + gout87 * d_14 + gout93 * d_15 +
                       gout99 * d_16 + gout105 * d_17,
                vk + (i0 + 3) + nao * (l0 + 0));
            reduce(gout4 * d_0 + gout10 * d_1 + gout16 * d_2 + gout22 * d_3 +
                       gout28 * d_4 + gout34 * d_5 + gout40 * d_6 +
                       gout46 * d_7 + gout52 * d_8 + gout58 * d_9 +
                       gout64 * d_10 + gout70 * d_11 + gout76 * d_12 +
                       gout82 * d_13 + gout88 * d_14 + gout94 * d_15 +
                       gout100 * d_16 + gout106 * d_17,
                vk + (i0 + 4) + nao * (l0 + 0));
            reduce(gout5 * d_0 + gout11 * d_1 + gout17 * d_2 + gout23 * d_3 +
                       gout29 * d_4 + gout35 * d_5 + gout41 * d_6 +
                       gout47 * d_7 + gout53 * d_8 + gout59 * d_9 +
                       gout65 * d_10 + gout71 * d_11 + gout77 * d_12 +
                       gout83 * d_13 + gout89 * d_14 + gout95 * d_15 +
                       gout101 * d_16 + gout107 * d_17,
                vk + (i0 + 5) + nao * (l0 + 0));
            // ijkl,il->jk
            d_0 = dm[(i0 + 0) + nao * (l0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (l0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (l0 + 0)];
            d_3 = dm[(i0 + 3) + nao * (l0 + 0)];
            d_4 = dm[(i0 + 4) + nao * (l0 + 0)];
            d_5 = dm[(i0 + 5) + nao * (l0 + 0)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout3 * d_3 +
                       gout4 * d_4 + gout5 * d_5,
                vk + (j0 + 0) + nao * (k0 + 0));
            reduce(gout6 * d_0 + gout7 * d_1 + gout8 * d_2 + gout9 * d_3 +
                       gout10 * d_4 + gout11 * d_5,
                vk + (j0 + 1) + nao * (k0 + 0));
            reduce(gout12 * d_0 + gout13 * d_1 + gout14 * d_2 + gout15 * d_3 +
                       gout16 * d_4 + gout17 * d_5,
                vk + (j0 + 2) + nao * (k0 + 0));
            reduce(gout18 * d_0 + gout19 * d_1 + gout20 * d_2 + gout21 * d_3 +
                       gout22 * d_4 + gout23 * d_5,
                vk + (j0 + 0) + nao * (k0 + 1));
            reduce(gout24 * d_0 + gout25 * d_1 + gout26 * d_2 + gout27 * d_3 +
                       gout28 * d_4 + gout29 * d_5,
                vk + (j0 + 1) + nao * (k0 + 1));
            reduce(gout30 * d_0 + gout31 * d_1 + gout32 * d_2 + gout33 * d_3 +
                       gout34 * d_4 + gout35 * d_5,
                vk + (j0 + 2) + nao * (k0 + 1));
            reduce(gout36 * d_0 + gout37 * d_1 + gout38 * d_2 + gout39 * d_3 +
                       gout40 * d_4 + gout41 * d_5,
                vk + (j0 + 0) + nao * (k0 + 2));
            reduce(gout42 * d_0 + gout43 * d_1 + gout44 * d_2 + gout45 * d_3 +
                       gout46 * d_4 + gout47 * d_5,
                vk + (j0 + 1) + nao * (k0 + 2));
            reduce(gout48 * d_0 + gout49 * d_1 + gout50 * d_2 + gout51 * d_3 +
                       gout52 * d_4 + gout53 * d_5,
                vk + (j0 + 2) + nao * (k0 + 2));
            reduce(gout54 * d_0 + gout55 * d_1 + gout56 * d_2 + gout57 * d_3 +
                       gout58 * d_4 + gout59 * d_5,
                vk + (j0 + 0) + nao * (k0 + 3));
            reduce(gout60 * d_0 + gout61 * d_1 + gout62 * d_2 + gout63 * d_3 +
                       gout64 * d_4 + gout65 * d_5,
                vk + (j0 + 1) + nao * (k0 + 3));
            reduce(gout66 * d_0 + gout67 * d_1 + gout68 * d_2 + gout69 * d_3 +
                       gout70 * d_4 + gout71 * d_5,
                vk + (j0 + 2) + nao * (k0 + 3));
            reduce(gout72 * d_0 + gout73 * d_1 + gout74 * d_2 + gout75 * d_3 +
                       gout76 * d_4 + gout77 * d_5,
                vk + (j0 + 0) + nao * (k0 + 4));
            reduce(gout78 * d_0 + gout79 * d_1 + gout80 * d_2 + gout81 * d_3 +
                       gout82 * d_4 + gout83 * d_5,
                vk + (j0 + 1) + nao * (k0 + 4));
            reduce(gout84 * d_0 + gout85 * d_1 + gout86 * d_2 + gout87 * d_3 +
                       gout88 * d_4 + gout89 * d_5,
                vk + (j0 + 2) + nao * (k0 + 4));
            reduce(gout90 * d_0 + gout91 * d_1 + gout92 * d_2 + gout93 * d_3 +
                       gout94 * d_4 + gout95 * d_5,
                vk + (j0 + 0) + nao * (k0 + 5));
            reduce(gout96 * d_0 + gout97 * d_1 + gout98 * d_2 + gout99 * d_3 +
                       gout100 * d_4 + gout101 * d_5,
                vk + (j0 + 1) + nao * (k0 + 5));
            reduce(gout102 * d_0 + gout103 * d_1 + gout104 * d_2 +
                       gout105 * d_3 + gout106 * d_4 + gout107 * d_5,
                vk + (j0 + 2) + nao * (k0 + 5));
            // ijkl,ik->jl
            d_0 = dm[(i0 + 0) + nao * (k0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (k0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (k0 + 0)];
            d_3 = dm[(i0 + 3) + nao * (k0 + 0)];
            d_4 = dm[(i0 + 4) + nao * (k0 + 0)];
            d_5 = dm[(i0 + 5) + nao * (k0 + 0)];
            d_6 = dm[(i0 + 0) + nao * (k0 + 1)];
            d_7 = dm[(i0 + 1) + nao * (k0 + 1)];
            d_8 = dm[(i0 + 2) + nao * (k0 + 1)];
            d_9 = dm[(i0 + 3) + nao * (k0 + 1)];
            d_10 = dm[(i0 + 4) + nao * (k0 + 1)];
            d_11 = dm[(i0 + 5) + nao * (k0 + 1)];
            d_12 = dm[(i0 + 0) + nao * (k0 + 2)];
            d_13 = dm[(i0 + 1) + nao * (k0 + 2)];
            d_14 = dm[(i0 + 2) + nao * (k0 + 2)];
            d_15 = dm[(i0 + 3) + nao * (k0 + 2)];
            d_16 = dm[(i0 + 4) + nao * (k0 + 2)];
            d_17 = dm[(i0 + 5) + nao * (k0 + 2)];
            d_18 = dm[(i0 + 0) + nao * (k0 + 3)];
            d_19 = dm[(i0 + 1) + nao * (k0 + 3)];
            d_20 = dm[(i0 + 2) + nao * (k0 + 3)];
            d_21 = dm[(i0 + 3) + nao * (k0 + 3)];
            d_22 = dm[(i0 + 4) + nao * (k0 + 3)];
            d_23 = dm[(i0 + 5) + nao * (k0 + 3)];
            d_24 = dm[(i0 + 0) + nao * (k0 + 4)];
            d_25 = dm[(i0 + 1) + nao * (k0 + 4)];
            d_26 = dm[(i0 + 2) + nao * (k0 + 4)];
            d_27 = dm[(i0 + 3) + nao * (k0 + 4)];
            d_28 = dm[(i0 + 4) + nao * (k0 + 4)];
            d_29 = dm[(i0 + 5) + nao * (k0 + 4)];
            d_30 = dm[(i0 + 0) + nao * (k0 + 5)];
            d_31 = dm[(i0 + 1) + nao * (k0 + 5)];
            d_32 = dm[(i0 + 2) + nao * (k0 + 5)];
            d_33 = dm[(i0 + 3) + nao * (k0 + 5)];
            d_34 = dm[(i0 + 4) + nao * (k0 + 5)];
            d_35 = dm[(i0 + 5) + nao * (k0 + 5)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout3 * d_3 +
                       gout4 * d_4 + gout5 * d_5 + gout18 * d_6 +
                       gout19 * d_7 + gout20 * d_8 + gout21 * d_9 +
                       gout22 * d_10 + gout23 * d_11 + gout36 * d_12 +
                       gout37 * d_13 + gout38 * d_14 + gout39 * d_15 +
                       gout40 * d_16 + gout41 * d_17 + gout54 * d_18 +
                       gout55 * d_19 + gout56 * d_20 + gout57 * d_21 +
                       gout58 * d_22 + gout59 * d_23 + gout72 * d_24 +
                       gout73 * d_25 + gout74 * d_26 + gout75 * d_27 +
                       gout76 * d_28 + gout77 * d_29 + gout90 * d_30 +
                       gout91 * d_31 + gout92 * d_32 + gout93 * d_33 +
                       gout94 * d_34 + gout95 * d_35,
                vk + (j0 + 0) + nao * (l0 + 0));
            reduce(gout6 * d_0 + gout7 * d_1 + gout8 * d_2 + gout9 * d_3 +
                       gout10 * d_4 + gout11 * d_5 + gout24 * d_6 +
                       gout25 * d_7 + gout26 * d_8 + gout27 * d_9 +
                       gout28 * d_10 + gout29 * d_11 + gout42 * d_12 +
                       gout43 * d_13 + gout44 * d_14 + gout45 * d_15 +
                       gout46 * d_16 + gout47 * d_17 + gout60 * d_18 +
                       gout61 * d_19 + gout62 * d_20 + gout63 * d_21 +
                       gout64 * d_22 + gout65 * d_23 + gout78 * d_24 +
                       gout79 * d_25 + gout80 * d_26 + gout81 * d_27 +
                       gout82 * d_28 + gout83 * d_29 + gout96 * d_30 +
                       gout97 * d_31 + gout98 * d_32 + gout99 * d_33 +
                       gout100 * d_34 + gout101 * d_35,
                vk + (j0 + 1) + nao * (l0 + 0));
            reduce(gout12 * d_0 + gout13 * d_1 + gout14 * d_2 + gout15 * d_3 +
                       gout16 * d_4 + gout17 * d_5 + gout30 * d_6 +
                       gout31 * d_7 + gout32 * d_8 + gout33 * d_9 +
                       gout34 * d_10 + gout35 * d_11 + gout48 * d_12 +
                       gout49 * d_13 + gout50 * d_14 + gout51 * d_15 +
                       gout52 * d_16 + gout53 * d_17 + gout66 * d_18 +
                       gout67 * d_19 + gout68 * d_20 + gout69 * d_21 +
                       gout70 * d_22 + gout71 * d_23 + gout84 * d_24 +
                       gout85 * d_25 + gout86 * d_26 + gout87 * d_27 +
                       gout88 * d_28 + gout89 * d_29 + gout102 * d_30 +
                       gout103 * d_31 + gout104 * d_32 + gout105 * d_33 +
                       gout106 * d_34 + gout107 * d_35,
                vk + (j0 + 2) + nao * (l0 + 0));
            vk += nao2;
        }
        dm += nao2;
    }
}

__global__ static void GINTint2e_jk_kernel2200(JKMatrix jk,
    BasisProdOffsets offsets, GINTEnvVars envs, BasisProdCache bpcache) {
    int ntasks_ij = offsets.ntasks_ij;
    long ntasks = ntasks_ij * offsets.ntasks_kl;
    long task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int igroup = nprim_ij * nprim_kl;
    ntasks *= igroup;
    if (task_ij >= ntasks)
        return;
    int kl = task_ij % nprim_kl;
    task_ij /= nprim_kl;
    int ij = task_ij % nprim_ij;
    task_ij /= nprim_ij;
    int task_kl = task_ij / ntasks_ij;
    task_ij = task_ij % ntasks_ij;

    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    if (bas_ij < bas_kl) {
        return;
    }
    double norm = envs.fac;
    if (bas_ij == bas_kl) {
        norm *= .5;
    }

    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *ao_loc = bpcache.ao_loc;
    int *bas_pair2bra = bpcache.bas_pair2bra;
    int *bas_pair2ket = bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    int k0 = ao_loc[ksh];
    int l0 = ao_loc[lsh];
    double *__restrict__ a12 = bpcache.a12;
    double *__restrict__ e12 = bpcache.e12;
    double *__restrict__ x12 = bpcache.x12;
    double *__restrict__ y12 = bpcache.y12;
    double *__restrict__ z12 = bpcache.z12;
    int i_dm;
    int nbas = bpcache.nbas;
    double *__restrict__ bas_x = bpcache.bas_coords;
    double *__restrict__ bas_y = bas_x + nbas;
    double *__restrict__ bas_z = bas_y + nbas;

    double gout0 = 0;
    double gout1 = 0;
    double gout2 = 0;
    double gout3 = 0;
    double gout4 = 0;
    double gout5 = 0;
    double gout6 = 0;
    double gout7 = 0;
    double gout8 = 0;
    double gout9 = 0;
    double gout10 = 0;
    double gout11 = 0;
    double gout12 = 0;
    double gout13 = 0;
    double gout14 = 0;
    double gout15 = 0;
    double gout16 = 0;
    double gout17 = 0;
    double gout18 = 0;
    double gout19 = 0;
    double gout20 = 0;
    double gout21 = 0;
    double gout22 = 0;
    double gout23 = 0;
    double gout24 = 0;
    double gout25 = 0;
    double gout26 = 0;
    double gout27 = 0;
    double gout28 = 0;
    double gout29 = 0;
    double gout30 = 0;
    double gout31 = 0;
    double gout32 = 0;
    double gout33 = 0;
    double gout34 = 0;
    double gout35 = 0;
    double xi = bas_x[ish];
    double yi = bas_y[ish];
    double zi = bas_z[ish];
    double xixj = xi - bas_x[jsh];
    double yiyj = yi - bas_y[jsh];
    double zizj = zi - bas_z[jsh];
    auto reduce = SegReduce<double>(igroup);
    ij += prim_ij;
    kl += prim_kl;
    double aij = a12[ij];
    double eij = e12[ij];
    double xij = x12[ij];
    double yij = y12[ij];
    double zij = z12[ij];
    double akl = a12[kl];
    double ekl = e12[kl];
    double xkl = x12[kl];
    double ykl = y12[kl];
    double zkl = z12[kl];
    double xijxkl = xij - xkl;
    double yijykl = yij - ykl;
    double zijzkl = zij - zkl;
    double aijkl = aij + akl;
    double a1 = aij * akl;
    double a0 = a1 / aijkl;
    double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
    double fac = norm * eij * ekl / (sqrt(aijkl) * a1);

    double rw[6];
    double root0, weight0;
    GINTrys_root<3>(x, rw);
    int irys;
    for (irys = 0; irys < 3; ++irys) {
        root0 = rw[irys];
        weight0 = rw[irys + 3];
        double u2 = a0 * root0;
        double tmp4 = .5 / (u2 * aijkl + a1);
        double b00 = u2 * tmp4;
        double tmp1 = 2 * b00;
        double tmp2 = tmp1 * akl;
        double b10 = b00 + tmp4 * akl;
        double c00x = xij - xi - tmp2 * xijxkl;
        double c00y = yij - yi - tmp2 * yijykl;
        double c00z = zij - zi - tmp2 * zijzkl;
        double g_0 = 1;
        double g_1 = c00x;
        double g_2 = c00x * c00x + b10;
        double g_3 = c00x + xixj;
        double g_4 = c00x * (c00x + xixj) + b10;
        double g_5 = c00x * (2 * b10 + g_2) + xixj * g_2;
        double g_6 = xixj * (xixj + c00x) + xixj * c00x + c00x * c00x + b10;
        double g_7 = xixj * (xixj * c00x + c00x * c00x + b10) + xixj * g_2 +
                     c00x * g_2 + 2 * b10 * c00x;
        double g_8 = xixj * (xixj * g_2 + c00x * g_2 + 2 * b10 * c00x) +
                     xixj * (c00x * g_2 + 2 * b10 * c00x) +
                     c00x * (c00x * g_2 + 2 * b10 * c00x) + 3 * b10 * g_2;
        double g_9 = 1;
        double g_10 = c00y;
        double g_11 = c00y * c00y + b10;
        double g_12 = c00y + yiyj;
        double g_13 = c00y * (c00y + yiyj) + b10;
        double g_14 = c00y * (2 * b10 + g_11) + yiyj * g_11;
        double g_15 = yiyj * (yiyj + c00y) + yiyj * c00y + c00y * c00y + b10;
        double g_16 = yiyj * (yiyj * c00y + c00y * c00y + b10) + yiyj * g_11 +
                      c00y * g_11 + 2 * b10 * c00y;
        double g_17 = yiyj * (yiyj * g_11 + c00y * g_11 + 2 * b10 * c00y) +
                      yiyj * (c00y * g_11 + 2 * b10 * c00y) +
                      c00y * (c00y * g_11 + 2 * b10 * c00y) + 3 * b10 * g_11;
        double g_18 = weight0 * fac;
        double g_19 = c00z * g_18;
        double g_20 = b10 * g_18 + c00z * g_19;
        double g_21 = g_18 * (c00z + zizj);
        double g_22 = b10 * g_18 + c00z * g_19 + zizj * g_19;
        double g_23 = 2 * b10 * g_19 + c00z * g_20 + zizj * g_20;
        double g_24 = zizj * (zizj * g_18 + c00z * g_18) + zizj * g_19 +
                      c00z * g_19 + b10 * g_18;
        double g_25 = zizj * (zizj * g_19 + c00z * g_19 + b10 * g_18) +
                      zizj * g_20 + c00z * g_20 + 2 * b10 * g_19;
        double g_26 = zizj * (zizj * g_20 + c00z * g_20 + 2 * b10 * g_19) +
                      zizj * (c00z * g_20 + 2 * b10 * g_19) +
                      c00z * (c00z * g_20 + 2 * b10 * g_19) + 3 * b10 * g_20;
        gout0 += g_8 * g_9 * g_18;
        gout1 += g_7 * g_10 * g_18;
        gout2 += g_7 * g_9 * g_19;
        gout3 += g_6 * g_11 * g_18;
        gout4 += g_6 * g_10 * g_19;
        gout5 += g_6 * g_9 * g_20;
        gout6 += g_5 * g_12 * g_18;
        gout7 += g_4 * g_13 * g_18;
        gout8 += g_4 * g_12 * g_19;
        gout9 += g_3 * g_14 * g_18;
        gout10 += g_3 * g_13 * g_19;
        gout11 += g_3 * g_12 * g_20;
        gout12 += g_5 * g_9 * g_21;
        gout13 += g_4 * g_10 * g_21;
        gout14 += g_4 * g_9 * g_22;
        gout15 += g_3 * g_11 * g_21;
        gout16 += g_3 * g_10 * g_22;
        gout17 += g_3 * g_9 * g_23;
        gout18 += g_2 * g_15 * g_18;
        gout19 += g_1 * g_16 * g_18;
        gout20 += g_1 * g_15 * g_19;
        gout21 += g_0 * g_17 * g_18;
        gout22 += g_0 * g_16 * g_19;
        gout23 += g_0 * g_15 * g_20;
        gout24 += g_2 * g_12 * g_21;
        gout25 += g_1 * g_13 * g_21;
        gout26 += g_1 * g_12 * g_22;
        gout27 += g_0 * g_14 * g_21;
        gout28 += g_0 * g_13 * g_22;
        gout29 += g_0 * g_12 * g_23;
        gout30 += g_2 * g_9 * g_24;
        gout31 += g_1 * g_10 * g_24;
        gout32 += g_1 * g_9 * g_25;
        gout33 += g_0 * g_11 * g_24;
        gout34 += g_0 * g_10 * g_25;
        gout35 += g_0 * g_9 * g_26;
    }
    double d_0, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, d_9;
    double d_10, d_11, d_12, d_13, d_14, d_15, d_16, d_17, d_18, d_19;
    double d_20, d_21, d_22, d_23, d_24, d_25, d_26, d_27, d_28, d_29;
    double d_30, d_31, d_32, d_33, d_34, d_35;
    int n_dm = jk.n_dm;
    int nao = jk.nao;
    size_t nao2 = nao * nao;
    double *__restrict__ dm = jk.dm;
    double *vj = jk.vj;
    double *vk = jk.vk;
    for (i_dm = 0; i_dm < n_dm; ++i_dm) {
        if (vj != NULL) {
            // ijkl,ij->kl
            d_0 = dm[(i0 + 0) + nao * (j0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (j0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (j0 + 0)];
            d_3 = dm[(i0 + 3) + nao * (j0 + 0)];
            d_4 = dm[(i0 + 4) + nao * (j0 + 0)];
            d_5 = dm[(i0 + 5) + nao * (j0 + 0)];
            d_6 = dm[(i0 + 0) + nao * (j0 + 1)];
            d_7 = dm[(i0 + 1) + nao * (j0 + 1)];
            d_8 = dm[(i0 + 2) + nao * (j0 + 1)];
            d_9 = dm[(i0 + 3) + nao * (j0 + 1)];
            d_10 = dm[(i0 + 4) + nao * (j0 + 1)];
            d_11 = dm[(i0 + 5) + nao * (j0 + 1)];
            d_12 = dm[(i0 + 0) + nao * (j0 + 2)];
            d_13 = dm[(i0 + 1) + nao * (j0 + 2)];
            d_14 = dm[(i0 + 2) + nao * (j0 + 2)];
            d_15 = dm[(i0 + 3) + nao * (j0 + 2)];
            d_16 = dm[(i0 + 4) + nao * (j0 + 2)];
            d_17 = dm[(i0 + 5) + nao * (j0 + 2)];
            d_18 = dm[(i0 + 0) + nao * (j0 + 3)];
            d_19 = dm[(i0 + 1) + nao * (j0 + 3)];
            d_20 = dm[(i0 + 2) + nao * (j0 + 3)];
            d_21 = dm[(i0 + 3) + nao * (j0 + 3)];
            d_22 = dm[(i0 + 4) + nao * (j0 + 3)];
            d_23 = dm[(i0 + 5) + nao * (j0 + 3)];
            d_24 = dm[(i0 + 0) + nao * (j0 + 4)];
            d_25 = dm[(i0 + 1) + nao * (j0 + 4)];
            d_26 = dm[(i0 + 2) + nao * (j0 + 4)];
            d_27 = dm[(i0 + 3) + nao * (j0 + 4)];
            d_28 = dm[(i0 + 4) + nao * (j0 + 4)];
            d_29 = dm[(i0 + 5) + nao * (j0 + 4)];
            d_30 = dm[(i0 + 0) + nao * (j0 + 5)];
            d_31 = dm[(i0 + 1) + nao * (j0 + 5)];
            d_32 = dm[(i0 + 2) + nao * (j0 + 5)];
            d_33 = dm[(i0 + 3) + nao * (j0 + 5)];
            d_34 = dm[(i0 + 4) + nao * (j0 + 5)];
            d_35 = dm[(i0 + 5) + nao * (j0 + 5)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout3 * d_3 +
                       gout4 * d_4 + gout5 * d_5 + gout6 * d_6 + gout7 * d_7 +
                       gout8 * d_8 + gout9 * d_9 + gout10 * d_10 +
                       gout11 * d_11 + gout12 * d_12 + gout13 * d_13 +
                       gout14 * d_14 + gout15 * d_15 + gout16 * d_16 +
                       gout17 * d_17 + gout18 * d_18 + gout19 * d_19 +
                       gout20 * d_20 + gout21 * d_21 + gout22 * d_22 +
                       gout23 * d_23 + gout24 * d_24 + gout25 * d_25 +
                       gout26 * d_26 + gout27 * d_27 + gout28 * d_28 +
                       gout29 * d_29 + gout30 * d_30 + gout31 * d_31 +
                       gout32 * d_32 + gout33 * d_33 + gout34 * d_34 +
                       gout35 * d_35,
                vj + (k0 + 0) + nao * (l0 + 0));
            // ijkl,kl->ij
            d_0 = dm[(k0 + 0) + nao * (l0 + 0)];
            reduce(gout0 * d_0, vj + (i0 + 0) + nao * (j0 + 0));
            reduce(gout1 * d_0, vj + (i0 + 1) + nao * (j0 + 0));
            reduce(gout2 * d_0, vj + (i0 + 2) + nao * (j0 + 0));
            reduce(gout3 * d_0, vj + (i0 + 3) + nao * (j0 + 0));
            reduce(gout4 * d_0, vj + (i0 + 4) + nao * (j0 + 0));
            reduce(gout5 * d_0, vj + (i0 + 5) + nao * (j0 + 0));
            reduce(gout6 * d_0, vj + (i0 + 0) + nao * (j0 + 1));
            reduce(gout7 * d_0, vj + (i0 + 1) + nao * (j0 + 1));
            reduce(gout8 * d_0, vj + (i0 + 2) + nao * (j0 + 1));
            reduce(gout9 * d_0, vj + (i0 + 3) + nao * (j0 + 1));
            reduce(gout10 * d_0, vj + (i0 + 4) + nao * (j0 + 1));
            reduce(gout11 * d_0, vj + (i0 + 5) + nao * (j0 + 1));
            reduce(gout12 * d_0, vj + (i0 + 0) + nao * (j0 + 2));
            reduce(gout13 * d_0, vj + (i0 + 1) + nao * (j0 + 2));
            reduce(gout14 * d_0, vj + (i0 + 2) + nao * (j0 + 2));
            reduce(gout15 * d_0, vj + (i0 + 3) + nao * (j0 + 2));
            reduce(gout16 * d_0, vj + (i0 + 4) + nao * (j0 + 2));
            reduce(gout17 * d_0, vj + (i0 + 5) + nao * (j0 + 2));
            reduce(gout18 * d_0, vj + (i0 + 0) + nao * (j0 + 3));
            reduce(gout19 * d_0, vj + (i0 + 1) + nao * (j0 + 3));
            reduce(gout20 * d_0, vj + (i0 + 2) + nao * (j0 + 3));
            reduce(gout21 * d_0, vj + (i0 + 3) + nao * (j0 + 3));
            reduce(gout22 * d_0, vj + (i0 + 4) + nao * (j0 + 3));
            reduce(gout23 * d_0, vj + (i0 + 5) + nao * (j0 + 3));
            reduce(gout24 * d_0, vj + (i0 + 0) + nao * (j0 + 4));
            reduce(gout25 * d_0, vj + (i0 + 1) + nao * (j0 + 4));
            reduce(gout26 * d_0, vj + (i0 + 2) + nao * (j0 + 4));
            reduce(gout27 * d_0, vj + (i0 + 3) + nao * (j0 + 4));
            reduce(gout28 * d_0, vj + (i0 + 4) + nao * (j0 + 4));
            reduce(gout29 * d_0, vj + (i0 + 5) + nao * (j0 + 4));
            reduce(gout30 * d_0, vj + (i0 + 0) + nao * (j0 + 5));
            reduce(gout31 * d_0, vj + (i0 + 1) + nao * (j0 + 5));
            reduce(gout32 * d_0, vj + (i0 + 2) + nao * (j0 + 5));
            reduce(gout33 * d_0, vj + (i0 + 3) + nao * (j0 + 5));
            reduce(gout34 * d_0, vj + (i0 + 4) + nao * (j0 + 5));
            reduce(gout35 * d_0, vj + (i0 + 5) + nao * (j0 + 5));
            vj += nao2;
        }
        if (vk != NULL) {
            // ijkl,jl->ik
            d_0 = dm[(j0 + 0) + nao * (l0 + 0)];
            d_1 = dm[(j0 + 1) + nao * (l0 + 0)];
            d_2 = dm[(j0 + 2) + nao * (l0 + 0)];
            d_3 = dm[(j0 + 3) + nao * (l0 + 0)];
            d_4 = dm[(j0 + 4) + nao * (l0 + 0)];
            d_5 = dm[(j0 + 5) + nao * (l0 + 0)];
            reduce(gout0 * d_0 + gout6 * d_1 + gout12 * d_2 + gout18 * d_3 +
                       gout24 * d_4 + gout30 * d_5,
                vk + (i0 + 0) + nao * (k0 + 0));
            reduce(gout1 * d_0 + gout7 * d_1 + gout13 * d_2 + gout19 * d_3 +
                       gout25 * d_4 + gout31 * d_5,
                vk + (i0 + 1) + nao * (k0 + 0));
            reduce(gout2 * d_0 + gout8 * d_1 + gout14 * d_2 + gout20 * d_3 +
                       gout26 * d_4 + gout32 * d_5,
                vk + (i0 + 2) + nao * (k0 + 0));
            reduce(gout3 * d_0 + gout9 * d_1 + gout15 * d_2 + gout21 * d_3 +
                       gout27 * d_4 + gout33 * d_5,
                vk + (i0 + 3) + nao * (k0 + 0));
            reduce(gout4 * d_0 + gout10 * d_1 + gout16 * d_2 + gout22 * d_3 +
                       gout28 * d_4 + gout34 * d_5,
                vk + (i0 + 4) + nao * (k0 + 0));
            reduce(gout5 * d_0 + gout11 * d_1 + gout17 * d_2 + gout23 * d_3 +
                       gout29 * d_4 + gout35 * d_5,
                vk + (i0 + 5) + nao * (k0 + 0));
            // ijkl,jk->il
            d_0 = dm[(j0 + 0) + nao * (k0 + 0)];
            d_1 = dm[(j0 + 1) + nao * (k0 + 0)];
            d_2 = dm[(j0 + 2) + nao * (k0 + 0)];
            d_3 = dm[(j0 + 3) + nao * (k0 + 0)];
            d_4 = dm[(j0 + 4) + nao * (k0 + 0)];
            d_5 = dm[(j0 + 5) + nao * (k0 + 0)];
            reduce(gout0 * d_0 + gout6 * d_1 + gout12 * d_2 + gout18 * d_3 +
                       gout24 * d_4 + gout30 * d_5,
                vk + (i0 + 0) + nao * (l0 + 0));
            reduce(gout1 * d_0 + gout7 * d_1 + gout13 * d_2 + gout19 * d_3 +
                       gout25 * d_4 + gout31 * d_5,
                vk + (i0 + 1) + nao * (l0 + 0));
            reduce(gout2 * d_0 + gout8 * d_1 + gout14 * d_2 + gout20 * d_3 +
                       gout26 * d_4 + gout32 * d_5,
                vk + (i0 + 2) + nao * (l0 + 0));
            reduce(gout3 * d_0 + gout9 * d_1 + gout15 * d_2 + gout21 * d_3 +
                       gout27 * d_4 + gout33 * d_5,
                vk + (i0 + 3) + nao * (l0 + 0));
            reduce(gout4 * d_0 + gout10 * d_1 + gout16 * d_2 + gout22 * d_3 +
                       gout28 * d_4 + gout34 * d_5,
                vk + (i0 + 4) + nao * (l0 + 0));
            reduce(gout5 * d_0 + gout11 * d_1 + gout17 * d_2 + gout23 * d_3 +
                       gout29 * d_4 + gout35 * d_5,
                vk + (i0 + 5) + nao * (l0 + 0));
            // ijkl,il->jk
            d_0 = dm[(i0 + 0) + nao * (l0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (l0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (l0 + 0)];
            d_3 = dm[(i0 + 3) + nao * (l0 + 0)];
            d_4 = dm[(i0 + 4) + nao * (l0 + 0)];
            d_5 = dm[(i0 + 5) + nao * (l0 + 0)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout3 * d_3 +
                       gout4 * d_4 + gout5 * d_5,
                vk + (j0 + 0) + nao * (k0 + 0));
            reduce(gout6 * d_0 + gout7 * d_1 + gout8 * d_2 + gout9 * d_3 +
                       gout10 * d_4 + gout11 * d_5,
                vk + (j0 + 1) + nao * (k0 + 0));
            reduce(gout12 * d_0 + gout13 * d_1 + gout14 * d_2 + gout15 * d_3 +
                       gout16 * d_4 + gout17 * d_5,
                vk + (j0 + 2) + nao * (k0 + 0));
            reduce(gout18 * d_0 + gout19 * d_1 + gout20 * d_2 + gout21 * d_3 +
                       gout22 * d_4 + gout23 * d_5,
                vk + (j0 + 3) + nao * (k0 + 0));
            reduce(gout24 * d_0 + gout25 * d_1 + gout26 * d_2 + gout27 * d_3 +
                       gout28 * d_4 + gout29 * d_5,
                vk + (j0 + 4) + nao * (k0 + 0));
            reduce(gout30 * d_0 + gout31 * d_1 + gout32 * d_2 + gout33 * d_3 +
                       gout34 * d_4 + gout35 * d_5,
                vk + (j0 + 5) + nao * (k0 + 0));
            // ijkl,ik->jl
            d_0 = dm[(i0 + 0) + nao * (k0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (k0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (k0 + 0)];
            d_3 = dm[(i0 + 3) + nao * (k0 + 0)];
            d_4 = dm[(i0 + 4) + nao * (k0 + 0)];
            d_5 = dm[(i0 + 5) + nao * (k0 + 0)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout3 * d_3 +
                       gout4 * d_4 + gout5 * d_5,
                vk + (j0 + 0) + nao * (l0 + 0));
            reduce(gout6 * d_0 + gout7 * d_1 + gout8 * d_2 + gout9 * d_3 +
                       gout10 * d_4 + gout11 * d_5,
                vk + (j0 + 1) + nao * (l0 + 0));
            reduce(gout12 * d_0 + gout13 * d_1 + gout14 * d_2 + gout15 * d_3 +
                       gout16 * d_4 + gout17 * d_5,
                vk + (j0 + 2) + nao * (l0 + 0));
            reduce(gout18 * d_0 + gout19 * d_1 + gout20 * d_2 + gout21 * d_3 +
                       gout22 * d_4 + gout23 * d_5,
                vk + (j0 + 3) + nao * (l0 + 0));
            reduce(gout24 * d_0 + gout25 * d_1 + gout26 * d_2 + gout27 * d_3 +
                       gout28 * d_4 + gout29 * d_5,
                vk + (j0 + 4) + nao * (l0 + 0));
            reduce(gout30 * d_0 + gout31 * d_1 + gout32 * d_2 + gout33 * d_3 +
                       gout34 * d_4 + gout35 * d_5,
                vk + (j0 + 5) + nao * (l0 + 0));
            vk += nao2;
        }
        dm += nao2;
    }
}

__global__ static void GINTint2e_jk_kernel2210(JKMatrix jk,
    BasisProdOffsets offsets, GINTEnvVars envs, BasisProdCache bpcache) {
    int ntasks_ij = offsets.ntasks_ij;
    long ntasks = ntasks_ij * offsets.ntasks_kl;
    long task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int igroup = nprim_ij * nprim_kl;
    ntasks *= igroup;
    if (task_ij >= ntasks)
        return;
    int kl = task_ij % nprim_kl;
    task_ij /= nprim_kl;
    int ij = task_ij % nprim_ij;
    task_ij /= nprim_ij;
    int task_kl = task_ij / ntasks_ij;
    task_ij = task_ij % ntasks_ij;

    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    if (bas_ij < bas_kl) {
        return;
    }
    double norm = envs.fac;
    if (bas_ij == bas_kl) {
        norm *= .5;
    }

    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *ao_loc = bpcache.ao_loc;
    int *bas_pair2bra = bpcache.bas_pair2bra;
    int *bas_pair2ket = bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    int k0 = ao_loc[ksh];
    int l0 = ao_loc[lsh];
    double *__restrict__ a12 = bpcache.a12;
    double *__restrict__ e12 = bpcache.e12;
    double *__restrict__ x12 = bpcache.x12;
    double *__restrict__ y12 = bpcache.y12;
    double *__restrict__ z12 = bpcache.z12;
    int i_dm;
    int nbas = bpcache.nbas;
    double *__restrict__ bas_x = bpcache.bas_coords;
    double *__restrict__ bas_y = bas_x + nbas;
    double *__restrict__ bas_z = bas_y + nbas;

    double gout0 = 0;
    double gout1 = 0;
    double gout2 = 0;
    double gout3 = 0;
    double gout4 = 0;
    double gout5 = 0;
    double gout6 = 0;
    double gout7 = 0;
    double gout8 = 0;
    double gout9 = 0;
    double gout10 = 0;
    double gout11 = 0;
    double gout12 = 0;
    double gout13 = 0;
    double gout14 = 0;
    double gout15 = 0;
    double gout16 = 0;
    double gout17 = 0;
    double gout18 = 0;
    double gout19 = 0;
    double gout20 = 0;
    double gout21 = 0;
    double gout22 = 0;
    double gout23 = 0;
    double gout24 = 0;
    double gout25 = 0;
    double gout26 = 0;
    double gout27 = 0;
    double gout28 = 0;
    double gout29 = 0;
    double gout30 = 0;
    double gout31 = 0;
    double gout32 = 0;
    double gout33 = 0;
    double gout34 = 0;
    double gout35 = 0;
    double gout36 = 0;
    double gout37 = 0;
    double gout38 = 0;
    double gout39 = 0;
    double gout40 = 0;
    double gout41 = 0;
    double gout42 = 0;
    double gout43 = 0;
    double gout44 = 0;
    double gout45 = 0;
    double gout46 = 0;
    double gout47 = 0;
    double gout48 = 0;
    double gout49 = 0;
    double gout50 = 0;
    double gout51 = 0;
    double gout52 = 0;
    double gout53 = 0;
    double gout54 = 0;
    double gout55 = 0;
    double gout56 = 0;
    double gout57 = 0;
    double gout58 = 0;
    double gout59 = 0;
    double gout60 = 0;
    double gout61 = 0;
    double gout62 = 0;
    double gout63 = 0;
    double gout64 = 0;
    double gout65 = 0;
    double gout66 = 0;
    double gout67 = 0;
    double gout68 = 0;
    double gout69 = 0;
    double gout70 = 0;
    double gout71 = 0;
    double gout72 = 0;
    double gout73 = 0;
    double gout74 = 0;
    double gout75 = 0;
    double gout76 = 0;
    double gout77 = 0;
    double gout78 = 0;
    double gout79 = 0;
    double gout80 = 0;
    double gout81 = 0;
    double gout82 = 0;
    double gout83 = 0;
    double gout84 = 0;
    double gout85 = 0;
    double gout86 = 0;
    double gout87 = 0;
    double gout88 = 0;
    double gout89 = 0;
    double gout90 = 0;
    double gout91 = 0;
    double gout92 = 0;
    double gout93 = 0;
    double gout94 = 0;
    double gout95 = 0;
    double gout96 = 0;
    double gout97 = 0;
    double gout98 = 0;
    double gout99 = 0;
    double gout100 = 0;
    double gout101 = 0;
    double gout102 = 0;
    double gout103 = 0;
    double gout104 = 0;
    double gout105 = 0;
    double gout106 = 0;
    double gout107 = 0;
    double xi = bas_x[ish];
    double yi = bas_y[ish];
    double zi = bas_z[ish];
    double xixj = xi - bas_x[jsh];
    double yiyj = yi - bas_y[jsh];
    double zizj = zi - bas_z[jsh];
    double xk = bas_x[ksh];
    double yk = bas_y[ksh];
    double zk = bas_z[ksh];
    auto reduce = SegReduce<double>(igroup);
    ij += prim_ij;
    kl += prim_kl;
    double aij = a12[ij];
    double eij = e12[ij];
    double xij = x12[ij];
    double yij = y12[ij];
    double zij = z12[ij];
    double akl = a12[kl];
    double ekl = e12[kl];
    double xkl = x12[kl];
    double ykl = y12[kl];
    double zkl = z12[kl];
    double xijxkl = xij - xkl;
    double yijykl = yij - ykl;
    double zijzkl = zij - zkl;
    double aijkl = aij + akl;
    double a1 = aij * akl;
    double a0 = a1 / aijkl;
    double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
    double fac = norm * eij * ekl / (sqrt(aijkl) * a1);

    double rw[6];
    double root0, weight0;
    GINTrys_root<3>(x, rw);
    int irys;
    for (irys = 0; irys < 3; ++irys) {
        root0 = rw[irys];
        weight0 = rw[irys + 3];
        double u2 = a0 * root0;
        double tmp4 = .5 / (u2 * aijkl + a1);
        double b00 = u2 * tmp4;
        double tmp1 = 2 * b00;
        double tmp2 = tmp1 * akl;
        double b10 = b00 + tmp4 * akl;
        double c00x = xij - xi - tmp2 * xijxkl;
        double c00y = yij - yi - tmp2 * yijykl;
        double c00z = zij - zi - tmp2 * zijzkl;
        double tmp3 = tmp1 * aij;
        double c0px = xkl - xk + tmp3 * xijxkl;
        double c0py = ykl - yk + tmp3 * yijykl;
        double c0pz = zkl - zk + tmp3 * zijzkl;
        double g_0 = 1;
        double g_1 = c00x;
        double g_2 = c00x * c00x + b10;
        double g_3 = c00x + xixj;
        double g_4 = c00x * (c00x + xixj) + b10;
        double g_5 = c00x * (2 * b10 + g_2) + xixj * g_2;
        double g_6 = xixj * (xixj + c00x) + xixj * c00x + c00x * c00x + b10;
        double g_7 = xixj * (xixj * c00x + c00x * c00x + b10) + xixj * g_2 +
                     c00x * g_2 + 2 * b10 * c00x;
        double g_8 = xixj * (xixj * g_2 + c00x * g_2 + 2 * b10 * c00x) +
                     xixj * (c00x * g_2 + 2 * b10 * c00x) +
                     c00x * (c00x * g_2 + 2 * b10 * c00x) + 3 * b10 * g_2;
        double g_9 = c0px;
        double g_10 = c0px * c00x + b00;
        double g_11 = b00 * c00x + b10 * c0px + c00x * g_10;
        double g_12 = c0px * (c00x + xixj) + b00;
        double g_13 = b00 * c00x + b10 * c0px + c00x * g_10 + xixj * g_10;
        double g_14 = 2 * b10 * g_10 + b00 * g_2 + c00x * g_11 + xixj * g_11;
        double g_15 = xixj * (xixj * c0px + c0px * c00x + b00) + xixj * g_10 +
                      c00x * g_10 + b10 * c0px + b00 * c00x;
        double g_16 =
            xixj * (xixj * g_10 + c00x * g_10 + b10 * c0px + b00 * c00x) +
            xixj * g_11 + c00x * g_11 + 2 * b10 * g_10 + b00 * g_2;
        double g_17 =
            xixj * (xixj * g_11 + c00x * g_11 + 2 * b10 * g_10 + b00 * g_2) +
            xixj * (c00x * g_11 + 2 * b10 * g_10 + b00 * g_2) +
            c00x * (c00x * g_11 + 2 * b10 * g_10 + b00 * g_2) +
            3 * b10 * g_11 + b00 * (c00x * g_2 + 2 * b10 * c00x);
        double g_18 = 1;
        double g_19 = c00y;
        double g_20 = c00y * c00y + b10;
        double g_21 = c00y + yiyj;
        double g_22 = c00y * (c00y + yiyj) + b10;
        double g_23 = c00y * (2 * b10 + g_20) + yiyj * g_20;
        double g_24 = yiyj * (yiyj + c00y) + yiyj * c00y + c00y * c00y + b10;
        double g_25 = yiyj * (yiyj * c00y + c00y * c00y + b10) + yiyj * g_20 +
                      c00y * g_20 + 2 * b10 * c00y;
        double g_26 = yiyj * (yiyj * g_20 + c00y * g_20 + 2 * b10 * c00y) +
                      yiyj * (c00y * g_20 + 2 * b10 * c00y) +
                      c00y * (c00y * g_20 + 2 * b10 * c00y) + 3 * b10 * g_20;
        double g_27 = c0py;
        double g_28 = c0py * c00y + b00;
        double g_29 = b00 * c00y + b10 * c0py + c00y * g_28;
        double g_30 = c0py * (c00y + yiyj) + b00;
        double g_31 = b00 * c00y + b10 * c0py + c00y * g_28 + yiyj * g_28;
        double g_32 = 2 * b10 * g_28 + b00 * g_20 + c00y * g_29 + yiyj * g_29;
        double g_33 = yiyj * (yiyj * c0py + c0py * c00y + b00) + yiyj * g_28 +
                      c00y * g_28 + b10 * c0py + b00 * c00y;
        double g_34 =
            yiyj * (yiyj * g_28 + c00y * g_28 + b10 * c0py + b00 * c00y) +
            yiyj * g_29 + c00y * g_29 + 2 * b10 * g_28 + b00 * g_20;
        double g_35 =
            yiyj * (yiyj * g_29 + c00y * g_29 + 2 * b10 * g_28 + b00 * g_20) +
            yiyj * (c00y * g_29 + 2 * b10 * g_28 + b00 * g_20) +
            c00y * (c00y * g_29 + 2 * b10 * g_28 + b00 * g_20) +
            3 * b10 * g_29 + b00 * (c00y * g_20 + 2 * b10 * c00y);
        double g_36 = weight0 * fac;
        double g_37 = c00z * g_36;
        double g_38 = b10 * g_36 + c00z * g_37;
        double g_39 = g_36 * (c00z + zizj);
        double g_40 = b10 * g_36 + c00z * g_37 + zizj * g_37;
        double g_41 = 2 * b10 * g_37 + c00z * g_38 + zizj * g_38;
        double g_42 = zizj * (zizj * g_36 + c00z * g_36) + zizj * g_37 +
                      c00z * g_37 + b10 * g_36;
        double g_43 = zizj * (zizj * g_37 + c00z * g_37 + b10 * g_36) +
                      zizj * g_38 + c00z * g_38 + 2 * b10 * g_37;
        double g_44 = zizj * (zizj * g_38 + c00z * g_38 + 2 * b10 * g_37) +
                      zizj * (c00z * g_38 + 2 * b10 * g_37) +
                      c00z * (c00z * g_38 + 2 * b10 * g_37) + 3 * b10 * g_38;
        double g_45 = c0pz * g_36;
        double g_46 = b00 * g_36 + c0pz * g_37;
        double g_47 = b00 * g_37 + b10 * g_45 + c00z * g_46;
        double g_48 = b00 * g_36 + c0pz * g_37 + zizj * g_45;
        double g_49 = b00 * g_37 + b10 * g_45 + c00z * g_46 + zizj * g_46;
        double g_50 = 2 * b10 * g_46 + b00 * g_38 + c00z * g_47 + zizj * g_47;
        double g_51 = zizj * (zizj * g_45 + c0pz * g_37 + b00 * g_36) +
                      zizj * g_46 + c00z * g_46 + b10 * g_45 + b00 * g_37;
        double g_52 =
            zizj * (zizj * g_46 + c00z * g_46 + b10 * g_45 + b00 * g_37) +
            zizj * g_47 + c00z * g_47 + 2 * b10 * g_46 + b00 * g_38;
        double g_53 =
            zizj * (zizj * g_47 + c00z * g_47 + 2 * b10 * g_46 + b00 * g_38) +
            zizj * (c00z * g_47 + 2 * b10 * g_46 + b00 * g_38) +
            c00z * (c00z * g_47 + 2 * b10 * g_46 + b00 * g_38) +
            3 * b10 * g_47 + b00 * (c00z * g_38 + 2 * b10 * g_37);
        gout0 += g_17 * g_18 * g_36;
        gout1 += g_16 * g_19 * g_36;
        gout2 += g_16 * g_18 * g_37;
        gout3 += g_15 * g_20 * g_36;
        gout4 += g_15 * g_19 * g_37;
        gout5 += g_15 * g_18 * g_38;
        gout6 += g_14 * g_21 * g_36;
        gout7 += g_13 * g_22 * g_36;
        gout8 += g_13 * g_21 * g_37;
        gout9 += g_12 * g_23 * g_36;
        gout10 += g_12 * g_22 * g_37;
        gout11 += g_12 * g_21 * g_38;
        gout12 += g_14 * g_18 * g_39;
        gout13 += g_13 * g_19 * g_39;
        gout14 += g_13 * g_18 * g_40;
        gout15 += g_12 * g_20 * g_39;
        gout16 += g_12 * g_19 * g_40;
        gout17 += g_12 * g_18 * g_41;
        gout18 += g_11 * g_24 * g_36;
        gout19 += g_10 * g_25 * g_36;
        gout20 += g_10 * g_24 * g_37;
        gout21 += g_9 * g_26 * g_36;
        gout22 += g_9 * g_25 * g_37;
        gout23 += g_9 * g_24 * g_38;
        gout24 += g_11 * g_21 * g_39;
        gout25 += g_10 * g_22 * g_39;
        gout26 += g_10 * g_21 * g_40;
        gout27 += g_9 * g_23 * g_39;
        gout28 += g_9 * g_22 * g_40;
        gout29 += g_9 * g_21 * g_41;
        gout30 += g_11 * g_18 * g_42;
        gout31 += g_10 * g_19 * g_42;
        gout32 += g_10 * g_18 * g_43;
        gout33 += g_9 * g_20 * g_42;
        gout34 += g_9 * g_19 * g_43;
        gout35 += g_9 * g_18 * g_44;
        gout36 += g_8 * g_27 * g_36;
        gout37 += g_7 * g_28 * g_36;
        gout38 += g_7 * g_27 * g_37;
        gout39 += g_6 * g_29 * g_36;
        gout40 += g_6 * g_28 * g_37;
        gout41 += g_6 * g_27 * g_38;
        gout42 += g_5 * g_30 * g_36;
        gout43 += g_4 * g_31 * g_36;
        gout44 += g_4 * g_30 * g_37;
        gout45 += g_3 * g_32 * g_36;
        gout46 += g_3 * g_31 * g_37;
        gout47 += g_3 * g_30 * g_38;
        gout48 += g_5 * g_27 * g_39;
        gout49 += g_4 * g_28 * g_39;
        gout50 += g_4 * g_27 * g_40;
        gout51 += g_3 * g_29 * g_39;
        gout52 += g_3 * g_28 * g_40;
        gout53 += g_3 * g_27 * g_41;
        gout54 += g_2 * g_33 * g_36;
        gout55 += g_1 * g_34 * g_36;
        gout56 += g_1 * g_33 * g_37;
        gout57 += g_0 * g_35 * g_36;
        gout58 += g_0 * g_34 * g_37;
        gout59 += g_0 * g_33 * g_38;
        gout60 += g_2 * g_30 * g_39;
        gout61 += g_1 * g_31 * g_39;
        gout62 += g_1 * g_30 * g_40;
        gout63 += g_0 * g_32 * g_39;
        gout64 += g_0 * g_31 * g_40;
        gout65 += g_0 * g_30 * g_41;
        gout66 += g_2 * g_27 * g_42;
        gout67 += g_1 * g_28 * g_42;
        gout68 += g_1 * g_27 * g_43;
        gout69 += g_0 * g_29 * g_42;
        gout70 += g_0 * g_28 * g_43;
        gout71 += g_0 * g_27 * g_44;
        gout72 += g_8 * g_18 * g_45;
        gout73 += g_7 * g_19 * g_45;
        gout74 += g_7 * g_18 * g_46;
        gout75 += g_6 * g_20 * g_45;
        gout76 += g_6 * g_19 * g_46;
        gout77 += g_6 * g_18 * g_47;
        gout78 += g_5 * g_21 * g_45;
        gout79 += g_4 * g_22 * g_45;
        gout80 += g_4 * g_21 * g_46;
        gout81 += g_3 * g_23 * g_45;
        gout82 += g_3 * g_22 * g_46;
        gout83 += g_3 * g_21 * g_47;
        gout84 += g_5 * g_18 * g_48;
        gout85 += g_4 * g_19 * g_48;
        gout86 += g_4 * g_18 * g_49;
        gout87 += g_3 * g_20 * g_48;
        gout88 += g_3 * g_19 * g_49;
        gout89 += g_3 * g_18 * g_50;
        gout90 += g_2 * g_24 * g_45;
        gout91 += g_1 * g_25 * g_45;
        gout92 += g_1 * g_24 * g_46;
        gout93 += g_0 * g_26 * g_45;
        gout94 += g_0 * g_25 * g_46;
        gout95 += g_0 * g_24 * g_47;
        gout96 += g_2 * g_21 * g_48;
        gout97 += g_1 * g_22 * g_48;
        gout98 += g_1 * g_21 * g_49;
        gout99 += g_0 * g_23 * g_48;
        gout100 += g_0 * g_22 * g_49;
        gout101 += g_0 * g_21 * g_50;
        gout102 += g_2 * g_18 * g_51;
        gout103 += g_1 * g_19 * g_51;
        gout104 += g_1 * g_18 * g_52;
        gout105 += g_0 * g_20 * g_51;
        gout106 += g_0 * g_19 * g_52;
        gout107 += g_0 * g_18 * g_53;
    }
    double d_0, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, d_9;
    double d_10, d_11, d_12, d_13, d_14, d_15, d_16, d_17, d_18, d_19;
    double d_20, d_21, d_22, d_23, d_24, d_25, d_26, d_27, d_28, d_29;
    double d_30, d_31, d_32, d_33, d_34, d_35;
    int n_dm = jk.n_dm;
    int nao = jk.nao;
    size_t nao2 = nao * nao;
    double *__restrict__ dm = jk.dm;
    double *vj = jk.vj;
    double *vk = jk.vk;
    for (i_dm = 0; i_dm < n_dm; ++i_dm) {
        if (vj != NULL) {
            // ijkl,ij->kl
            d_0 = dm[(i0 + 0) + nao * (j0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (j0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (j0 + 0)];
            d_3 = dm[(i0 + 3) + nao * (j0 + 0)];
            d_4 = dm[(i0 + 4) + nao * (j0 + 0)];
            d_5 = dm[(i0 + 5) + nao * (j0 + 0)];
            d_6 = dm[(i0 + 0) + nao * (j0 + 1)];
            d_7 = dm[(i0 + 1) + nao * (j0 + 1)];
            d_8 = dm[(i0 + 2) + nao * (j0 + 1)];
            d_9 = dm[(i0 + 3) + nao * (j0 + 1)];
            d_10 = dm[(i0 + 4) + nao * (j0 + 1)];
            d_11 = dm[(i0 + 5) + nao * (j0 + 1)];
            d_12 = dm[(i0 + 0) + nao * (j0 + 2)];
            d_13 = dm[(i0 + 1) + nao * (j0 + 2)];
            d_14 = dm[(i0 + 2) + nao * (j0 + 2)];
            d_15 = dm[(i0 + 3) + nao * (j0 + 2)];
            d_16 = dm[(i0 + 4) + nao * (j0 + 2)];
            d_17 = dm[(i0 + 5) + nao * (j0 + 2)];
            d_18 = dm[(i0 + 0) + nao * (j0 + 3)];
            d_19 = dm[(i0 + 1) + nao * (j0 + 3)];
            d_20 = dm[(i0 + 2) + nao * (j0 + 3)];
            d_21 = dm[(i0 + 3) + nao * (j0 + 3)];
            d_22 = dm[(i0 + 4) + nao * (j0 + 3)];
            d_23 = dm[(i0 + 5) + nao * (j0 + 3)];
            d_24 = dm[(i0 + 0) + nao * (j0 + 4)];
            d_25 = dm[(i0 + 1) + nao * (j0 + 4)];
            d_26 = dm[(i0 + 2) + nao * (j0 + 4)];
            d_27 = dm[(i0 + 3) + nao * (j0 + 4)];
            d_28 = dm[(i0 + 4) + nao * (j0 + 4)];
            d_29 = dm[(i0 + 5) + nao * (j0 + 4)];
            d_30 = dm[(i0 + 0) + nao * (j0 + 5)];
            d_31 = dm[(i0 + 1) + nao * (j0 + 5)];
            d_32 = dm[(i0 + 2) + nao * (j0 + 5)];
            d_33 = dm[(i0 + 3) + nao * (j0 + 5)];
            d_34 = dm[(i0 + 4) + nao * (j0 + 5)];
            d_35 = dm[(i0 + 5) + nao * (j0 + 5)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout3 * d_3 +
                       gout4 * d_4 + gout5 * d_5 + gout6 * d_6 + gout7 * d_7 +
                       gout8 * d_8 + gout9 * d_9 + gout10 * d_10 +
                       gout11 * d_11 + gout12 * d_12 + gout13 * d_13 +
                       gout14 * d_14 + gout15 * d_15 + gout16 * d_16 +
                       gout17 * d_17 + gout18 * d_18 + gout19 * d_19 +
                       gout20 * d_20 + gout21 * d_21 + gout22 * d_22 +
                       gout23 * d_23 + gout24 * d_24 + gout25 * d_25 +
                       gout26 * d_26 + gout27 * d_27 + gout28 * d_28 +
                       gout29 * d_29 + gout30 * d_30 + gout31 * d_31 +
                       gout32 * d_32 + gout33 * d_33 + gout34 * d_34 +
                       gout35 * d_35,
                vj + (k0 + 0) + nao * (l0 + 0));
            reduce(gout36 * d_0 + gout37 * d_1 + gout38 * d_2 + gout39 * d_3 +
                       gout40 * d_4 + gout41 * d_5 + gout42 * d_6 +
                       gout43 * d_7 + gout44 * d_8 + gout45 * d_9 +
                       gout46 * d_10 + gout47 * d_11 + gout48 * d_12 +
                       gout49 * d_13 + gout50 * d_14 + gout51 * d_15 +
                       gout52 * d_16 + gout53 * d_17 + gout54 * d_18 +
                       gout55 * d_19 + gout56 * d_20 + gout57 * d_21 +
                       gout58 * d_22 + gout59 * d_23 + gout60 * d_24 +
                       gout61 * d_25 + gout62 * d_26 + gout63 * d_27 +
                       gout64 * d_28 + gout65 * d_29 + gout66 * d_30 +
                       gout67 * d_31 + gout68 * d_32 + gout69 * d_33 +
                       gout70 * d_34 + gout71 * d_35,
                vj + (k0 + 1) + nao * (l0 + 0));
            reduce(gout72 * d_0 + gout73 * d_1 + gout74 * d_2 + gout75 * d_3 +
                       gout76 * d_4 + gout77 * d_5 + gout78 * d_6 +
                       gout79 * d_7 + gout80 * d_8 + gout81 * d_9 +
                       gout82 * d_10 + gout83 * d_11 + gout84 * d_12 +
                       gout85 * d_13 + gout86 * d_14 + gout87 * d_15 +
                       gout88 * d_16 + gout89 * d_17 + gout90 * d_18 +
                       gout91 * d_19 + gout92 * d_20 + gout93 * d_21 +
                       gout94 * d_22 + gout95 * d_23 + gout96 * d_24 +
                       gout97 * d_25 + gout98 * d_26 + gout99 * d_27 +
                       gout100 * d_28 + gout101 * d_29 + gout102 * d_30 +
                       gout103 * d_31 + gout104 * d_32 + gout105 * d_33 +
                       gout106 * d_34 + gout107 * d_35,
                vj + (k0 + 2) + nao * (l0 + 0));
            // ijkl,kl->ij
            d_0 = dm[(k0 + 0) + nao * (l0 + 0)];
            d_1 = dm[(k0 + 1) + nao * (l0 + 0)];
            d_2 = dm[(k0 + 2) + nao * (l0 + 0)];
            reduce(gout0 * d_0 + gout36 * d_1 + gout72 * d_2,
                vj + (i0 + 0) + nao * (j0 + 0));
            reduce(gout1 * d_0 + gout37 * d_1 + gout73 * d_2,
                vj + (i0 + 1) + nao * (j0 + 0));
            reduce(gout2 * d_0 + gout38 * d_1 + gout74 * d_2,
                vj + (i0 + 2) + nao * (j0 + 0));
            reduce(gout3 * d_0 + gout39 * d_1 + gout75 * d_2,
                vj + (i0 + 3) + nao * (j0 + 0));
            reduce(gout4 * d_0 + gout40 * d_1 + gout76 * d_2,
                vj + (i0 + 4) + nao * (j0 + 0));
            reduce(gout5 * d_0 + gout41 * d_1 + gout77 * d_2,
                vj + (i0 + 5) + nao * (j0 + 0));
            reduce(gout6 * d_0 + gout42 * d_1 + gout78 * d_2,
                vj + (i0 + 0) + nao * (j0 + 1));
            reduce(gout7 * d_0 + gout43 * d_1 + gout79 * d_2,
                vj + (i0 + 1) + nao * (j0 + 1));
            reduce(gout8 * d_0 + gout44 * d_1 + gout80 * d_2,
                vj + (i0 + 2) + nao * (j0 + 1));
            reduce(gout9 * d_0 + gout45 * d_1 + gout81 * d_2,
                vj + (i0 + 3) + nao * (j0 + 1));
            reduce(gout10 * d_0 + gout46 * d_1 + gout82 * d_2,
                vj + (i0 + 4) + nao * (j0 + 1));
            reduce(gout11 * d_0 + gout47 * d_1 + gout83 * d_2,
                vj + (i0 + 5) + nao * (j0 + 1));
            reduce(gout12 * d_0 + gout48 * d_1 + gout84 * d_2,
                vj + (i0 + 0) + nao * (j0 + 2));
            reduce(gout13 * d_0 + gout49 * d_1 + gout85 * d_2,
                vj + (i0 + 1) + nao * (j0 + 2));
            reduce(gout14 * d_0 + gout50 * d_1 + gout86 * d_2,
                vj + (i0 + 2) + nao * (j0 + 2));
            reduce(gout15 * d_0 + gout51 * d_1 + gout87 * d_2,
                vj + (i0 + 3) + nao * (j0 + 2));
            reduce(gout16 * d_0 + gout52 * d_1 + gout88 * d_2,
                vj + (i0 + 4) + nao * (j0 + 2));
            reduce(gout17 * d_0 + gout53 * d_1 + gout89 * d_2,
                vj + (i0 + 5) + nao * (j0 + 2));
            reduce(gout18 * d_0 + gout54 * d_1 + gout90 * d_2,
                vj + (i0 + 0) + nao * (j0 + 3));
            reduce(gout19 * d_0 + gout55 * d_1 + gout91 * d_2,
                vj + (i0 + 1) + nao * (j0 + 3));
            reduce(gout20 * d_0 + gout56 * d_1 + gout92 * d_2,
                vj + (i0 + 2) + nao * (j0 + 3));
            reduce(gout21 * d_0 + gout57 * d_1 + gout93 * d_2,
                vj + (i0 + 3) + nao * (j0 + 3));
            reduce(gout22 * d_0 + gout58 * d_1 + gout94 * d_2,
                vj + (i0 + 4) + nao * (j0 + 3));
            reduce(gout23 * d_0 + gout59 * d_1 + gout95 * d_2,
                vj + (i0 + 5) + nao * (j0 + 3));
            reduce(gout24 * d_0 + gout60 * d_1 + gout96 * d_2,
                vj + (i0 + 0) + nao * (j0 + 4));
            reduce(gout25 * d_0 + gout61 * d_1 + gout97 * d_2,
                vj + (i0 + 1) + nao * (j0 + 4));
            reduce(gout26 * d_0 + gout62 * d_1 + gout98 * d_2,
                vj + (i0 + 2) + nao * (j0 + 4));
            reduce(gout27 * d_0 + gout63 * d_1 + gout99 * d_2,
                vj + (i0 + 3) + nao * (j0 + 4));
            reduce(gout28 * d_0 + gout64 * d_1 + gout100 * d_2,
                vj + (i0 + 4) + nao * (j0 + 4));
            reduce(gout29 * d_0 + gout65 * d_1 + gout101 * d_2,
                vj + (i0 + 5) + nao * (j0 + 4));
            reduce(gout30 * d_0 + gout66 * d_1 + gout102 * d_2,
                vj + (i0 + 0) + nao * (j0 + 5));
            reduce(gout31 * d_0 + gout67 * d_1 + gout103 * d_2,
                vj + (i0 + 1) + nao * (j0 + 5));
            reduce(gout32 * d_0 + gout68 * d_1 + gout104 * d_2,
                vj + (i0 + 2) + nao * (j0 + 5));
            reduce(gout33 * d_0 + gout69 * d_1 + gout105 * d_2,
                vj + (i0 + 3) + nao * (j0 + 5));
            reduce(gout34 * d_0 + gout70 * d_1 + gout106 * d_2,
                vj + (i0 + 4) + nao * (j0 + 5));
            reduce(gout35 * d_0 + gout71 * d_1 + gout107 * d_2,
                vj + (i0 + 5) + nao * (j0 + 5));
            vj += nao2;
        }
        if (vk != NULL) {
            // ijkl,jl->ik
            d_0 = dm[(j0 + 0) + nao * (l0 + 0)];
            d_1 = dm[(j0 + 1) + nao * (l0 + 0)];
            d_2 = dm[(j0 + 2) + nao * (l0 + 0)];
            d_3 = dm[(j0 + 3) + nao * (l0 + 0)];
            d_4 = dm[(j0 + 4) + nao * (l0 + 0)];
            d_5 = dm[(j0 + 5) + nao * (l0 + 0)];
            reduce(gout0 * d_0 + gout6 * d_1 + gout12 * d_2 + gout18 * d_3 +
                       gout24 * d_4 + gout30 * d_5,
                vk + (i0 + 0) + nao * (k0 + 0));
            reduce(gout1 * d_0 + gout7 * d_1 + gout13 * d_2 + gout19 * d_3 +
                       gout25 * d_4 + gout31 * d_5,
                vk + (i0 + 1) + nao * (k0 + 0));
            reduce(gout2 * d_0 + gout8 * d_1 + gout14 * d_2 + gout20 * d_3 +
                       gout26 * d_4 + gout32 * d_5,
                vk + (i0 + 2) + nao * (k0 + 0));
            reduce(gout3 * d_0 + gout9 * d_1 + gout15 * d_2 + gout21 * d_3 +
                       gout27 * d_4 + gout33 * d_5,
                vk + (i0 + 3) + nao * (k0 + 0));
            reduce(gout4 * d_0 + gout10 * d_1 + gout16 * d_2 + gout22 * d_3 +
                       gout28 * d_4 + gout34 * d_5,
                vk + (i0 + 4) + nao * (k0 + 0));
            reduce(gout5 * d_0 + gout11 * d_1 + gout17 * d_2 + gout23 * d_3 +
                       gout29 * d_4 + gout35 * d_5,
                vk + (i0 + 5) + nao * (k0 + 0));
            reduce(gout36 * d_0 + gout42 * d_1 + gout48 * d_2 + gout54 * d_3 +
                       gout60 * d_4 + gout66 * d_5,
                vk + (i0 + 0) + nao * (k0 + 1));
            reduce(gout37 * d_0 + gout43 * d_1 + gout49 * d_2 + gout55 * d_3 +
                       gout61 * d_4 + gout67 * d_5,
                vk + (i0 + 1) + nao * (k0 + 1));
            reduce(gout38 * d_0 + gout44 * d_1 + gout50 * d_2 + gout56 * d_3 +
                       gout62 * d_4 + gout68 * d_5,
                vk + (i0 + 2) + nao * (k0 + 1));
            reduce(gout39 * d_0 + gout45 * d_1 + gout51 * d_2 + gout57 * d_3 +
                       gout63 * d_4 + gout69 * d_5,
                vk + (i0 + 3) + nao * (k0 + 1));
            reduce(gout40 * d_0 + gout46 * d_1 + gout52 * d_2 + gout58 * d_3 +
                       gout64 * d_4 + gout70 * d_5,
                vk + (i0 + 4) + nao * (k0 + 1));
            reduce(gout41 * d_0 + gout47 * d_1 + gout53 * d_2 + gout59 * d_3 +
                       gout65 * d_4 + gout71 * d_5,
                vk + (i0 + 5) + nao * (k0 + 1));
            reduce(gout72 * d_0 + gout78 * d_1 + gout84 * d_2 + gout90 * d_3 +
                       gout96 * d_4 + gout102 * d_5,
                vk + (i0 + 0) + nao * (k0 + 2));
            reduce(gout73 * d_0 + gout79 * d_1 + gout85 * d_2 + gout91 * d_3 +
                       gout97 * d_4 + gout103 * d_5,
                vk + (i0 + 1) + nao * (k0 + 2));
            reduce(gout74 * d_0 + gout80 * d_1 + gout86 * d_2 + gout92 * d_3 +
                       gout98 * d_4 + gout104 * d_5,
                vk + (i0 + 2) + nao * (k0 + 2));
            reduce(gout75 * d_0 + gout81 * d_1 + gout87 * d_2 + gout93 * d_3 +
                       gout99 * d_4 + gout105 * d_5,
                vk + (i0 + 3) + nao * (k0 + 2));
            reduce(gout76 * d_0 + gout82 * d_1 + gout88 * d_2 + gout94 * d_3 +
                       gout100 * d_4 + gout106 * d_5,
                vk + (i0 + 4) + nao * (k0 + 2));
            reduce(gout77 * d_0 + gout83 * d_1 + gout89 * d_2 + gout95 * d_3 +
                       gout101 * d_4 + gout107 * d_5,
                vk + (i0 + 5) + nao * (k0 + 2));
            // ijkl,jk->il
            d_0 = dm[(j0 + 0) + nao * (k0 + 0)];
            d_1 = dm[(j0 + 1) + nao * (k0 + 0)];
            d_2 = dm[(j0 + 2) + nao * (k0 + 0)];
            d_3 = dm[(j0 + 3) + nao * (k0 + 0)];
            d_4 = dm[(j0 + 4) + nao * (k0 + 0)];
            d_5 = dm[(j0 + 5) + nao * (k0 + 0)];
            d_6 = dm[(j0 + 0) + nao * (k0 + 1)];
            d_7 = dm[(j0 + 1) + nao * (k0 + 1)];
            d_8 = dm[(j0 + 2) + nao * (k0 + 1)];
            d_9 = dm[(j0 + 3) + nao * (k0 + 1)];
            d_10 = dm[(j0 + 4) + nao * (k0 + 1)];
            d_11 = dm[(j0 + 5) + nao * (k0 + 1)];
            d_12 = dm[(j0 + 0) + nao * (k0 + 2)];
            d_13 = dm[(j0 + 1) + nao * (k0 + 2)];
            d_14 = dm[(j0 + 2) + nao * (k0 + 2)];
            d_15 = dm[(j0 + 3) + nao * (k0 + 2)];
            d_16 = dm[(j0 + 4) + nao * (k0 + 2)];
            d_17 = dm[(j0 + 5) + nao * (k0 + 2)];
            reduce(gout0 * d_0 + gout6 * d_1 + gout12 * d_2 + gout18 * d_3 +
                       gout24 * d_4 + gout30 * d_5 + gout36 * d_6 +
                       gout42 * d_7 + gout48 * d_8 + gout54 * d_9 +
                       gout60 * d_10 + gout66 * d_11 + gout72 * d_12 +
                       gout78 * d_13 + gout84 * d_14 + gout90 * d_15 +
                       gout96 * d_16 + gout102 * d_17,
                vk + (i0 + 0) + nao * (l0 + 0));
            reduce(gout1 * d_0 + gout7 * d_1 + gout13 * d_2 + gout19 * d_3 +
                       gout25 * d_4 + gout31 * d_5 + gout37 * d_6 +
                       gout43 * d_7 + gout49 * d_8 + gout55 * d_9 +
                       gout61 * d_10 + gout67 * d_11 + gout73 * d_12 +
                       gout79 * d_13 + gout85 * d_14 + gout91 * d_15 +
                       gout97 * d_16 + gout103 * d_17,
                vk + (i0 + 1) + nao * (l0 + 0));
            reduce(gout2 * d_0 + gout8 * d_1 + gout14 * d_2 + gout20 * d_3 +
                       gout26 * d_4 + gout32 * d_5 + gout38 * d_6 +
                       gout44 * d_7 + gout50 * d_8 + gout56 * d_9 +
                       gout62 * d_10 + gout68 * d_11 + gout74 * d_12 +
                       gout80 * d_13 + gout86 * d_14 + gout92 * d_15 +
                       gout98 * d_16 + gout104 * d_17,
                vk + (i0 + 2) + nao * (l0 + 0));
            reduce(gout3 * d_0 + gout9 * d_1 + gout15 * d_2 + gout21 * d_3 +
                       gout27 * d_4 + gout33 * d_5 + gout39 * d_6 +
                       gout45 * d_7 + gout51 * d_8 + gout57 * d_9 +
                       gout63 * d_10 + gout69 * d_11 + gout75 * d_12 +
                       gout81 * d_13 + gout87 * d_14 + gout93 * d_15 +
                       gout99 * d_16 + gout105 * d_17,
                vk + (i0 + 3) + nao * (l0 + 0));
            reduce(gout4 * d_0 + gout10 * d_1 + gout16 * d_2 + gout22 * d_3 +
                       gout28 * d_4 + gout34 * d_5 + gout40 * d_6 +
                       gout46 * d_7 + gout52 * d_8 + gout58 * d_9 +
                       gout64 * d_10 + gout70 * d_11 + gout76 * d_12 +
                       gout82 * d_13 + gout88 * d_14 + gout94 * d_15 +
                       gout100 * d_16 + gout106 * d_17,
                vk + (i0 + 4) + nao * (l0 + 0));
            reduce(gout5 * d_0 + gout11 * d_1 + gout17 * d_2 + gout23 * d_3 +
                       gout29 * d_4 + gout35 * d_5 + gout41 * d_6 +
                       gout47 * d_7 + gout53 * d_8 + gout59 * d_9 +
                       gout65 * d_10 + gout71 * d_11 + gout77 * d_12 +
                       gout83 * d_13 + gout89 * d_14 + gout95 * d_15 +
                       gout101 * d_16 + gout107 * d_17,
                vk + (i0 + 5) + nao * (l0 + 0));
            // ijkl,il->jk
            d_0 = dm[(i0 + 0) + nao * (l0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (l0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (l0 + 0)];
            d_3 = dm[(i0 + 3) + nao * (l0 + 0)];
            d_4 = dm[(i0 + 4) + nao * (l0 + 0)];
            d_5 = dm[(i0 + 5) + nao * (l0 + 0)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout3 * d_3 +
                       gout4 * d_4 + gout5 * d_5,
                vk + (j0 + 0) + nao * (k0 + 0));
            reduce(gout6 * d_0 + gout7 * d_1 + gout8 * d_2 + gout9 * d_3 +
                       gout10 * d_4 + gout11 * d_5,
                vk + (j0 + 1) + nao * (k0 + 0));
            reduce(gout12 * d_0 + gout13 * d_1 + gout14 * d_2 + gout15 * d_3 +
                       gout16 * d_4 + gout17 * d_5,
                vk + (j0 + 2) + nao * (k0 + 0));
            reduce(gout18 * d_0 + gout19 * d_1 + gout20 * d_2 + gout21 * d_3 +
                       gout22 * d_4 + gout23 * d_5,
                vk + (j0 + 3) + nao * (k0 + 0));
            reduce(gout24 * d_0 + gout25 * d_1 + gout26 * d_2 + gout27 * d_3 +
                       gout28 * d_4 + gout29 * d_5,
                vk + (j0 + 4) + nao * (k0 + 0));
            reduce(gout30 * d_0 + gout31 * d_1 + gout32 * d_2 + gout33 * d_3 +
                       gout34 * d_4 + gout35 * d_5,
                vk + (j0 + 5) + nao * (k0 + 0));
            reduce(gout36 * d_0 + gout37 * d_1 + gout38 * d_2 + gout39 * d_3 +
                       gout40 * d_4 + gout41 * d_5,
                vk + (j0 + 0) + nao * (k0 + 1));
            reduce(gout42 * d_0 + gout43 * d_1 + gout44 * d_2 + gout45 * d_3 +
                       gout46 * d_4 + gout47 * d_5,
                vk + (j0 + 1) + nao * (k0 + 1));
            reduce(gout48 * d_0 + gout49 * d_1 + gout50 * d_2 + gout51 * d_3 +
                       gout52 * d_4 + gout53 * d_5,
                vk + (j0 + 2) + nao * (k0 + 1));
            reduce(gout54 * d_0 + gout55 * d_1 + gout56 * d_2 + gout57 * d_3 +
                       gout58 * d_4 + gout59 * d_5,
                vk + (j0 + 3) + nao * (k0 + 1));
            reduce(gout60 * d_0 + gout61 * d_1 + gout62 * d_2 + gout63 * d_3 +
                       gout64 * d_4 + gout65 * d_5,
                vk + (j0 + 4) + nao * (k0 + 1));
            reduce(gout66 * d_0 + gout67 * d_1 + gout68 * d_2 + gout69 * d_3 +
                       gout70 * d_4 + gout71 * d_5,
                vk + (j0 + 5) + nao * (k0 + 1));
            reduce(gout72 * d_0 + gout73 * d_1 + gout74 * d_2 + gout75 * d_3 +
                       gout76 * d_4 + gout77 * d_5,
                vk + (j0 + 0) + nao * (k0 + 2));
            reduce(gout78 * d_0 + gout79 * d_1 + gout80 * d_2 + gout81 * d_3 +
                       gout82 * d_4 + gout83 * d_5,
                vk + (j0 + 1) + nao * (k0 + 2));
            reduce(gout84 * d_0 + gout85 * d_1 + gout86 * d_2 + gout87 * d_3 +
                       gout88 * d_4 + gout89 * d_5,
                vk + (j0 + 2) + nao * (k0 + 2));
            reduce(gout90 * d_0 + gout91 * d_1 + gout92 * d_2 + gout93 * d_3 +
                       gout94 * d_4 + gout95 * d_5,
                vk + (j0 + 3) + nao * (k0 + 2));
            reduce(gout96 * d_0 + gout97 * d_1 + gout98 * d_2 + gout99 * d_3 +
                       gout100 * d_4 + gout101 * d_5,
                vk + (j0 + 4) + nao * (k0 + 2));
            reduce(gout102 * d_0 + gout103 * d_1 + gout104 * d_2 +
                       gout105 * d_3 + gout106 * d_4 + gout107 * d_5,
                vk + (j0 + 5) + nao * (k0 + 2));
            // ijkl,ik->jl
            d_0 = dm[(i0 + 0) + nao * (k0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (k0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (k0 + 0)];
            d_3 = dm[(i0 + 3) + nao * (k0 + 0)];
            d_4 = dm[(i0 + 4) + nao * (k0 + 0)];
            d_5 = dm[(i0 + 5) + nao * (k0 + 0)];
            d_6 = dm[(i0 + 0) + nao * (k0 + 1)];
            d_7 = dm[(i0 + 1) + nao * (k0 + 1)];
            d_8 = dm[(i0 + 2) + nao * (k0 + 1)];
            d_9 = dm[(i0 + 3) + nao * (k0 + 1)];
            d_10 = dm[(i0 + 4) + nao * (k0 + 1)];
            d_11 = dm[(i0 + 5) + nao * (k0 + 1)];
            d_12 = dm[(i0 + 0) + nao * (k0 + 2)];
            d_13 = dm[(i0 + 1) + nao * (k0 + 2)];
            d_14 = dm[(i0 + 2) + nao * (k0 + 2)];
            d_15 = dm[(i0 + 3) + nao * (k0 + 2)];
            d_16 = dm[(i0 + 4) + nao * (k0 + 2)];
            d_17 = dm[(i0 + 5) + nao * (k0 + 2)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout3 * d_3 +
                       gout4 * d_4 + gout5 * d_5 + gout36 * d_6 +
                       gout37 * d_7 + gout38 * d_8 + gout39 * d_9 +
                       gout40 * d_10 + gout41 * d_11 + gout72 * d_12 +
                       gout73 * d_13 + gout74 * d_14 + gout75 * d_15 +
                       gout76 * d_16 + gout77 * d_17,
                vk + (j0 + 0) + nao * (l0 + 0));
            reduce(gout6 * d_0 + gout7 * d_1 + gout8 * d_2 + gout9 * d_3 +
                       gout10 * d_4 + gout11 * d_5 + gout42 * d_6 +
                       gout43 * d_7 + gout44 * d_8 + gout45 * d_9 +
                       gout46 * d_10 + gout47 * d_11 + gout78 * d_12 +
                       gout79 * d_13 + gout80 * d_14 + gout81 * d_15 +
                       gout82 * d_16 + gout83 * d_17,
                vk + (j0 + 1) + nao * (l0 + 0));
            reduce(gout12 * d_0 + gout13 * d_1 + gout14 * d_2 + gout15 * d_3 +
                       gout16 * d_4 + gout17 * d_5 + gout48 * d_6 +
                       gout49 * d_7 + gout50 * d_8 + gout51 * d_9 +
                       gout52 * d_10 + gout53 * d_11 + gout84 * d_12 +
                       gout85 * d_13 + gout86 * d_14 + gout87 * d_15 +
                       gout88 * d_16 + gout89 * d_17,
                vk + (j0 + 2) + nao * (l0 + 0));
            reduce(gout18 * d_0 + gout19 * d_1 + gout20 * d_2 + gout21 * d_3 +
                       gout22 * d_4 + gout23 * d_5 + gout54 * d_6 +
                       gout55 * d_7 + gout56 * d_8 + gout57 * d_9 +
                       gout58 * d_10 + gout59 * d_11 + gout90 * d_12 +
                       gout91 * d_13 + gout92 * d_14 + gout93 * d_15 +
                       gout94 * d_16 + gout95 * d_17,
                vk + (j0 + 3) + nao * (l0 + 0));
            reduce(gout24 * d_0 + gout25 * d_1 + gout26 * d_2 + gout27 * d_3 +
                       gout28 * d_4 + gout29 * d_5 + gout60 * d_6 +
                       gout61 * d_7 + gout62 * d_8 + gout63 * d_9 +
                       gout64 * d_10 + gout65 * d_11 + gout96 * d_12 +
                       gout97 * d_13 + gout98 * d_14 + gout99 * d_15 +
                       gout100 * d_16 + gout101 * d_17,
                vk + (j0 + 4) + nao * (l0 + 0));
            reduce(gout30 * d_0 + gout31 * d_1 + gout32 * d_2 + gout33 * d_3 +
                       gout34 * d_4 + gout35 * d_5 + gout66 * d_6 +
                       gout67 * d_7 + gout68 * d_8 + gout69 * d_9 +
                       gout70 * d_10 + gout71 * d_11 + gout102 * d_12 +
                       gout103 * d_13 + gout104 * d_14 + gout105 * d_15 +
                       gout106 * d_16 + gout107 * d_17,
                vk + (j0 + 5) + nao * (l0 + 0));
            vk += nao2;
        }
        dm += nao2;
    }
}

__global__ static void GINTint2e_jk_kernel3010(JKMatrix jk,
    BasisProdOffsets offsets, GINTEnvVars envs, BasisProdCache bpcache) {
    int ntasks_ij = offsets.ntasks_ij;
    long ntasks = ntasks_ij * offsets.ntasks_kl;
    long task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int igroup = nprim_ij * nprim_kl;
    ntasks *= igroup;
    if (task_ij >= ntasks)
        return;
    int kl = task_ij % nprim_kl;
    task_ij /= nprim_kl;
    int ij = task_ij % nprim_ij;
    task_ij /= nprim_ij;
    int task_kl = task_ij / ntasks_ij;
    task_ij = task_ij % ntasks_ij;

    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    if (bas_ij < bas_kl) {
        return;
    }
    double norm = envs.fac;
    if (bas_ij == bas_kl) {
        norm *= .5;
    }

    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *ao_loc = bpcache.ao_loc;
    int *bas_pair2bra = bpcache.bas_pair2bra;
    int *bas_pair2ket = bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    int k0 = ao_loc[ksh];
    int l0 = ao_loc[lsh];
    double *__restrict__ a12 = bpcache.a12;
    double *__restrict__ e12 = bpcache.e12;
    double *__restrict__ x12 = bpcache.x12;
    double *__restrict__ y12 = bpcache.y12;
    double *__restrict__ z12 = bpcache.z12;
    int i_dm;
    int nbas = bpcache.nbas;
    double *__restrict__ bas_x = bpcache.bas_coords;
    double *__restrict__ bas_y = bas_x + nbas;
    double *__restrict__ bas_z = bas_y + nbas;

    double gout0 = 0;
    double gout1 = 0;
    double gout2 = 0;
    double gout3 = 0;
    double gout4 = 0;
    double gout5 = 0;
    double gout6 = 0;
    double gout7 = 0;
    double gout8 = 0;
    double gout9 = 0;
    double gout10 = 0;
    double gout11 = 0;
    double gout12 = 0;
    double gout13 = 0;
    double gout14 = 0;
    double gout15 = 0;
    double gout16 = 0;
    double gout17 = 0;
    double gout18 = 0;
    double gout19 = 0;
    double gout20 = 0;
    double gout21 = 0;
    double gout22 = 0;
    double gout23 = 0;
    double gout24 = 0;
    double gout25 = 0;
    double gout26 = 0;
    double gout27 = 0;
    double gout28 = 0;
    double gout29 = 0;
    double xi = bas_x[ish];
    double yi = bas_y[ish];
    double zi = bas_z[ish];
    double xk = bas_x[ksh];
    double yk = bas_y[ksh];
    double zk = bas_z[ksh];
    auto reduce = SegReduce<double>(igroup);
    ij += prim_ij;
    kl += prim_kl;
    double aij = a12[ij];
    double eij = e12[ij];
    double xij = x12[ij];
    double yij = y12[ij];
    double zij = z12[ij];
    double akl = a12[kl];
    double ekl = e12[kl];
    double xkl = x12[kl];
    double ykl = y12[kl];
    double zkl = z12[kl];
    double xijxkl = xij - xkl;
    double yijykl = yij - ykl;
    double zijzkl = zij - zkl;
    double aijkl = aij + akl;
    double a1 = aij * akl;
    double a0 = a1 / aijkl;
    double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
    double fac = norm * eij * ekl / (sqrt(aijkl) * a1);

    double rw[6];
    double root0, weight0;
    GINTrys_root<3>(x, rw);
    int irys;
    for (irys = 0; irys < 3; ++irys) {
        root0 = rw[irys];
        weight0 = rw[irys + 3];
        double u2 = a0 * root0;
        double tmp4 = .5 / (u2 * aijkl + a1);
        double b00 = u2 * tmp4;
        double tmp1 = 2 * b00;
        double tmp2 = tmp1 * akl;
        double b10 = b00 + tmp4 * akl;
        double c00x = xij - xi - tmp2 * xijxkl;
        double c00y = yij - yi - tmp2 * yijykl;
        double c00z = zij - zi - tmp2 * zijzkl;
        double tmp3 = tmp1 * aij;
        double c0px = xkl - xk + tmp3 * xijxkl;
        double c0py = ykl - yk + tmp3 * yijykl;
        double c0pz = zkl - zk + tmp3 * zijzkl;
        double g_0 = 1;
        double g_1 = c00x;
        double g_2 = c00x * c00x + b10;
        double g_3 = c00x * (2 * b10 + g_2);
        double g_4 = c0px;
        double g_5 = c0px * c00x + b00;
        double g_6 = b00 * c00x + b10 * c0px + c00x * g_5;
        double g_7 = 2 * b10 * g_5 + b00 * g_2 + c00x * g_6;
        double g_8 = 1;
        double g_9 = c00y;
        double g_10 = c00y * c00y + b10;
        double g_11 = c00y * (2 * b10 + g_10);
        double g_12 = c0py;
        double g_13 = c0py * c00y + b00;
        double g_14 = b00 * c00y + b10 * c0py + c00y * g_13;
        double g_15 = 2 * b10 * g_13 + b00 * g_10 + c00y * g_14;
        double g_16 = weight0 * fac;
        double g_17 = c00z * g_16;
        double g_18 = b10 * g_16 + c00z * g_17;
        double g_19 = 2 * b10 * g_17 + c00z * g_18;
        double g_20 = c0pz * g_16;
        double g_21 = b00 * g_16 + c0pz * g_17;
        double g_22 = b00 * g_17 + b10 * g_20 + c00z * g_21;
        double g_23 = 2 * b10 * g_21 + b00 * g_18 + c00z * g_22;
        gout0 += g_7 * g_8 * g_16;
        gout1 += g_6 * g_9 * g_16;
        gout2 += g_6 * g_8 * g_17;
        gout3 += g_5 * g_10 * g_16;
        gout4 += g_5 * g_9 * g_17;
        gout5 += g_5 * g_8 * g_18;
        gout6 += g_4 * g_11 * g_16;
        gout7 += g_4 * g_10 * g_17;
        gout8 += g_4 * g_9 * g_18;
        gout9 += g_4 * g_8 * g_19;
        gout10 += g_3 * g_12 * g_16;
        gout11 += g_2 * g_13 * g_16;
        gout12 += g_2 * g_12 * g_17;
        gout13 += g_1 * g_14 * g_16;
        gout14 += g_1 * g_13 * g_17;
        gout15 += g_1 * g_12 * g_18;
        gout16 += g_0 * g_15 * g_16;
        gout17 += g_0 * g_14 * g_17;
        gout18 += g_0 * g_13 * g_18;
        gout19 += g_0 * g_12 * g_19;
        gout20 += g_3 * g_8 * g_20;
        gout21 += g_2 * g_9 * g_20;
        gout22 += g_2 * g_8 * g_21;
        gout23 += g_1 * g_10 * g_20;
        gout24 += g_1 * g_9 * g_21;
        gout25 += g_1 * g_8 * g_22;
        gout26 += g_0 * g_11 * g_20;
        gout27 += g_0 * g_10 * g_21;
        gout28 += g_0 * g_9 * g_22;
        gout29 += g_0 * g_8 * g_23;
    }
    double d_0, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, d_9;
    double d_10, d_11, d_12, d_13, d_14, d_15, d_16, d_17, d_18, d_19;
    double d_20, d_21, d_22, d_23, d_24, d_25, d_26, d_27, d_28, d_29;
    int n_dm = jk.n_dm;
    int nao = jk.nao;
    size_t nao2 = nao * nao;
    double *__restrict__ dm = jk.dm;
    double *vj = jk.vj;
    double *vk = jk.vk;
    for (i_dm = 0; i_dm < n_dm; ++i_dm) {
        if (vj != NULL) {
            // ijkl,ij->kl
            d_0 = dm[(i0 + 0) + nao * (j0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (j0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (j0 + 0)];
            d_3 = dm[(i0 + 3) + nao * (j0 + 0)];
            d_4 = dm[(i0 + 4) + nao * (j0 + 0)];
            d_5 = dm[(i0 + 5) + nao * (j0 + 0)];
            d_6 = dm[(i0 + 6) + nao * (j0 + 0)];
            d_7 = dm[(i0 + 7) + nao * (j0 + 0)];
            d_8 = dm[(i0 + 8) + nao * (j0 + 0)];
            d_9 = dm[(i0 + 9) + nao * (j0 + 0)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout3 * d_3 +
                       gout4 * d_4 + gout5 * d_5 + gout6 * d_6 + gout7 * d_7 +
                       gout8 * d_8 + gout9 * d_9,
                vj + (k0 + 0) + nao * (l0 + 0));
            reduce(gout10 * d_0 + gout11 * d_1 + gout12 * d_2 + gout13 * d_3 +
                       gout14 * d_4 + gout15 * d_5 + gout16 * d_6 +
                       gout17 * d_7 + gout18 * d_8 + gout19 * d_9,
                vj + (k0 + 1) + nao * (l0 + 0));
            reduce(gout20 * d_0 + gout21 * d_1 + gout22 * d_2 + gout23 * d_3 +
                       gout24 * d_4 + gout25 * d_5 + gout26 * d_6 +
                       gout27 * d_7 + gout28 * d_8 + gout29 * d_9,
                vj + (k0 + 2) + nao * (l0 + 0));
            // ijkl,kl->ij
            d_0 = dm[(k0 + 0) + nao * (l0 + 0)];
            d_1 = dm[(k0 + 1) + nao * (l0 + 0)];
            d_2 = dm[(k0 + 2) + nao * (l0 + 0)];
            reduce(gout0 * d_0 + gout10 * d_1 + gout20 * d_2,
                vj + (i0 + 0) + nao * (j0 + 0));
            reduce(gout1 * d_0 + gout11 * d_1 + gout21 * d_2,
                vj + (i0 + 1) + nao * (j0 + 0));
            reduce(gout2 * d_0 + gout12 * d_1 + gout22 * d_2,
                vj + (i0 + 2) + nao * (j0 + 0));
            reduce(gout3 * d_0 + gout13 * d_1 + gout23 * d_2,
                vj + (i0 + 3) + nao * (j0 + 0));
            reduce(gout4 * d_0 + gout14 * d_1 + gout24 * d_2,
                vj + (i0 + 4) + nao * (j0 + 0));
            reduce(gout5 * d_0 + gout15 * d_1 + gout25 * d_2,
                vj + (i0 + 5) + nao * (j0 + 0));
            reduce(gout6 * d_0 + gout16 * d_1 + gout26 * d_2,
                vj + (i0 + 6) + nao * (j0 + 0));
            reduce(gout7 * d_0 + gout17 * d_1 + gout27 * d_2,
                vj + (i0 + 7) + nao * (j0 + 0));
            reduce(gout8 * d_0 + gout18 * d_1 + gout28 * d_2,
                vj + (i0 + 8) + nao * (j0 + 0));
            reduce(gout9 * d_0 + gout19 * d_1 + gout29 * d_2,
                vj + (i0 + 9) + nao * (j0 + 0));
            vj += nao2;
        }
        if (vk != NULL) {
            // ijkl,jl->ik
            d_0 = dm[(j0 + 0) + nao * (l0 + 0)];
            reduce(gout0 * d_0, vk + (i0 + 0) + nao * (k0 + 0));
            reduce(gout1 * d_0, vk + (i0 + 1) + nao * (k0 + 0));
            reduce(gout2 * d_0, vk + (i0 + 2) + nao * (k0 + 0));
            reduce(gout3 * d_0, vk + (i0 + 3) + nao * (k0 + 0));
            reduce(gout4 * d_0, vk + (i0 + 4) + nao * (k0 + 0));
            reduce(gout5 * d_0, vk + (i0 + 5) + nao * (k0 + 0));
            reduce(gout6 * d_0, vk + (i0 + 6) + nao * (k0 + 0));
            reduce(gout7 * d_0, vk + (i0 + 7) + nao * (k0 + 0));
            reduce(gout8 * d_0, vk + (i0 + 8) + nao * (k0 + 0));
            reduce(gout9 * d_0, vk + (i0 + 9) + nao * (k0 + 0));
            reduce(gout10 * d_0, vk + (i0 + 0) + nao * (k0 + 1));
            reduce(gout11 * d_0, vk + (i0 + 1) + nao * (k0 + 1));
            reduce(gout12 * d_0, vk + (i0 + 2) + nao * (k0 + 1));
            reduce(gout13 * d_0, vk + (i0 + 3) + nao * (k0 + 1));
            reduce(gout14 * d_0, vk + (i0 + 4) + nao * (k0 + 1));
            reduce(gout15 * d_0, vk + (i0 + 5) + nao * (k0 + 1));
            reduce(gout16 * d_0, vk + (i0 + 6) + nao * (k0 + 1));
            reduce(gout17 * d_0, vk + (i0 + 7) + nao * (k0 + 1));
            reduce(gout18 * d_0, vk + (i0 + 8) + nao * (k0 + 1));
            reduce(gout19 * d_0, vk + (i0 + 9) + nao * (k0 + 1));
            reduce(gout20 * d_0, vk + (i0 + 0) + nao * (k0 + 2));
            reduce(gout21 * d_0, vk + (i0 + 1) + nao * (k0 + 2));
            reduce(gout22 * d_0, vk + (i0 + 2) + nao * (k0 + 2));
            reduce(gout23 * d_0, vk + (i0 + 3) + nao * (k0 + 2));
            reduce(gout24 * d_0, vk + (i0 + 4) + nao * (k0 + 2));
            reduce(gout25 * d_0, vk + (i0 + 5) + nao * (k0 + 2));
            reduce(gout26 * d_0, vk + (i0 + 6) + nao * (k0 + 2));
            reduce(gout27 * d_0, vk + (i0 + 7) + nao * (k0 + 2));
            reduce(gout28 * d_0, vk + (i0 + 8) + nao * (k0 + 2));
            reduce(gout29 * d_0, vk + (i0 + 9) + nao * (k0 + 2));
            // ijkl,jk->il
            d_0 = dm[(j0 + 0) + nao * (k0 + 0)];
            d_1 = dm[(j0 + 0) + nao * (k0 + 1)];
            d_2 = dm[(j0 + 0) + nao * (k0 + 2)];
            reduce(gout0 * d_0 + gout10 * d_1 + gout20 * d_2,
                vk + (i0 + 0) + nao * (l0 + 0));
            reduce(gout1 * d_0 + gout11 * d_1 + gout21 * d_2,
                vk + (i0 + 1) + nao * (l0 + 0));
            reduce(gout2 * d_0 + gout12 * d_1 + gout22 * d_2,
                vk + (i0 + 2) + nao * (l0 + 0));
            reduce(gout3 * d_0 + gout13 * d_1 + gout23 * d_2,
                vk + (i0 + 3) + nao * (l0 + 0));
            reduce(gout4 * d_0 + gout14 * d_1 + gout24 * d_2,
                vk + (i0 + 4) + nao * (l0 + 0));
            reduce(gout5 * d_0 + gout15 * d_1 + gout25 * d_2,
                vk + (i0 + 5) + nao * (l0 + 0));
            reduce(gout6 * d_0 + gout16 * d_1 + gout26 * d_2,
                vk + (i0 + 6) + nao * (l0 + 0));
            reduce(gout7 * d_0 + gout17 * d_1 + gout27 * d_2,
                vk + (i0 + 7) + nao * (l0 + 0));
            reduce(gout8 * d_0 + gout18 * d_1 + gout28 * d_2,
                vk + (i0 + 8) + nao * (l0 + 0));
            reduce(gout9 * d_0 + gout19 * d_1 + gout29 * d_2,
                vk + (i0 + 9) + nao * (l0 + 0));
            // ijkl,il->jk
            d_0 = dm[(i0 + 0) + nao * (l0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (l0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (l0 + 0)];
            d_3 = dm[(i0 + 3) + nao * (l0 + 0)];
            d_4 = dm[(i0 + 4) + nao * (l0 + 0)];
            d_5 = dm[(i0 + 5) + nao * (l0 + 0)];
            d_6 = dm[(i0 + 6) + nao * (l0 + 0)];
            d_7 = dm[(i0 + 7) + nao * (l0 + 0)];
            d_8 = dm[(i0 + 8) + nao * (l0 + 0)];
            d_9 = dm[(i0 + 9) + nao * (l0 + 0)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout3 * d_3 +
                       gout4 * d_4 + gout5 * d_5 + gout6 * d_6 + gout7 * d_7 +
                       gout8 * d_8 + gout9 * d_9,
                vk + (j0 + 0) + nao * (k0 + 0));
            reduce(gout10 * d_0 + gout11 * d_1 + gout12 * d_2 + gout13 * d_3 +
                       gout14 * d_4 + gout15 * d_5 + gout16 * d_6 +
                       gout17 * d_7 + gout18 * d_8 + gout19 * d_9,
                vk + (j0 + 0) + nao * (k0 + 1));
            reduce(gout20 * d_0 + gout21 * d_1 + gout22 * d_2 + gout23 * d_3 +
                       gout24 * d_4 + gout25 * d_5 + gout26 * d_6 +
                       gout27 * d_7 + gout28 * d_8 + gout29 * d_9,
                vk + (j0 + 0) + nao * (k0 + 2));
            // ijkl,ik->jl
            d_0 = dm[(i0 + 0) + nao * (k0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (k0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (k0 + 0)];
            d_3 = dm[(i0 + 3) + nao * (k0 + 0)];
            d_4 = dm[(i0 + 4) + nao * (k0 + 0)];
            d_5 = dm[(i0 + 5) + nao * (k0 + 0)];
            d_6 = dm[(i0 + 6) + nao * (k0 + 0)];
            d_7 = dm[(i0 + 7) + nao * (k0 + 0)];
            d_8 = dm[(i0 + 8) + nao * (k0 + 0)];
            d_9 = dm[(i0 + 9) + nao * (k0 + 0)];
            d_10 = dm[(i0 + 0) + nao * (k0 + 1)];
            d_11 = dm[(i0 + 1) + nao * (k0 + 1)];
            d_12 = dm[(i0 + 2) + nao * (k0 + 1)];
            d_13 = dm[(i0 + 3) + nao * (k0 + 1)];
            d_14 = dm[(i0 + 4) + nao * (k0 + 1)];
            d_15 = dm[(i0 + 5) + nao * (k0 + 1)];
            d_16 = dm[(i0 + 6) + nao * (k0 + 1)];
            d_17 = dm[(i0 + 7) + nao * (k0 + 1)];
            d_18 = dm[(i0 + 8) + nao * (k0 + 1)];
            d_19 = dm[(i0 + 9) + nao * (k0 + 1)];
            d_20 = dm[(i0 + 0) + nao * (k0 + 2)];
            d_21 = dm[(i0 + 1) + nao * (k0 + 2)];
            d_22 = dm[(i0 + 2) + nao * (k0 + 2)];
            d_23 = dm[(i0 + 3) + nao * (k0 + 2)];
            d_24 = dm[(i0 + 4) + nao * (k0 + 2)];
            d_25 = dm[(i0 + 5) + nao * (k0 + 2)];
            d_26 = dm[(i0 + 6) + nao * (k0 + 2)];
            d_27 = dm[(i0 + 7) + nao * (k0 + 2)];
            d_28 = dm[(i0 + 8) + nao * (k0 + 2)];
            d_29 = dm[(i0 + 9) + nao * (k0 + 2)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout3 * d_3 +
                       gout4 * d_4 + gout5 * d_5 + gout6 * d_6 + gout7 * d_7 +
                       gout8 * d_8 + gout9 * d_9 + gout10 * d_10 +
                       gout11 * d_11 + gout12 * d_12 + gout13 * d_13 +
                       gout14 * d_14 + gout15 * d_15 + gout16 * d_16 +
                       gout17 * d_17 + gout18 * d_18 + gout19 * d_19 +
                       gout20 * d_20 + gout21 * d_21 + gout22 * d_22 +
                       gout23 * d_23 + gout24 * d_24 + gout25 * d_25 +
                       gout26 * d_26 + gout27 * d_27 + gout28 * d_28 +
                       gout29 * d_29,
                vk + (j0 + 0) + nao * (l0 + 0));
            vk += nao2;
        }
        dm += nao2;
    }
}

__global__ static void GINTint2e_jk_kernel3011(JKMatrix jk,
    BasisProdOffsets offsets, GINTEnvVars envs, BasisProdCache bpcache) {
    int ntasks_ij = offsets.ntasks_ij;
    long ntasks = ntasks_ij * offsets.ntasks_kl;
    long task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int igroup = nprim_ij * nprim_kl;
    ntasks *= igroup;
    if (task_ij >= ntasks)
        return;
    int kl = task_ij % nprim_kl;
    task_ij /= nprim_kl;
    int ij = task_ij % nprim_ij;
    task_ij /= nprim_ij;
    int task_kl = task_ij / ntasks_ij;
    task_ij = task_ij % ntasks_ij;

    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    if (bas_ij < bas_kl) {
        return;
    }
    double norm = envs.fac;
    if (bas_ij == bas_kl) {
        norm *= .5;
    }

    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *ao_loc = bpcache.ao_loc;
    int *bas_pair2bra = bpcache.bas_pair2bra;
    int *bas_pair2ket = bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    int k0 = ao_loc[ksh];
    int l0 = ao_loc[lsh];
    double *__restrict__ a12 = bpcache.a12;
    double *__restrict__ e12 = bpcache.e12;
    double *__restrict__ x12 = bpcache.x12;
    double *__restrict__ y12 = bpcache.y12;
    double *__restrict__ z12 = bpcache.z12;
    int i_dm;
    int nbas = bpcache.nbas;
    double *__restrict__ bas_x = bpcache.bas_coords;
    double *__restrict__ bas_y = bas_x + nbas;
    double *__restrict__ bas_z = bas_y + nbas;

    double gout0 = 0;
    double gout1 = 0;
    double gout2 = 0;
    double gout3 = 0;
    double gout4 = 0;
    double gout5 = 0;
    double gout6 = 0;
    double gout7 = 0;
    double gout8 = 0;
    double gout9 = 0;
    double gout10 = 0;
    double gout11 = 0;
    double gout12 = 0;
    double gout13 = 0;
    double gout14 = 0;
    double gout15 = 0;
    double gout16 = 0;
    double gout17 = 0;
    double gout18 = 0;
    double gout19 = 0;
    double gout20 = 0;
    double gout21 = 0;
    double gout22 = 0;
    double gout23 = 0;
    double gout24 = 0;
    double gout25 = 0;
    double gout26 = 0;
    double gout27 = 0;
    double gout28 = 0;
    double gout29 = 0;
    double gout30 = 0;
    double gout31 = 0;
    double gout32 = 0;
    double gout33 = 0;
    double gout34 = 0;
    double gout35 = 0;
    double gout36 = 0;
    double gout37 = 0;
    double gout38 = 0;
    double gout39 = 0;
    double gout40 = 0;
    double gout41 = 0;
    double gout42 = 0;
    double gout43 = 0;
    double gout44 = 0;
    double gout45 = 0;
    double gout46 = 0;
    double gout47 = 0;
    double gout48 = 0;
    double gout49 = 0;
    double gout50 = 0;
    double gout51 = 0;
    double gout52 = 0;
    double gout53 = 0;
    double gout54 = 0;
    double gout55 = 0;
    double gout56 = 0;
    double gout57 = 0;
    double gout58 = 0;
    double gout59 = 0;
    double gout60 = 0;
    double gout61 = 0;
    double gout62 = 0;
    double gout63 = 0;
    double gout64 = 0;
    double gout65 = 0;
    double gout66 = 0;
    double gout67 = 0;
    double gout68 = 0;
    double gout69 = 0;
    double gout70 = 0;
    double gout71 = 0;
    double gout72 = 0;
    double gout73 = 0;
    double gout74 = 0;
    double gout75 = 0;
    double gout76 = 0;
    double gout77 = 0;
    double gout78 = 0;
    double gout79 = 0;
    double gout80 = 0;
    double gout81 = 0;
    double gout82 = 0;
    double gout83 = 0;
    double gout84 = 0;
    double gout85 = 0;
    double gout86 = 0;
    double gout87 = 0;
    double gout88 = 0;
    double gout89 = 0;
    double xi = bas_x[ish];
    double yi = bas_y[ish];
    double zi = bas_z[ish];
    double xk = bas_x[ksh];
    double yk = bas_y[ksh];
    double zk = bas_z[ksh];
    double xkxl = xk - bas_x[lsh];
    double ykyl = yk - bas_y[lsh];
    double zkzl = zk - bas_z[lsh];
    auto reduce = SegReduce<double>(igroup);
    ij += prim_ij;
    kl += prim_kl;
    double aij = a12[ij];
    double eij = e12[ij];
    double xij = x12[ij];
    double yij = y12[ij];
    double zij = z12[ij];
    double akl = a12[kl];
    double ekl = e12[kl];
    double xkl = x12[kl];
    double ykl = y12[kl];
    double zkl = z12[kl];
    double xijxkl = xij - xkl;
    double yijykl = yij - ykl;
    double zijzkl = zij - zkl;
    double aijkl = aij + akl;
    double a1 = aij * akl;
    double a0 = a1 / aijkl;
    double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
    double fac = norm * eij * ekl / (sqrt(aijkl) * a1);

    double rw[6];
    double root0, weight0;
    GINTrys_root<3>(x, rw);
    int irys;
    for (irys = 0; irys < 3; ++irys) {
        root0 = rw[irys];
        weight0 = rw[irys + 3];
        double u2 = a0 * root0;
        double tmp4 = .5 / (u2 * aijkl + a1);
        double b00 = u2 * tmp4;
        double tmp1 = 2 * b00;
        double tmp2 = tmp1 * akl;
        double b10 = b00 + tmp4 * akl;
        double c00x = xij - xi - tmp2 * xijxkl;
        double c00y = yij - yi - tmp2 * yijykl;
        double c00z = zij - zi - tmp2 * zijzkl;
        double tmp3 = tmp1 * aij;
        double b01 = b00 + tmp4 * aij;
        double c0px = xkl - xk + tmp3 * xijxkl;
        double c0py = ykl - yk + tmp3 * yijykl;
        double c0pz = zkl - zk + tmp3 * zijzkl;
        double g_0 = 1;
        double g_1 = c00x;
        double g_2 = c00x * c00x + b10;
        double g_3 = c00x * (2 * b10 + g_2);
        double g_4 = c0px;
        double g_5 = c0px * c00x + b00;
        double g_6 = b00 * c00x + b10 * c0px + c00x * g_5;
        double g_7 = 2 * b10 * g_5 + b00 * g_2 + c00x * g_6;
        double g_8 = c0px + xkxl;
        double g_9 = c00x * (c0px + xkxl) + b00;
        double g_10 = b00 * c00x + b10 * c0px + c00x * g_5 + xkxl * g_2;
        double g_11 = 2 * b10 * g_5 + b00 * g_2 + c00x * g_6 + xkxl * g_3;
        double g_12 = c0px * (c0px + xkxl) + b01;
        double g_13 = b00 * c0px + b01 * c00x + c0px * g_5 + xkxl * g_5;
        double g_14 = xkxl * g_6 +
                      c00x * (c0px * g_5 + b01 * c00x + b00 * c0px) +
                      b10 * (c0px * c0px + b01) + 2 * b00 * g_5;
        double g_15 = xkxl * g_7 +
                      c00x * (c00x * (c0px * g_5 + b01 * c00x + b00 * c0px) +
                                 b10 * (c0px * c0px + b01) + 2 * b00 * g_5) +
                      2 * b10 * (c0px * g_5 + b01 * c00x + b00 * c0px) +
                      2 * b00 * g_6;
        double g_16 = 1;
        double g_17 = c00y;
        double g_18 = c00y * c00y + b10;
        double g_19 = c00y * (2 * b10 + g_18);
        double g_20 = c0py;
        double g_21 = c0py * c00y + b00;
        double g_22 = b00 * c00y + b10 * c0py + c00y * g_21;
        double g_23 = 2 * b10 * g_21 + b00 * g_18 + c00y * g_22;
        double g_24 = c0py + ykyl;
        double g_25 = c00y * (c0py + ykyl) + b00;
        double g_26 = b00 * c00y + b10 * c0py + c00y * g_21 + ykyl * g_18;
        double g_27 = 2 * b10 * g_21 + b00 * g_18 + c00y * g_22 + ykyl * g_19;
        double g_28 = c0py * (c0py + ykyl) + b01;
        double g_29 = b00 * c0py + b01 * c00y + c0py * g_21 + ykyl * g_21;
        double g_30 = ykyl * g_22 +
                      c00y * (c0py * g_21 + b01 * c00y + b00 * c0py) +
                      b10 * (c0py * c0py + b01) + 2 * b00 * g_21;
        double g_31 = ykyl * g_23 +
                      c00y * (c00y * (c0py * g_21 + b01 * c00y + b00 * c0py) +
                                 b10 * (c0py * c0py + b01) + 2 * b00 * g_21) +
                      2 * b10 * (c0py * g_21 + b01 * c00y + b00 * c0py) +
                      2 * b00 * g_22;
        double g_32 = weight0 * fac;
        double g_33 = c00z * g_32;
        double g_34 = b10 * g_32 + c00z * g_33;
        double g_35 = 2 * b10 * g_33 + c00z * g_34;
        double g_36 = c0pz * g_32;
        double g_37 = b00 * g_32 + c0pz * g_33;
        double g_38 = b00 * g_33 + b10 * g_36 + c00z * g_37;
        double g_39 = 2 * b10 * g_37 + b00 * g_34 + c00z * g_38;
        double g_40 = g_32 * (c0pz + zkzl);
        double g_41 = b00 * g_32 + c0pz * g_33 + zkzl * g_33;
        double g_42 = b00 * g_33 + b10 * g_36 + c00z * g_37 + zkzl * g_34;
        double g_43 = 2 * b10 * g_37 + b00 * g_34 + c00z * g_38 + zkzl * g_35;
        double g_44 = b01 * g_32 + c0pz * g_36 + zkzl * g_36;
        double g_45 = b00 * g_36 + b01 * g_33 + c0pz * g_37 + zkzl * g_37;
        double g_46 = zkzl * g_38 +
                      c00z * (c0pz * g_37 + b01 * g_33 + b00 * g_36) +
                      b10 * (c0pz * g_36 + b01 * g_32) + 2 * b00 * g_37;
        double g_47 =
            zkzl * g_39 +
            c00z * (c00z * (c0pz * g_37 + b01 * g_33 + b00 * g_36) +
                       b10 * (c0pz * g_36 + b01 * g_32) + 2 * b00 * g_37) +
            2 * b10 * (c0pz * g_37 + b01 * g_33 + b00 * g_36) + 2 * b00 * g_38;
        gout0 += g_15 * g_16 * g_32;
        gout1 += g_14 * g_17 * g_32;
        gout2 += g_14 * g_16 * g_33;
        gout3 += g_13 * g_18 * g_32;
        gout4 += g_13 * g_17 * g_33;
        gout5 += g_13 * g_16 * g_34;
        gout6 += g_12 * g_19 * g_32;
        gout7 += g_12 * g_18 * g_33;
        gout8 += g_12 * g_17 * g_34;
        gout9 += g_12 * g_16 * g_35;
        gout10 += g_11 * g_20 * g_32;
        gout11 += g_10 * g_21 * g_32;
        gout12 += g_10 * g_20 * g_33;
        gout13 += g_9 * g_22 * g_32;
        gout14 += g_9 * g_21 * g_33;
        gout15 += g_9 * g_20 * g_34;
        gout16 += g_8 * g_23 * g_32;
        gout17 += g_8 * g_22 * g_33;
        gout18 += g_8 * g_21 * g_34;
        gout19 += g_8 * g_20 * g_35;
        gout20 += g_11 * g_16 * g_36;
        gout21 += g_10 * g_17 * g_36;
        gout22 += g_10 * g_16 * g_37;
        gout23 += g_9 * g_18 * g_36;
        gout24 += g_9 * g_17 * g_37;
        gout25 += g_9 * g_16 * g_38;
        gout26 += g_8 * g_19 * g_36;
        gout27 += g_8 * g_18 * g_37;
        gout28 += g_8 * g_17 * g_38;
        gout29 += g_8 * g_16 * g_39;
        gout30 += g_7 * g_24 * g_32;
        gout31 += g_6 * g_25 * g_32;
        gout32 += g_6 * g_24 * g_33;
        gout33 += g_5 * g_26 * g_32;
        gout34 += g_5 * g_25 * g_33;
        gout35 += g_5 * g_24 * g_34;
        gout36 += g_4 * g_27 * g_32;
        gout37 += g_4 * g_26 * g_33;
        gout38 += g_4 * g_25 * g_34;
        gout39 += g_4 * g_24 * g_35;
        gout40 += g_3 * g_28 * g_32;
        gout41 += g_2 * g_29 * g_32;
        gout42 += g_2 * g_28 * g_33;
        gout43 += g_1 * g_30 * g_32;
        gout44 += g_1 * g_29 * g_33;
        gout45 += g_1 * g_28 * g_34;
        gout46 += g_0 * g_31 * g_32;
        gout47 += g_0 * g_30 * g_33;
        gout48 += g_0 * g_29 * g_34;
        gout49 += g_0 * g_28 * g_35;
        gout50 += g_3 * g_24 * g_36;
        gout51 += g_2 * g_25 * g_36;
        gout52 += g_2 * g_24 * g_37;
        gout53 += g_1 * g_26 * g_36;
        gout54 += g_1 * g_25 * g_37;
        gout55 += g_1 * g_24 * g_38;
        gout56 += g_0 * g_27 * g_36;
        gout57 += g_0 * g_26 * g_37;
        gout58 += g_0 * g_25 * g_38;
        gout59 += g_0 * g_24 * g_39;
        gout60 += g_7 * g_16 * g_40;
        gout61 += g_6 * g_17 * g_40;
        gout62 += g_6 * g_16 * g_41;
        gout63 += g_5 * g_18 * g_40;
        gout64 += g_5 * g_17 * g_41;
        gout65 += g_5 * g_16 * g_42;
        gout66 += g_4 * g_19 * g_40;
        gout67 += g_4 * g_18 * g_41;
        gout68 += g_4 * g_17 * g_42;
        gout69 += g_4 * g_16 * g_43;
        gout70 += g_3 * g_20 * g_40;
        gout71 += g_2 * g_21 * g_40;
        gout72 += g_2 * g_20 * g_41;
        gout73 += g_1 * g_22 * g_40;
        gout74 += g_1 * g_21 * g_41;
        gout75 += g_1 * g_20 * g_42;
        gout76 += g_0 * g_23 * g_40;
        gout77 += g_0 * g_22 * g_41;
        gout78 += g_0 * g_21 * g_42;
        gout79 += g_0 * g_20 * g_43;
        gout80 += g_3 * g_16 * g_44;
        gout81 += g_2 * g_17 * g_44;
        gout82 += g_2 * g_16 * g_45;
        gout83 += g_1 * g_18 * g_44;
        gout84 += g_1 * g_17 * g_45;
        gout85 += g_1 * g_16 * g_46;
        gout86 += g_0 * g_19 * g_44;
        gout87 += g_0 * g_18 * g_45;
        gout88 += g_0 * g_17 * g_46;
        gout89 += g_0 * g_16 * g_47;
    }
    double d_0, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, d_9;
    double d_10, d_11, d_12, d_13, d_14, d_15, d_16, d_17, d_18, d_19;
    double d_20, d_21, d_22, d_23, d_24, d_25, d_26, d_27, d_28, d_29;
    int n_dm = jk.n_dm;
    int nao = jk.nao;
    size_t nao2 = nao * nao;
    double *__restrict__ dm = jk.dm;
    double *vj = jk.vj;
    double *vk = jk.vk;
    for (i_dm = 0; i_dm < n_dm; ++i_dm) {
        if (vj != NULL) {
            // ijkl,ij->kl
            d_0 = dm[(i0 + 0) + nao * (j0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (j0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (j0 + 0)];
            d_3 = dm[(i0 + 3) + nao * (j0 + 0)];
            d_4 = dm[(i0 + 4) + nao * (j0 + 0)];
            d_5 = dm[(i0 + 5) + nao * (j0 + 0)];
            d_6 = dm[(i0 + 6) + nao * (j0 + 0)];
            d_7 = dm[(i0 + 7) + nao * (j0 + 0)];
            d_8 = dm[(i0 + 8) + nao * (j0 + 0)];
            d_9 = dm[(i0 + 9) + nao * (j0 + 0)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout3 * d_3 +
                       gout4 * d_4 + gout5 * d_5 + gout6 * d_6 + gout7 * d_7 +
                       gout8 * d_8 + gout9 * d_9,
                vj + (k0 + 0) + nao * (l0 + 0));
            reduce(gout10 * d_0 + gout11 * d_1 + gout12 * d_2 + gout13 * d_3 +
                       gout14 * d_4 + gout15 * d_5 + gout16 * d_6 +
                       gout17 * d_7 + gout18 * d_8 + gout19 * d_9,
                vj + (k0 + 1) + nao * (l0 + 0));
            reduce(gout20 * d_0 + gout21 * d_1 + gout22 * d_2 + gout23 * d_3 +
                       gout24 * d_4 + gout25 * d_5 + gout26 * d_6 +
                       gout27 * d_7 + gout28 * d_8 + gout29 * d_9,
                vj + (k0 + 2) + nao * (l0 + 0));
            reduce(gout30 * d_0 + gout31 * d_1 + gout32 * d_2 + gout33 * d_3 +
                       gout34 * d_4 + gout35 * d_5 + gout36 * d_6 +
                       gout37 * d_7 + gout38 * d_8 + gout39 * d_9,
                vj + (k0 + 0) + nao * (l0 + 1));
            reduce(gout40 * d_0 + gout41 * d_1 + gout42 * d_2 + gout43 * d_3 +
                       gout44 * d_4 + gout45 * d_5 + gout46 * d_6 +
                       gout47 * d_7 + gout48 * d_8 + gout49 * d_9,
                vj + (k0 + 1) + nao * (l0 + 1));
            reduce(gout50 * d_0 + gout51 * d_1 + gout52 * d_2 + gout53 * d_3 +
                       gout54 * d_4 + gout55 * d_5 + gout56 * d_6 +
                       gout57 * d_7 + gout58 * d_8 + gout59 * d_9,
                vj + (k0 + 2) + nao * (l0 + 1));
            reduce(gout60 * d_0 + gout61 * d_1 + gout62 * d_2 + gout63 * d_3 +
                       gout64 * d_4 + gout65 * d_5 + gout66 * d_6 +
                       gout67 * d_7 + gout68 * d_8 + gout69 * d_9,
                vj + (k0 + 0) + nao * (l0 + 2));
            reduce(gout70 * d_0 + gout71 * d_1 + gout72 * d_2 + gout73 * d_3 +
                       gout74 * d_4 + gout75 * d_5 + gout76 * d_6 +
                       gout77 * d_7 + gout78 * d_8 + gout79 * d_9,
                vj + (k0 + 1) + nao * (l0 + 2));
            reduce(gout80 * d_0 + gout81 * d_1 + gout82 * d_2 + gout83 * d_3 +
                       gout84 * d_4 + gout85 * d_5 + gout86 * d_6 +
                       gout87 * d_7 + gout88 * d_8 + gout89 * d_9,
                vj + (k0 + 2) + nao * (l0 + 2));
            // ijkl,kl->ij
            d_0 = dm[(k0 + 0) + nao * (l0 + 0)];
            d_1 = dm[(k0 + 1) + nao * (l0 + 0)];
            d_2 = dm[(k0 + 2) + nao * (l0 + 0)];
            d_3 = dm[(k0 + 0) + nao * (l0 + 1)];
            d_4 = dm[(k0 + 1) + nao * (l0 + 1)];
            d_5 = dm[(k0 + 2) + nao * (l0 + 1)];
            d_6 = dm[(k0 + 0) + nao * (l0 + 2)];
            d_7 = dm[(k0 + 1) + nao * (l0 + 2)];
            d_8 = dm[(k0 + 2) + nao * (l0 + 2)];
            reduce(gout0 * d_0 + gout10 * d_1 + gout20 * d_2 + gout30 * d_3 +
                       gout40 * d_4 + gout50 * d_5 + gout60 * d_6 +
                       gout70 * d_7 + gout80 * d_8,
                vj + (i0 + 0) + nao * (j0 + 0));
            reduce(gout1 * d_0 + gout11 * d_1 + gout21 * d_2 + gout31 * d_3 +
                       gout41 * d_4 + gout51 * d_5 + gout61 * d_6 +
                       gout71 * d_7 + gout81 * d_8,
                vj + (i0 + 1) + nao * (j0 + 0));
            reduce(gout2 * d_0 + gout12 * d_1 + gout22 * d_2 + gout32 * d_3 +
                       gout42 * d_4 + gout52 * d_5 + gout62 * d_6 +
                       gout72 * d_7 + gout82 * d_8,
                vj + (i0 + 2) + nao * (j0 + 0));
            reduce(gout3 * d_0 + gout13 * d_1 + gout23 * d_2 + gout33 * d_3 +
                       gout43 * d_4 + gout53 * d_5 + gout63 * d_6 +
                       gout73 * d_7 + gout83 * d_8,
                vj + (i0 + 3) + nao * (j0 + 0));
            reduce(gout4 * d_0 + gout14 * d_1 + gout24 * d_2 + gout34 * d_3 +
                       gout44 * d_4 + gout54 * d_5 + gout64 * d_6 +
                       gout74 * d_7 + gout84 * d_8,
                vj + (i0 + 4) + nao * (j0 + 0));
            reduce(gout5 * d_0 + gout15 * d_1 + gout25 * d_2 + gout35 * d_3 +
                       gout45 * d_4 + gout55 * d_5 + gout65 * d_6 +
                       gout75 * d_7 + gout85 * d_8,
                vj + (i0 + 5) + nao * (j0 + 0));
            reduce(gout6 * d_0 + gout16 * d_1 + gout26 * d_2 + gout36 * d_3 +
                       gout46 * d_4 + gout56 * d_5 + gout66 * d_6 +
                       gout76 * d_7 + gout86 * d_8,
                vj + (i0 + 6) + nao * (j0 + 0));
            reduce(gout7 * d_0 + gout17 * d_1 + gout27 * d_2 + gout37 * d_3 +
                       gout47 * d_4 + gout57 * d_5 + gout67 * d_6 +
                       gout77 * d_7 + gout87 * d_8,
                vj + (i0 + 7) + nao * (j0 + 0));
            reduce(gout8 * d_0 + gout18 * d_1 + gout28 * d_2 + gout38 * d_3 +
                       gout48 * d_4 + gout58 * d_5 + gout68 * d_6 +
                       gout78 * d_7 + gout88 * d_8,
                vj + (i0 + 8) + nao * (j0 + 0));
            reduce(gout9 * d_0 + gout19 * d_1 + gout29 * d_2 + gout39 * d_3 +
                       gout49 * d_4 + gout59 * d_5 + gout69 * d_6 +
                       gout79 * d_7 + gout89 * d_8,
                vj + (i0 + 9) + nao * (j0 + 0));
            vj += nao2;
        }
        if (vk != NULL) {
            // ijkl,jl->ik
            d_0 = dm[(j0 + 0) + nao * (l0 + 0)];
            d_1 = dm[(j0 + 0) + nao * (l0 + 1)];
            d_2 = dm[(j0 + 0) + nao * (l0 + 2)];
            reduce(gout0 * d_0 + gout30 * d_1 + gout60 * d_2,
                vk + (i0 + 0) + nao * (k0 + 0));
            reduce(gout1 * d_0 + gout31 * d_1 + gout61 * d_2,
                vk + (i0 + 1) + nao * (k0 + 0));
            reduce(gout2 * d_0 + gout32 * d_1 + gout62 * d_2,
                vk + (i0 + 2) + nao * (k0 + 0));
            reduce(gout3 * d_0 + gout33 * d_1 + gout63 * d_2,
                vk + (i0 + 3) + nao * (k0 + 0));
            reduce(gout4 * d_0 + gout34 * d_1 + gout64 * d_2,
                vk + (i0 + 4) + nao * (k0 + 0));
            reduce(gout5 * d_0 + gout35 * d_1 + gout65 * d_2,
                vk + (i0 + 5) + nao * (k0 + 0));
            reduce(gout6 * d_0 + gout36 * d_1 + gout66 * d_2,
                vk + (i0 + 6) + nao * (k0 + 0));
            reduce(gout7 * d_0 + gout37 * d_1 + gout67 * d_2,
                vk + (i0 + 7) + nao * (k0 + 0));
            reduce(gout8 * d_0 + gout38 * d_1 + gout68 * d_2,
                vk + (i0 + 8) + nao * (k0 + 0));
            reduce(gout9 * d_0 + gout39 * d_1 + gout69 * d_2,
                vk + (i0 + 9) + nao * (k0 + 0));
            reduce(gout10 * d_0 + gout40 * d_1 + gout70 * d_2,
                vk + (i0 + 0) + nao * (k0 + 1));
            reduce(gout11 * d_0 + gout41 * d_1 + gout71 * d_2,
                vk + (i0 + 1) + nao * (k0 + 1));
            reduce(gout12 * d_0 + gout42 * d_1 + gout72 * d_2,
                vk + (i0 + 2) + nao * (k0 + 1));
            reduce(gout13 * d_0 + gout43 * d_1 + gout73 * d_2,
                vk + (i0 + 3) + nao * (k0 + 1));
            reduce(gout14 * d_0 + gout44 * d_1 + gout74 * d_2,
                vk + (i0 + 4) + nao * (k0 + 1));
            reduce(gout15 * d_0 + gout45 * d_1 + gout75 * d_2,
                vk + (i0 + 5) + nao * (k0 + 1));
            reduce(gout16 * d_0 + gout46 * d_1 + gout76 * d_2,
                vk + (i0 + 6) + nao * (k0 + 1));
            reduce(gout17 * d_0 + gout47 * d_1 + gout77 * d_2,
                vk + (i0 + 7) + nao * (k0 + 1));
            reduce(gout18 * d_0 + gout48 * d_1 + gout78 * d_2,
                vk + (i0 + 8) + nao * (k0 + 1));
            reduce(gout19 * d_0 + gout49 * d_1 + gout79 * d_2,
                vk + (i0 + 9) + nao * (k0 + 1));
            reduce(gout20 * d_0 + gout50 * d_1 + gout80 * d_2,
                vk + (i0 + 0) + nao * (k0 + 2));
            reduce(gout21 * d_0 + gout51 * d_1 + gout81 * d_2,
                vk + (i0 + 1) + nao * (k0 + 2));
            reduce(gout22 * d_0 + gout52 * d_1 + gout82 * d_2,
                vk + (i0 + 2) + nao * (k0 + 2));
            reduce(gout23 * d_0 + gout53 * d_1 + gout83 * d_2,
                vk + (i0 + 3) + nao * (k0 + 2));
            reduce(gout24 * d_0 + gout54 * d_1 + gout84 * d_2,
                vk + (i0 + 4) + nao * (k0 + 2));
            reduce(gout25 * d_0 + gout55 * d_1 + gout85 * d_2,
                vk + (i0 + 5) + nao * (k0 + 2));
            reduce(gout26 * d_0 + gout56 * d_1 + gout86 * d_2,
                vk + (i0 + 6) + nao * (k0 + 2));
            reduce(gout27 * d_0 + gout57 * d_1 + gout87 * d_2,
                vk + (i0 + 7) + nao * (k0 + 2));
            reduce(gout28 * d_0 + gout58 * d_1 + gout88 * d_2,
                vk + (i0 + 8) + nao * (k0 + 2));
            reduce(gout29 * d_0 + gout59 * d_1 + gout89 * d_2,
                vk + (i0 + 9) + nao * (k0 + 2));
            // ijkl,jk->il
            d_0 = dm[(j0 + 0) + nao * (k0 + 0)];
            d_1 = dm[(j0 + 0) + nao * (k0 + 1)];
            d_2 = dm[(j0 + 0) + nao * (k0 + 2)];
            reduce(gout0 * d_0 + gout10 * d_1 + gout20 * d_2,
                vk + (i0 + 0) + nao * (l0 + 0));
            reduce(gout1 * d_0 + gout11 * d_1 + gout21 * d_2,
                vk + (i0 + 1) + nao * (l0 + 0));
            reduce(gout2 * d_0 + gout12 * d_1 + gout22 * d_2,
                vk + (i0 + 2) + nao * (l0 + 0));
            reduce(gout3 * d_0 + gout13 * d_1 + gout23 * d_2,
                vk + (i0 + 3) + nao * (l0 + 0));
            reduce(gout4 * d_0 + gout14 * d_1 + gout24 * d_2,
                vk + (i0 + 4) + nao * (l0 + 0));
            reduce(gout5 * d_0 + gout15 * d_1 + gout25 * d_2,
                vk + (i0 + 5) + nao * (l0 + 0));
            reduce(gout6 * d_0 + gout16 * d_1 + gout26 * d_2,
                vk + (i0 + 6) + nao * (l0 + 0));
            reduce(gout7 * d_0 + gout17 * d_1 + gout27 * d_2,
                vk + (i0 + 7) + nao * (l0 + 0));
            reduce(gout8 * d_0 + gout18 * d_1 + gout28 * d_2,
                vk + (i0 + 8) + nao * (l0 + 0));
            reduce(gout9 * d_0 + gout19 * d_1 + gout29 * d_2,
                vk + (i0 + 9) + nao * (l0 + 0));
            reduce(gout30 * d_0 + gout40 * d_1 + gout50 * d_2,
                vk + (i0 + 0) + nao * (l0 + 1));
            reduce(gout31 * d_0 + gout41 * d_1 + gout51 * d_2,
                vk + (i0 + 1) + nao * (l0 + 1));
            reduce(gout32 * d_0 + gout42 * d_1 + gout52 * d_2,
                vk + (i0 + 2) + nao * (l0 + 1));
            reduce(gout33 * d_0 + gout43 * d_1 + gout53 * d_2,
                vk + (i0 + 3) + nao * (l0 + 1));
            reduce(gout34 * d_0 + gout44 * d_1 + gout54 * d_2,
                vk + (i0 + 4) + nao * (l0 + 1));
            reduce(gout35 * d_0 + gout45 * d_1 + gout55 * d_2,
                vk + (i0 + 5) + nao * (l0 + 1));
            reduce(gout36 * d_0 + gout46 * d_1 + gout56 * d_2,
                vk + (i0 + 6) + nao * (l0 + 1));
            reduce(gout37 * d_0 + gout47 * d_1 + gout57 * d_2,
                vk + (i0 + 7) + nao * (l0 + 1));
            reduce(gout38 * d_0 + gout48 * d_1 + gout58 * d_2,
                vk + (i0 + 8) + nao * (l0 + 1));
            reduce(gout39 * d_0 + gout49 * d_1 + gout59 * d_2,
                vk + (i0 + 9) + nao * (l0 + 1));
            reduce(gout60 * d_0 + gout70 * d_1 + gout80 * d_2,
                vk + (i0 + 0) + nao * (l0 + 2));
            reduce(gout61 * d_0 + gout71 * d_1 + gout81 * d_2,
                vk + (i0 + 1) + nao * (l0 + 2));
            reduce(gout62 * d_0 + gout72 * d_1 + gout82 * d_2,
                vk + (i0 + 2) + nao * (l0 + 2));
            reduce(gout63 * d_0 + gout73 * d_1 + gout83 * d_2,
                vk + (i0 + 3) + nao * (l0 + 2));
            reduce(gout64 * d_0 + gout74 * d_1 + gout84 * d_2,
                vk + (i0 + 4) + nao * (l0 + 2));
            reduce(gout65 * d_0 + gout75 * d_1 + gout85 * d_2,
                vk + (i0 + 5) + nao * (l0 + 2));
            reduce(gout66 * d_0 + gout76 * d_1 + gout86 * d_2,
                vk + (i0 + 6) + nao * (l0 + 2));
            reduce(gout67 * d_0 + gout77 * d_1 + gout87 * d_2,
                vk + (i0 + 7) + nao * (l0 + 2));
            reduce(gout68 * d_0 + gout78 * d_1 + gout88 * d_2,
                vk + (i0 + 8) + nao * (l0 + 2));
            reduce(gout69 * d_0 + gout79 * d_1 + gout89 * d_2,
                vk + (i0 + 9) + nao * (l0 + 2));
            // ijkl,il->jk
            d_0 = dm[(i0 + 0) + nao * (l0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (l0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (l0 + 0)];
            d_3 = dm[(i0 + 3) + nao * (l0 + 0)];
            d_4 = dm[(i0 + 4) + nao * (l0 + 0)];
            d_5 = dm[(i0 + 5) + nao * (l0 + 0)];
            d_6 = dm[(i0 + 6) + nao * (l0 + 0)];
            d_7 = dm[(i0 + 7) + nao * (l0 + 0)];
            d_8 = dm[(i0 + 8) + nao * (l0 + 0)];
            d_9 = dm[(i0 + 9) + nao * (l0 + 0)];
            d_10 = dm[(i0 + 0) + nao * (l0 + 1)];
            d_11 = dm[(i0 + 1) + nao * (l0 + 1)];
            d_12 = dm[(i0 + 2) + nao * (l0 + 1)];
            d_13 = dm[(i0 + 3) + nao * (l0 + 1)];
            d_14 = dm[(i0 + 4) + nao * (l0 + 1)];
            d_15 = dm[(i0 + 5) + nao * (l0 + 1)];
            d_16 = dm[(i0 + 6) + nao * (l0 + 1)];
            d_17 = dm[(i0 + 7) + nao * (l0 + 1)];
            d_18 = dm[(i0 + 8) + nao * (l0 + 1)];
            d_19 = dm[(i0 + 9) + nao * (l0 + 1)];
            d_20 = dm[(i0 + 0) + nao * (l0 + 2)];
            d_21 = dm[(i0 + 1) + nao * (l0 + 2)];
            d_22 = dm[(i0 + 2) + nao * (l0 + 2)];
            d_23 = dm[(i0 + 3) + nao * (l0 + 2)];
            d_24 = dm[(i0 + 4) + nao * (l0 + 2)];
            d_25 = dm[(i0 + 5) + nao * (l0 + 2)];
            d_26 = dm[(i0 + 6) + nao * (l0 + 2)];
            d_27 = dm[(i0 + 7) + nao * (l0 + 2)];
            d_28 = dm[(i0 + 8) + nao * (l0 + 2)];
            d_29 = dm[(i0 + 9) + nao * (l0 + 2)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout3 * d_3 +
                       gout4 * d_4 + gout5 * d_5 + gout6 * d_6 + gout7 * d_7 +
                       gout8 * d_8 + gout9 * d_9 + gout30 * d_10 +
                       gout31 * d_11 + gout32 * d_12 + gout33 * d_13 +
                       gout34 * d_14 + gout35 * d_15 + gout36 * d_16 +
                       gout37 * d_17 + gout38 * d_18 + gout39 * d_19 +
                       gout60 * d_20 + gout61 * d_21 + gout62 * d_22 +
                       gout63 * d_23 + gout64 * d_24 + gout65 * d_25 +
                       gout66 * d_26 + gout67 * d_27 + gout68 * d_28 +
                       gout69 * d_29,
                vk + (j0 + 0) + nao * (k0 + 0));
            reduce(gout10 * d_0 + gout11 * d_1 + gout12 * d_2 + gout13 * d_3 +
                       gout14 * d_4 + gout15 * d_5 + gout16 * d_6 +
                       gout17 * d_7 + gout18 * d_8 + gout19 * d_9 +
                       gout40 * d_10 + gout41 * d_11 + gout42 * d_12 +
                       gout43 * d_13 + gout44 * d_14 + gout45 * d_15 +
                       gout46 * d_16 + gout47 * d_17 + gout48 * d_18 +
                       gout49 * d_19 + gout70 * d_20 + gout71 * d_21 +
                       gout72 * d_22 + gout73 * d_23 + gout74 * d_24 +
                       gout75 * d_25 + gout76 * d_26 + gout77 * d_27 +
                       gout78 * d_28 + gout79 * d_29,
                vk + (j0 + 0) + nao * (k0 + 1));
            reduce(gout20 * d_0 + gout21 * d_1 + gout22 * d_2 + gout23 * d_3 +
                       gout24 * d_4 + gout25 * d_5 + gout26 * d_6 +
                       gout27 * d_7 + gout28 * d_8 + gout29 * d_9 +
                       gout50 * d_10 + gout51 * d_11 + gout52 * d_12 +
                       gout53 * d_13 + gout54 * d_14 + gout55 * d_15 +
                       gout56 * d_16 + gout57 * d_17 + gout58 * d_18 +
                       gout59 * d_19 + gout80 * d_20 + gout81 * d_21 +
                       gout82 * d_22 + gout83 * d_23 + gout84 * d_24 +
                       gout85 * d_25 + gout86 * d_26 + gout87 * d_27 +
                       gout88 * d_28 + gout89 * d_29,
                vk + (j0 + 0) + nao * (k0 + 2));
            // ijkl,ik->jl
            d_0 = dm[(i0 + 0) + nao * (k0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (k0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (k0 + 0)];
            d_3 = dm[(i0 + 3) + nao * (k0 + 0)];
            d_4 = dm[(i0 + 4) + nao * (k0 + 0)];
            d_5 = dm[(i0 + 5) + nao * (k0 + 0)];
            d_6 = dm[(i0 + 6) + nao * (k0 + 0)];
            d_7 = dm[(i0 + 7) + nao * (k0 + 0)];
            d_8 = dm[(i0 + 8) + nao * (k0 + 0)];
            d_9 = dm[(i0 + 9) + nao * (k0 + 0)];
            d_10 = dm[(i0 + 0) + nao * (k0 + 1)];
            d_11 = dm[(i0 + 1) + nao * (k0 + 1)];
            d_12 = dm[(i0 + 2) + nao * (k0 + 1)];
            d_13 = dm[(i0 + 3) + nao * (k0 + 1)];
            d_14 = dm[(i0 + 4) + nao * (k0 + 1)];
            d_15 = dm[(i0 + 5) + nao * (k0 + 1)];
            d_16 = dm[(i0 + 6) + nao * (k0 + 1)];
            d_17 = dm[(i0 + 7) + nao * (k0 + 1)];
            d_18 = dm[(i0 + 8) + nao * (k0 + 1)];
            d_19 = dm[(i0 + 9) + nao * (k0 + 1)];
            d_20 = dm[(i0 + 0) + nao * (k0 + 2)];
            d_21 = dm[(i0 + 1) + nao * (k0 + 2)];
            d_22 = dm[(i0 + 2) + nao * (k0 + 2)];
            d_23 = dm[(i0 + 3) + nao * (k0 + 2)];
            d_24 = dm[(i0 + 4) + nao * (k0 + 2)];
            d_25 = dm[(i0 + 5) + nao * (k0 + 2)];
            d_26 = dm[(i0 + 6) + nao * (k0 + 2)];
            d_27 = dm[(i0 + 7) + nao * (k0 + 2)];
            d_28 = dm[(i0 + 8) + nao * (k0 + 2)];
            d_29 = dm[(i0 + 9) + nao * (k0 + 2)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout3 * d_3 +
                       gout4 * d_4 + gout5 * d_5 + gout6 * d_6 + gout7 * d_7 +
                       gout8 * d_8 + gout9 * d_9 + gout10 * d_10 +
                       gout11 * d_11 + gout12 * d_12 + gout13 * d_13 +
                       gout14 * d_14 + gout15 * d_15 + gout16 * d_16 +
                       gout17 * d_17 + gout18 * d_18 + gout19 * d_19 +
                       gout20 * d_20 + gout21 * d_21 + gout22 * d_22 +
                       gout23 * d_23 + gout24 * d_24 + gout25 * d_25 +
                       gout26 * d_26 + gout27 * d_27 + gout28 * d_28 +
                       gout29 * d_29,
                vk + (j0 + 0) + nao * (l0 + 0));
            reduce(gout30 * d_0 + gout31 * d_1 + gout32 * d_2 + gout33 * d_3 +
                       gout34 * d_4 + gout35 * d_5 + gout36 * d_6 +
                       gout37 * d_7 + gout38 * d_8 + gout39 * d_9 +
                       gout40 * d_10 + gout41 * d_11 + gout42 * d_12 +
                       gout43 * d_13 + gout44 * d_14 + gout45 * d_15 +
                       gout46 * d_16 + gout47 * d_17 + gout48 * d_18 +
                       gout49 * d_19 + gout50 * d_20 + gout51 * d_21 +
                       gout52 * d_22 + gout53 * d_23 + gout54 * d_24 +
                       gout55 * d_25 + gout56 * d_26 + gout57 * d_27 +
                       gout58 * d_28 + gout59 * d_29,
                vk + (j0 + 0) + nao * (l0 + 1));
            reduce(gout60 * d_0 + gout61 * d_1 + gout62 * d_2 + gout63 * d_3 +
                       gout64 * d_4 + gout65 * d_5 + gout66 * d_6 +
                       gout67 * d_7 + gout68 * d_8 + gout69 * d_9 +
                       gout70 * d_10 + gout71 * d_11 + gout72 * d_12 +
                       gout73 * d_13 + gout74 * d_14 + gout75 * d_15 +
                       gout76 * d_16 + gout77 * d_17 + gout78 * d_18 +
                       gout79 * d_19 + gout80 * d_20 + gout81 * d_21 +
                       gout82 * d_22 + gout83 * d_23 + gout84 * d_24 +
                       gout85 * d_25 + gout86 * d_26 + gout87 * d_27 +
                       gout88 * d_28 + gout89 * d_29,
                vk + (j0 + 0) + nao * (l0 + 2));
            vk += nao2;
        }
        dm += nao2;
    }
}

__global__ static void GINTint2e_jk_kernel3020(JKMatrix jk,
    BasisProdOffsets offsets, GINTEnvVars envs, BasisProdCache bpcache) {
    int ntasks_ij = offsets.ntasks_ij;
    long ntasks = ntasks_ij * offsets.ntasks_kl;
    long task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int igroup = nprim_ij * nprim_kl;
    ntasks *= igroup;
    if (task_ij >= ntasks)
        return;
    int kl = task_ij % nprim_kl;
    task_ij /= nprim_kl;
    int ij = task_ij % nprim_ij;
    task_ij /= nprim_ij;
    int task_kl = task_ij / ntasks_ij;
    task_ij = task_ij % ntasks_ij;

    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    if (bas_ij < bas_kl) {
        return;
    }
    double norm = envs.fac;
    if (bas_ij == bas_kl) {
        norm *= .5;
    }

    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *ao_loc = bpcache.ao_loc;
    int *bas_pair2bra = bpcache.bas_pair2bra;
    int *bas_pair2ket = bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    int k0 = ao_loc[ksh];
    int l0 = ao_loc[lsh];
    double *__restrict__ a12 = bpcache.a12;
    double *__restrict__ e12 = bpcache.e12;
    double *__restrict__ x12 = bpcache.x12;
    double *__restrict__ y12 = bpcache.y12;
    double *__restrict__ z12 = bpcache.z12;
    int i_dm;
    int nbas = bpcache.nbas;
    double *__restrict__ bas_x = bpcache.bas_coords;
    double *__restrict__ bas_y = bas_x + nbas;
    double *__restrict__ bas_z = bas_y + nbas;

    double gout0 = 0;
    double gout1 = 0;
    double gout2 = 0;
    double gout3 = 0;
    double gout4 = 0;
    double gout5 = 0;
    double gout6 = 0;
    double gout7 = 0;
    double gout8 = 0;
    double gout9 = 0;
    double gout10 = 0;
    double gout11 = 0;
    double gout12 = 0;
    double gout13 = 0;
    double gout14 = 0;
    double gout15 = 0;
    double gout16 = 0;
    double gout17 = 0;
    double gout18 = 0;
    double gout19 = 0;
    double gout20 = 0;
    double gout21 = 0;
    double gout22 = 0;
    double gout23 = 0;
    double gout24 = 0;
    double gout25 = 0;
    double gout26 = 0;
    double gout27 = 0;
    double gout28 = 0;
    double gout29 = 0;
    double gout30 = 0;
    double gout31 = 0;
    double gout32 = 0;
    double gout33 = 0;
    double gout34 = 0;
    double gout35 = 0;
    double gout36 = 0;
    double gout37 = 0;
    double gout38 = 0;
    double gout39 = 0;
    double gout40 = 0;
    double gout41 = 0;
    double gout42 = 0;
    double gout43 = 0;
    double gout44 = 0;
    double gout45 = 0;
    double gout46 = 0;
    double gout47 = 0;
    double gout48 = 0;
    double gout49 = 0;
    double gout50 = 0;
    double gout51 = 0;
    double gout52 = 0;
    double gout53 = 0;
    double gout54 = 0;
    double gout55 = 0;
    double gout56 = 0;
    double gout57 = 0;
    double gout58 = 0;
    double gout59 = 0;
    double xi = bas_x[ish];
    double yi = bas_y[ish];
    double zi = bas_z[ish];
    double xk = bas_x[ksh];
    double yk = bas_y[ksh];
    double zk = bas_z[ksh];
    auto reduce = SegReduce<double>(igroup);
    ij += prim_ij;
    kl += prim_kl;
    double aij = a12[ij];
    double eij = e12[ij];
    double xij = x12[ij];
    double yij = y12[ij];
    double zij = z12[ij];
    double akl = a12[kl];
    double ekl = e12[kl];
    double xkl = x12[kl];
    double ykl = y12[kl];
    double zkl = z12[kl];
    double xijxkl = xij - xkl;
    double yijykl = yij - ykl;
    double zijzkl = zij - zkl;
    double aijkl = aij + akl;
    double a1 = aij * akl;
    double a0 = a1 / aijkl;
    double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
    double fac = norm * eij * ekl / (sqrt(aijkl) * a1);

    double rw[6];
    double root0, weight0;
    GINTrys_root<3>(x, rw);
    int irys;
    for (irys = 0; irys < 3; ++irys) {
        root0 = rw[irys];
        weight0 = rw[irys + 3];
        double u2 = a0 * root0;
        double tmp4 = .5 / (u2 * aijkl + a1);
        double b00 = u2 * tmp4;
        double tmp1 = 2 * b00;
        double tmp2 = tmp1 * akl;
        double b10 = b00 + tmp4 * akl;
        double c00x = xij - xi - tmp2 * xijxkl;
        double c00y = yij - yi - tmp2 * yijykl;
        double c00z = zij - zi - tmp2 * zijzkl;
        double tmp3 = tmp1 * aij;
        double b01 = b00 + tmp4 * aij;
        double c0px = xkl - xk + tmp3 * xijxkl;
        double c0py = ykl - yk + tmp3 * yijykl;
        double c0pz = zkl - zk + tmp3 * zijzkl;
        double g_0 = 1;
        double g_1 = c00x;
        double g_2 = c00x * c00x + b10;
        double g_3 = c00x * (2 * b10 + g_2);
        double g_4 = c0px;
        double g_5 = c0px * c00x + b00;
        double g_6 = b00 * c00x + b10 * c0px + c00x * g_5;
        double g_7 = 2 * b10 * g_5 + b00 * g_2 + c00x * g_6;
        double g_8 = c0px * c0px + b01;
        double g_9 = b00 * c0px + b01 * c00x + c0px * g_5;
        double g_10 = 2 * b00 * g_5 + b10 * g_8 + c00x * g_9;
        double g_11 = 2 * (b00 * g_6 + b10 * g_9) + c00x * g_10;
        double g_12 = 1;
        double g_13 = c00y;
        double g_14 = c00y * c00y + b10;
        double g_15 = c00y * (2 * b10 + g_14);
        double g_16 = c0py;
        double g_17 = c0py * c00y + b00;
        double g_18 = b00 * c00y + b10 * c0py + c00y * g_17;
        double g_19 = 2 * b10 * g_17 + b00 * g_14 + c00y * g_18;
        double g_20 = c0py * c0py + b01;
        double g_21 = b00 * c0py + b01 * c00y + c0py * g_17;
        double g_22 = 2 * b00 * g_17 + b10 * g_20 + c00y * g_21;
        double g_23 = 2 * (b00 * g_18 + b10 * g_21) + c00y * g_22;
        double g_24 = weight0 * fac;
        double g_25 = c00z * g_24;
        double g_26 = b10 * g_24 + c00z * g_25;
        double g_27 = 2 * b10 * g_25 + c00z * g_26;
        double g_28 = c0pz * g_24;
        double g_29 = b00 * g_24 + c0pz * g_25;
        double g_30 = b00 * g_25 + b10 * g_28 + c00z * g_29;
        double g_31 = 2 * b10 * g_29 + b00 * g_26 + c00z * g_30;
        double g_32 = b01 * g_24 + c0pz * g_28;
        double g_33 = b00 * g_28 + b01 * g_25 + c0pz * g_29;
        double g_34 = 2 * b00 * g_29 + b10 * g_32 + c00z * g_33;
        double g_35 = 2 * (b00 * g_30 + b10 * g_33) + c00z * g_34;
        gout0 += g_11 * g_12 * g_24;
        gout1 += g_10 * g_13 * g_24;
        gout2 += g_10 * g_12 * g_25;
        gout3 += g_9 * g_14 * g_24;
        gout4 += g_9 * g_13 * g_25;
        gout5 += g_9 * g_12 * g_26;
        gout6 += g_8 * g_15 * g_24;
        gout7 += g_8 * g_14 * g_25;
        gout8 += g_8 * g_13 * g_26;
        gout9 += g_8 * g_12 * g_27;
        gout10 += g_7 * g_16 * g_24;
        gout11 += g_6 * g_17 * g_24;
        gout12 += g_6 * g_16 * g_25;
        gout13 += g_5 * g_18 * g_24;
        gout14 += g_5 * g_17 * g_25;
        gout15 += g_5 * g_16 * g_26;
        gout16 += g_4 * g_19 * g_24;
        gout17 += g_4 * g_18 * g_25;
        gout18 += g_4 * g_17 * g_26;
        gout19 += g_4 * g_16 * g_27;
        gout20 += g_7 * g_12 * g_28;
        gout21 += g_6 * g_13 * g_28;
        gout22 += g_6 * g_12 * g_29;
        gout23 += g_5 * g_14 * g_28;
        gout24 += g_5 * g_13 * g_29;
        gout25 += g_5 * g_12 * g_30;
        gout26 += g_4 * g_15 * g_28;
        gout27 += g_4 * g_14 * g_29;
        gout28 += g_4 * g_13 * g_30;
        gout29 += g_4 * g_12 * g_31;
        gout30 += g_3 * g_20 * g_24;
        gout31 += g_2 * g_21 * g_24;
        gout32 += g_2 * g_20 * g_25;
        gout33 += g_1 * g_22 * g_24;
        gout34 += g_1 * g_21 * g_25;
        gout35 += g_1 * g_20 * g_26;
        gout36 += g_0 * g_23 * g_24;
        gout37 += g_0 * g_22 * g_25;
        gout38 += g_0 * g_21 * g_26;
        gout39 += g_0 * g_20 * g_27;
        gout40 += g_3 * g_16 * g_28;
        gout41 += g_2 * g_17 * g_28;
        gout42 += g_2 * g_16 * g_29;
        gout43 += g_1 * g_18 * g_28;
        gout44 += g_1 * g_17 * g_29;
        gout45 += g_1 * g_16 * g_30;
        gout46 += g_0 * g_19 * g_28;
        gout47 += g_0 * g_18 * g_29;
        gout48 += g_0 * g_17 * g_30;
        gout49 += g_0 * g_16 * g_31;
        gout50 += g_3 * g_12 * g_32;
        gout51 += g_2 * g_13 * g_32;
        gout52 += g_2 * g_12 * g_33;
        gout53 += g_1 * g_14 * g_32;
        gout54 += g_1 * g_13 * g_33;
        gout55 += g_1 * g_12 * g_34;
        gout56 += g_0 * g_15 * g_32;
        gout57 += g_0 * g_14 * g_33;
        gout58 += g_0 * g_13 * g_34;
        gout59 += g_0 * g_12 * g_35;
    }
    double d_0, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, d_9;
    double d_10, d_11, d_12, d_13, d_14, d_15, d_16, d_17, d_18, d_19;
    double d_20, d_21, d_22, d_23, d_24, d_25, d_26, d_27, d_28, d_29;
    double d_30, d_31, d_32, d_33, d_34, d_35, d_36, d_37, d_38, d_39;
    double d_40, d_41, d_42, d_43, d_44, d_45, d_46, d_47, d_48, d_49;
    double d_50, d_51, d_52, d_53, d_54, d_55, d_56, d_57, d_58, d_59;
    int n_dm = jk.n_dm;
    int nao = jk.nao;
    size_t nao2 = nao * nao;
    double *__restrict__ dm = jk.dm;
    double *vj = jk.vj;
    double *vk = jk.vk;
    for (i_dm = 0; i_dm < n_dm; ++i_dm) {
        if (vj != NULL) {
            // ijkl,ij->kl
            d_0 = dm[(i0 + 0) + nao * (j0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (j0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (j0 + 0)];
            d_3 = dm[(i0 + 3) + nao * (j0 + 0)];
            d_4 = dm[(i0 + 4) + nao * (j0 + 0)];
            d_5 = dm[(i0 + 5) + nao * (j0 + 0)];
            d_6 = dm[(i0 + 6) + nao * (j0 + 0)];
            d_7 = dm[(i0 + 7) + nao * (j0 + 0)];
            d_8 = dm[(i0 + 8) + nao * (j0 + 0)];
            d_9 = dm[(i0 + 9) + nao * (j0 + 0)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout3 * d_3 +
                       gout4 * d_4 + gout5 * d_5 + gout6 * d_6 + gout7 * d_7 +
                       gout8 * d_8 + gout9 * d_9,
                vj + (k0 + 0) + nao * (l0 + 0));
            reduce(gout10 * d_0 + gout11 * d_1 + gout12 * d_2 + gout13 * d_3 +
                       gout14 * d_4 + gout15 * d_5 + gout16 * d_6 +
                       gout17 * d_7 + gout18 * d_8 + gout19 * d_9,
                vj + (k0 + 1) + nao * (l0 + 0));
            reduce(gout20 * d_0 + gout21 * d_1 + gout22 * d_2 + gout23 * d_3 +
                       gout24 * d_4 + gout25 * d_5 + gout26 * d_6 +
                       gout27 * d_7 + gout28 * d_8 + gout29 * d_9,
                vj + (k0 + 2) + nao * (l0 + 0));
            reduce(gout30 * d_0 + gout31 * d_1 + gout32 * d_2 + gout33 * d_3 +
                       gout34 * d_4 + gout35 * d_5 + gout36 * d_6 +
                       gout37 * d_7 + gout38 * d_8 + gout39 * d_9,
                vj + (k0 + 3) + nao * (l0 + 0));
            reduce(gout40 * d_0 + gout41 * d_1 + gout42 * d_2 + gout43 * d_3 +
                       gout44 * d_4 + gout45 * d_5 + gout46 * d_6 +
                       gout47 * d_7 + gout48 * d_8 + gout49 * d_9,
                vj + (k0 + 4) + nao * (l0 + 0));
            reduce(gout50 * d_0 + gout51 * d_1 + gout52 * d_2 + gout53 * d_3 +
                       gout54 * d_4 + gout55 * d_5 + gout56 * d_6 +
                       gout57 * d_7 + gout58 * d_8 + gout59 * d_9,
                vj + (k0 + 5) + nao * (l0 + 0));
            // ijkl,kl->ij
            d_0 = dm[(k0 + 0) + nao * (l0 + 0)];
            d_1 = dm[(k0 + 1) + nao * (l0 + 0)];
            d_2 = dm[(k0 + 2) + nao * (l0 + 0)];
            d_3 = dm[(k0 + 3) + nao * (l0 + 0)];
            d_4 = dm[(k0 + 4) + nao * (l0 + 0)];
            d_5 = dm[(k0 + 5) + nao * (l0 + 0)];
            reduce(gout0 * d_0 + gout10 * d_1 + gout20 * d_2 + gout30 * d_3 +
                       gout40 * d_4 + gout50 * d_5,
                vj + (i0 + 0) + nao * (j0 + 0));
            reduce(gout1 * d_0 + gout11 * d_1 + gout21 * d_2 + gout31 * d_3 +
                       gout41 * d_4 + gout51 * d_5,
                vj + (i0 + 1) + nao * (j0 + 0));
            reduce(gout2 * d_0 + gout12 * d_1 + gout22 * d_2 + gout32 * d_3 +
                       gout42 * d_4 + gout52 * d_5,
                vj + (i0 + 2) + nao * (j0 + 0));
            reduce(gout3 * d_0 + gout13 * d_1 + gout23 * d_2 + gout33 * d_3 +
                       gout43 * d_4 + gout53 * d_5,
                vj + (i0 + 3) + nao * (j0 + 0));
            reduce(gout4 * d_0 + gout14 * d_1 + gout24 * d_2 + gout34 * d_3 +
                       gout44 * d_4 + gout54 * d_5,
                vj + (i0 + 4) + nao * (j0 + 0));
            reduce(gout5 * d_0 + gout15 * d_1 + gout25 * d_2 + gout35 * d_3 +
                       gout45 * d_4 + gout55 * d_5,
                vj + (i0 + 5) + nao * (j0 + 0));
            reduce(gout6 * d_0 + gout16 * d_1 + gout26 * d_2 + gout36 * d_3 +
                       gout46 * d_4 + gout56 * d_5,
                vj + (i0 + 6) + nao * (j0 + 0));
            reduce(gout7 * d_0 + gout17 * d_1 + gout27 * d_2 + gout37 * d_3 +
                       gout47 * d_4 + gout57 * d_5,
                vj + (i0 + 7) + nao * (j0 + 0));
            reduce(gout8 * d_0 + gout18 * d_1 + gout28 * d_2 + gout38 * d_3 +
                       gout48 * d_4 + gout58 * d_5,
                vj + (i0 + 8) + nao * (j0 + 0));
            reduce(gout9 * d_0 + gout19 * d_1 + gout29 * d_2 + gout39 * d_3 +
                       gout49 * d_4 + gout59 * d_5,
                vj + (i0 + 9) + nao * (j0 + 0));
            vj += nao2;
        }
        if (vk != NULL) {
            // ijkl,jl->ik
            d_0 = dm[(j0 + 0) + nao * (l0 + 0)];
            reduce(gout0 * d_0, vk + (i0 + 0) + nao * (k0 + 0));
            reduce(gout1 * d_0, vk + (i0 + 1) + nao * (k0 + 0));
            reduce(gout2 * d_0, vk + (i0 + 2) + nao * (k0 + 0));
            reduce(gout3 * d_0, vk + (i0 + 3) + nao * (k0 + 0));
            reduce(gout4 * d_0, vk + (i0 + 4) + nao * (k0 + 0));
            reduce(gout5 * d_0, vk + (i0 + 5) + nao * (k0 + 0));
            reduce(gout6 * d_0, vk + (i0 + 6) + nao * (k0 + 0));
            reduce(gout7 * d_0, vk + (i0 + 7) + nao * (k0 + 0));
            reduce(gout8 * d_0, vk + (i0 + 8) + nao * (k0 + 0));
            reduce(gout9 * d_0, vk + (i0 + 9) + nao * (k0 + 0));
            reduce(gout10 * d_0, vk + (i0 + 0) + nao * (k0 + 1));
            reduce(gout11 * d_0, vk + (i0 + 1) + nao * (k0 + 1));
            reduce(gout12 * d_0, vk + (i0 + 2) + nao * (k0 + 1));
            reduce(gout13 * d_0, vk + (i0 + 3) + nao * (k0 + 1));
            reduce(gout14 * d_0, vk + (i0 + 4) + nao * (k0 + 1));
            reduce(gout15 * d_0, vk + (i0 + 5) + nao * (k0 + 1));
            reduce(gout16 * d_0, vk + (i0 + 6) + nao * (k0 + 1));
            reduce(gout17 * d_0, vk + (i0 + 7) + nao * (k0 + 1));
            reduce(gout18 * d_0, vk + (i0 + 8) + nao * (k0 + 1));
            reduce(gout19 * d_0, vk + (i0 + 9) + nao * (k0 + 1));
            reduce(gout20 * d_0, vk + (i0 + 0) + nao * (k0 + 2));
            reduce(gout21 * d_0, vk + (i0 + 1) + nao * (k0 + 2));
            reduce(gout22 * d_0, vk + (i0 + 2) + nao * (k0 + 2));
            reduce(gout23 * d_0, vk + (i0 + 3) + nao * (k0 + 2));
            reduce(gout24 * d_0, vk + (i0 + 4) + nao * (k0 + 2));
            reduce(gout25 * d_0, vk + (i0 + 5) + nao * (k0 + 2));
            reduce(gout26 * d_0, vk + (i0 + 6) + nao * (k0 + 2));
            reduce(gout27 * d_0, vk + (i0 + 7) + nao * (k0 + 2));
            reduce(gout28 * d_0, vk + (i0 + 8) + nao * (k0 + 2));
            reduce(gout29 * d_0, vk + (i0 + 9) + nao * (k0 + 2));
            reduce(gout30 * d_0, vk + (i0 + 0) + nao * (k0 + 3));
            reduce(gout31 * d_0, vk + (i0 + 1) + nao * (k0 + 3));
            reduce(gout32 * d_0, vk + (i0 + 2) + nao * (k0 + 3));
            reduce(gout33 * d_0, vk + (i0 + 3) + nao * (k0 + 3));
            reduce(gout34 * d_0, vk + (i0 + 4) + nao * (k0 + 3));
            reduce(gout35 * d_0, vk + (i0 + 5) + nao * (k0 + 3));
            reduce(gout36 * d_0, vk + (i0 + 6) + nao * (k0 + 3));
            reduce(gout37 * d_0, vk + (i0 + 7) + nao * (k0 + 3));
            reduce(gout38 * d_0, vk + (i0 + 8) + nao * (k0 + 3));
            reduce(gout39 * d_0, vk + (i0 + 9) + nao * (k0 + 3));
            reduce(gout40 * d_0, vk + (i0 + 0) + nao * (k0 + 4));
            reduce(gout41 * d_0, vk + (i0 + 1) + nao * (k0 + 4));
            reduce(gout42 * d_0, vk + (i0 + 2) + nao * (k0 + 4));
            reduce(gout43 * d_0, vk + (i0 + 3) + nao * (k0 + 4));
            reduce(gout44 * d_0, vk + (i0 + 4) + nao * (k0 + 4));
            reduce(gout45 * d_0, vk + (i0 + 5) + nao * (k0 + 4));
            reduce(gout46 * d_0, vk + (i0 + 6) + nao * (k0 + 4));
            reduce(gout47 * d_0, vk + (i0 + 7) + nao * (k0 + 4));
            reduce(gout48 * d_0, vk + (i0 + 8) + nao * (k0 + 4));
            reduce(gout49 * d_0, vk + (i0 + 9) + nao * (k0 + 4));
            reduce(gout50 * d_0, vk + (i0 + 0) + nao * (k0 + 5));
            reduce(gout51 * d_0, vk + (i0 + 1) + nao * (k0 + 5));
            reduce(gout52 * d_0, vk + (i0 + 2) + nao * (k0 + 5));
            reduce(gout53 * d_0, vk + (i0 + 3) + nao * (k0 + 5));
            reduce(gout54 * d_0, vk + (i0 + 4) + nao * (k0 + 5));
            reduce(gout55 * d_0, vk + (i0 + 5) + nao * (k0 + 5));
            reduce(gout56 * d_0, vk + (i0 + 6) + nao * (k0 + 5));
            reduce(gout57 * d_0, vk + (i0 + 7) + nao * (k0 + 5));
            reduce(gout58 * d_0, vk + (i0 + 8) + nao * (k0 + 5));
            reduce(gout59 * d_0, vk + (i0 + 9) + nao * (k0 + 5));
            // ijkl,jk->il
            d_0 = dm[(j0 + 0) + nao * (k0 + 0)];
            d_1 = dm[(j0 + 0) + nao * (k0 + 1)];
            d_2 = dm[(j0 + 0) + nao * (k0 + 2)];
            d_3 = dm[(j0 + 0) + nao * (k0 + 3)];
            d_4 = dm[(j0 + 0) + nao * (k0 + 4)];
            d_5 = dm[(j0 + 0) + nao * (k0 + 5)];
            reduce(gout0 * d_0 + gout10 * d_1 + gout20 * d_2 + gout30 * d_3 +
                       gout40 * d_4 + gout50 * d_5,
                vk + (i0 + 0) + nao * (l0 + 0));
            reduce(gout1 * d_0 + gout11 * d_1 + gout21 * d_2 + gout31 * d_3 +
                       gout41 * d_4 + gout51 * d_5,
                vk + (i0 + 1) + nao * (l0 + 0));
            reduce(gout2 * d_0 + gout12 * d_1 + gout22 * d_2 + gout32 * d_3 +
                       gout42 * d_4 + gout52 * d_5,
                vk + (i0 + 2) + nao * (l0 + 0));
            reduce(gout3 * d_0 + gout13 * d_1 + gout23 * d_2 + gout33 * d_3 +
                       gout43 * d_4 + gout53 * d_5,
                vk + (i0 + 3) + nao * (l0 + 0));
            reduce(gout4 * d_0 + gout14 * d_1 + gout24 * d_2 + gout34 * d_3 +
                       gout44 * d_4 + gout54 * d_5,
                vk + (i0 + 4) + nao * (l0 + 0));
            reduce(gout5 * d_0 + gout15 * d_1 + gout25 * d_2 + gout35 * d_3 +
                       gout45 * d_4 + gout55 * d_5,
                vk + (i0 + 5) + nao * (l0 + 0));
            reduce(gout6 * d_0 + gout16 * d_1 + gout26 * d_2 + gout36 * d_3 +
                       gout46 * d_4 + gout56 * d_5,
                vk + (i0 + 6) + nao * (l0 + 0));
            reduce(gout7 * d_0 + gout17 * d_1 + gout27 * d_2 + gout37 * d_3 +
                       gout47 * d_4 + gout57 * d_5,
                vk + (i0 + 7) + nao * (l0 + 0));
            reduce(gout8 * d_0 + gout18 * d_1 + gout28 * d_2 + gout38 * d_3 +
                       gout48 * d_4 + gout58 * d_5,
                vk + (i0 + 8) + nao * (l0 + 0));
            reduce(gout9 * d_0 + gout19 * d_1 + gout29 * d_2 + gout39 * d_3 +
                       gout49 * d_4 + gout59 * d_5,
                vk + (i0 + 9) + nao * (l0 + 0));
            // ijkl,il->jk
            d_0 = dm[(i0 + 0) + nao * (l0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (l0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (l0 + 0)];
            d_3 = dm[(i0 + 3) + nao * (l0 + 0)];
            d_4 = dm[(i0 + 4) + nao * (l0 + 0)];
            d_5 = dm[(i0 + 5) + nao * (l0 + 0)];
            d_6 = dm[(i0 + 6) + nao * (l0 + 0)];
            d_7 = dm[(i0 + 7) + nao * (l0 + 0)];
            d_8 = dm[(i0 + 8) + nao * (l0 + 0)];
            d_9 = dm[(i0 + 9) + nao * (l0 + 0)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout3 * d_3 +
                       gout4 * d_4 + gout5 * d_5 + gout6 * d_6 + gout7 * d_7 +
                       gout8 * d_8 + gout9 * d_9,
                vk + (j0 + 0) + nao * (k0 + 0));
            reduce(gout10 * d_0 + gout11 * d_1 + gout12 * d_2 + gout13 * d_3 +
                       gout14 * d_4 + gout15 * d_5 + gout16 * d_6 +
                       gout17 * d_7 + gout18 * d_8 + gout19 * d_9,
                vk + (j0 + 0) + nao * (k0 + 1));
            reduce(gout20 * d_0 + gout21 * d_1 + gout22 * d_2 + gout23 * d_3 +
                       gout24 * d_4 + gout25 * d_5 + gout26 * d_6 +
                       gout27 * d_7 + gout28 * d_8 + gout29 * d_9,
                vk + (j0 + 0) + nao * (k0 + 2));
            reduce(gout30 * d_0 + gout31 * d_1 + gout32 * d_2 + gout33 * d_3 +
                       gout34 * d_4 + gout35 * d_5 + gout36 * d_6 +
                       gout37 * d_7 + gout38 * d_8 + gout39 * d_9,
                vk + (j0 + 0) + nao * (k0 + 3));
            reduce(gout40 * d_0 + gout41 * d_1 + gout42 * d_2 + gout43 * d_3 +
                       gout44 * d_4 + gout45 * d_5 + gout46 * d_6 +
                       gout47 * d_7 + gout48 * d_8 + gout49 * d_9,
                vk + (j0 + 0) + nao * (k0 + 4));
            reduce(gout50 * d_0 + gout51 * d_1 + gout52 * d_2 + gout53 * d_3 +
                       gout54 * d_4 + gout55 * d_5 + gout56 * d_6 +
                       gout57 * d_7 + gout58 * d_8 + gout59 * d_9,
                vk + (j0 + 0) + nao * (k0 + 5));
            // ijkl,ik->jl
            d_0 = dm[(i0 + 0) + nao * (k0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (k0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (k0 + 0)];
            d_3 = dm[(i0 + 3) + nao * (k0 + 0)];
            d_4 = dm[(i0 + 4) + nao * (k0 + 0)];
            d_5 = dm[(i0 + 5) + nao * (k0 + 0)];
            d_6 = dm[(i0 + 6) + nao * (k0 + 0)];
            d_7 = dm[(i0 + 7) + nao * (k0 + 0)];
            d_8 = dm[(i0 + 8) + nao * (k0 + 0)];
            d_9 = dm[(i0 + 9) + nao * (k0 + 0)];
            d_10 = dm[(i0 + 0) + nao * (k0 + 1)];
            d_11 = dm[(i0 + 1) + nao * (k0 + 1)];
            d_12 = dm[(i0 + 2) + nao * (k0 + 1)];
            d_13 = dm[(i0 + 3) + nao * (k0 + 1)];
            d_14 = dm[(i0 + 4) + nao * (k0 + 1)];
            d_15 = dm[(i0 + 5) + nao * (k0 + 1)];
            d_16 = dm[(i0 + 6) + nao * (k0 + 1)];
            d_17 = dm[(i0 + 7) + nao * (k0 + 1)];
            d_18 = dm[(i0 + 8) + nao * (k0 + 1)];
            d_19 = dm[(i0 + 9) + nao * (k0 + 1)];
            d_20 = dm[(i0 + 0) + nao * (k0 + 2)];
            d_21 = dm[(i0 + 1) + nao * (k0 + 2)];
            d_22 = dm[(i0 + 2) + nao * (k0 + 2)];
            d_23 = dm[(i0 + 3) + nao * (k0 + 2)];
            d_24 = dm[(i0 + 4) + nao * (k0 + 2)];
            d_25 = dm[(i0 + 5) + nao * (k0 + 2)];
            d_26 = dm[(i0 + 6) + nao * (k0 + 2)];
            d_27 = dm[(i0 + 7) + nao * (k0 + 2)];
            d_28 = dm[(i0 + 8) + nao * (k0 + 2)];
            d_29 = dm[(i0 + 9) + nao * (k0 + 2)];
            d_30 = dm[(i0 + 0) + nao * (k0 + 3)];
            d_31 = dm[(i0 + 1) + nao * (k0 + 3)];
            d_32 = dm[(i0 + 2) + nao * (k0 + 3)];
            d_33 = dm[(i0 + 3) + nao * (k0 + 3)];
            d_34 = dm[(i0 + 4) + nao * (k0 + 3)];
            d_35 = dm[(i0 + 5) + nao * (k0 + 3)];
            d_36 = dm[(i0 + 6) + nao * (k0 + 3)];
            d_37 = dm[(i0 + 7) + nao * (k0 + 3)];
            d_38 = dm[(i0 + 8) + nao * (k0 + 3)];
            d_39 = dm[(i0 + 9) + nao * (k0 + 3)];
            d_40 = dm[(i0 + 0) + nao * (k0 + 4)];
            d_41 = dm[(i0 + 1) + nao * (k0 + 4)];
            d_42 = dm[(i0 + 2) + nao * (k0 + 4)];
            d_43 = dm[(i0 + 3) + nao * (k0 + 4)];
            d_44 = dm[(i0 + 4) + nao * (k0 + 4)];
            d_45 = dm[(i0 + 5) + nao * (k0 + 4)];
            d_46 = dm[(i0 + 6) + nao * (k0 + 4)];
            d_47 = dm[(i0 + 7) + nao * (k0 + 4)];
            d_48 = dm[(i0 + 8) + nao * (k0 + 4)];
            d_49 = dm[(i0 + 9) + nao * (k0 + 4)];
            d_50 = dm[(i0 + 0) + nao * (k0 + 5)];
            d_51 = dm[(i0 + 1) + nao * (k0 + 5)];
            d_52 = dm[(i0 + 2) + nao * (k0 + 5)];
            d_53 = dm[(i0 + 3) + nao * (k0 + 5)];
            d_54 = dm[(i0 + 4) + nao * (k0 + 5)];
            d_55 = dm[(i0 + 5) + nao * (k0 + 5)];
            d_56 = dm[(i0 + 6) + nao * (k0 + 5)];
            d_57 = dm[(i0 + 7) + nao * (k0 + 5)];
            d_58 = dm[(i0 + 8) + nao * (k0 + 5)];
            d_59 = dm[(i0 + 9) + nao * (k0 + 5)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout3 * d_3 +
                       gout4 * d_4 + gout5 * d_5 + gout6 * d_6 + gout7 * d_7 +
                       gout8 * d_8 + gout9 * d_9 + gout10 * d_10 +
                       gout11 * d_11 + gout12 * d_12 + gout13 * d_13 +
                       gout14 * d_14 + gout15 * d_15 + gout16 * d_16 +
                       gout17 * d_17 + gout18 * d_18 + gout19 * d_19 +
                       gout20 * d_20 + gout21 * d_21 + gout22 * d_22 +
                       gout23 * d_23 + gout24 * d_24 + gout25 * d_25 +
                       gout26 * d_26 + gout27 * d_27 + gout28 * d_28 +
                       gout29 * d_29 + gout30 * d_30 + gout31 * d_31 +
                       gout32 * d_32 + gout33 * d_33 + gout34 * d_34 +
                       gout35 * d_35 + gout36 * d_36 + gout37 * d_37 +
                       gout38 * d_38 + gout39 * d_39 + gout40 * d_40 +
                       gout41 * d_41 + gout42 * d_42 + gout43 * d_43 +
                       gout44 * d_44 + gout45 * d_45 + gout46 * d_46 +
                       gout47 * d_47 + gout48 * d_48 + gout49 * d_49 +
                       gout50 * d_50 + gout51 * d_51 + gout52 * d_52 +
                       gout53 * d_53 + gout54 * d_54 + gout55 * d_55 +
                       gout56 * d_56 + gout57 * d_57 + gout58 * d_58 +
                       gout59 * d_59,
                vk + (j0 + 0) + nao * (l0 + 0));
            vk += nao2;
        }
        dm += nao2;
    }
}

__global__ static void GINTint2e_jk_kernel3100(JKMatrix jk,
    BasisProdOffsets offsets, GINTEnvVars envs, BasisProdCache bpcache) {
    int ntasks_ij = offsets.ntasks_ij;
    long ntasks = ntasks_ij * offsets.ntasks_kl;
    long task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int igroup = nprim_ij * nprim_kl;
    ntasks *= igroup;
    if (task_ij >= ntasks)
        return;
    int kl = task_ij % nprim_kl;
    task_ij /= nprim_kl;
    int ij = task_ij % nprim_ij;
    task_ij /= nprim_ij;
    int task_kl = task_ij / ntasks_ij;
    task_ij = task_ij % ntasks_ij;

    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    if (bas_ij < bas_kl) {
        return;
    }
    double norm = envs.fac;
    if (bas_ij == bas_kl) {
        norm *= .5;
    }

    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *ao_loc = bpcache.ao_loc;
    int *bas_pair2bra = bpcache.bas_pair2bra;
    int *bas_pair2ket = bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    int k0 = ao_loc[ksh];
    int l0 = ao_loc[lsh];
    double *__restrict__ a12 = bpcache.a12;
    double *__restrict__ e12 = bpcache.e12;
    double *__restrict__ x12 = bpcache.x12;
    double *__restrict__ y12 = bpcache.y12;
    double *__restrict__ z12 = bpcache.z12;
    int i_dm;
    int nbas = bpcache.nbas;
    double *__restrict__ bas_x = bpcache.bas_coords;
    double *__restrict__ bas_y = bas_x + nbas;
    double *__restrict__ bas_z = bas_y + nbas;

    double gout0 = 0;
    double gout1 = 0;
    double gout2 = 0;
    double gout3 = 0;
    double gout4 = 0;
    double gout5 = 0;
    double gout6 = 0;
    double gout7 = 0;
    double gout8 = 0;
    double gout9 = 0;
    double gout10 = 0;
    double gout11 = 0;
    double gout12 = 0;
    double gout13 = 0;
    double gout14 = 0;
    double gout15 = 0;
    double gout16 = 0;
    double gout17 = 0;
    double gout18 = 0;
    double gout19 = 0;
    double gout20 = 0;
    double gout21 = 0;
    double gout22 = 0;
    double gout23 = 0;
    double gout24 = 0;
    double gout25 = 0;
    double gout26 = 0;
    double gout27 = 0;
    double gout28 = 0;
    double gout29 = 0;
    double xi = bas_x[ish];
    double yi = bas_y[ish];
    double zi = bas_z[ish];
    double xixj = xi - bas_x[jsh];
    double yiyj = yi - bas_y[jsh];
    double zizj = zi - bas_z[jsh];
    auto reduce = SegReduce<double>(igroup);
    ij += prim_ij;
    kl += prim_kl;
    double aij = a12[ij];
    double eij = e12[ij];
    double xij = x12[ij];
    double yij = y12[ij];
    double zij = z12[ij];
    double akl = a12[kl];
    double ekl = e12[kl];
    double xkl = x12[kl];
    double ykl = y12[kl];
    double zkl = z12[kl];
    double xijxkl = xij - xkl;
    double yijykl = yij - ykl;
    double zijzkl = zij - zkl;
    double aijkl = aij + akl;
    double a1 = aij * akl;
    double a0 = a1 / aijkl;
    double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
    double fac = norm * eij * ekl / (sqrt(aijkl) * a1);

    double rw[6];
    double root0, weight0;
    GINTrys_root<3>(x, rw);
    int irys;
    for (irys = 0; irys < 3; ++irys) {
        root0 = rw[irys];
        weight0 = rw[irys + 3];
        double u2 = a0 * root0;
        double tmp4 = .5 / (u2 * aijkl + a1);
        double b00 = u2 * tmp4;
        double tmp1 = 2 * b00;
        double tmp2 = tmp1 * akl;
        double b10 = b00 + tmp4 * akl;
        double c00x = xij - xi - tmp2 * xijxkl;
        double c00y = yij - yi - tmp2 * yijykl;
        double c00z = zij - zi - tmp2 * zijzkl;
        double g_0 = 1;
        double g_1 = c00x;
        double g_2 = c00x * c00x + b10;
        double g_3 = c00x * (2 * b10 + g_2);
        double g_4 = c00x + xixj;
        double g_5 = c00x * (c00x + xixj) + b10;
        double g_6 = c00x * (2 * b10 + g_2) + xixj * g_2;
        double g_7 = 3 * b10 * g_2 + c00x * g_3 + xixj * g_3;
        double g_8 = 1;
        double g_9 = c00y;
        double g_10 = c00y * c00y + b10;
        double g_11 = c00y * (2 * b10 + g_10);
        double g_12 = c00y + yiyj;
        double g_13 = c00y * (c00y + yiyj) + b10;
        double g_14 = c00y * (2 * b10 + g_10) + yiyj * g_10;
        double g_15 = 3 * b10 * g_10 + c00y * g_11 + yiyj * g_11;
        double g_16 = weight0 * fac;
        double g_17 = c00z * g_16;
        double g_18 = b10 * g_16 + c00z * g_17;
        double g_19 = 2 * b10 * g_17 + c00z * g_18;
        double g_20 = g_16 * (c00z + zizj);
        double g_21 = b10 * g_16 + c00z * g_17 + zizj * g_17;
        double g_22 = 2 * b10 * g_17 + c00z * g_18 + zizj * g_18;
        double g_23 = 3 * b10 * g_18 + c00z * g_19 + zizj * g_19;
        gout0 += g_7 * g_8 * g_16;
        gout1 += g_6 * g_9 * g_16;
        gout2 += g_6 * g_8 * g_17;
        gout3 += g_5 * g_10 * g_16;
        gout4 += g_5 * g_9 * g_17;
        gout5 += g_5 * g_8 * g_18;
        gout6 += g_4 * g_11 * g_16;
        gout7 += g_4 * g_10 * g_17;
        gout8 += g_4 * g_9 * g_18;
        gout9 += g_4 * g_8 * g_19;
        gout10 += g_3 * g_12 * g_16;
        gout11 += g_2 * g_13 * g_16;
        gout12 += g_2 * g_12 * g_17;
        gout13 += g_1 * g_14 * g_16;
        gout14 += g_1 * g_13 * g_17;
        gout15 += g_1 * g_12 * g_18;
        gout16 += g_0 * g_15 * g_16;
        gout17 += g_0 * g_14 * g_17;
        gout18 += g_0 * g_13 * g_18;
        gout19 += g_0 * g_12 * g_19;
        gout20 += g_3 * g_8 * g_20;
        gout21 += g_2 * g_9 * g_20;
        gout22 += g_2 * g_8 * g_21;
        gout23 += g_1 * g_10 * g_20;
        gout24 += g_1 * g_9 * g_21;
        gout25 += g_1 * g_8 * g_22;
        gout26 += g_0 * g_11 * g_20;
        gout27 += g_0 * g_10 * g_21;
        gout28 += g_0 * g_9 * g_22;
        gout29 += g_0 * g_8 * g_23;
    }
    double d_0, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, d_9;
    double d_10, d_11, d_12, d_13, d_14, d_15, d_16, d_17, d_18, d_19;
    double d_20, d_21, d_22, d_23, d_24, d_25, d_26, d_27, d_28, d_29;
    int n_dm = jk.n_dm;
    int nao = jk.nao;
    size_t nao2 = nao * nao;
    double *__restrict__ dm = jk.dm;
    double *vj = jk.vj;
    double *vk = jk.vk;
    for (i_dm = 0; i_dm < n_dm; ++i_dm) {
        if (vj != NULL) {
            // ijkl,ij->kl
            d_0 = dm[(i0 + 0) + nao * (j0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (j0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (j0 + 0)];
            d_3 = dm[(i0 + 3) + nao * (j0 + 0)];
            d_4 = dm[(i0 + 4) + nao * (j0 + 0)];
            d_5 = dm[(i0 + 5) + nao * (j0 + 0)];
            d_6 = dm[(i0 + 6) + nao * (j0 + 0)];
            d_7 = dm[(i0 + 7) + nao * (j0 + 0)];
            d_8 = dm[(i0 + 8) + nao * (j0 + 0)];
            d_9 = dm[(i0 + 9) + nao * (j0 + 0)];
            d_10 = dm[(i0 + 0) + nao * (j0 + 1)];
            d_11 = dm[(i0 + 1) + nao * (j0 + 1)];
            d_12 = dm[(i0 + 2) + nao * (j0 + 1)];
            d_13 = dm[(i0 + 3) + nao * (j0 + 1)];
            d_14 = dm[(i0 + 4) + nao * (j0 + 1)];
            d_15 = dm[(i0 + 5) + nao * (j0 + 1)];
            d_16 = dm[(i0 + 6) + nao * (j0 + 1)];
            d_17 = dm[(i0 + 7) + nao * (j0 + 1)];
            d_18 = dm[(i0 + 8) + nao * (j0 + 1)];
            d_19 = dm[(i0 + 9) + nao * (j0 + 1)];
            d_20 = dm[(i0 + 0) + nao * (j0 + 2)];
            d_21 = dm[(i0 + 1) + nao * (j0 + 2)];
            d_22 = dm[(i0 + 2) + nao * (j0 + 2)];
            d_23 = dm[(i0 + 3) + nao * (j0 + 2)];
            d_24 = dm[(i0 + 4) + nao * (j0 + 2)];
            d_25 = dm[(i0 + 5) + nao * (j0 + 2)];
            d_26 = dm[(i0 + 6) + nao * (j0 + 2)];
            d_27 = dm[(i0 + 7) + nao * (j0 + 2)];
            d_28 = dm[(i0 + 8) + nao * (j0 + 2)];
            d_29 = dm[(i0 + 9) + nao * (j0 + 2)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout3 * d_3 +
                       gout4 * d_4 + gout5 * d_5 + gout6 * d_6 + gout7 * d_7 +
                       gout8 * d_8 + gout9 * d_9 + gout10 * d_10 +
                       gout11 * d_11 + gout12 * d_12 + gout13 * d_13 +
                       gout14 * d_14 + gout15 * d_15 + gout16 * d_16 +
                       gout17 * d_17 + gout18 * d_18 + gout19 * d_19 +
                       gout20 * d_20 + gout21 * d_21 + gout22 * d_22 +
                       gout23 * d_23 + gout24 * d_24 + gout25 * d_25 +
                       gout26 * d_26 + gout27 * d_27 + gout28 * d_28 +
                       gout29 * d_29,
                vj + (k0 + 0) + nao * (l0 + 0));
            // ijkl,kl->ij
            d_0 = dm[(k0 + 0) + nao * (l0 + 0)];
            reduce(gout0 * d_0, vj + (i0 + 0) + nao * (j0 + 0));
            reduce(gout1 * d_0, vj + (i0 + 1) + nao * (j0 + 0));
            reduce(gout2 * d_0, vj + (i0 + 2) + nao * (j0 + 0));
            reduce(gout3 * d_0, vj + (i0 + 3) + nao * (j0 + 0));
            reduce(gout4 * d_0, vj + (i0 + 4) + nao * (j0 + 0));
            reduce(gout5 * d_0, vj + (i0 + 5) + nao * (j0 + 0));
            reduce(gout6 * d_0, vj + (i0 + 6) + nao * (j0 + 0));
            reduce(gout7 * d_0, vj + (i0 + 7) + nao * (j0 + 0));
            reduce(gout8 * d_0, vj + (i0 + 8) + nao * (j0 + 0));
            reduce(gout9 * d_0, vj + (i0 + 9) + nao * (j0 + 0));
            reduce(gout10 * d_0, vj + (i0 + 0) + nao * (j0 + 1));
            reduce(gout11 * d_0, vj + (i0 + 1) + nao * (j0 + 1));
            reduce(gout12 * d_0, vj + (i0 + 2) + nao * (j0 + 1));
            reduce(gout13 * d_0, vj + (i0 + 3) + nao * (j0 + 1));
            reduce(gout14 * d_0, vj + (i0 + 4) + nao * (j0 + 1));
            reduce(gout15 * d_0, vj + (i0 + 5) + nao * (j0 + 1));
            reduce(gout16 * d_0, vj + (i0 + 6) + nao * (j0 + 1));
            reduce(gout17 * d_0, vj + (i0 + 7) + nao * (j0 + 1));
            reduce(gout18 * d_0, vj + (i0 + 8) + nao * (j0 + 1));
            reduce(gout19 * d_0, vj + (i0 + 9) + nao * (j0 + 1));
            reduce(gout20 * d_0, vj + (i0 + 0) + nao * (j0 + 2));
            reduce(gout21 * d_0, vj + (i0 + 1) + nao * (j0 + 2));
            reduce(gout22 * d_0, vj + (i0 + 2) + nao * (j0 + 2));
            reduce(gout23 * d_0, vj + (i0 + 3) + nao * (j0 + 2));
            reduce(gout24 * d_0, vj + (i0 + 4) + nao * (j0 + 2));
            reduce(gout25 * d_0, vj + (i0 + 5) + nao * (j0 + 2));
            reduce(gout26 * d_0, vj + (i0 + 6) + nao * (j0 + 2));
            reduce(gout27 * d_0, vj + (i0 + 7) + nao * (j0 + 2));
            reduce(gout28 * d_0, vj + (i0 + 8) + nao * (j0 + 2));
            reduce(gout29 * d_0, vj + (i0 + 9) + nao * (j0 + 2));
            vj += nao2;
        }
        if (vk != NULL) {
            // ijkl,jl->ik
            d_0 = dm[(j0 + 0) + nao * (l0 + 0)];
            d_1 = dm[(j0 + 1) + nao * (l0 + 0)];
            d_2 = dm[(j0 + 2) + nao * (l0 + 0)];
            reduce(gout0 * d_0 + gout10 * d_1 + gout20 * d_2,
                vk + (i0 + 0) + nao * (k0 + 0));
            reduce(gout1 * d_0 + gout11 * d_1 + gout21 * d_2,
                vk + (i0 + 1) + nao * (k0 + 0));
            reduce(gout2 * d_0 + gout12 * d_1 + gout22 * d_2,
                vk + (i0 + 2) + nao * (k0 + 0));
            reduce(gout3 * d_0 + gout13 * d_1 + gout23 * d_2,
                vk + (i0 + 3) + nao * (k0 + 0));
            reduce(gout4 * d_0 + gout14 * d_1 + gout24 * d_2,
                vk + (i0 + 4) + nao * (k0 + 0));
            reduce(gout5 * d_0 + gout15 * d_1 + gout25 * d_2,
                vk + (i0 + 5) + nao * (k0 + 0));
            reduce(gout6 * d_0 + gout16 * d_1 + gout26 * d_2,
                vk + (i0 + 6) + nao * (k0 + 0));
            reduce(gout7 * d_0 + gout17 * d_1 + gout27 * d_2,
                vk + (i0 + 7) + nao * (k0 + 0));
            reduce(gout8 * d_0 + gout18 * d_1 + gout28 * d_2,
                vk + (i0 + 8) + nao * (k0 + 0));
            reduce(gout9 * d_0 + gout19 * d_1 + gout29 * d_2,
                vk + (i0 + 9) + nao * (k0 + 0));
            // ijkl,jk->il
            d_0 = dm[(j0 + 0) + nao * (k0 + 0)];
            d_1 = dm[(j0 + 1) + nao * (k0 + 0)];
            d_2 = dm[(j0 + 2) + nao * (k0 + 0)];
            reduce(gout0 * d_0 + gout10 * d_1 + gout20 * d_2,
                vk + (i0 + 0) + nao * (l0 + 0));
            reduce(gout1 * d_0 + gout11 * d_1 + gout21 * d_2,
                vk + (i0 + 1) + nao * (l0 + 0));
            reduce(gout2 * d_0 + gout12 * d_1 + gout22 * d_2,
                vk + (i0 + 2) + nao * (l0 + 0));
            reduce(gout3 * d_0 + gout13 * d_1 + gout23 * d_2,
                vk + (i0 + 3) + nao * (l0 + 0));
            reduce(gout4 * d_0 + gout14 * d_1 + gout24 * d_2,
                vk + (i0 + 4) + nao * (l0 + 0));
            reduce(gout5 * d_0 + gout15 * d_1 + gout25 * d_2,
                vk + (i0 + 5) + nao * (l0 + 0));
            reduce(gout6 * d_0 + gout16 * d_1 + gout26 * d_2,
                vk + (i0 + 6) + nao * (l0 + 0));
            reduce(gout7 * d_0 + gout17 * d_1 + gout27 * d_2,
                vk + (i0 + 7) + nao * (l0 + 0));
            reduce(gout8 * d_0 + gout18 * d_1 + gout28 * d_2,
                vk + (i0 + 8) + nao * (l0 + 0));
            reduce(gout9 * d_0 + gout19 * d_1 + gout29 * d_2,
                vk + (i0 + 9) + nao * (l0 + 0));
            // ijkl,il->jk
            d_0 = dm[(i0 + 0) + nao * (l0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (l0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (l0 + 0)];
            d_3 = dm[(i0 + 3) + nao * (l0 + 0)];
            d_4 = dm[(i0 + 4) + nao * (l0 + 0)];
            d_5 = dm[(i0 + 5) + nao * (l0 + 0)];
            d_6 = dm[(i0 + 6) + nao * (l0 + 0)];
            d_7 = dm[(i0 + 7) + nao * (l0 + 0)];
            d_8 = dm[(i0 + 8) + nao * (l0 + 0)];
            d_9 = dm[(i0 + 9) + nao * (l0 + 0)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout3 * d_3 +
                       gout4 * d_4 + gout5 * d_5 + gout6 * d_6 + gout7 * d_7 +
                       gout8 * d_8 + gout9 * d_9,
                vk + (j0 + 0) + nao * (k0 + 0));
            reduce(gout10 * d_0 + gout11 * d_1 + gout12 * d_2 + gout13 * d_3 +
                       gout14 * d_4 + gout15 * d_5 + gout16 * d_6 +
                       gout17 * d_7 + gout18 * d_8 + gout19 * d_9,
                vk + (j0 + 1) + nao * (k0 + 0));
            reduce(gout20 * d_0 + gout21 * d_1 + gout22 * d_2 + gout23 * d_3 +
                       gout24 * d_4 + gout25 * d_5 + gout26 * d_6 +
                       gout27 * d_7 + gout28 * d_8 + gout29 * d_9,
                vk + (j0 + 2) + nao * (k0 + 0));
            // ijkl,ik->jl
            d_0 = dm[(i0 + 0) + nao * (k0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (k0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (k0 + 0)];
            d_3 = dm[(i0 + 3) + nao * (k0 + 0)];
            d_4 = dm[(i0 + 4) + nao * (k0 + 0)];
            d_5 = dm[(i0 + 5) + nao * (k0 + 0)];
            d_6 = dm[(i0 + 6) + nao * (k0 + 0)];
            d_7 = dm[(i0 + 7) + nao * (k0 + 0)];
            d_8 = dm[(i0 + 8) + nao * (k0 + 0)];
            d_9 = dm[(i0 + 9) + nao * (k0 + 0)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout3 * d_3 +
                       gout4 * d_4 + gout5 * d_5 + gout6 * d_6 + gout7 * d_7 +
                       gout8 * d_8 + gout9 * d_9,
                vk + (j0 + 0) + nao * (l0 + 0));
            reduce(gout10 * d_0 + gout11 * d_1 + gout12 * d_2 + gout13 * d_3 +
                       gout14 * d_4 + gout15 * d_5 + gout16 * d_6 +
                       gout17 * d_7 + gout18 * d_8 + gout19 * d_9,
                vk + (j0 + 1) + nao * (l0 + 0));
            reduce(gout20 * d_0 + gout21 * d_1 + gout22 * d_2 + gout23 * d_3 +
                       gout24 * d_4 + gout25 * d_5 + gout26 * d_6 +
                       gout27 * d_7 + gout28 * d_8 + gout29 * d_9,
                vk + (j0 + 2) + nao * (l0 + 0));
            vk += nao2;
        }
        dm += nao2;
    }
}

__global__ static void GINTint2e_jk_kernel3110(JKMatrix jk,
    BasisProdOffsets offsets, GINTEnvVars envs, BasisProdCache bpcache) {
    int ntasks_ij = offsets.ntasks_ij;
    long ntasks = ntasks_ij * offsets.ntasks_kl;
    long task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int igroup = nprim_ij * nprim_kl;
    ntasks *= igroup;
    if (task_ij >= ntasks)
        return;
    int kl = task_ij % nprim_kl;
    task_ij /= nprim_kl;
    int ij = task_ij % nprim_ij;
    task_ij /= nprim_ij;
    int task_kl = task_ij / ntasks_ij;
    task_ij = task_ij % ntasks_ij;

    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    if (bas_ij < bas_kl) {
        return;
    }
    double norm = envs.fac;
    if (bas_ij == bas_kl) {
        norm *= .5;
    }

    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *ao_loc = bpcache.ao_loc;
    int *bas_pair2bra = bpcache.bas_pair2bra;
    int *bas_pair2ket = bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    int k0 = ao_loc[ksh];
    int l0 = ao_loc[lsh];
    double *__restrict__ a12 = bpcache.a12;
    double *__restrict__ e12 = bpcache.e12;
    double *__restrict__ x12 = bpcache.x12;
    double *__restrict__ y12 = bpcache.y12;
    double *__restrict__ z12 = bpcache.z12;
    int i_dm;
    int nbas = bpcache.nbas;
    double *__restrict__ bas_x = bpcache.bas_coords;
    double *__restrict__ bas_y = bas_x + nbas;
    double *__restrict__ bas_z = bas_y + nbas;

    double gout0 = 0;
    double gout1 = 0;
    double gout2 = 0;
    double gout3 = 0;
    double gout4 = 0;
    double gout5 = 0;
    double gout6 = 0;
    double gout7 = 0;
    double gout8 = 0;
    double gout9 = 0;
    double gout10 = 0;
    double gout11 = 0;
    double gout12 = 0;
    double gout13 = 0;
    double gout14 = 0;
    double gout15 = 0;
    double gout16 = 0;
    double gout17 = 0;
    double gout18 = 0;
    double gout19 = 0;
    double gout20 = 0;
    double gout21 = 0;
    double gout22 = 0;
    double gout23 = 0;
    double gout24 = 0;
    double gout25 = 0;
    double gout26 = 0;
    double gout27 = 0;
    double gout28 = 0;
    double gout29 = 0;
    double gout30 = 0;
    double gout31 = 0;
    double gout32 = 0;
    double gout33 = 0;
    double gout34 = 0;
    double gout35 = 0;
    double gout36 = 0;
    double gout37 = 0;
    double gout38 = 0;
    double gout39 = 0;
    double gout40 = 0;
    double gout41 = 0;
    double gout42 = 0;
    double gout43 = 0;
    double gout44 = 0;
    double gout45 = 0;
    double gout46 = 0;
    double gout47 = 0;
    double gout48 = 0;
    double gout49 = 0;
    double gout50 = 0;
    double gout51 = 0;
    double gout52 = 0;
    double gout53 = 0;
    double gout54 = 0;
    double gout55 = 0;
    double gout56 = 0;
    double gout57 = 0;
    double gout58 = 0;
    double gout59 = 0;
    double gout60 = 0;
    double gout61 = 0;
    double gout62 = 0;
    double gout63 = 0;
    double gout64 = 0;
    double gout65 = 0;
    double gout66 = 0;
    double gout67 = 0;
    double gout68 = 0;
    double gout69 = 0;
    double gout70 = 0;
    double gout71 = 0;
    double gout72 = 0;
    double gout73 = 0;
    double gout74 = 0;
    double gout75 = 0;
    double gout76 = 0;
    double gout77 = 0;
    double gout78 = 0;
    double gout79 = 0;
    double gout80 = 0;
    double gout81 = 0;
    double gout82 = 0;
    double gout83 = 0;
    double gout84 = 0;
    double gout85 = 0;
    double gout86 = 0;
    double gout87 = 0;
    double gout88 = 0;
    double gout89 = 0;
    double xi = bas_x[ish];
    double yi = bas_y[ish];
    double zi = bas_z[ish];
    double xixj = xi - bas_x[jsh];
    double yiyj = yi - bas_y[jsh];
    double zizj = zi - bas_z[jsh];
    double xk = bas_x[ksh];
    double yk = bas_y[ksh];
    double zk = bas_z[ksh];
    auto reduce = SegReduce<double>(igroup);
    ij += prim_ij;
    kl += prim_kl;
    double aij = a12[ij];
    double eij = e12[ij];
    double xij = x12[ij];
    double yij = y12[ij];
    double zij = z12[ij];
    double akl = a12[kl];
    double ekl = e12[kl];
    double xkl = x12[kl];
    double ykl = y12[kl];
    double zkl = z12[kl];
    double xijxkl = xij - xkl;
    double yijykl = yij - ykl;
    double zijzkl = zij - zkl;
    double aijkl = aij + akl;
    double a1 = aij * akl;
    double a0 = a1 / aijkl;
    double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
    double fac = norm * eij * ekl / (sqrt(aijkl) * a1);

    double rw[6];
    double root0, weight0;
    GINTrys_root<3>(x, rw);
    int irys;
    for (irys = 0; irys < 3; ++irys) {
        root0 = rw[irys];
        weight0 = rw[irys + 3];
        double u2 = a0 * root0;
        double tmp4 = .5 / (u2 * aijkl + a1);
        double b00 = u2 * tmp4;
        double tmp1 = 2 * b00;
        double tmp2 = tmp1 * akl;
        double b10 = b00 + tmp4 * akl;
        double c00x = xij - xi - tmp2 * xijxkl;
        double c00y = yij - yi - tmp2 * yijykl;
        double c00z = zij - zi - tmp2 * zijzkl;
        double tmp3 = tmp1 * aij;
        double c0px = xkl - xk + tmp3 * xijxkl;
        double c0py = ykl - yk + tmp3 * yijykl;
        double c0pz = zkl - zk + tmp3 * zijzkl;
        double g_0 = 1;
        double g_1 = c00x;
        double g_2 = c00x * c00x + b10;
        double g_3 = c00x * (2 * b10 + g_2);
        double g_4 = c00x + xixj;
        double g_5 = c00x * (c00x + xixj) + b10;
        double g_6 = c00x * (2 * b10 + g_2) + xixj * g_2;
        double g_7 = 3 * b10 * g_2 + c00x * g_3 + xixj * g_3;
        double g_8 = c0px;
        double g_9 = c0px * c00x + b00;
        double g_10 = b00 * c00x + b10 * c0px + c00x * g_9;
        double g_11 = 2 * b10 * g_9 + b00 * g_2 + c00x * g_10;
        double g_12 = c0px * (c00x + xixj) + b00;
        double g_13 = b00 * c00x + b10 * c0px + c00x * g_9 + xixj * g_9;
        double g_14 = 2 * b10 * g_9 + b00 * g_2 + c00x * g_10 + xixj * g_10;
        double g_15 = 3 * b10 * g_10 + b00 * g_3 + c00x * g_11 + xixj * g_11;
        double g_16 = 1;
        double g_17 = c00y;
        double g_18 = c00y * c00y + b10;
        double g_19 = c00y * (2 * b10 + g_18);
        double g_20 = c00y + yiyj;
        double g_21 = c00y * (c00y + yiyj) + b10;
        double g_22 = c00y * (2 * b10 + g_18) + yiyj * g_18;
        double g_23 = 3 * b10 * g_18 + c00y * g_19 + yiyj * g_19;
        double g_24 = c0py;
        double g_25 = c0py * c00y + b00;
        double g_26 = b00 * c00y + b10 * c0py + c00y * g_25;
        double g_27 = 2 * b10 * g_25 + b00 * g_18 + c00y * g_26;
        double g_28 = c0py * (c00y + yiyj) + b00;
        double g_29 = b00 * c00y + b10 * c0py + c00y * g_25 + yiyj * g_25;
        double g_30 = 2 * b10 * g_25 + b00 * g_18 + c00y * g_26 + yiyj * g_26;
        double g_31 = 3 * b10 * g_26 + b00 * g_19 + c00y * g_27 + yiyj * g_27;
        double g_32 = weight0 * fac;
        double g_33 = c00z * g_32;
        double g_34 = b10 * g_32 + c00z * g_33;
        double g_35 = 2 * b10 * g_33 + c00z * g_34;
        double g_36 = g_32 * (c00z + zizj);
        double g_37 = b10 * g_32 + c00z * g_33 + zizj * g_33;
        double g_38 = 2 * b10 * g_33 + c00z * g_34 + zizj * g_34;
        double g_39 = 3 * b10 * g_34 + c00z * g_35 + zizj * g_35;
        double g_40 = c0pz * g_32;
        double g_41 = b00 * g_32 + c0pz * g_33;
        double g_42 = b00 * g_33 + b10 * g_40 + c00z * g_41;
        double g_43 = 2 * b10 * g_41 + b00 * g_34 + c00z * g_42;
        double g_44 = b00 * g_32 + c0pz * g_33 + zizj * g_40;
        double g_45 = b00 * g_33 + b10 * g_40 + c00z * g_41 + zizj * g_41;
        double g_46 = 2 * b10 * g_41 + b00 * g_34 + c00z * g_42 + zizj * g_42;
        double g_47 = 3 * b10 * g_42 + b00 * g_35 + c00z * g_43 + zizj * g_43;
        gout0 += g_15 * g_16 * g_32;
        gout1 += g_14 * g_17 * g_32;
        gout2 += g_14 * g_16 * g_33;
        gout3 += g_13 * g_18 * g_32;
        gout4 += g_13 * g_17 * g_33;
        gout5 += g_13 * g_16 * g_34;
        gout6 += g_12 * g_19 * g_32;
        gout7 += g_12 * g_18 * g_33;
        gout8 += g_12 * g_17 * g_34;
        gout9 += g_12 * g_16 * g_35;
        gout10 += g_11 * g_20 * g_32;
        gout11 += g_10 * g_21 * g_32;
        gout12 += g_10 * g_20 * g_33;
        gout13 += g_9 * g_22 * g_32;
        gout14 += g_9 * g_21 * g_33;
        gout15 += g_9 * g_20 * g_34;
        gout16 += g_8 * g_23 * g_32;
        gout17 += g_8 * g_22 * g_33;
        gout18 += g_8 * g_21 * g_34;
        gout19 += g_8 * g_20 * g_35;
        gout20 += g_11 * g_16 * g_36;
        gout21 += g_10 * g_17 * g_36;
        gout22 += g_10 * g_16 * g_37;
        gout23 += g_9 * g_18 * g_36;
        gout24 += g_9 * g_17 * g_37;
        gout25 += g_9 * g_16 * g_38;
        gout26 += g_8 * g_19 * g_36;
        gout27 += g_8 * g_18 * g_37;
        gout28 += g_8 * g_17 * g_38;
        gout29 += g_8 * g_16 * g_39;
        gout30 += g_7 * g_24 * g_32;
        gout31 += g_6 * g_25 * g_32;
        gout32 += g_6 * g_24 * g_33;
        gout33 += g_5 * g_26 * g_32;
        gout34 += g_5 * g_25 * g_33;
        gout35 += g_5 * g_24 * g_34;
        gout36 += g_4 * g_27 * g_32;
        gout37 += g_4 * g_26 * g_33;
        gout38 += g_4 * g_25 * g_34;
        gout39 += g_4 * g_24 * g_35;
        gout40 += g_3 * g_28 * g_32;
        gout41 += g_2 * g_29 * g_32;
        gout42 += g_2 * g_28 * g_33;
        gout43 += g_1 * g_30 * g_32;
        gout44 += g_1 * g_29 * g_33;
        gout45 += g_1 * g_28 * g_34;
        gout46 += g_0 * g_31 * g_32;
        gout47 += g_0 * g_30 * g_33;
        gout48 += g_0 * g_29 * g_34;
        gout49 += g_0 * g_28 * g_35;
        gout50 += g_3 * g_24 * g_36;
        gout51 += g_2 * g_25 * g_36;
        gout52 += g_2 * g_24 * g_37;
        gout53 += g_1 * g_26 * g_36;
        gout54 += g_1 * g_25 * g_37;
        gout55 += g_1 * g_24 * g_38;
        gout56 += g_0 * g_27 * g_36;
        gout57 += g_0 * g_26 * g_37;
        gout58 += g_0 * g_25 * g_38;
        gout59 += g_0 * g_24 * g_39;
        gout60 += g_7 * g_16 * g_40;
        gout61 += g_6 * g_17 * g_40;
        gout62 += g_6 * g_16 * g_41;
        gout63 += g_5 * g_18 * g_40;
        gout64 += g_5 * g_17 * g_41;
        gout65 += g_5 * g_16 * g_42;
        gout66 += g_4 * g_19 * g_40;
        gout67 += g_4 * g_18 * g_41;
        gout68 += g_4 * g_17 * g_42;
        gout69 += g_4 * g_16 * g_43;
        gout70 += g_3 * g_20 * g_40;
        gout71 += g_2 * g_21 * g_40;
        gout72 += g_2 * g_20 * g_41;
        gout73 += g_1 * g_22 * g_40;
        gout74 += g_1 * g_21 * g_41;
        gout75 += g_1 * g_20 * g_42;
        gout76 += g_0 * g_23 * g_40;
        gout77 += g_0 * g_22 * g_41;
        gout78 += g_0 * g_21 * g_42;
        gout79 += g_0 * g_20 * g_43;
        gout80 += g_3 * g_16 * g_44;
        gout81 += g_2 * g_17 * g_44;
        gout82 += g_2 * g_16 * g_45;
        gout83 += g_1 * g_18 * g_44;
        gout84 += g_1 * g_17 * g_45;
        gout85 += g_1 * g_16 * g_46;
        gout86 += g_0 * g_19 * g_44;
        gout87 += g_0 * g_18 * g_45;
        gout88 += g_0 * g_17 * g_46;
        gout89 += g_0 * g_16 * g_47;
    }
    double d_0, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, d_9;
    double d_10, d_11, d_12, d_13, d_14, d_15, d_16, d_17, d_18, d_19;
    double d_20, d_21, d_22, d_23, d_24, d_25, d_26, d_27, d_28, d_29;
    int n_dm = jk.n_dm;
    int nao = jk.nao;
    size_t nao2 = nao * nao;
    double *__restrict__ dm = jk.dm;
    double *vj = jk.vj;
    double *vk = jk.vk;
    for (i_dm = 0; i_dm < n_dm; ++i_dm) {
        if (vj != NULL) {
            // ijkl,ij->kl
            d_0 = dm[(i0 + 0) + nao * (j0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (j0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (j0 + 0)];
            d_3 = dm[(i0 + 3) + nao * (j0 + 0)];
            d_4 = dm[(i0 + 4) + nao * (j0 + 0)];
            d_5 = dm[(i0 + 5) + nao * (j0 + 0)];
            d_6 = dm[(i0 + 6) + nao * (j0 + 0)];
            d_7 = dm[(i0 + 7) + nao * (j0 + 0)];
            d_8 = dm[(i0 + 8) + nao * (j0 + 0)];
            d_9 = dm[(i0 + 9) + nao * (j0 + 0)];
            d_10 = dm[(i0 + 0) + nao * (j0 + 1)];
            d_11 = dm[(i0 + 1) + nao * (j0 + 1)];
            d_12 = dm[(i0 + 2) + nao * (j0 + 1)];
            d_13 = dm[(i0 + 3) + nao * (j0 + 1)];
            d_14 = dm[(i0 + 4) + nao * (j0 + 1)];
            d_15 = dm[(i0 + 5) + nao * (j0 + 1)];
            d_16 = dm[(i0 + 6) + nao * (j0 + 1)];
            d_17 = dm[(i0 + 7) + nao * (j0 + 1)];
            d_18 = dm[(i0 + 8) + nao * (j0 + 1)];
            d_19 = dm[(i0 + 9) + nao * (j0 + 1)];
            d_20 = dm[(i0 + 0) + nao * (j0 + 2)];
            d_21 = dm[(i0 + 1) + nao * (j0 + 2)];
            d_22 = dm[(i0 + 2) + nao * (j0 + 2)];
            d_23 = dm[(i0 + 3) + nao * (j0 + 2)];
            d_24 = dm[(i0 + 4) + nao * (j0 + 2)];
            d_25 = dm[(i0 + 5) + nao * (j0 + 2)];
            d_26 = dm[(i0 + 6) + nao * (j0 + 2)];
            d_27 = dm[(i0 + 7) + nao * (j0 + 2)];
            d_28 = dm[(i0 + 8) + nao * (j0 + 2)];
            d_29 = dm[(i0 + 9) + nao * (j0 + 2)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout3 * d_3 +
                       gout4 * d_4 + gout5 * d_5 + gout6 * d_6 + gout7 * d_7 +
                       gout8 * d_8 + gout9 * d_9 + gout10 * d_10 +
                       gout11 * d_11 + gout12 * d_12 + gout13 * d_13 +
                       gout14 * d_14 + gout15 * d_15 + gout16 * d_16 +
                       gout17 * d_17 + gout18 * d_18 + gout19 * d_19 +
                       gout20 * d_20 + gout21 * d_21 + gout22 * d_22 +
                       gout23 * d_23 + gout24 * d_24 + gout25 * d_25 +
                       gout26 * d_26 + gout27 * d_27 + gout28 * d_28 +
                       gout29 * d_29,
                vj + (k0 + 0) + nao * (l0 + 0));
            reduce(gout30 * d_0 + gout31 * d_1 + gout32 * d_2 + gout33 * d_3 +
                       gout34 * d_4 + gout35 * d_5 + gout36 * d_6 +
                       gout37 * d_7 + gout38 * d_8 + gout39 * d_9 +
                       gout40 * d_10 + gout41 * d_11 + gout42 * d_12 +
                       gout43 * d_13 + gout44 * d_14 + gout45 * d_15 +
                       gout46 * d_16 + gout47 * d_17 + gout48 * d_18 +
                       gout49 * d_19 + gout50 * d_20 + gout51 * d_21 +
                       gout52 * d_22 + gout53 * d_23 + gout54 * d_24 +
                       gout55 * d_25 + gout56 * d_26 + gout57 * d_27 +
                       gout58 * d_28 + gout59 * d_29,
                vj + (k0 + 1) + nao * (l0 + 0));
            reduce(gout60 * d_0 + gout61 * d_1 + gout62 * d_2 + gout63 * d_3 +
                       gout64 * d_4 + gout65 * d_5 + gout66 * d_6 +
                       gout67 * d_7 + gout68 * d_8 + gout69 * d_9 +
                       gout70 * d_10 + gout71 * d_11 + gout72 * d_12 +
                       gout73 * d_13 + gout74 * d_14 + gout75 * d_15 +
                       gout76 * d_16 + gout77 * d_17 + gout78 * d_18 +
                       gout79 * d_19 + gout80 * d_20 + gout81 * d_21 +
                       gout82 * d_22 + gout83 * d_23 + gout84 * d_24 +
                       gout85 * d_25 + gout86 * d_26 + gout87 * d_27 +
                       gout88 * d_28 + gout89 * d_29,
                vj + (k0 + 2) + nao * (l0 + 0));
            // ijkl,kl->ij
            d_0 = dm[(k0 + 0) + nao * (l0 + 0)];
            d_1 = dm[(k0 + 1) + nao * (l0 + 0)];
            d_2 = dm[(k0 + 2) + nao * (l0 + 0)];
            reduce(gout0 * d_0 + gout30 * d_1 + gout60 * d_2,
                vj + (i0 + 0) + nao * (j0 + 0));
            reduce(gout1 * d_0 + gout31 * d_1 + gout61 * d_2,
                vj + (i0 + 1) + nao * (j0 + 0));
            reduce(gout2 * d_0 + gout32 * d_1 + gout62 * d_2,
                vj + (i0 + 2) + nao * (j0 + 0));
            reduce(gout3 * d_0 + gout33 * d_1 + gout63 * d_2,
                vj + (i0 + 3) + nao * (j0 + 0));
            reduce(gout4 * d_0 + gout34 * d_1 + gout64 * d_2,
                vj + (i0 + 4) + nao * (j0 + 0));
            reduce(gout5 * d_0 + gout35 * d_1 + gout65 * d_2,
                vj + (i0 + 5) + nao * (j0 + 0));
            reduce(gout6 * d_0 + gout36 * d_1 + gout66 * d_2,
                vj + (i0 + 6) + nao * (j0 + 0));
            reduce(gout7 * d_0 + gout37 * d_1 + gout67 * d_2,
                vj + (i0 + 7) + nao * (j0 + 0));
            reduce(gout8 * d_0 + gout38 * d_1 + gout68 * d_2,
                vj + (i0 + 8) + nao * (j0 + 0));
            reduce(gout9 * d_0 + gout39 * d_1 + gout69 * d_2,
                vj + (i0 + 9) + nao * (j0 + 0));
            reduce(gout10 * d_0 + gout40 * d_1 + gout70 * d_2,
                vj + (i0 + 0) + nao * (j0 + 1));
            reduce(gout11 * d_0 + gout41 * d_1 + gout71 * d_2,
                vj + (i0 + 1) + nao * (j0 + 1));
            reduce(gout12 * d_0 + gout42 * d_1 + gout72 * d_2,
                vj + (i0 + 2) + nao * (j0 + 1));
            reduce(gout13 * d_0 + gout43 * d_1 + gout73 * d_2,
                vj + (i0 + 3) + nao * (j0 + 1));
            reduce(gout14 * d_0 + gout44 * d_1 + gout74 * d_2,
                vj + (i0 + 4) + nao * (j0 + 1));
            reduce(gout15 * d_0 + gout45 * d_1 + gout75 * d_2,
                vj + (i0 + 5) + nao * (j0 + 1));
            reduce(gout16 * d_0 + gout46 * d_1 + gout76 * d_2,
                vj + (i0 + 6) + nao * (j0 + 1));
            reduce(gout17 * d_0 + gout47 * d_1 + gout77 * d_2,
                vj + (i0 + 7) + nao * (j0 + 1));
            reduce(gout18 * d_0 + gout48 * d_1 + gout78 * d_2,
                vj + (i0 + 8) + nao * (j0 + 1));
            reduce(gout19 * d_0 + gout49 * d_1 + gout79 * d_2,
                vj + (i0 + 9) + nao * (j0 + 1));
            reduce(gout20 * d_0 + gout50 * d_1 + gout80 * d_2,
                vj + (i0 + 0) + nao * (j0 + 2));
            reduce(gout21 * d_0 + gout51 * d_1 + gout81 * d_2,
                vj + (i0 + 1) + nao * (j0 + 2));
            reduce(gout22 * d_0 + gout52 * d_1 + gout82 * d_2,
                vj + (i0 + 2) + nao * (j0 + 2));
            reduce(gout23 * d_0 + gout53 * d_1 + gout83 * d_2,
                vj + (i0 + 3) + nao * (j0 + 2));
            reduce(gout24 * d_0 + gout54 * d_1 + gout84 * d_2,
                vj + (i0 + 4) + nao * (j0 + 2));
            reduce(gout25 * d_0 + gout55 * d_1 + gout85 * d_2,
                vj + (i0 + 5) + nao * (j0 + 2));
            reduce(gout26 * d_0 + gout56 * d_1 + gout86 * d_2,
                vj + (i0 + 6) + nao * (j0 + 2));
            reduce(gout27 * d_0 + gout57 * d_1 + gout87 * d_2,
                vj + (i0 + 7) + nao * (j0 + 2));
            reduce(gout28 * d_0 + gout58 * d_1 + gout88 * d_2,
                vj + (i0 + 8) + nao * (j0 + 2));
            reduce(gout29 * d_0 + gout59 * d_1 + gout89 * d_2,
                vj + (i0 + 9) + nao * (j0 + 2));
            vj += nao2;
        }
        if (vk != NULL) {
            // ijkl,jl->ik
            d_0 = dm[(j0 + 0) + nao * (l0 + 0)];
            d_1 = dm[(j0 + 1) + nao * (l0 + 0)];
            d_2 = dm[(j0 + 2) + nao * (l0 + 0)];
            reduce(gout0 * d_0 + gout10 * d_1 + gout20 * d_2,
                vk + (i0 + 0) + nao * (k0 + 0));
            reduce(gout1 * d_0 + gout11 * d_1 + gout21 * d_2,
                vk + (i0 + 1) + nao * (k0 + 0));
            reduce(gout2 * d_0 + gout12 * d_1 + gout22 * d_2,
                vk + (i0 + 2) + nao * (k0 + 0));
            reduce(gout3 * d_0 + gout13 * d_1 + gout23 * d_2,
                vk + (i0 + 3) + nao * (k0 + 0));
            reduce(gout4 * d_0 + gout14 * d_1 + gout24 * d_2,
                vk + (i0 + 4) + nao * (k0 + 0));
            reduce(gout5 * d_0 + gout15 * d_1 + gout25 * d_2,
                vk + (i0 + 5) + nao * (k0 + 0));
            reduce(gout6 * d_0 + gout16 * d_1 + gout26 * d_2,
                vk + (i0 + 6) + nao * (k0 + 0));
            reduce(gout7 * d_0 + gout17 * d_1 + gout27 * d_2,
                vk + (i0 + 7) + nao * (k0 + 0));
            reduce(gout8 * d_0 + gout18 * d_1 + gout28 * d_2,
                vk + (i0 + 8) + nao * (k0 + 0));
            reduce(gout9 * d_0 + gout19 * d_1 + gout29 * d_2,
                vk + (i0 + 9) + nao * (k0 + 0));
            reduce(gout30 * d_0 + gout40 * d_1 + gout50 * d_2,
                vk + (i0 + 0) + nao * (k0 + 1));
            reduce(gout31 * d_0 + gout41 * d_1 + gout51 * d_2,
                vk + (i0 + 1) + nao * (k0 + 1));
            reduce(gout32 * d_0 + gout42 * d_1 + gout52 * d_2,
                vk + (i0 + 2) + nao * (k0 + 1));
            reduce(gout33 * d_0 + gout43 * d_1 + gout53 * d_2,
                vk + (i0 + 3) + nao * (k0 + 1));
            reduce(gout34 * d_0 + gout44 * d_1 + gout54 * d_2,
                vk + (i0 + 4) + nao * (k0 + 1));
            reduce(gout35 * d_0 + gout45 * d_1 + gout55 * d_2,
                vk + (i0 + 5) + nao * (k0 + 1));
            reduce(gout36 * d_0 + gout46 * d_1 + gout56 * d_2,
                vk + (i0 + 6) + nao * (k0 + 1));
            reduce(gout37 * d_0 + gout47 * d_1 + gout57 * d_2,
                vk + (i0 + 7) + nao * (k0 + 1));
            reduce(gout38 * d_0 + gout48 * d_1 + gout58 * d_2,
                vk + (i0 + 8) + nao * (k0 + 1));
            reduce(gout39 * d_0 + gout49 * d_1 + gout59 * d_2,
                vk + (i0 + 9) + nao * (k0 + 1));
            reduce(gout60 * d_0 + gout70 * d_1 + gout80 * d_2,
                vk + (i0 + 0) + nao * (k0 + 2));
            reduce(gout61 * d_0 + gout71 * d_1 + gout81 * d_2,
                vk + (i0 + 1) + nao * (k0 + 2));
            reduce(gout62 * d_0 + gout72 * d_1 + gout82 * d_2,
                vk + (i0 + 2) + nao * (k0 + 2));
            reduce(gout63 * d_0 + gout73 * d_1 + gout83 * d_2,
                vk + (i0 + 3) + nao * (k0 + 2));
            reduce(gout64 * d_0 + gout74 * d_1 + gout84 * d_2,
                vk + (i0 + 4) + nao * (k0 + 2));
            reduce(gout65 * d_0 + gout75 * d_1 + gout85 * d_2,
                vk + (i0 + 5) + nao * (k0 + 2));
            reduce(gout66 * d_0 + gout76 * d_1 + gout86 * d_2,
                vk + (i0 + 6) + nao * (k0 + 2));
            reduce(gout67 * d_0 + gout77 * d_1 + gout87 * d_2,
                vk + (i0 + 7) + nao * (k0 + 2));
            reduce(gout68 * d_0 + gout78 * d_1 + gout88 * d_2,
                vk + (i0 + 8) + nao * (k0 + 2));
            reduce(gout69 * d_0 + gout79 * d_1 + gout89 * d_2,
                vk + (i0 + 9) + nao * (k0 + 2));
            // ijkl,jk->il
            d_0 = dm[(j0 + 0) + nao * (k0 + 0)];
            d_1 = dm[(j0 + 1) + nao * (k0 + 0)];
            d_2 = dm[(j0 + 2) + nao * (k0 + 0)];
            d_3 = dm[(j0 + 0) + nao * (k0 + 1)];
            d_4 = dm[(j0 + 1) + nao * (k0 + 1)];
            d_5 = dm[(j0 + 2) + nao * (k0 + 1)];
            d_6 = dm[(j0 + 0) + nao * (k0 + 2)];
            d_7 = dm[(j0 + 1) + nao * (k0 + 2)];
            d_8 = dm[(j0 + 2) + nao * (k0 + 2)];
            reduce(gout0 * d_0 + gout10 * d_1 + gout20 * d_2 + gout30 * d_3 +
                       gout40 * d_4 + gout50 * d_5 + gout60 * d_6 +
                       gout70 * d_7 + gout80 * d_8,
                vk + (i0 + 0) + nao * (l0 + 0));
            reduce(gout1 * d_0 + gout11 * d_1 + gout21 * d_2 + gout31 * d_3 +
                       gout41 * d_4 + gout51 * d_5 + gout61 * d_6 +
                       gout71 * d_7 + gout81 * d_8,
                vk + (i0 + 1) + nao * (l0 + 0));
            reduce(gout2 * d_0 + gout12 * d_1 + gout22 * d_2 + gout32 * d_3 +
                       gout42 * d_4 + gout52 * d_5 + gout62 * d_6 +
                       gout72 * d_7 + gout82 * d_8,
                vk + (i0 + 2) + nao * (l0 + 0));
            reduce(gout3 * d_0 + gout13 * d_1 + gout23 * d_2 + gout33 * d_3 +
                       gout43 * d_4 + gout53 * d_5 + gout63 * d_6 +
                       gout73 * d_7 + gout83 * d_8,
                vk + (i0 + 3) + nao * (l0 + 0));
            reduce(gout4 * d_0 + gout14 * d_1 + gout24 * d_2 + gout34 * d_3 +
                       gout44 * d_4 + gout54 * d_5 + gout64 * d_6 +
                       gout74 * d_7 + gout84 * d_8,
                vk + (i0 + 4) + nao * (l0 + 0));
            reduce(gout5 * d_0 + gout15 * d_1 + gout25 * d_2 + gout35 * d_3 +
                       gout45 * d_4 + gout55 * d_5 + gout65 * d_6 +
                       gout75 * d_7 + gout85 * d_8,
                vk + (i0 + 5) + nao * (l0 + 0));
            reduce(gout6 * d_0 + gout16 * d_1 + gout26 * d_2 + gout36 * d_3 +
                       gout46 * d_4 + gout56 * d_5 + gout66 * d_6 +
                       gout76 * d_7 + gout86 * d_8,
                vk + (i0 + 6) + nao * (l0 + 0));
            reduce(gout7 * d_0 + gout17 * d_1 + gout27 * d_2 + gout37 * d_3 +
                       gout47 * d_4 + gout57 * d_5 + gout67 * d_6 +
                       gout77 * d_7 + gout87 * d_8,
                vk + (i0 + 7) + nao * (l0 + 0));
            reduce(gout8 * d_0 + gout18 * d_1 + gout28 * d_2 + gout38 * d_3 +
                       gout48 * d_4 + gout58 * d_5 + gout68 * d_6 +
                       gout78 * d_7 + gout88 * d_8,
                vk + (i0 + 8) + nao * (l0 + 0));
            reduce(gout9 * d_0 + gout19 * d_1 + gout29 * d_2 + gout39 * d_3 +
                       gout49 * d_4 + gout59 * d_5 + gout69 * d_6 +
                       gout79 * d_7 + gout89 * d_8,
                vk + (i0 + 9) + nao * (l0 + 0));
            // ijkl,il->jk
            d_0 = dm[(i0 + 0) + nao * (l0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (l0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (l0 + 0)];
            d_3 = dm[(i0 + 3) + nao * (l0 + 0)];
            d_4 = dm[(i0 + 4) + nao * (l0 + 0)];
            d_5 = dm[(i0 + 5) + nao * (l0 + 0)];
            d_6 = dm[(i0 + 6) + nao * (l0 + 0)];
            d_7 = dm[(i0 + 7) + nao * (l0 + 0)];
            d_8 = dm[(i0 + 8) + nao * (l0 + 0)];
            d_9 = dm[(i0 + 9) + nao * (l0 + 0)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout3 * d_3 +
                       gout4 * d_4 + gout5 * d_5 + gout6 * d_6 + gout7 * d_7 +
                       gout8 * d_8 + gout9 * d_9,
                vk + (j0 + 0) + nao * (k0 + 0));
            reduce(gout10 * d_0 + gout11 * d_1 + gout12 * d_2 + gout13 * d_3 +
                       gout14 * d_4 + gout15 * d_5 + gout16 * d_6 +
                       gout17 * d_7 + gout18 * d_8 + gout19 * d_9,
                vk + (j0 + 1) + nao * (k0 + 0));
            reduce(gout20 * d_0 + gout21 * d_1 + gout22 * d_2 + gout23 * d_3 +
                       gout24 * d_4 + gout25 * d_5 + gout26 * d_6 +
                       gout27 * d_7 + gout28 * d_8 + gout29 * d_9,
                vk + (j0 + 2) + nao * (k0 + 0));
            reduce(gout30 * d_0 + gout31 * d_1 + gout32 * d_2 + gout33 * d_3 +
                       gout34 * d_4 + gout35 * d_5 + gout36 * d_6 +
                       gout37 * d_7 + gout38 * d_8 + gout39 * d_9,
                vk + (j0 + 0) + nao * (k0 + 1));
            reduce(gout40 * d_0 + gout41 * d_1 + gout42 * d_2 + gout43 * d_3 +
                       gout44 * d_4 + gout45 * d_5 + gout46 * d_6 +
                       gout47 * d_7 + gout48 * d_8 + gout49 * d_9,
                vk + (j0 + 1) + nao * (k0 + 1));
            reduce(gout50 * d_0 + gout51 * d_1 + gout52 * d_2 + gout53 * d_3 +
                       gout54 * d_4 + gout55 * d_5 + gout56 * d_6 +
                       gout57 * d_7 + gout58 * d_8 + gout59 * d_9,
                vk + (j0 + 2) + nao * (k0 + 1));
            reduce(gout60 * d_0 + gout61 * d_1 + gout62 * d_2 + gout63 * d_3 +
                       gout64 * d_4 + gout65 * d_5 + gout66 * d_6 +
                       gout67 * d_7 + gout68 * d_8 + gout69 * d_9,
                vk + (j0 + 0) + nao * (k0 + 2));
            reduce(gout70 * d_0 + gout71 * d_1 + gout72 * d_2 + gout73 * d_3 +
                       gout74 * d_4 + gout75 * d_5 + gout76 * d_6 +
                       gout77 * d_7 + gout78 * d_8 + gout79 * d_9,
                vk + (j0 + 1) + nao * (k0 + 2));
            reduce(gout80 * d_0 + gout81 * d_1 + gout82 * d_2 + gout83 * d_3 +
                       gout84 * d_4 + gout85 * d_5 + gout86 * d_6 +
                       gout87 * d_7 + gout88 * d_8 + gout89 * d_9,
                vk + (j0 + 2) + nao * (k0 + 2));
            // ijkl,ik->jl
            d_0 = dm[(i0 + 0) + nao * (k0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (k0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (k0 + 0)];
            d_3 = dm[(i0 + 3) + nao * (k0 + 0)];
            d_4 = dm[(i0 + 4) + nao * (k0 + 0)];
            d_5 = dm[(i0 + 5) + nao * (k0 + 0)];
            d_6 = dm[(i0 + 6) + nao * (k0 + 0)];
            d_7 = dm[(i0 + 7) + nao * (k0 + 0)];
            d_8 = dm[(i0 + 8) + nao * (k0 + 0)];
            d_9 = dm[(i0 + 9) + nao * (k0 + 0)];
            d_10 = dm[(i0 + 0) + nao * (k0 + 1)];
            d_11 = dm[(i0 + 1) + nao * (k0 + 1)];
            d_12 = dm[(i0 + 2) + nao * (k0 + 1)];
            d_13 = dm[(i0 + 3) + nao * (k0 + 1)];
            d_14 = dm[(i0 + 4) + nao * (k0 + 1)];
            d_15 = dm[(i0 + 5) + nao * (k0 + 1)];
            d_16 = dm[(i0 + 6) + nao * (k0 + 1)];
            d_17 = dm[(i0 + 7) + nao * (k0 + 1)];
            d_18 = dm[(i0 + 8) + nao * (k0 + 1)];
            d_19 = dm[(i0 + 9) + nao * (k0 + 1)];
            d_20 = dm[(i0 + 0) + nao * (k0 + 2)];
            d_21 = dm[(i0 + 1) + nao * (k0 + 2)];
            d_22 = dm[(i0 + 2) + nao * (k0 + 2)];
            d_23 = dm[(i0 + 3) + nao * (k0 + 2)];
            d_24 = dm[(i0 + 4) + nao * (k0 + 2)];
            d_25 = dm[(i0 + 5) + nao * (k0 + 2)];
            d_26 = dm[(i0 + 6) + nao * (k0 + 2)];
            d_27 = dm[(i0 + 7) + nao * (k0 + 2)];
            d_28 = dm[(i0 + 8) + nao * (k0 + 2)];
            d_29 = dm[(i0 + 9) + nao * (k0 + 2)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout3 * d_3 +
                       gout4 * d_4 + gout5 * d_5 + gout6 * d_6 + gout7 * d_7 +
                       gout8 * d_8 + gout9 * d_9 + gout30 * d_10 +
                       gout31 * d_11 + gout32 * d_12 + gout33 * d_13 +
                       gout34 * d_14 + gout35 * d_15 + gout36 * d_16 +
                       gout37 * d_17 + gout38 * d_18 + gout39 * d_19 +
                       gout60 * d_20 + gout61 * d_21 + gout62 * d_22 +
                       gout63 * d_23 + gout64 * d_24 + gout65 * d_25 +
                       gout66 * d_26 + gout67 * d_27 + gout68 * d_28 +
                       gout69 * d_29,
                vk + (j0 + 0) + nao * (l0 + 0));
            reduce(gout10 * d_0 + gout11 * d_1 + gout12 * d_2 + gout13 * d_3 +
                       gout14 * d_4 + gout15 * d_5 + gout16 * d_6 +
                       gout17 * d_7 + gout18 * d_8 + gout19 * d_9 +
                       gout40 * d_10 + gout41 * d_11 + gout42 * d_12 +
                       gout43 * d_13 + gout44 * d_14 + gout45 * d_15 +
                       gout46 * d_16 + gout47 * d_17 + gout48 * d_18 +
                       gout49 * d_19 + gout70 * d_20 + gout71 * d_21 +
                       gout72 * d_22 + gout73 * d_23 + gout74 * d_24 +
                       gout75 * d_25 + gout76 * d_26 + gout77 * d_27 +
                       gout78 * d_28 + gout79 * d_29,
                vk + (j0 + 1) + nao * (l0 + 0));
            reduce(gout20 * d_0 + gout21 * d_1 + gout22 * d_2 + gout23 * d_3 +
                       gout24 * d_4 + gout25 * d_5 + gout26 * d_6 +
                       gout27 * d_7 + gout28 * d_8 + gout29 * d_9 +
                       gout50 * d_10 + gout51 * d_11 + gout52 * d_12 +
                       gout53 * d_13 + gout54 * d_14 + gout55 * d_15 +
                       gout56 * d_16 + gout57 * d_17 + gout58 * d_18 +
                       gout59 * d_19 + gout80 * d_20 + gout81 * d_21 +
                       gout82 * d_22 + gout83 * d_23 + gout84 * d_24 +
                       gout85 * d_25 + gout86 * d_26 + gout87 * d_27 +
                       gout88 * d_28 + gout89 * d_29,
                vk + (j0 + 2) + nao * (l0 + 0));
            vk += nao2;
        }
        dm += nao2;
    }
}

__global__ static void GINTint2e_jk_kernel3200(JKMatrix jk,
    BasisProdOffsets offsets, GINTEnvVars envs, BasisProdCache bpcache) {
    int ntasks_ij = offsets.ntasks_ij;
    long ntasks = ntasks_ij * offsets.ntasks_kl;
    long task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    int nprim_ij = envs.nprim_ij;
    int nprim_kl = envs.nprim_kl;
    int igroup = nprim_ij * nprim_kl;
    ntasks *= igroup;
    if (task_ij >= ntasks)
        return;
    int kl = task_ij % nprim_kl;
    task_ij /= nprim_kl;
    int ij = task_ij % nprim_ij;
    task_ij /= nprim_ij;
    int task_kl = task_ij / ntasks_ij;
    task_ij = task_ij % ntasks_ij;

    int bas_ij = offsets.bas_ij + task_ij;
    int bas_kl = offsets.bas_kl + task_kl;
    if (bas_ij < bas_kl) {
        return;
    }
    double norm = envs.fac;
    if (bas_ij == bas_kl) {
        norm *= .5;
    }

    int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
    int *ao_loc = bpcache.ao_loc;
    int *bas_pair2bra = bpcache.bas_pair2bra;
    int *bas_pair2ket = bpcache.bas_pair2ket;
    int ish = bas_pair2bra[bas_ij];
    int jsh = bas_pair2ket[bas_ij];
    int ksh = bas_pair2bra[bas_kl];
    int lsh = bas_pair2ket[bas_kl];
    int i0 = ao_loc[ish];
    int j0 = ao_loc[jsh];
    int k0 = ao_loc[ksh];
    int l0 = ao_loc[lsh];
    double *__restrict__ a12 = bpcache.a12;
    double *__restrict__ e12 = bpcache.e12;
    double *__restrict__ x12 = bpcache.x12;
    double *__restrict__ y12 = bpcache.y12;
    double *__restrict__ z12 = bpcache.z12;
    int i_dm;
    int nbas = bpcache.nbas;
    double *__restrict__ bas_x = bpcache.bas_coords;
    double *__restrict__ bas_y = bas_x + nbas;
    double *__restrict__ bas_z = bas_y + nbas;

    double gout0 = 0;
    double gout1 = 0;
    double gout2 = 0;
    double gout3 = 0;
    double gout4 = 0;
    double gout5 = 0;
    double gout6 = 0;
    double gout7 = 0;
    double gout8 = 0;
    double gout9 = 0;
    double gout10 = 0;
    double gout11 = 0;
    double gout12 = 0;
    double gout13 = 0;
    double gout14 = 0;
    double gout15 = 0;
    double gout16 = 0;
    double gout17 = 0;
    double gout18 = 0;
    double gout19 = 0;
    double gout20 = 0;
    double gout21 = 0;
    double gout22 = 0;
    double gout23 = 0;
    double gout24 = 0;
    double gout25 = 0;
    double gout26 = 0;
    double gout27 = 0;
    double gout28 = 0;
    double gout29 = 0;
    double gout30 = 0;
    double gout31 = 0;
    double gout32 = 0;
    double gout33 = 0;
    double gout34 = 0;
    double gout35 = 0;
    double gout36 = 0;
    double gout37 = 0;
    double gout38 = 0;
    double gout39 = 0;
    double gout40 = 0;
    double gout41 = 0;
    double gout42 = 0;
    double gout43 = 0;
    double gout44 = 0;
    double gout45 = 0;
    double gout46 = 0;
    double gout47 = 0;
    double gout48 = 0;
    double gout49 = 0;
    double gout50 = 0;
    double gout51 = 0;
    double gout52 = 0;
    double gout53 = 0;
    double gout54 = 0;
    double gout55 = 0;
    double gout56 = 0;
    double gout57 = 0;
    double gout58 = 0;
    double gout59 = 0;
    double xi = bas_x[ish];
    double yi = bas_y[ish];
    double zi = bas_z[ish];
    double xixj = xi - bas_x[jsh];
    double yiyj = yi - bas_y[jsh];
    double zizj = zi - bas_z[jsh];
    auto reduce = SegReduce<double>(igroup);
    ij += prim_ij;
    kl += prim_kl;
    double aij = a12[ij];
    double eij = e12[ij];
    double xij = x12[ij];
    double yij = y12[ij];
    double zij = z12[ij];
    double akl = a12[kl];
    double ekl = e12[kl];
    double xkl = x12[kl];
    double ykl = y12[kl];
    double zkl = z12[kl];
    double xijxkl = xij - xkl;
    double yijykl = yij - ykl;
    double zijzkl = zij - zkl;
    double aijkl = aij + akl;
    double a1 = aij * akl;
    double a0 = a1 / aijkl;
    double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
    double fac = norm * eij * ekl / (sqrt(aijkl) * a1);

    double rw[6];
    double root0, weight0;
    GINTrys_root<3>(x, rw);
    int irys;
    for (irys = 0; irys < 3; ++irys) {
        root0 = rw[irys];
        weight0 = rw[irys + 3];
        double u2 = a0 * root0;
        double tmp4 = .5 / (u2 * aijkl + a1);
        double b00 = u2 * tmp4;
        double tmp1 = 2 * b00;
        double tmp2 = tmp1 * akl;
        double b10 = b00 + tmp4 * akl;
        double c00x = xij - xi - tmp2 * xijxkl;
        double c00y = yij - yi - tmp2 * yijykl;
        double c00z = zij - zi - tmp2 * zijzkl;
        double g_0 = 1;
        double g_1 = c00x;
        double g_2 = c00x * c00x + b10;
        double g_3 = c00x * (2 * b10 + g_2);
        double g_4 = c00x + xixj;
        double g_5 = c00x * (c00x + xixj) + b10;
        double g_6 = c00x * (2 * b10 + g_2) + xixj * g_2;
        double g_7 = 3 * b10 * g_2 + c00x * g_3 + xixj * g_3;
        double g_8 = xixj * (xixj + c00x) + xixj * c00x + c00x * c00x + b10;
        double g_9 = xixj * (xixj * c00x + c00x * c00x + b10) + xixj * g_2 +
                     c00x * g_2 + 2 * b10 * c00x;
        double g_10 = xixj * (xixj * g_2 + c00x * g_2 + 2 * b10 * c00x) +
                      xixj * g_3 + c00x * g_3 + 3 * b10 * g_2;
        double g_11 = xixj * (xixj * g_3 + c00x * g_3 + 3 * b10 * g_2) +
                      xixj * (c00x * g_3 + 3 * b10 * g_2) +
                      c00x * (c00x * g_3 + 3 * b10 * g_2) + 4 * b10 * g_3;
        double g_12 = 1;
        double g_13 = c00y;
        double g_14 = c00y * c00y + b10;
        double g_15 = c00y * (2 * b10 + g_14);
        double g_16 = c00y + yiyj;
        double g_17 = c00y * (c00y + yiyj) + b10;
        double g_18 = c00y * (2 * b10 + g_14) + yiyj * g_14;
        double g_19 = 3 * b10 * g_14 + c00y * g_15 + yiyj * g_15;
        double g_20 = yiyj * (yiyj + c00y) + yiyj * c00y + c00y * c00y + b10;
        double g_21 = yiyj * (yiyj * c00y + c00y * c00y + b10) + yiyj * g_14 +
                      c00y * g_14 + 2 * b10 * c00y;
        double g_22 = yiyj * (yiyj * g_14 + c00y * g_14 + 2 * b10 * c00y) +
                      yiyj * g_15 + c00y * g_15 + 3 * b10 * g_14;
        double g_23 = yiyj * (yiyj * g_15 + c00y * g_15 + 3 * b10 * g_14) +
                      yiyj * (c00y * g_15 + 3 * b10 * g_14) +
                      c00y * (c00y * g_15 + 3 * b10 * g_14) + 4 * b10 * g_15;
        double g_24 = weight0 * fac;
        double g_25 = c00z * g_24;
        double g_26 = b10 * g_24 + c00z * g_25;
        double g_27 = 2 * b10 * g_25 + c00z * g_26;
        double g_28 = g_24 * (c00z + zizj);
        double g_29 = b10 * g_24 + c00z * g_25 + zizj * g_25;
        double g_30 = 2 * b10 * g_25 + c00z * g_26 + zizj * g_26;
        double g_31 = 3 * b10 * g_26 + c00z * g_27 + zizj * g_27;
        double g_32 = zizj * (zizj * g_24 + c00z * g_24) + zizj * g_25 +
                      c00z * g_25 + b10 * g_24;
        double g_33 = zizj * (zizj * g_25 + c00z * g_25 + b10 * g_24) +
                      zizj * g_26 + c00z * g_26 + 2 * b10 * g_25;
        double g_34 = zizj * (zizj * g_26 + c00z * g_26 + 2 * b10 * g_25) +
                      zizj * g_27 + c00z * g_27 + 3 * b10 * g_26;
        double g_35 = zizj * (zizj * g_27 + c00z * g_27 + 3 * b10 * g_26) +
                      zizj * (c00z * g_27 + 3 * b10 * g_26) +
                      c00z * (c00z * g_27 + 3 * b10 * g_26) + 4 * b10 * g_27;
        gout0 += g_11 * g_12 * g_24;
        gout1 += g_10 * g_13 * g_24;
        gout2 += g_10 * g_12 * g_25;
        gout3 += g_9 * g_14 * g_24;
        gout4 += g_9 * g_13 * g_25;
        gout5 += g_9 * g_12 * g_26;
        gout6 += g_8 * g_15 * g_24;
        gout7 += g_8 * g_14 * g_25;
        gout8 += g_8 * g_13 * g_26;
        gout9 += g_8 * g_12 * g_27;
        gout10 += g_7 * g_16 * g_24;
        gout11 += g_6 * g_17 * g_24;
        gout12 += g_6 * g_16 * g_25;
        gout13 += g_5 * g_18 * g_24;
        gout14 += g_5 * g_17 * g_25;
        gout15 += g_5 * g_16 * g_26;
        gout16 += g_4 * g_19 * g_24;
        gout17 += g_4 * g_18 * g_25;
        gout18 += g_4 * g_17 * g_26;
        gout19 += g_4 * g_16 * g_27;
        gout20 += g_7 * g_12 * g_28;
        gout21 += g_6 * g_13 * g_28;
        gout22 += g_6 * g_12 * g_29;
        gout23 += g_5 * g_14 * g_28;
        gout24 += g_5 * g_13 * g_29;
        gout25 += g_5 * g_12 * g_30;
        gout26 += g_4 * g_15 * g_28;
        gout27 += g_4 * g_14 * g_29;
        gout28 += g_4 * g_13 * g_30;
        gout29 += g_4 * g_12 * g_31;
        gout30 += g_3 * g_20 * g_24;
        gout31 += g_2 * g_21 * g_24;
        gout32 += g_2 * g_20 * g_25;
        gout33 += g_1 * g_22 * g_24;
        gout34 += g_1 * g_21 * g_25;
        gout35 += g_1 * g_20 * g_26;
        gout36 += g_0 * g_23 * g_24;
        gout37 += g_0 * g_22 * g_25;
        gout38 += g_0 * g_21 * g_26;
        gout39 += g_0 * g_20 * g_27;
        gout40 += g_3 * g_16 * g_28;
        gout41 += g_2 * g_17 * g_28;
        gout42 += g_2 * g_16 * g_29;
        gout43 += g_1 * g_18 * g_28;
        gout44 += g_1 * g_17 * g_29;
        gout45 += g_1 * g_16 * g_30;
        gout46 += g_0 * g_19 * g_28;
        gout47 += g_0 * g_18 * g_29;
        gout48 += g_0 * g_17 * g_30;
        gout49 += g_0 * g_16 * g_31;
        gout50 += g_3 * g_12 * g_32;
        gout51 += g_2 * g_13 * g_32;
        gout52 += g_2 * g_12 * g_33;
        gout53 += g_1 * g_14 * g_32;
        gout54 += g_1 * g_13 * g_33;
        gout55 += g_1 * g_12 * g_34;
        gout56 += g_0 * g_15 * g_32;
        gout57 += g_0 * g_14 * g_33;
        gout58 += g_0 * g_13 * g_34;
        gout59 += g_0 * g_12 * g_35;
    }
    double d_0, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, d_9;
    double d_10, d_11, d_12, d_13, d_14, d_15, d_16, d_17, d_18, d_19;
    double d_20, d_21, d_22, d_23, d_24, d_25, d_26, d_27, d_28, d_29;
    double d_30, d_31, d_32, d_33, d_34, d_35, d_36, d_37, d_38, d_39;
    double d_40, d_41, d_42, d_43, d_44, d_45, d_46, d_47, d_48, d_49;
    double d_50, d_51, d_52, d_53, d_54, d_55, d_56, d_57, d_58, d_59;
    int n_dm = jk.n_dm;
    int nao = jk.nao;
    size_t nao2 = nao * nao;
    double *__restrict__ dm = jk.dm;
    double *vj = jk.vj;
    double *vk = jk.vk;
    for (i_dm = 0; i_dm < n_dm; ++i_dm) {
        if (vj != NULL) {
            // ijkl,ij->kl
            d_0 = dm[(i0 + 0) + nao * (j0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (j0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (j0 + 0)];
            d_3 = dm[(i0 + 3) + nao * (j0 + 0)];
            d_4 = dm[(i0 + 4) + nao * (j0 + 0)];
            d_5 = dm[(i0 + 5) + nao * (j0 + 0)];
            d_6 = dm[(i0 + 6) + nao * (j0 + 0)];
            d_7 = dm[(i0 + 7) + nao * (j0 + 0)];
            d_8 = dm[(i0 + 8) + nao * (j0 + 0)];
            d_9 = dm[(i0 + 9) + nao * (j0 + 0)];
            d_10 = dm[(i0 + 0) + nao * (j0 + 1)];
            d_11 = dm[(i0 + 1) + nao * (j0 + 1)];
            d_12 = dm[(i0 + 2) + nao * (j0 + 1)];
            d_13 = dm[(i0 + 3) + nao * (j0 + 1)];
            d_14 = dm[(i0 + 4) + nao * (j0 + 1)];
            d_15 = dm[(i0 + 5) + nao * (j0 + 1)];
            d_16 = dm[(i0 + 6) + nao * (j0 + 1)];
            d_17 = dm[(i0 + 7) + nao * (j0 + 1)];
            d_18 = dm[(i0 + 8) + nao * (j0 + 1)];
            d_19 = dm[(i0 + 9) + nao * (j0 + 1)];
            d_20 = dm[(i0 + 0) + nao * (j0 + 2)];
            d_21 = dm[(i0 + 1) + nao * (j0 + 2)];
            d_22 = dm[(i0 + 2) + nao * (j0 + 2)];
            d_23 = dm[(i0 + 3) + nao * (j0 + 2)];
            d_24 = dm[(i0 + 4) + nao * (j0 + 2)];
            d_25 = dm[(i0 + 5) + nao * (j0 + 2)];
            d_26 = dm[(i0 + 6) + nao * (j0 + 2)];
            d_27 = dm[(i0 + 7) + nao * (j0 + 2)];
            d_28 = dm[(i0 + 8) + nao * (j0 + 2)];
            d_29 = dm[(i0 + 9) + nao * (j0 + 2)];
            d_30 = dm[(i0 + 0) + nao * (j0 + 3)];
            d_31 = dm[(i0 + 1) + nao * (j0 + 3)];
            d_32 = dm[(i0 + 2) + nao * (j0 + 3)];
            d_33 = dm[(i0 + 3) + nao * (j0 + 3)];
            d_34 = dm[(i0 + 4) + nao * (j0 + 3)];
            d_35 = dm[(i0 + 5) + nao * (j0 + 3)];
            d_36 = dm[(i0 + 6) + nao * (j0 + 3)];
            d_37 = dm[(i0 + 7) + nao * (j0 + 3)];
            d_38 = dm[(i0 + 8) + nao * (j0 + 3)];
            d_39 = dm[(i0 + 9) + nao * (j0 + 3)];
            d_40 = dm[(i0 + 0) + nao * (j0 + 4)];
            d_41 = dm[(i0 + 1) + nao * (j0 + 4)];
            d_42 = dm[(i0 + 2) + nao * (j0 + 4)];
            d_43 = dm[(i0 + 3) + nao * (j0 + 4)];
            d_44 = dm[(i0 + 4) + nao * (j0 + 4)];
            d_45 = dm[(i0 + 5) + nao * (j0 + 4)];
            d_46 = dm[(i0 + 6) + nao * (j0 + 4)];
            d_47 = dm[(i0 + 7) + nao * (j0 + 4)];
            d_48 = dm[(i0 + 8) + nao * (j0 + 4)];
            d_49 = dm[(i0 + 9) + nao * (j0 + 4)];
            d_50 = dm[(i0 + 0) + nao * (j0 + 5)];
            d_51 = dm[(i0 + 1) + nao * (j0 + 5)];
            d_52 = dm[(i0 + 2) + nao * (j0 + 5)];
            d_53 = dm[(i0 + 3) + nao * (j0 + 5)];
            d_54 = dm[(i0 + 4) + nao * (j0 + 5)];
            d_55 = dm[(i0 + 5) + nao * (j0 + 5)];
            d_56 = dm[(i0 + 6) + nao * (j0 + 5)];
            d_57 = dm[(i0 + 7) + nao * (j0 + 5)];
            d_58 = dm[(i0 + 8) + nao * (j0 + 5)];
            d_59 = dm[(i0 + 9) + nao * (j0 + 5)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout3 * d_3 +
                       gout4 * d_4 + gout5 * d_5 + gout6 * d_6 + gout7 * d_7 +
                       gout8 * d_8 + gout9 * d_9 + gout10 * d_10 +
                       gout11 * d_11 + gout12 * d_12 + gout13 * d_13 +
                       gout14 * d_14 + gout15 * d_15 + gout16 * d_16 +
                       gout17 * d_17 + gout18 * d_18 + gout19 * d_19 +
                       gout20 * d_20 + gout21 * d_21 + gout22 * d_22 +
                       gout23 * d_23 + gout24 * d_24 + gout25 * d_25 +
                       gout26 * d_26 + gout27 * d_27 + gout28 * d_28 +
                       gout29 * d_29 + gout30 * d_30 + gout31 * d_31 +
                       gout32 * d_32 + gout33 * d_33 + gout34 * d_34 +
                       gout35 * d_35 + gout36 * d_36 + gout37 * d_37 +
                       gout38 * d_38 + gout39 * d_39 + gout40 * d_40 +
                       gout41 * d_41 + gout42 * d_42 + gout43 * d_43 +
                       gout44 * d_44 + gout45 * d_45 + gout46 * d_46 +
                       gout47 * d_47 + gout48 * d_48 + gout49 * d_49 +
                       gout50 * d_50 + gout51 * d_51 + gout52 * d_52 +
                       gout53 * d_53 + gout54 * d_54 + gout55 * d_55 +
                       gout56 * d_56 + gout57 * d_57 + gout58 * d_58 +
                       gout59 * d_59,
                vj + (k0 + 0) + nao * (l0 + 0));
            // ijkl,kl->ij
            d_0 = dm[(k0 + 0) + nao * (l0 + 0)];
            reduce(gout0 * d_0, vj + (i0 + 0) + nao * (j0 + 0));
            reduce(gout1 * d_0, vj + (i0 + 1) + nao * (j0 + 0));
            reduce(gout2 * d_0, vj + (i0 + 2) + nao * (j0 + 0));
            reduce(gout3 * d_0, vj + (i0 + 3) + nao * (j0 + 0));
            reduce(gout4 * d_0, vj + (i0 + 4) + nao * (j0 + 0));
            reduce(gout5 * d_0, vj + (i0 + 5) + nao * (j0 + 0));
            reduce(gout6 * d_0, vj + (i0 + 6) + nao * (j0 + 0));
            reduce(gout7 * d_0, vj + (i0 + 7) + nao * (j0 + 0));
            reduce(gout8 * d_0, vj + (i0 + 8) + nao * (j0 + 0));
            reduce(gout9 * d_0, vj + (i0 + 9) + nao * (j0 + 0));
            reduce(gout10 * d_0, vj + (i0 + 0) + nao * (j0 + 1));
            reduce(gout11 * d_0, vj + (i0 + 1) + nao * (j0 + 1));
            reduce(gout12 * d_0, vj + (i0 + 2) + nao * (j0 + 1));
            reduce(gout13 * d_0, vj + (i0 + 3) + nao * (j0 + 1));
            reduce(gout14 * d_0, vj + (i0 + 4) + nao * (j0 + 1));
            reduce(gout15 * d_0, vj + (i0 + 5) + nao * (j0 + 1));
            reduce(gout16 * d_0, vj + (i0 + 6) + nao * (j0 + 1));
            reduce(gout17 * d_0, vj + (i0 + 7) + nao * (j0 + 1));
            reduce(gout18 * d_0, vj + (i0 + 8) + nao * (j0 + 1));
            reduce(gout19 * d_0, vj + (i0 + 9) + nao * (j0 + 1));
            reduce(gout20 * d_0, vj + (i0 + 0) + nao * (j0 + 2));
            reduce(gout21 * d_0, vj + (i0 + 1) + nao * (j0 + 2));
            reduce(gout22 * d_0, vj + (i0 + 2) + nao * (j0 + 2));
            reduce(gout23 * d_0, vj + (i0 + 3) + nao * (j0 + 2));
            reduce(gout24 * d_0, vj + (i0 + 4) + nao * (j0 + 2));
            reduce(gout25 * d_0, vj + (i0 + 5) + nao * (j0 + 2));
            reduce(gout26 * d_0, vj + (i0 + 6) + nao * (j0 + 2));
            reduce(gout27 * d_0, vj + (i0 + 7) + nao * (j0 + 2));
            reduce(gout28 * d_0, vj + (i0 + 8) + nao * (j0 + 2));
            reduce(gout29 * d_0, vj + (i0 + 9) + nao * (j0 + 2));
            reduce(gout30 * d_0, vj + (i0 + 0) + nao * (j0 + 3));
            reduce(gout31 * d_0, vj + (i0 + 1) + nao * (j0 + 3));
            reduce(gout32 * d_0, vj + (i0 + 2) + nao * (j0 + 3));
            reduce(gout33 * d_0, vj + (i0 + 3) + nao * (j0 + 3));
            reduce(gout34 * d_0, vj + (i0 + 4) + nao * (j0 + 3));
            reduce(gout35 * d_0, vj + (i0 + 5) + nao * (j0 + 3));
            reduce(gout36 * d_0, vj + (i0 + 6) + nao * (j0 + 3));
            reduce(gout37 * d_0, vj + (i0 + 7) + nao * (j0 + 3));
            reduce(gout38 * d_0, vj + (i0 + 8) + nao * (j0 + 3));
            reduce(gout39 * d_0, vj + (i0 + 9) + nao * (j0 + 3));
            reduce(gout40 * d_0, vj + (i0 + 0) + nao * (j0 + 4));
            reduce(gout41 * d_0, vj + (i0 + 1) + nao * (j0 + 4));
            reduce(gout42 * d_0, vj + (i0 + 2) + nao * (j0 + 4));
            reduce(gout43 * d_0, vj + (i0 + 3) + nao * (j0 + 4));
            reduce(gout44 * d_0, vj + (i0 + 4) + nao * (j0 + 4));
            reduce(gout45 * d_0, vj + (i0 + 5) + nao * (j0 + 4));
            reduce(gout46 * d_0, vj + (i0 + 6) + nao * (j0 + 4));
            reduce(gout47 * d_0, vj + (i0 + 7) + nao * (j0 + 4));
            reduce(gout48 * d_0, vj + (i0 + 8) + nao * (j0 + 4));
            reduce(gout49 * d_0, vj + (i0 + 9) + nao * (j0 + 4));
            reduce(gout50 * d_0, vj + (i0 + 0) + nao * (j0 + 5));
            reduce(gout51 * d_0, vj + (i0 + 1) + nao * (j0 + 5));
            reduce(gout52 * d_0, vj + (i0 + 2) + nao * (j0 + 5));
            reduce(gout53 * d_0, vj + (i0 + 3) + nao * (j0 + 5));
            reduce(gout54 * d_0, vj + (i0 + 4) + nao * (j0 + 5));
            reduce(gout55 * d_0, vj + (i0 + 5) + nao * (j0 + 5));
            reduce(gout56 * d_0, vj + (i0 + 6) + nao * (j0 + 5));
            reduce(gout57 * d_0, vj + (i0 + 7) + nao * (j0 + 5));
            reduce(gout58 * d_0, vj + (i0 + 8) + nao * (j0 + 5));
            reduce(gout59 * d_0, vj + (i0 + 9) + nao * (j0 + 5));
            vj += nao2;
        }
        if (vk != NULL) {
            // ijkl,jl->ik
            d_0 = dm[(j0 + 0) + nao * (l0 + 0)];
            d_1 = dm[(j0 + 1) + nao * (l0 + 0)];
            d_2 = dm[(j0 + 2) + nao * (l0 + 0)];
            d_3 = dm[(j0 + 3) + nao * (l0 + 0)];
            d_4 = dm[(j0 + 4) + nao * (l0 + 0)];
            d_5 = dm[(j0 + 5) + nao * (l0 + 0)];
            reduce(gout0 * d_0 + gout10 * d_1 + gout20 * d_2 + gout30 * d_3 +
                       gout40 * d_4 + gout50 * d_5,
                vk + (i0 + 0) + nao * (k0 + 0));
            reduce(gout1 * d_0 + gout11 * d_1 + gout21 * d_2 + gout31 * d_3 +
                       gout41 * d_4 + gout51 * d_5,
                vk + (i0 + 1) + nao * (k0 + 0));
            reduce(gout2 * d_0 + gout12 * d_1 + gout22 * d_2 + gout32 * d_3 +
                       gout42 * d_4 + gout52 * d_5,
                vk + (i0 + 2) + nao * (k0 + 0));
            reduce(gout3 * d_0 + gout13 * d_1 + gout23 * d_2 + gout33 * d_3 +
                       gout43 * d_4 + gout53 * d_5,
                vk + (i0 + 3) + nao * (k0 + 0));
            reduce(gout4 * d_0 + gout14 * d_1 + gout24 * d_2 + gout34 * d_3 +
                       gout44 * d_4 + gout54 * d_5,
                vk + (i0 + 4) + nao * (k0 + 0));
            reduce(gout5 * d_0 + gout15 * d_1 + gout25 * d_2 + gout35 * d_3 +
                       gout45 * d_4 + gout55 * d_5,
                vk + (i0 + 5) + nao * (k0 + 0));
            reduce(gout6 * d_0 + gout16 * d_1 + gout26 * d_2 + gout36 * d_3 +
                       gout46 * d_4 + gout56 * d_5,
                vk + (i0 + 6) + nao * (k0 + 0));
            reduce(gout7 * d_0 + gout17 * d_1 + gout27 * d_2 + gout37 * d_3 +
                       gout47 * d_4 + gout57 * d_5,
                vk + (i0 + 7) + nao * (k0 + 0));
            reduce(gout8 * d_0 + gout18 * d_1 + gout28 * d_2 + gout38 * d_3 +
                       gout48 * d_4 + gout58 * d_5,
                vk + (i0 + 8) + nao * (k0 + 0));
            reduce(gout9 * d_0 + gout19 * d_1 + gout29 * d_2 + gout39 * d_3 +
                       gout49 * d_4 + gout59 * d_5,
                vk + (i0 + 9) + nao * (k0 + 0));
            // ijkl,jk->il
            d_0 = dm[(j0 + 0) + nao * (k0 + 0)];
            d_1 = dm[(j0 + 1) + nao * (k0 + 0)];
            d_2 = dm[(j0 + 2) + nao * (k0 + 0)];
            d_3 = dm[(j0 + 3) + nao * (k0 + 0)];
            d_4 = dm[(j0 + 4) + nao * (k0 + 0)];
            d_5 = dm[(j0 + 5) + nao * (k0 + 0)];
            reduce(gout0 * d_0 + gout10 * d_1 + gout20 * d_2 + gout30 * d_3 +
                       gout40 * d_4 + gout50 * d_5,
                vk + (i0 + 0) + nao * (l0 + 0));
            reduce(gout1 * d_0 + gout11 * d_1 + gout21 * d_2 + gout31 * d_3 +
                       gout41 * d_4 + gout51 * d_5,
                vk + (i0 + 1) + nao * (l0 + 0));
            reduce(gout2 * d_0 + gout12 * d_1 + gout22 * d_2 + gout32 * d_3 +
                       gout42 * d_4 + gout52 * d_5,
                vk + (i0 + 2) + nao * (l0 + 0));
            reduce(gout3 * d_0 + gout13 * d_1 + gout23 * d_2 + gout33 * d_3 +
                       gout43 * d_4 + gout53 * d_5,
                vk + (i0 + 3) + nao * (l0 + 0));
            reduce(gout4 * d_0 + gout14 * d_1 + gout24 * d_2 + gout34 * d_3 +
                       gout44 * d_4 + gout54 * d_5,
                vk + (i0 + 4) + nao * (l0 + 0));
            reduce(gout5 * d_0 + gout15 * d_1 + gout25 * d_2 + gout35 * d_3 +
                       gout45 * d_4 + gout55 * d_5,
                vk + (i0 + 5) + nao * (l0 + 0));
            reduce(gout6 * d_0 + gout16 * d_1 + gout26 * d_2 + gout36 * d_3 +
                       gout46 * d_4 + gout56 * d_5,
                vk + (i0 + 6) + nao * (l0 + 0));
            reduce(gout7 * d_0 + gout17 * d_1 + gout27 * d_2 + gout37 * d_3 +
                       gout47 * d_4 + gout57 * d_5,
                vk + (i0 + 7) + nao * (l0 + 0));
            reduce(gout8 * d_0 + gout18 * d_1 + gout28 * d_2 + gout38 * d_3 +
                       gout48 * d_4 + gout58 * d_5,
                vk + (i0 + 8) + nao * (l0 + 0));
            reduce(gout9 * d_0 + gout19 * d_1 + gout29 * d_2 + gout39 * d_3 +
                       gout49 * d_4 + gout59 * d_5,
                vk + (i0 + 9) + nao * (l0 + 0));
            // ijkl,il->jk
            d_0 = dm[(i0 + 0) + nao * (l0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (l0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (l0 + 0)];
            d_3 = dm[(i0 + 3) + nao * (l0 + 0)];
            d_4 = dm[(i0 + 4) + nao * (l0 + 0)];
            d_5 = dm[(i0 + 5) + nao * (l0 + 0)];
            d_6 = dm[(i0 + 6) + nao * (l0 + 0)];
            d_7 = dm[(i0 + 7) + nao * (l0 + 0)];
            d_8 = dm[(i0 + 8) + nao * (l0 + 0)];
            d_9 = dm[(i0 + 9) + nao * (l0 + 0)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout3 * d_3 +
                       gout4 * d_4 + gout5 * d_5 + gout6 * d_6 + gout7 * d_7 +
                       gout8 * d_8 + gout9 * d_9,
                vk + (j0 + 0) + nao * (k0 + 0));
            reduce(gout10 * d_0 + gout11 * d_1 + gout12 * d_2 + gout13 * d_3 +
                       gout14 * d_4 + gout15 * d_5 + gout16 * d_6 +
                       gout17 * d_7 + gout18 * d_8 + gout19 * d_9,
                vk + (j0 + 1) + nao * (k0 + 0));
            reduce(gout20 * d_0 + gout21 * d_1 + gout22 * d_2 + gout23 * d_3 +
                       gout24 * d_4 + gout25 * d_5 + gout26 * d_6 +
                       gout27 * d_7 + gout28 * d_8 + gout29 * d_9,
                vk + (j0 + 2) + nao * (k0 + 0));
            reduce(gout30 * d_0 + gout31 * d_1 + gout32 * d_2 + gout33 * d_3 +
                       gout34 * d_4 + gout35 * d_5 + gout36 * d_6 +
                       gout37 * d_7 + gout38 * d_8 + gout39 * d_9,
                vk + (j0 + 3) + nao * (k0 + 0));
            reduce(gout40 * d_0 + gout41 * d_1 + gout42 * d_2 + gout43 * d_3 +
                       gout44 * d_4 + gout45 * d_5 + gout46 * d_6 +
                       gout47 * d_7 + gout48 * d_8 + gout49 * d_9,
                vk + (j0 + 4) + nao * (k0 + 0));
            reduce(gout50 * d_0 + gout51 * d_1 + gout52 * d_2 + gout53 * d_3 +
                       gout54 * d_4 + gout55 * d_5 + gout56 * d_6 +
                       gout57 * d_7 + gout58 * d_8 + gout59 * d_9,
                vk + (j0 + 5) + nao * (k0 + 0));
            // ijkl,ik->jl
            d_0 = dm[(i0 + 0) + nao * (k0 + 0)];
            d_1 = dm[(i0 + 1) + nao * (k0 + 0)];
            d_2 = dm[(i0 + 2) + nao * (k0 + 0)];
            d_3 = dm[(i0 + 3) + nao * (k0 + 0)];
            d_4 = dm[(i0 + 4) + nao * (k0 + 0)];
            d_5 = dm[(i0 + 5) + nao * (k0 + 0)];
            d_6 = dm[(i0 + 6) + nao * (k0 + 0)];
            d_7 = dm[(i0 + 7) + nao * (k0 + 0)];
            d_8 = dm[(i0 + 8) + nao * (k0 + 0)];
            d_9 = dm[(i0 + 9) + nao * (k0 + 0)];
            reduce(gout0 * d_0 + gout1 * d_1 + gout2 * d_2 + gout3 * d_3 +
                       gout4 * d_4 + gout5 * d_5 + gout6 * d_6 + gout7 * d_7 +
                       gout8 * d_8 + gout9 * d_9,
                vk + (j0 + 0) + nao * (l0 + 0));
            reduce(gout10 * d_0 + gout11 * d_1 + gout12 * d_2 + gout13 * d_3 +
                       gout14 * d_4 + gout15 * d_5 + gout16 * d_6 +
                       gout17 * d_7 + gout18 * d_8 + gout19 * d_9,
                vk + (j0 + 1) + nao * (l0 + 0));
            reduce(gout20 * d_0 + gout21 * d_1 + gout22 * d_2 + gout23 * d_3 +
                       gout24 * d_4 + gout25 * d_5 + gout26 * d_6 +
                       gout27 * d_7 + gout28 * d_8 + gout29 * d_9,
                vk + (j0 + 2) + nao * (l0 + 0));
            reduce(gout30 * d_0 + gout31 * d_1 + gout32 * d_2 + gout33 * d_3 +
                       gout34 * d_4 + gout35 * d_5 + gout36 * d_6 +
                       gout37 * d_7 + gout38 * d_8 + gout39 * d_9,
                vk + (j0 + 3) + nao * (l0 + 0));
            reduce(gout40 * d_0 + gout41 * d_1 + gout42 * d_2 + gout43 * d_3 +
                       gout44 * d_4 + gout45 * d_5 + gout46 * d_6 +
                       gout47 * d_7 + gout48 * d_8 + gout49 * d_9,
                vk + (j0 + 4) + nao * (l0 + 0));
            reduce(gout50 * d_0 + gout51 * d_1 + gout52 * d_2 + gout53 * d_3 +
                       gout54 * d_4 + gout55 * d_5 + gout56 * d_6 +
                       gout57 * d_7 + gout58 * d_8 + gout59 * d_9,
                vk + (j0 + 5) + nao * (l0 + 0));
            vk += nao2;
        }
        dm += nao2;
    }
}
