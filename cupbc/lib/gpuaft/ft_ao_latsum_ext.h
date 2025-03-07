/*
Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
This file is part of ByteQC.

ByteQC is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ByteQC is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

template <bool, int, int>
void _PBC_ft_latsum(const int, cuDoubleComplex *, const int, const int,
    const int, double *, const double *, const cuDoubleComplex *, const int *,
    const int *, const double *, const double *, const int *, const int *,
    const int, int *, const int, int *, const int);

#define FUNC_LATSUM(F, I, J)                                                   \
    extern template void _PBC_ft_latsum<F, I, J>(const int, cuDoubleComplex *, \
        const int, const int, const int, double *, const double *,             \
        const cuDoubleComplex *, const int *, const int *, const double *,     \
        const double *, const int *, const int *, const int, int *, const int, \
        int *, const int);

#define FUNC_LATSUM_F(I, J) FUNC_LATSUM(true, I, J) FUNC_LATSUM(false, I, J)

#define FUNC_LATSUM_F_0(I) FUNC_LATSUM_F(I, 0)
#define FUNC_LATSUM_F_1(I) FUNC_LATSUM_F(I, 1) FUNC_LATSUM_F_0(I)
#define FUNC_LATSUM_F_2(I) FUNC_LATSUM_F(I, 2) FUNC_LATSUM_F_1(I)
#define FUNC_LATSUM_F_3(I) FUNC_LATSUM_F(I, 3) FUNC_LATSUM_F_2(I)
#define FUNC_LATSUM_F_4(I) FUNC_LATSUM_F(I, 4) FUNC_LATSUM_F_3(I)
#define FUNC_LATSUM_F_5(I) FUNC_LATSUM_F(I, 5) FUNC_LATSUM_F_4(I)

#define FUNC_LATSUM_F_J(I) FUNC_LATSUM_F_##I(I)

FUNC_LATSUM_F_J(0)
FUNC_LATSUM_F_J(1)
FUNC_LATSUM_F_J(2)
FUNC_LATSUM_F_J(3)
FUNC_LATSUM_F_J(4)
FUNC_LATSUM_F_J(5)
