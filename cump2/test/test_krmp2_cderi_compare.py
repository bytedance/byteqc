#!/usr/bin/env python3
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0.

"""Standalone periodic CDERI/Lov comparison for the draft KRMP2 path.

This script compares two ways of producing periodic KRMP2-ready ``Lov`` data:

1. A gpu4pyscf reference path that keeps each q-block of unpacked CDERI in
   memory and performs the AO->MO transform in one shot.
2. The current draft path in ``byteqc/cump2/krmp2_cderi.py`` that forces small
   auxiliary slices and writes the transformed data to disk incrementally.

Known interface gap
-------------------
The current draft module does not dump AO-level PySCF-style ``j3c``.  It writes
MO-space ``Lov`` directly.  This comparison therefore validates the current
draft's on-disk ``Lov`` layout against an in-memory gpu4pyscf ``Lov``
reference, not AO-level ``j3c`` files.

System choice
-------------
The test cell contains 8 water molecules in a periodic 2x2x2 arrangement.
The k-mesh is intentionally reduced to gamma only.  This remains a periodic
cell, not an isolated molecular calculation.  The gamma restriction is only to
keep the 8-water regression/smoke test runnable while still forcing multiple
auxiliary slices.
"""

from __future__ import annotations

import argparse
import os, sys
import tempfile

import numpy as np


def build_test_cell(pbcgto):
    """Periodic 8-water cell in a regular 2x2x2 arrangement."""
    cell = pbcgto.Cell()
    cell.unit = 'A'
    cell.a = np.array([
        [7.76991940, 0.0, 0.0],
        [3.88495970, 6.72894716, 0.0],
        [0.0, 0.0, 7.32108974],
    ]),
    cell.charge = 0
    cell.spin = 0
    cell.atom = [
        ('H', (5.184886000, 4.477113000, 1.451336000)),
        ('H', (6.485207000, 6.728804000, 5.111840000)),
        ('H', (5.646641000, 3.677423000, 0.151458000)),
        ('H', (7.408643000, 6.728826000, 3.811934000)),
        ('H', (6.006336000, 5.930382000, 7.206082000)),
        ('H', (4.254016000, 4.492389000, 3.545621000)),
        ('H', (5.637134000, 5.290906000, 3.545561000)),
        ('H', (6.006351500, 7.527438239, 7.206046000)),
        ('H', (5.169661000, 0.000059000, 1.451250000)),
        ('H', (6.469712000, 2.251911000, 5.111942000)),
        ('H', (4.246191000, 0.000036000, 0.151391000)),
        ('H', (6.007997000, 3.051532000, 3.812004000)),
        ('H', (7.400629000, 2.236535000, 7.206186000)),
        ('H', (6.017510000, 1.438031000, 7.206104000)),
        ('H', (5.648503000, 0.798475000, 3.545549000)),
        ('H', (5.648467500, -0.798606239, 3.545486000)),
        ('H', (1.300070000, 2.251708000, 1.451386000)),
        ('H', (1.761705000, 3.051525000, 0.151538000)),
        ('H', (1.752253000, 1.437762000, 3.545639000)),
        ('H', (0.369153000, 2.236324000, 3.545642000)),
        ('H', (3.515578000, 4.492448000, 7.206132000)),
        ('H', (2.584635000, 4.476974000, 5.111941000)),
        ('H', (2.123031000, 3.677207000, 3.812053000)),
        ('H', (2.132479000, 5.290987000, 7.206126000)),
        ('O', (5.170923000, 4.501277000, 0.454136000)),
        ('O', (5.190110000, 4.468058000, 3.200629000)),
        ('O', (6.457320500, 6.728763239, 4.114636000)),
        ('O', (6.495460500, 6.728896239, 6.861091000)),
        ('O', (5.197547000, 0.000077000, 0.454036000)),
        ('O', (6.483716000, 2.227704000, 4.114745000)),
        ('O', (6.464527000, 2.260908000, 6.861236000)),
        ('O', (5.159428000, -0.000040000, 3.200505000)),
        ('O', (1.286108000, 2.227586000, 0.454183000)),
        ('O', (1.305270000, 2.260631000, 3.200708000)),
        ('O', (2.598628000, 4.501124000, 4.114745000)),
        ('O', (2.579445000, 4.468082000, 6.861261000)),
    ]

    cell.basis = 'sto-3g'
    cell.verbose = 4
    cell.build()
    return cell


def build_test_auxbasis():
    """Use a standard fitting basis; slicing is enforced via aux_blksize."""
    return 'weigend'


def require_gpu_stack():
    try:
        import cupy  # pylint: disable=import-error
    except ModuleNotFoundError as err:
        raise RuntimeError(
            'This standalone script requires cupy and a working gpu4pyscf '
            'runtime. The current environment does not provide cupy.') from err

    try:
        from byteqc.cump2.krmp2_cderi import build_gpu_cderi_df, build_krmp2_lov_store
    except Exception as err:  # pragma: no cover - environment dependent
        raise RuntimeError(
            'Unable to import the draft KRMP2 CDERI module '
            '`byteqc.cump2.krmp2_cderi`.') from err

    return cupy, build_gpu_cderi_df, build_krmp2_lov_store


def require_pyscf_stack():
    try:
        from pyscf import lib
        from pyscf.pbc import gto as pbcgto
        from gpu4pyscf.pbc import scf as pbcscf
        from pyscf.pbc.lib.kpts_helper import gamma_point
        from pyscf.pbc.tools import k2gamma
    except Exception as err:  # pragma: no cover - environment dependent
        raise RuntimeError(
            'This standalone script requires a working PySCF periodic stack, '
            'including scipy.') from err
    return lib, pbcgto, pbcscf, gamma_point, k2gamma


def run_reference_krhf(cell, kpts, *, pbcscf):
    mf = pbcscf.KRHF(cell, kpts=kpts).density_fit()
    mf.conv_tol = 1e-8
    mf.max_cycle = 80
    mf.kernel()
    if not mf.converged:
        raise RuntimeError('KRHF did not converge for the standalone CDERI test.')
    return mf


def build_pair_mapping(kmesh, *, k2gamma_module):
    nkpts = int(np.prod(kmesh))
    kk_conserv = k2gamma_module.double_translation_indices(kmesh)
    pair_table = np.empty((nkpts * nkpts, 3), dtype=np.int64)
    pair_mapping = {}
    for q in range(nkpts):
        ki_idx, ka_idx = np.where(kk_conserv == q)
        order = np.argsort(ki_idx)
        records = []
        for blk_pos, pos in enumerate(order):
            ki = int(ki_idx[pos])
            ka = int(ka_idx[pos])
            pair_id = ki * nkpts + ka
            pair_table[pair_id] = (q, ki, ka)
            records.append((blk_pos, ki, ka, pair_id))
        pair_mapping[q] = records
    return pair_mapping, pair_table


def split_occ_vir(mo_coeff, mo_occ):
    occ_coeff = []
    vir_coeff = []
    nocc = []
    nvir = []
    for coeff_k, occ_k in zip(mo_coeff, mo_occ):
        occ_idx = np.where(np.asarray(occ_k) > 0)[0]
        vir_idx = np.where(np.asarray(occ_k) <= 0)[0]
        occ_coeff.append(np.asarray(coeff_k[:, occ_idx]))
        vir_coeff.append(np.asarray(coeff_k[:, vir_idx]))
        nocc.append(len(occ_idx))
        nvir.append(len(vir_idx))
    return occ_coeff, vir_coeff, np.asarray(nocc), np.asarray(nvir)


def infer_store_dtype(kpts, mo_coeff, *, gamma_point_fn):
    if gamma_point_fn(kpts) and not any(np.iscomplexobj(x) for x in mo_coeff):
        return np.dtype(np.float64)
    return np.dtype(np.complex128)


def naux_by_q(cderi_dict, nkpts):
    out = np.zeros(nkpts, dtype=np.int64)
    for q in range(nkpts):
        if cderi_dict is not None and q in cderi_dict:
            out[q] = int(cderi_dict[q].shape[0])
    return out


def build_reference_lov(with_df, mo_coeff, mo_occ, *, cupy_module,
                        gamma_point_fn, k2gamma_module):
    kpts = np.asarray(with_df.kpts)
    kmesh = np.asarray(with_df.kmesh, dtype=np.int64)
    nkpts = int(np.prod(kmesh))
    occ_coeff, vir_coeff, nocc_per_k, nvir_per_k = split_occ_vir(mo_coeff, mo_occ)
    store_dtype = infer_store_dtype(kpts, mo_coeff, gamma_point_fn=gamma_point_fn)
    pair_mapping, pair_table = build_pair_mapping(kmesh, k2gamma_module=k2gamma_module)
    naux_pos_by_q = naux_by_q(with_df._cderi, nkpts)

    ref_lov = {}
    max_naux = int(naux_pos_by_q.max())
    seen_q = set()
    for q, cderi_block, sign in with_df.loop(blksize=max_naux, unpack=True):
        if sign < 0:
            continue
        if q in seen_q:
            raise RuntimeError(
                f'Reference path unexpectedly sliced q={q}; '
                f'blksize={max_naux} did not keep the full block in memory.')
        seen_q.add(int(q))
        block = cupy_module.asnumpy(cderi_block).astype(store_dtype, copy=False)
        if block.shape[1] != naux_pos_by_q[q]:
            raise RuntimeError(
                f'Unexpected aux length for q={q}: got {block.shape[1]}, '
                f'expected {naux_pos_by_q[q]}.')
        for blk_pos, ki, ka, pair_id in pair_mapping[q]:
            ao_pair = np.asarray(block[blk_pos], dtype=store_dtype)
            lov = np.einsum(
                'Lij,io,jv->Lov',
                ao_pair,
                np.asarray(occ_coeff[ki], dtype=store_dtype),
                np.asarray(vir_coeff[ka], dtype=store_dtype),
                optimize=True,
            )
            ref_lov[pair_id] = lov

    if len(seen_q) != nkpts:
        raise RuntimeError(f'Reference path only saw q blocks {sorted(seen_q)} of {nkpts}.')

    return {
        'kpts': kpts,
        'kmesh': kmesh,
        'k2gamma_module': k2gamma_module,
        'pair_table': pair_table,
        'nocc_per_k': nocc_per_k,
        'nvir_per_k': nvir_per_k,
        'naux_pos_by_q': naux_pos_by_q,
        'store_dtype': store_dtype,
        'lov': ref_lov,
    }


def read_disk_lov(path):
    from byteqc.lib import FileMp

    with FileMp(path, 'r') as f:
        pair_table = np.asarray(f['pair_table'][()])
        lov = {}
        lov_group = f['lov']
        for pair_id in range(len(pair_table)):
            lov[pair_id] = np.asarray(lov_group[str(pair_id)][:])
        store_dtype = np.dtype(bytes(f['store_dtype'][()]).decode())
        return {
            'kpts': np.asarray(f['kpts'][()]),
            'kmesh': np.asarray(f['kmesh'][()]),
            'pair_table': pair_table,
            'nocc_per_k': np.asarray(f['nocc_per_k'][()]),
            'nvir_per_k': np.asarray(f['nvir_per_k'][()]),
            'naux_pos_by_q': np.asarray(f['naux_pos_by_q'][()]),
            'naux_neg_by_q': np.asarray(f['naux_neg_by_q'][()]),
            'store_dtype': store_dtype,
            'lov': lov,
        }


def compare_results(reference, disk_data, *, aux_blksize, atol):
    _, expected_pair_table = build_pair_mapping(
        reference['kmesh'], k2gamma_module=reference['k2gamma_module'])

    if not np.array_equal(reference['pair_table'], expected_pair_table):
        raise AssertionError('Reference pair_table does not match the expected q/pair mapping.')
    if not np.array_equal(disk_data['pair_table'], expected_pair_table):
        raise AssertionError('Disk pair_table does not match the expected q/pair mapping.')
    if not np.array_equal(disk_data['kmesh'], reference['kmesh']):
        raise AssertionError('kmesh metadata mismatch.')
    if disk_data['store_dtype'] != reference['store_dtype']:
        raise AssertionError(
            f'dtype mismatch: disk={disk_data["store_dtype"]} '
            f'ref={reference["store_dtype"]}')
    if not np.array_equal(disk_data['naux_pos_by_q'], reference['naux_pos_by_q']):
        raise AssertionError('naux_by_q mismatch.')
    if np.any(disk_data['naux_neg_by_q'] != 0):
        raise AssertionError('This standalone test expects no negative-metric branch.')
    if not np.array_equal(disk_data['nocc_per_k'], reference['nocc_per_k']):
        raise AssertionError('nocc_per_k mismatch.')
    if not np.array_equal(disk_data['nvir_per_k'], reference['nvir_per_k']):
        raise AssertionError('nvir_per_k mismatch.')

    if int(reference['naux_pos_by_q'].max()) <= int(aux_blksize):
        raise AssertionError(
            'Aux slicing did not really happen: '
            f'max_naux={int(reference["naux_pos_by_q"].max())}, '
            f'aux_blksize={int(aux_blksize)}.')

    max_abs = 0.0
    sq_norm = 0.0
    nval = 0
    for pair_id, ref_lov in reference['lov'].items():
        disk_lov = disk_data['lov'][pair_id]
        if disk_lov.shape != ref_lov.shape:
            raise AssertionError(
                f'Shape mismatch for pair_id={pair_id}: '
                f'disk={disk_lov.shape} ref={ref_lov.shape}')
        if disk_lov.dtype != reference['store_dtype']:
            raise AssertionError(
                f'Dataset dtype mismatch for pair_id={pair_id}: '
                f'disk={disk_lov.dtype} ref={reference["store_dtype"]}')
        diff = disk_lov - ref_lov
        pair_max = float(np.max(np.abs(diff))) if diff.size else 0.0
        max_abs = max(max_abs, pair_max)
        sq_norm += float(np.vdot(diff.reshape(-1), diff.reshape(-1)).real)
        nval += diff.size

    rms = 0.0 if nval == 0 else float(np.sqrt(sq_norm / nval))
    print(f'aux_blksize={aux_blksize}')
    print(f'naux_pos_by_q={reference["naux_pos_by_q"].tolist()}')
    print(f'max_abs_diff={max_abs:.6e}')
    print(f'rms_diff={rms:.6e}')

    if max_abs > atol:
        raise AssertionError(f'max_abs_diff={max_abs:.6e} exceeds atol={atol:.6e}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aux-blksize', type=int, default=16,
                        help='Small aux block size used to force multiple disk writes.')
    parser.add_argument('--atol', type=float, default=1e-10,
                        help='Absolute tolerance for Lov comparison.')
    args = parser.parse_args()

    lib, pbcgto, pbcscf, gamma_point_fn, k2gamma_module = require_pyscf_stack()
    cell = build_test_cell(pbcgto)
    auxbasis = build_test_auxbasis()
    kmesh = np.asarray([1, 1, 1], dtype=np.int64)
    kpts = cell.make_kpts(kmesh, wrap_around=True)
    mf = run_reference_krhf(cell, kpts, pbcscf=pbcscf)
    cupy_module, build_gpu_cderi_df, build_krmp2_lov_store = require_gpu_stack()

    with_df = build_gpu_cderi_df(cell, kpts=kpts, auxbasis=auxbasis)
    reference = build_reference_lov(
        with_df, mf.mo_coeff, mf.mo_occ,
        cupy_module=cupy_module,
        gamma_point_fn=gamma_point_fn,
        k2gamma_module=k2gamma_module)

    with tempfile.TemporaryDirectory(prefix='krmp2_cderi_compare_', dir=lib.param.TMPDIR) as tmpdir:
        out_file = os.path.join(tmpdir, 'lov_slice_compare.h5')
        result = build_krmp2_lov_store(
            cell,
            mf.mo_coeff,
            kpts=kpts,
            mo_occ=mf.mo_occ,
            auxbasis=auxbasis,
            aux_blksize=args.aux_blksize,
            out_file=out_file,
            verbose=cell.verbose,
        )
        print(f'lov_store={result["out_file"]}')
        disk_data = read_disk_lov(out_file)
        compare_results(reference, disk_data, aux_blksize=args.aux_blksize, atol=args.atol)


if __name__ == '__main__':
    main()
