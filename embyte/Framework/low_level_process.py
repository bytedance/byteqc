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

import h5py
import os
import time
from byteqc import lib
from byteqc.embyte.Localization import iao_pao_localization
from functools import reduce
from multiprocessing import Pool
import numpy
from byteqc.embyte.ERI import eri_trans
from byteqc.cuobc import scf
import pyscf
from pyscf import df
import cupy
cupy.cuda.set_pinned_memory_allocator(None)


def _normalize_kpts(kpts):
    if kpts is None:
        return None
    if hasattr(kpts, 'kpts'):
        kpts = kpts.kpts
    kpts = numpy.asarray(kpts, dtype=float)
    if kpts.ndim != 2 or kpts.shape[1] != 3:
        raise ValueError(f'Invalid kpts shape {kpts.shape}, expected (nkpts, 3)')
    return kpts


def _kpts_match(kpts_ref, kpts_given, tol=1e-9):
    kpts_ref = _normalize_kpts(kpts_ref)
    kpts_given = _normalize_kpts(kpts_given)
    if kpts_ref is None or kpts_given is None:
        return False
    return kpts_ref.shape == kpts_given.shape and numpy.allclose(
        kpts_ref, kpts_given, atol=tol, rtol=0.0)


def _load_or_transform_pbc_ao_matrix(data_or_path, cell, kpts, kmesh, nao_sc, label, cache=None):
    from byteqc.embyte.Tools import k2gamma as pbc_k2gamma

    if isinstance(data_or_path, str):
        arr = numpy.load(data_or_path, mmap_mode='r')
    else:
        arr = numpy.asarray(data_or_path)

    nkpts = len(kpts)
    nao = int(cell.nao_nr())
    if arr.shape == (nao_sc, nao_sc):
        if numpy.iscomplexobj(arr):
            if numpy.max(numpy.abs(arr.imag)) > 1e-8:
                raise ValueError(f'{label} is complex with non-negligible imaginary part')
            arr = arr.real
        return numpy.asarray(arr, dtype=numpy.float64)
    if arr.shape == (nkpts, nao, nao):
        return pbc_k2gamma.to_supercell_ao_integrals(
            cell, kpts, arr, kmesh=kmesh, force_real=True, cache=cache)
    raise ValueError(
        f'{label} has incompatible shape {arr.shape}; '
        f'expected either ({nao_sc}, {nao_sc}) or ({nkpts}, {nao}, {nao})'
    )


def _fix_orbital_sign_gpu(mo_coeff):
    absmax = cupy.argmax(cupy.abs(mo_coeff), axis=0)
    cols = cupy.arange(mo_coeff.shape[1])
    signs = cupy.where(mo_coeff[absmax, cols] < 0, -1.0, 1.0)
    mo_coeff *= signs[None, :]
    return mo_coeff


def _free_gpu_memory():
    cupy.cuda.get_current_stream().synchronize()
    lib.free_all_blocks()
    cupy.get_default_memory_pool().free_all_blocks()
    cupy.get_default_pinned_memory_pool().free_all_blocks()


def _as_float64_cpu(arr):
    arr = numpy.asarray(arr)
    if numpy.iscomplexobj(arr):
        if numpy.max(numpy.abs(arr.imag)) > 1e-8:
            raise ValueError('Cannot silently discard non-negligible imaginary part')
        arr = arr.real
    return numpy.asarray(arr, dtype=numpy.float64, order='C')


def _ao_matrix_to_lo_cpu(ao_mat_cpu, aolo_gpu):
    ao_mat_gpu = cupy.asarray(_as_float64_cpu(ao_mat_cpu))
    tmp = cupy.dot(ao_mat_gpu, aolo_gpu)
    lo_mat = cupy.dot(aolo_gpu.T, tmp).get()
    del ao_mat_gpu, tmp
    _free_gpu_memory()
    return lo_mat


def _remove_mf_ewald_shift(mo_energy, mo_occ, madelung):
    mo_energy = numpy.asarray(mo_energy, dtype=numpy.float64).copy()
    mo_occ = numpy.asarray(mo_occ)
    mo_energy[mo_occ > 0] += madelung
    return mo_energy


def _env_flag(name, default='0'):
    return bool(int(os.environ.get(name, default)))


def _sample_indices(n, nsample):
    n = int(n)
    nsample = max(0, min(int(nsample), n))
    if nsample == 0:
        return numpy.empty(0, dtype=numpy.int64)
    if nsample == n:
        return numpy.arange(n, dtype=numpy.int64)
    return numpy.unique(numpy.linspace(0, n - 1, nsample, dtype=numpy.int64))


def _metric_deviation(metric, target_diag=1.0):
    metric = numpy.asarray(metric)
    diag = numpy.diag(metric)
    offdiag = metric.copy()
    offdiag[numpy.diag_indices_from(offdiag)] = 0.0
    target = numpy.asarray(target_diag, dtype=diag.dtype)
    if target.ndim == 0:
        diag_err = numpy.max(numpy.abs(diag - target.item())) if diag.size else 0.0
    else:
        diag_err = numpy.max(numpy.abs(diag - target)) if diag.size else 0.0
    offdiag_err = numpy.max(numpy.abs(offdiag)) if offdiag.size else 0.0
    return float(max(diag_err, offdiag_err)), float(diag_err), float(offdiag_err)


def _log_orth_check(logger, label, err, diag_err, offdiag_err, tol):
    msg = (
        f'----------- {label}: sampled metric max_err={err:.3e}, '
        f'diag_err={diag_err:.3e}, offdiag_err={offdiag_err:.3e}, tol={tol:.3e}'
    )
    if err > tol:
        logger.warning(msg)
    else:
        logger.info(msg)


def _sample_s_metric_check_gpu(logger, label, coeff, ovlp, nsample, tol):
    idx = _sample_indices(coeff.shape[1], nsample)
    if idx.size == 0:
        return
    coeff_sample = cupy.ascontiguousarray(coeff[:, idx])
    s_coeff = cupy.dot(ovlp, coeff_sample)
    metric = cupy.dot(coeff_sample.T.conj(), s_coeff).get()
    del coeff_sample, s_coeff
    _free_gpu_memory()
    err, diag_err, offdiag_err = _metric_deviation(metric, 1.0)
    _log_orth_check(logger, label, err, diag_err, offdiag_err, tol)
    del metric


def _sample_euclidean_metric_check_gpu(logger, label, coeff, nsample, tol):
    idx = _sample_indices(coeff.shape[1], nsample)
    if idx.size == 0:
        return
    coeff_sample = cupy.ascontiguousarray(coeff[:, idx])
    metric = cupy.dot(coeff_sample.T.conj(), coeff_sample).get()
    del coeff_sample
    _free_gpu_memory()
    err, diag_err, offdiag_err = _metric_deviation(metric, 1.0)
    _log_orth_check(logger, label, err, diag_err, offdiag_err, tol)
    del metric


def _sample_lomo_occ_metric_check_gpu(logger, LOMO, mf_mo_occ, nsample, tol):
    occ_idx = numpy.where(numpy.asarray(mf_mo_occ) > 0)[0]
    idx = occ_idx[_sample_indices(len(occ_idx), nsample)]
    if idx.size == 0:
        return
    lomo_sample = cupy.ascontiguousarray(LOMO[:, idx])
    metric = cupy.dot(lomo_sample.T.conj(), lomo_sample).get()
    del lomo_sample
    _free_gpu_memory()
    err, diag_err, offdiag_err = _metric_deviation(metric, 1.0)
    _log_orth_check(logger, 'LOMO occupied Euclidean orthogonality',
                    err, diag_err, offdiag_err, tol)
    del metric


def _sample_overlap_difference_check(logger, mol, ovlp, nsample, tol):
    if not _env_flag('BYTEQC_LOW_LEVEL_DIRECT_S_CHECK'):
        return
    if not hasattr(mol, 'pbc_intor'):
        return
    idx = _sample_indices(mol.nao_nr(), nsample)
    if idx.size == 0:
        return
    logger.info(
        '----------- direct supercell overlap sample check is enabled '
        '(BYTEQC_LOW_LEVEL_DIRECT_S_CHECK=1)')
    s_direct = mol.pbc_intor('int1e_ovlp', hermi=1, kpts=None)
    s_direct = _as_float64_cpu(s_direct)
    if isinstance(ovlp, cupy.ndarray):
        idx_gpu = cupy.asarray(idx)
        s_sample = ovlp[idx_gpu[:, None], idx_gpu].get()
        del idx_gpu
    else:
        s_sample = numpy.asarray(ovlp)[numpy.ix_(idx, idx)]
    diff = s_sample - s_direct[numpy.ix_(idx, idx)]
    max_diff = float(numpy.max(numpy.abs(diff))) if diff.size else 0.0
    msg = (
        f'----------- S(k2gamma) vs direct supercell S sampled max_abs_diff='
        f'{max_diff:.3e}, tol={tol:.3e}'
    )
    if max_diff > tol:
        logger.warning(msg)
    else:
        logger.info(msg)
    del s_direct, s_sample, diff
    _free_gpu_memory()


def _sample_complex_k2gamma_mo_check(logger, primitive_mol, kpts, kmesh, primitive_mf,
                                     ovlp, nsample, tol):
    if not _env_flag('BYTEQC_LOW_LEVEL_COMPLEX_K2G_MO_CHECK'):
        return
    from byteqc.embyte.Tools import k2gamma as pbc_k2gamma

    logger.info(
        '----------- complex k2gamma MO sample check is enabled '
        '(BYTEQC_LOW_LEVEL_COMPLEX_K2G_MO_CHECK=1)')
    cache = pbc_k2gamma.build_k2gamma_cache(
        primitive_mol, kpts, kmesh=kmesh, wrap_around=False)
    try:
        _scell, _e_g, c_gamma, _mo_phase = pbc_k2gamma.mo_k2gamma(
            primitive_mol,
            primitive_mf.mo_energy,
            primitive_mf.mo_coeff,
            kpts,
            kmesh=kmesh,
            cache=cache,
            with_mo_phase=False,
            force_real=False,
            realify_if_needed=False,
        )
        idx = _sample_indices(c_gamma.shape[1], nsample)
        if idx.size == 0:
            return
        c_sample = cupy.asarray(
            numpy.asarray(c_gamma[:, idx], dtype=numpy.complex128, order='C'))
        s_c = cupy.dot(ovlp, c_sample)
        metric = cupy.dot(c_sample.T.conj(), s_c).get()
        del c_sample, s_c
        _free_gpu_memory()
        err, diag_err, offdiag_err = _metric_deviation(metric, 1.0)
        _log_orth_check(
            logger, 'complex k2gamma MO S-orthogonality',
            err, diag_err, offdiag_err, tol)
        del metric
    finally:
        try:
            del cache
        except UnboundLocalError:
            pass
        try:
            del c_gamma
        except UnboundLocalError:
            pass
        _free_gpu_memory()


def _run_low_level_orthogonality_checks(logger, mol, ao_ovlp, aolo, mf_mo_coeff,
                                        mf_mo_occ, lomo=None):
    if not _env_flag('BYTEQC_LOW_LEVEL_ORTH_CHECK', '1'):
        return
    nsample = int(os.environ.get('BYTEQC_LOW_LEVEL_ORTH_CHECK_COLS', '2048'))
    tol = float(os.environ.get('BYTEQC_LOW_LEVEL_ORTH_CHECK_TOL', '1e-6'))
    logger.info(
        '----------- low-level sampled orthogonality check enabled: '
        'nsample=%d tol=%.3e',
        nsample,
        tol,
    )
    _sample_overlap_difference_check(logger, mol, ao_ovlp, nsample, tol)
    _sample_s_metric_check_gpu(logger, 'AOLO S-orthogonality', aolo, ao_ovlp, nsample, tol)
    _sample_s_metric_check_gpu(logger, 'MO S-orthogonality', mf_mo_coeff, ao_ovlp, nsample, tol)
    if lomo is not None:
        _sample_euclidean_metric_check_gpu(logger, 'LOMO Euclidean orthogonality', lomo, nsample, tol)
        _sample_lomo_occ_metric_check_gpu(logger, lomo, mf_mo_occ, nsample, tol)


_LOW_LEVEL_MATRIX_ATTRS = (
    'AOLO',
    'onerdm_low_ao',
    'low_scf_fock',
    'ao_ovlp',
    'LOMO',
    # 'AOMO',
    'onerdm_low',
    'oei_LO',
    'fock_LO',
    'MOLO',
)


class low_level_info:
    '''
    Collect all the low level information from the mean field calculation.
    Commonly the mean field comes from the converged HF check file which is provided by pyscf.
    The low-level calculation would not be done in SIE workflow.
    '''

    def __getattribute__(self, name):
        if not name.startswith('_'):
            try:
                disk_arrays = object.__getattribute__(self, '_disk_arrays')
            except AttributeError:
                disk_arrays = None
            if disk_arrays is not None and name in disk_arrays:
                return object.__getattribute__(self, '_load_disk_matrix')(name)
        return object.__getattribute__(self, name)

    def _init_disk_matrix_store(self, disk_matrix_file):
        self._disk_matrix_file = disk_matrix_file
        self._disk_arrays = {}
        if disk_matrix_file is None:
            return
        dirname = os.path.dirname(disk_matrix_file)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with lib.FileMp(disk_matrix_file, 'w'):
            pass

    def _matrix_file_block_size(self, shape):
        nproc = max(int(getattr(lib, 'NumFileProcess', 1)), 1)
        return (max(int((shape[0] + nproc - 1) // nproc), 1),)

    def _write_matrix_to_disk(self, filemp, name, arr):
        is_gpu_array = isinstance(arr, cupy.ndarray)
        if is_gpu_array:
            arr_cpu = cupy.asnumpy(arr)
        else:
            arr_cpu = numpy.asarray(arr)
        if arr_cpu.ndim != 2:
            return False
        arr_cpu = numpy.asarray(arr_cpu, order='C')
        if name in filemp:
            del filemp[name]
        dataset = filemp.create_dataset(
            name,
            shape=arr_cpu.shape,
            dtype=arr_cpu.dtype,
            blksizes=self._matrix_file_block_size(arr_cpu.shape),
        )
        pool = Pool(processes=lib.NumFileProcess)
        try:
            waits = dataset.setitem(numpy.s_[:], arr_cpu, pool=pool)
            for wait in waits:
                wait.wait()
        finally:
            pool.close()
            pool.join()
        self._disk_arrays[name] = {
            'path': self._disk_matrix_file,
            'dataset': name,
            'shape': tuple(arr_cpu.shape),
            'dtype': numpy.dtype(arr_cpu.dtype).str,
            'as_gpu': False,
        }
        del arr_cpu
        return True

    def _offload_low_level_matrices(self):
        if self._disk_matrix_file is None:
            return
        offloaded = []
        with lib.FileMp(self._disk_matrix_file, 'a') as filemp:
            for name in _LOW_LEVEL_MATRIX_ATTRS:
                if name not in self.__dict__:
                    continue
                arr = self.__dict__[name]
                if arr is None or not hasattr(arr, 'ndim') or arr.ndim != 2:
                    continue
                if self._write_matrix_to_disk(filemp, name, arr):
                    del self.__dict__[name]
                    offloaded.append(name)
        if offloaded:
            _free_gpu_memory()

    def _offload_matrix_attr(self, name):
        if self._disk_matrix_file is None or name not in self.__dict__:
            return
        arr = self.__dict__[name]
        if arr is None or not hasattr(arr, 'ndim') or arr.ndim != 2:
            return
        with lib.FileMp(self._disk_matrix_file, 'a') as filemp:
            if self._write_matrix_to_disk(filemp, name, arr):
                del self.__dict__[name]
                _free_gpu_memory()

    def _load_disk_matrix(self, name):
        import cupyx

        meta = self._disk_arrays[name]
        shape = tuple(meta['shape'])
        dtype = numpy.dtype(meta['dtype'])
        buf = cupyx.empty_pinned(shape, dtype=dtype)
        with lib.FileMp(meta['path'], 'r') as filemp:
            dataset = filemp[meta['dataset']]
            pool = Pool(processes=lib.NumFileProcess)
            try:
                arr = dataset.getitem(numpy.s_[:], pool=pool, buf=buf)
                arr.wait()
            finally:
                pool.close()
                pool.join()
        arr_cpu = numpy.asarray(buf)
        return arr_cpu

    def __init__(self, mol,
                 mf,
                 LG,
                 aux_basis=None,
                 jk_file=None,
                 with_eri=False,
                 oei=None,
                 local_orb_path=None,
                 ewald_correct=False,
                 kpts=None,
                 mf_with_ewald=None,
                 disk_matrix_file=None,
                 ):

        logger = LG.logger
        if disk_matrix_file is None:
            disk_matrix_file = os.path.join(LG.filepath, 'low_level_info_matrices.h5')
        self._init_disk_matrix_store(disk_matrix_file)
        primitive_mol = mol
        primitive_mf = mf
        self.kpts = _normalize_kpts(kpts)
        self.kmesh = None
        self.eri_mol = primitive_mol
        self.eri_auxmol = None
        loaded_kpts = getattr(primitive_mf, 'kpts', None)
        if getattr(primitive_mol, 'pbc_intor', None) is not None and self.kpts is None:
            try:
                loaded_kpts = _normalize_kpts(loaded_kpts)
            except Exception:
                loaded_kpts = None
            if loaded_kpts is not None and len(loaded_kpts) > 1:
                raise ValueError(
                    'A multi-k mean-field / chk was detected. Please provide low_level_info(..., kpts=...)'
                )

        if getattr(primitive_mol, 'pbc_intor', None) is not None and self.kpts is not None:
            from byteqc.embyte.Tools import k2gamma as gpu_k2gamma

            if loaded_kpts is None:
                raise ValueError(
                    'kpts was provided, but the mean-field object does not contain k-point information'
                )
            if not _kpts_match(loaded_kpts, self.kpts):
                raise ValueError(
                    'The provided kpts does not match the kpts stored in the HF checkpoint / mean-field object'
                )

            self.kmesh = numpy.asarray(
                gpu_k2gamma.kpts_to_kmesh(primitive_mol, self.kpts),
                dtype=int,
            )
            mf = gpu_k2gamma.k2gamma(primitive_mf, kmesh=self.kmesh)
            mf.e_tot = primitive_mf.e_tot * len(self.kpts)
            mol = mf.cell
            logger.info(
                '----------- Use multi-k low-level path, kmesh=%s, nkpts=%d',
                tuple(self.kmesh.tolist()),
                len(self.kpts),
            )

        if getattr(primitive_mol, 'pbc_intor', None) is not None:
            if jk_file is None or oei is None:
                raise ValueError(
                    'Periodic systems require both jk_file and oei to be provided explicitly.'
                )
            if mf_with_ewald is None:
                raise ValueError(
                    'Periodic low_level_info requires mf_with_ewald=True/False because '
                    'external JK/OEI are assumed to be generated without Ewald exxdiv.'
                )

        if aux_basis is None:
            self.auxmol = df.addons.make_auxmol(mol, df.make_auxbasis(mol))
            if getattr(primitive_mol, 'pbc_intor', None) is not None and self.kpts is not None:
                self.eri_auxmol = df.addons.make_auxmol(
                    primitive_mol, df.make_auxbasis(primitive_mol))
        else:
            self.auxmol = df.addons.make_auxmol(mol, aux_basis)
            if getattr(primitive_mol, 'pbc_intor', None) is not None and self.kpts is not None:
                self.eri_auxmol = df.addons.make_auxmol(primitive_mol, aux_basis)

        if self.eri_auxmol is None:
            self.eri_auxmol = self.auxmol

        self.mol_full = mol

        mempool = cupy.get_default_memory_pool()
        mempool.free_all_blocks()

        blksize = int((lib.gpu_avail_bytes() / (8 * 2)) ** (1 / 3))
        if with_eri is False and getattr(mol, 'pbc_intor', None) is None:
            self.j2c = os.path.join(LG.filepath + 'j2c')
            lib.free_all_blocks()
            with h5py.File(self.j2c, 'w') as f:
                logger.info("--- start to get j2c")
                j2c, self.nL_un, self.nL = eri_trans.get_j2c(
                    logger, mol=self.mol_full, auxmol=self.auxmol)
                f.create_dataset('j2c', data=j2c, dtype='f8')
                del j2c
                logger.info(f'--- Save j2c to the disk {self.j2c}')
        else:

            self.j2c = None

        lib.free_all_blocks()

        if getattr(mol, 'pbc_intor', None) is not None and ewald_correct:
            from pyscf.pbc import tools
            self.madelung = tools.madelung(mf.cell, mf.kpt)
            self.ewald_correct = True
            logger.info("----------- Using Ewald correction. The results may not be reliable.")
        else:
            self.ewald_correct = False

        t_localized = time.time()

        if local_orb_path == 'lowdin':
            logger.info("----------- Using meta_lowdin")
            self.AOLO = cupy.asarray(
                pyscf.lo.orth.orth_ao(
                    mf, method='meta_lowdin'))
        elif local_orb_path is not None:
            logger.info("----------- Using load AOLO from file")
            self.AOLO = cupy.asarray(numpy.load(local_orb_path))
        else:
            logger.info("----------- Using IAO+PAO localizer")
            self.AOLO = _fix_orbital_sign_gpu(iao_pao_localization(mol, mf))

        logger.info(
            '----------- localization_function time cost is %s' %
            (time.time() - t_localized))

        self.low_scf_energy = mf.e_tot
        mf_mo_occ = numpy.asarray(mf.mo_occ)
        oei_ao_cpu = None
        low_scf_fock_cpu = None
        ao_ovlp_cpu = None
        pbc_lo_mats_done = False
        if self.kpts is None:
            mf_mo_coeff = cupy.asarray(mf.mo_coeff)
            mf.mo_coeff = None
            occ_idx = mf_mo_occ > 0
            occ_coeff = mf_mo_coeff[:, occ_idx] * cupy.sqrt(
                cupy.asarray(mf_mo_occ[occ_idx]))[None, :]
            low_scf_dm = cupy.dot(occ_coeff, occ_coeff.T)
            del occ_coeff
            self.onerdm_low_ao = low_scf_dm.get()
            self._offload_matrix_attr('onerdm_low_ao')
            if getattr(mol, 'pbc_intor', None) is None:
                if jk_file is None:
                    blksize = int((lib.gpu_avail_bytes() / (8 * 2)) ** (1 / 4))
                    vhfopt = scf.hf._VHFOpt(mol, 'int2e')
                    vhfopt.build(group_size=blksize)
                    t_get_JK = time.time()
                    j, k = scf.hf.get_jk(mol, low_scf_dm.get(), vhfopt=vhfopt)
                    logger.info(
                        '----------- get_jk time cost is %s' %
                        (time.time() - t_get_JK))
                    del vhfopt
                    lib.free_all_blocks()
                    low_scf_twoint = cupy.asarray(j - 0.5 * k)
                else:
                    logger.info('----------- load JK from %s' % jk_file)
                    low_scf_twoint = cupy.asarray(numpy.load(jk_file))

                if oei is None:
                    self.low_scf_fock = cupy.asarray(
                        mf.get_hcore(mol) + low_scf_twoint.get())
                else:
                    assert isinstance(oei, str), f'The oei type is not right, expect str but {type(oei)} provided'
                    self.low_scf_fock = cupy.asarray(
                        numpy.load(oei) + low_scf_twoint.get())
            else:
                logger.info('----------- load JK from %s' % jk_file)
                low_scf_twoint = cupy.asarray(numpy.load(jk_file))
                assert isinstance(oei, str), f'The oei type is not right, expect str but {type(oei)} provided'
                self.low_scf_fock = cupy.asarray(
                    numpy.load(oei) + low_scf_twoint.get())

            del low_scf_dm
            self.ao_ovlp = cupy.asarray(mf.get_ovlp(mol))
        else:
            from byteqc.embyte.Tools import k2gamma as pbc_k2gamma

            nao_sc = int(self.mol_full.nao_nr())
            pbc_cache = pbc_k2gamma.build_k2gamma_cache(
                primitive_mol, self.kpts, kmesh=self.kmesh)
            low_scf_dm_k = numpy.asarray(primitive_mf.make_rdm1())
            self.onerdm_low_ao = pbc_k2gamma.to_supercell_ao_integrals(
                primitive_mol, self.kpts, low_scf_dm_k, kmesh=self.kmesh,
                force_real=True, cache=pbc_cache)
            del low_scf_dm_k
            self._offload_matrix_attr('onerdm_low_ao')

            logger.info('----------- load JK from %s and transform it to gamma supercell' % jk_file)
            jk_ao_cpu = _load_or_transform_pbc_ao_matrix(
                jk_file, primitive_mol, self.kpts, self.kmesh, nao_sc, 'jk_file',
                cache=pbc_cache
            )
            oei_ao_cpu = _load_or_transform_pbc_ao_matrix(
                oei, primitive_mol, self.kpts, self.kmesh, nao_sc, 'oei',
                cache=pbc_cache
            )
            low_scf_fock_cpu = numpy.array(jk_ao_cpu, dtype=numpy.float64, order='C', copy=True)
            low_scf_fock_cpu += oei_ao_cpu
            del jk_ao_cpu
            low_scf_twoint = None
            self.oei_LO = _ao_matrix_to_lo_cpu(oei_ao_cpu, self.AOLO)
            self._offload_matrix_attr('oei_LO')
            del oei_ao_cpu
            self.fock_LO = _ao_matrix_to_lo_cpu(low_scf_fock_cpu, self.AOLO)
            self._offload_matrix_attr('fock_LO')
            self.low_scf_fock = low_scf_fock_cpu
            self._offload_matrix_attr('low_scf_fock')
            del low_scf_fock_cpu
            pbc_lo_mats_done = True

            s_k = numpy.asarray(primitive_mf.get_ovlp(primitive_mol))
            ao_ovlp_cpu = pbc_k2gamma.to_supercell_ao_integrals(
                primitive_mol, self.kpts, s_k, kmesh=self.kmesh,
                force_real=True, cache=pbc_cache
            )
            del s_k, pbc_cache
            self.ao_ovlp = cupy.asarray(ao_ovlp_cpu)
            if _env_flag('BYTEQC_LOW_LEVEL_COMPLEX_K2G_MO_CHECK'):
                nsample = int(os.environ.get('BYTEQC_LOW_LEVEL_ORTH_CHECK_COLS', '2048'))
                tol = float(os.environ.get('BYTEQC_LOW_LEVEL_ORTH_CHECK_TOL', '1e-6'))
                _sample_complex_k2gamma_mo_check(
                    logger,
                    primitive_mol,
                    self.kpts,
                    self.kmesh,
                    primitive_mf,
                    self.ao_ovlp,
                    nsample,
                    tol,
                )
            mf_mo_coeff = cupy.asarray(numpy.asarray(mf.mo_coeff, dtype=numpy.float64))
            mf.mo_coeff = None
            _free_gpu_memory()

        # Multi-k supercells are too large for a full gamma Fock diagonalization.
        # The k2gamma mean-field already carries the k-sorted orbital energies.
        from byteqc.embyte.Tools import k2gamma as pbc_k2gamma
        if self.kpts is None:
            self.mo_energy = pbc_k2gamma.gpu_generalized_eigvalsh(
                self.low_scf_fock, self.ao_ovlp, logger=logger)
        else:
            self.mo_energy = numpy.asarray(mf.mo_energy, dtype=numpy.float64).copy()
            if mf_with_ewald:
                from pyscf.pbc import tools
                madelung_mf = tools.madelung(primitive_mol, self.kpts)
                self.mo_energy = _remove_mf_ewald_shift(
                    self.mo_energy, mf_mo_occ, madelung_mf)
                logger.info(
                    '----------- Remove Ewald occupied-orbital shift from mf.mo_energy: +%s',
                    madelung_mf)

        _run_low_level_orthogonality_checks(
            logger,
            mol,
            self.ao_ovlp,
            self.AOLO,
            mf_mo_coeff,
            mf_mo_occ,
        )

        s_mo = cupy.dot(self.ao_ovlp, mf_mo_coeff)
        self.LOMO = cupy.dot(self.AOLO.T, s_mo)
        del s_mo
        self.LOMO = _fix_orbital_sign_gpu(self.LOMO)
        if _env_flag('BYTEQC_LOW_LEVEL_ORTH_CHECK', '1'):
            nsample = int(os.environ.get('BYTEQC_LOW_LEVEL_ORTH_CHECK_COLS', '2048'))
            tol = float(os.environ.get('BYTEQC_LOW_LEVEL_ORTH_CHECK_TOL', '1e-6'))
            _sample_euclidean_metric_check_gpu(
                logger, 'LOMO Euclidean orthogonality', self.LOMO, nsample, tol)
            _sample_lomo_occ_metric_check_gpu(
                logger, self.LOMO, mf_mo_occ, nsample, tol)
        occ_idx = mf_mo_occ > 0
        lomo_occ = self.LOMO[:, occ_idx] * cupy.sqrt(
            cupy.asarray(mf_mo_occ[occ_idx]))[None, :]
        self.onerdm_low = cupy.dot(lomo_occ, lomo_occ.T).get()
        del lomo_occ
        self._offload_matrix_attr('onerdm_low')
        self.LOMO = self.LOMO.get()
        # self.AOMO = mf_mo_coeff.get()
        # self._offload_matrix_attr('AOMO')
        del mf_mo_coeff
        if self.kpts is not None:
            del self.ao_ovlp
            self.ao_ovlp = ao_ovlp_cpu
            self._offload_matrix_attr('ao_ovlp')
            del ao_ovlp_cpu
        else:
            self.ao_ovlp = self.ao_ovlp.get()
            self._offload_matrix_attr('ao_ovlp')
        _free_gpu_memory()

        self.core_constant_energy = mf.mol.energy_nuc()
        if self.kpts is None:
            self.oei_LO = reduce(
                cupy.dot,
                (self.AOLO.T,
                 self.low_scf_fock
                 - low_scf_twoint,
                 self.AOLO)).get()
            self.fock_LO = reduce(
                cupy.dot,
                (self.AOLO.T,
                 self.low_scf_fock,
                 self.AOLO)).get()
            self.low_scf_fock = self.low_scf_fock.get()
        else:
            if not pbc_lo_mats_done:
                self.oei_LO = _ao_matrix_to_lo_cpu(oei_ao_cpu, self.AOLO)
                del oei_ao_cpu
                self.fock_LO = _ao_matrix_to_lo_cpu(low_scf_fock_cpu, self.AOLO)
                self.low_scf_fock = low_scf_fock_cpu
                del low_scf_fock_cpu

        self.num_occ = int(mol.nelectron / 2)
        self.MOLO = self.LOMO.conj().T

        self._offload_low_level_matrices()

        del mf
        if hasattr(self.auxmol, 'stdout'):
            try:
                del self.auxmol.stdout
            except:
                pass
