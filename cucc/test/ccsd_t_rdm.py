from byteqc.cucc import dfccsd
from byteqc.cucc.ccsd_t_rdm import make_rdm1
from byteqc.cucc.ccsd_t_lambda import kernel
from pyscf.cc import dfccsd as ccsd_pyscf
from pyscf.cc.ccsd_t_rdm_slow import make_rdm1 as cpu_make_rdm1
from pyscf.cc.ccsd_t_lambda_slow import kernel as cpu_kernel
from pyscf import gto,scf
import cupy
### Required PySCF version: 2.8.0! (pip install pyscf==2.8.0)

mol = gto.Mole()
mol.verbose = 0
mol.atom = [
    [8 , (0. , 0.     , 0.)],
    [1 , (0. , -0.757 , 0.587)],
    [1 , (0. , 0.757  , 0.587)],    
    [8 , (-6.31504583, -6.96366310, -2.72627950)],
    [1 , (-5.79312801, -6.26504898, -2.37597966)],
    [1 , (-7.14357233, -6.71134567, -2.32921052)]]
basis_set = ['sto-3g','cc-pvdz','cc-pvtz','cc-pvqz']
for basis in basis_set:
    mol.basis = basis
    mol.build()
    rhf = scf.RHF(mol).density_fit()
    rhf.conv_tol = 1e-14
    rhf.verbose = 0
    rhf.scf()
    print("Finish scf.")
    # run byteqc
    mcc = dfccsd.RCCSD(rhf)
    mcc.conv_tol = 1e-12
    ecc,t1,t2 = mcc.kernel()
    print("Finish ccsd.")
    conv,l1,l2 = kernel(mcc)
    rdm1 = make_rdm1(mcc, t1, t2, l1, l2)
    print("Finish GPU-ccsd_t_rdm.")

    # run pyscf as reference
    print("Start reference pyscf calculation.")
    cpu_cc = ccsd_pyscf.RCCSD(rhf)
    cpu_cc.conv_tol = 1e-12
    ecc,t1,t2 = cpu_cc.kernel()
    conv,l1_cpu,l2_cpu = cpu_kernel(cpu_cc)
    ref_rdm1 = cpu_make_rdm1(cpu_cc, t1, t2, l1_cpu, l2_cpu)
    ref_rdm1 = cupy.asarray(ref_rdm1)

    l1_cpu = cupy.asarray(l1_cpu)
    l2_cpu = cupy.asarray(l2_cpu)
    print("Basis:",basis)
    print("l1 error:",cupy.linalg.norm(l1 - l1_cpu))
    print("l1 mae:",cupy.max(cupy.abs(l1 - l1_cpu)))
    print("l2 error:",cupy.linalg.norm(l2 - l2_cpu))
    print("l2 mae:",cupy.max(cupy.abs(l2 - l2_cpu)))
    print("Tr(byteqc_rdm) = ",cupy.trace(rdm1))
    print("rdm1 error:",cupy.linalg.norm(rdm1 - ref_rdm1))
    print("rdm1 mae:",cupy.max(cupy.abs(rdm1 - ref_rdm1)))