from byteqc import lib
from pyscf.pbc import gto
from pyscf.pbc import scf as cscf
from byteqc.cupbc import scf as pscf
import os
import sys
import numpy
import time
os.environ['PYSCF_EXT_PATH'] = '/mnt/bn/jemiry-nas2/code/CPU_RSDF'

# lib.Mg.set_gpus(1)
name = './cupbc.txt'
kmeshs = [[1, 1, 1], [1, 1, 2], [1, 2, 2], [2, 2, 2]]
basises = ['6-31g', 'ccpvdz', 'ccpvtz', 'ccpvqz', 'ccpv5z']
omega = 0.1
xc = 'PBE'
max_cycle = 2
verbose = 5
atom = '''
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
    '''
a = '''
    7.3899998665 0.0000000000 0.0000000000
    0.0000000000 9.4200000763 0.0000000000
    0.0000000000 0.0000000000 6.8099999428
    '''


class DoubleOutput:
    def __init__(self, name="tmp.txt", isinit=True):
        self.name = name
        self.encoding = sys.stdout.encoding
        if isinit:
            self.init()

    def __del__(self):
        self.restore

    def init(self):
        self.file = open(self.name, "w")
        self.stdout_old = sys.stdout
        sys.stdout = self
        self.time = time.time()
        self.write(time.strftime(
            "Start logging at %m%d-%H:%M:%S\n", time.localtime()))

    def restore(self):
        sys.stdout = self.stdout_old
        self.file.close()

    def difftime(self):
        t2 = time.time()
        dt = t2 - self.time
        if dt < 1:
            return "%.3fs" % dt
        if dt < 60:
            return "%.1fs" % dt
        dt = round(dt)
        strtime = '%ds' % (dt % 60)
        dt //= 60
        strtime = '%dm' % (dt % 60) + strtime
        if dt < 60:
            return strtime
        dt //= 60
        strtime = '%dh' % (dt % 24) + strtime
        if dt < 24:
            return strtime
        dt //= 24
        return '%dd' % dt + strtime

    def write(self, data):
        if data and data.isspace():
            self.file.write(data)
        else:
            str = "[%s]" % self.difftime()
            self.file.write(f"{str}{data}")
        self.stdout_old.write(data)

    def flush(self):
        self.stdout_old.flush()
        self.file.flush()


def run_rsdf(scf, cell, kpts, max_cycle=0, verbose=4, **kargs):
    mf = scf.KRKS(cell, kpts=kpts).rs_density_fit(**kargs)
    mf.xc = xc
    mf.verbose = verbose
    mf.with_df.verbose = verbose
    mf.max_memory = 700000
    mf.with_df.max_memory = 700000
    mf.max_cycle = max_cycle
    mf.with_df.verbose = 0
    mf.with_df.omega = omega
    mf.with_df.direct = True
    mf.with_df.ksym = 's1'
    mf.with_df.use_bvk = [True, True]
    get_occ_ = mf.get_occ
    mf.dump_flags = lambda *args: None

    def get_occ(*args):
        v = mf.verbose
        mf.verbose = 0
        occ = get_occ_(*args)
        mf.verbose = v
        return occ
    mf.get_occ = get_occ
    if scf == cscf:
        from pyscf.pbc.df import df_jk
        from pyscf.rspbc.df.rsdf_direct_jk import get_j_kpts
        df_jk.get_j_kpts = get_j_kpts
    mf.kernel()
    return mf.e_tot


device = eval(os.getenv('DEFAULT_DEVICE', '0'))
logout = DoubleOutput(isinit=True, name=name)
for kmesh in kmeshs:
    for basis in basises:
        if basis != 'ccpvdz' or kmesh != [1,1,1]:
            continue
        cell = gto.Cell(atom=atom, basis=basis, a=a, pseudo=None)
        cell.verbose = 0
        cell.build()
        print("nbas:%s nao:%d kmesh:%s basis:%s" %
              (cell.nbas, cell.nao, str(kmesh), basis))
        kpts = cell.make_kpts(kmesh)
        if device == 0:
            e = run_rsdf(pscf, cell, kpts, max_cycle, verbose)
        else:
            e = run_rsdf(cscf, cell, kpts, max_cycle, verbose)
        print("Energy:", e, "\n")
