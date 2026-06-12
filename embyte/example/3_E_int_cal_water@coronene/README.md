# SIE+CCSD(T) Framework, Take Water(2-leg)@Coronene as an Example

## Run the Scripts
Please run the examples in order of their index.
```
python 1_HF.py
python 2_MP2.py
python 3-1_SIE.py or 3-2_SIE_with_symm.py
```
 - `1_HF.py` run GPU accelerated HF calculation for Water@Coronene. The HF calculation is supported by `byteqc.cuobc.scf.RHF`.
 - `2_MP2.py` run GPU accelerated MP2 calculation for Water@Coronene. The MP2 calculation is supported by `byteqc.cump2.DFMP2`.
 - `3-1_SIE.py` run GPU accelerated SIE calculation for Water@Coronene.
 - `3-2_SIE_with_symm.py` run GPU accelerated SIE calculation for Water@Coronene by using the symmetry of this system, which could save about 3/4 computational cost compared with `3-1_SIE.py`.

## Get Energy
As discussed in [our work](https://arxiv.org/abs/2412.18553), the total energy for SIE+CCSD(T) with bath truncation error correction is
$$
E_{\text{tot}} = E_{\text{HF}} + E_{\text{corr}},
$$ 
where $E_{\text{HF}}$ is the HF energy coming from the `1_HF.py`. and $E_{\text{corr}}$ is the correlation energy coming from the post-HF method. In SIE+CCSD(T), $E_{\text{corr}}$ is
$$
E_{\text{corr}} = E_{\text{SIE+CCSD(T)}} + (E_{\text{MP2}} - E_{\text{SIE+MP2}}),
$$
where the $E_{\text{SIE+CCSD(T)}}$ is the correlation energy comes from bare SIE+CCSD(T) calculation and the difference from MP2 and SIE+MP2, $(E_{\text{MP2}} - E_{\text{SIE+MP2}})$, is for the bath truncation error correction. $E_{\text{MP2}}$ comes from the `2_MP2.py`.

The $E_{\text{SIE+CCSD(T)}}$ could be obtained from the `3-1_SIE.py` or `3-2_SIE_with_symm.py` by setting the parameter `if_MP2` to `False` and `SIE_class.in_situ_T` to `True`. The $E_{\text{SIE+MP2}}$ could be also obtained from the `3-1_SIE.py` or `3-2_SIE_with_symm.py` by setting the parameter `if_MP2` to `True`.

Notably, $E_{\text{SIE+CCSD(T)}}$ and $E_{\text{SIE+MP2}}$ should be obtained at the same threshold setting. And the option of `SIE_class.RDM` is highly recommended to be `True` to using global 1-RDM and in-cluster 2-RDM to obtained correlation energy, which is more accurate than it turning off.

## Interacting Energy Calculation & Basis Set Superposition Error Correction
The interacting energy between water and coronene with basis set superposition error correction is defined as
$$
E_{\text{int}} = E_{\text{tot}}(\text{water + coronene}) - E_{\text{tot}}(\text{water + ghost-coronene}) - E_{\text{tot}}(\text{ghost-water + coronene}),
$$
where the $E_{\text{tot}}$ is calculated in the workflow above, the ghost-water or ghost-coronene means the water or coronene is set as the ghost atom. It is very easy to achieve in all scripts by just setting the `mol_type` to `2`, `1` or `0`. `mol_type = 2` the scripts will execute the calculation for water + coronene system, `mol_type = 1` is for ghost-water + coronene and `mol_type = 0` is for water + ghost-coronene. Note that, when `mol_type` is modified, the corresponding fragments would also be modified by excluding the ghost fragment which do not contain any occupied orbitals.
