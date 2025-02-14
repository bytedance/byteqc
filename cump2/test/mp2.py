# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# This file is part of ByteQC.
#
# ByteQC is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ByteQC is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import time
from byteqc import cump2
from pyscf.mp.dfmp2_native import DFMP2
from pyscf import gto
import os


def get_mol(nwater=8, basis='ccpvdz'):
    mol = gto.M()
    mol.atom = '''
    O	0.000000	0.000000	0.000000
    H	-0.623370	-0.719980	-0.273990
    H	-0.417950	0.335830	0.839320
    O	-1.462640	0.669480	2.227100
    H	-1.234510	1.513890	2.692860
    H	-1.325880	-0.011370	2.941000
    O	2.620180	-0.682850	0.464750
    H	2.929570	0.258200	0.491690
    H	1.641560	-0.581160	0.310220
    O	-1.741160	-2.140280	-0.369960
    H	-2.486110	-2.170650	-1.020140
    H	-2.163310	-2.254310	0.522870
    O	0.958470	1.834770	-1.956860
    H	0.540530	1.246680	-1.280060
    H	1.852600	1.439040	-2.136530
    O	-2.513520	1.029700	-1.812940
    H	-1.961230	0.330540	-2.242760
    H	-3.024280	1.455940	-2.551720
    O	-1.020430	-0.887900	-3.298690
    H	-1.329820	-1.828950	-3.325630
    H	-0.041810	-0.989590	-3.144170
    O	-1.542670	3.194420	-0.356830
    H	-1.770800	2.350000	-0.822590
    H	-1.679430	3.875270	-1.070740
    O	0.641290	-3.405520	-0.877080
    H	1.059230	-2.817420	-1.553880
    H	-0.252840	-3.009790	-0.697410
    O	1.599760	-1.570740	-2.833940
    H	2.223120	-0.850770	-2.559960
    H	2.017700	-1.906570	-3.673260
    O	0.777110	3.294370	1.293090
    H	0.032170	3.263990	0.642910
    H	0.354970	3.180330	2.185920
    O	3.159560	2.029120	0.785970
    H	3.577500	2.617220	0.109170
    H	2.265430	2.424850	0.965640
    O	-3.782420	0.569530	0.577170
    H	-3.037480	0.599900	1.227350
    H	-3.360280	0.683570	-0.315650
    O	3.340910	0.569530	-2.463980
    H	4.085860	0.599900	-1.813800
    H	3.763060	0.683570	-3.356810
    O	-3.010060	-2.600450	2.020150
    H	-3.562360	-1.901290	2.449970
    H	-2.499300	-3.026680	2.758930
    O	3.062400	-2.240220	2.684320
    H	2.834260	-3.084640	2.218560
    H	2.925640	-1.559370	1.970420
    O	-1.020430	-0.887900	4.446670
    H	-1.329820	-1.828950	4.419730
    H	-0.041810	-0.989590	4.601190
    O	-0.491790	2.834190	3.683200
    H	-1.044080	3.533360	4.113020
    H	0.018970	2.407960	4.421980
    O	4.113280	-2.600450	-1.021000
    H	3.560980	-1.901290	-0.591190
    H	4.624040	-3.026680	-0.282220
    O	-0.491790	2.834190	-4.062150
    H	-1.044080	3.533360	-3.632340
    H	0.018970	2.407960	-3.323370
    O	2.091540	-4.404940	1.228210
    H	2.643840	-5.104100	0.798400
    H	1.580780	-3.978700	0.489430
    O	-4.060940	-2.240220	-2.019890
    H	-4.289080	-3.084640	-2.485640
    H	-4.197700	-1.559370	-2.733790
    O	1.497850	4.546740	-1.635640
    H	1.188460	3.605690	-1.662580
    H	2.476470	4.445050	-1.481120
    O	-3.005310	3.863900	1.870260
    H	-2.381940	4.583870	2.144250
    H	-2.587370	3.528070	1.030940
    O	-1.559810	-3.599870	-3.619910
    H	-1.977750	-4.187970	-2.943110
    H	-0.665680	-3.995600	-3.799580
    O	1.599760	-1.570740	4.911410
    H	2.223120	-0.850770	5.185400
    H	2.017700	-1.906570	4.072100
    O	4.609810	1.029700	2.891260
    H	5.162110	0.330540	2.461450
    H	4.099050	1.455940	2.152480
    O	-1.559810	-3.599870	4.125450
    H	-1.977750	-4.187970	4.802250
    H	-0.665680	-3.995600	3.945780
    O	-5.523580	-1.570740	0.207210
    H	-4.900220	-0.850770	0.481200
    H	-5.105640	-1.906570	-0.632110
    O	-4.503160	-0.682850	3.505900
    H	-4.193770	0.258200	3.532850
    H	-5.481780	-0.581160	3.351380
    O	-1.462640	0.669480	-5.518260
    H	-1.234510	1.513890	-5.052500
    H	-1.325880	-0.011370	-4.804360
    O	5.660700	0.669480	-0.814060
    H	5.888830	1.513890	-0.348300
    H	5.797460	-0.011370	-0.100150
    O	-3.963780	2.029120	3.827120
    H	-3.545840	2.617220	3.150320
    H	-4.857910	2.424850	4.006800
    O	-3.963780	2.029120	-3.918240
    H	-3.545840	2.617220	-4.595040
    H	-4.857910	2.424850	-3.738560
    O	0.822640	-4.865110	3.618320
    H	1.567590	-4.834740	4.268500
    H	1.244790	-4.751070	2.725500
    O	0.958470	1.834770	5.788500
    H	0.540530	1.246680	6.465300
    H	1.852600	1.439040	5.608830
    O	3.142420	-4.765170	-2.477110
    H	3.370560	-3.920750	-2.011350
    H	3.279180	-5.446020	-1.763200
    O	-4.503160	-0.682850	-4.239460
    H	-4.193770	0.258200	-4.212510
    H	-5.481780	-0.581160	-4.393980
    O	0.101910	-6.117490	-1.198310
    H	0.411300	-5.176440	-1.171360
    H	-0.876710	-6.015800	-1.352830
    O	3.340910	0.569530	5.281380
    H	4.085860	0.599900	5.931550
    H	3.763060	0.683570	4.388550
    O	6.102910	-0.887900	1.405520
    H	5.793520	-1.828950	1.378570
    H	7.081530	-0.989590	1.560040
    O	3.062400	-2.240220	-5.061040
    H	2.834260	-3.084640	-5.526800
    H	2.925640	-1.559370	-5.774940
    O	0.822640	-4.865110	-4.127030
    H	1.567590	-4.834740	-3.476850
    H	1.244790	-4.751070	-5.019860
    O	-6.164870	1.834770	1.084290
    H	-6.582810	1.246680	1.761100
    H	-5.270740	1.439040	0.904620
    O	-2.513520	1.029700	5.932420
    H	-1.961230	0.330540	5.502600
    H	-3.024280	1.455940	5.193640
    O	5.563530	-3.599870	1.084290
    H	5.145590	-4.187970	1.761100
    H	6.457660	-3.995600	0.904620
    O	5.382180	-2.140280	-3.411110
    H	4.637230	-2.170650	-4.061290
    H	4.960030	-2.254310	-2.518290
    O	4.609810	1.029700	-4.854090
    H	5.162110	0.330540	-5.283910
    H	4.099050	1.455940	-5.592880
    O	-3.010060	-2.600450	-5.725210
    H	-3.562360	-1.901290	-5.295390
    H	-2.499300	-3.026680	-4.986430
    O	5.382180	-2.140280	4.334250
    H	4.637230	-2.170650	3.684070
    H	4.960030	-2.254310	5.227070
    O	0.777110	3.294370	-6.452270
    H	0.032170	3.263990	-7.102440
    H	0.354970	3.180330	-5.559440
    O	-4.060940	-2.240220	5.725470
    H	-4.289080	-3.084640	5.259720
    H	-4.197700	-1.559370	5.011570
    O	-5.625490	4.546740	1.405520
    H	-5.934880	3.605690	1.378570
    H	-4.646870	4.445050	1.560040
    O	-3.005310	3.863900	-5.875100
    H	-2.381940	4.583870	-5.601110
    H	-2.587370	3.528070	-6.714420
    O	0.641290	-3.405520	6.868280
    H	1.059230	-2.817420	6.191470
    H	-0.252840	-3.009790	7.047950
    O	7.123340	0.000000	-3.041150
    H	6.499970	-0.719980	-3.315140
    H	6.705390	0.335830	-2.201830
    O	0.000000	0.000000	-7.745360
    H	-0.623370	-0.719980	-8.019340
    H	-0.417950	0.335830	-6.906040
    O	0.000000	0.000000	7.745360
    H	-0.623370	-0.719980	7.471370
    H	-0.417950	0.335830	8.584680
    O	4.605060	-5.434650	3.041160
    H	3.981700	-6.154620	2.767170
    H	4.187120	-5.098820	3.880480
    O	3.142420	-4.765170	5.268250
    H	3.370560	-3.920750	5.734010
    H	3.279180	-5.446020	5.982150
    O	2.620180	-0.682850	-7.280610
    H	2.929570	0.258200	-7.253670
    H	1.641560	-0.581160	-7.435130
    O	-1.741160	-2.140280	7.375400
    H	-2.486110	-2.170650	6.725220
    H	-2.163310	-2.254310	8.268220
    O	3.159560	2.029120	-6.959390
    H	3.577500	2.617220	-7.636190
    H	2.265430	2.424850	-6.779720
    O	-6.346230	3.294370	-3.411110
    H	-7.091170	3.263990	-4.061290
    H	-6.768370	3.180330	-2.518290
    O	-3.782420	0.569530	-7.168190
    H	-3.037480	0.599900	-6.518010
    H	-3.360280	0.683570	-8.061010
    O	2.091540	-4.404940	-6.517150
    H	2.643840	-5.104100	-6.946960
    H	1.580780	-3.978700	-7.255930
    O	-7.615130	2.834190	-1.021000
    H	-8.167420	3.533360	-0.591190
    H	-7.104370	2.407960	-0.282220
    O	-1.542670	3.194420	7.388520
    H	-1.770800	2.350000	6.922770
    H	-1.679430	3.875270	6.674620
    O	4.113280	-2.600450	6.724360
    H	3.560980	-1.901290	7.154170
    H	4.624040	-3.026680	7.463140
    O	-6.346230	3.294370	4.334250
    H	-7.091170	3.263990	3.684070
    H	-6.768370	3.180330	5.227070
    O	4.605060	-5.434650	-4.704200
    H	3.981700	-6.154620	-4.978190
    H	4.187120	-5.098820	-3.864880
    O	7.123340	0.000000	4.704210
    H	6.499970	-0.719980	4.430220
    H	6.705390	0.335830	5.543530
    O	-1.741160	-2.140280	-8.115320
    H	-2.486110	-2.170650	-8.765490
    H	-2.163310	-2.254310	-7.222490
    O	2.620180	-0.682850	8.210100
    H	2.929570	0.258200	8.237050
    H	1.641560	-0.581160	8.055580
    O	6.102910	-0.887900	-6.339840
    H	5.793520	-1.828950	-6.366790
    H	7.081530	-0.989590	-6.185320
    O	-1.542670	3.194420	-8.102190
    H	-1.770800	2.350000	-8.567950
    H	-1.679430	3.875270	-8.816100
    O	0.101910	-6.117490	6.547050
    H	0.411300	-5.176440	6.574000
    H	-0.876710	-6.015800	6.392530
    O	-3.782420	0.569530	8.322530
    H	-3.037480	0.599900	8.972710
    H	-3.360280	0.683570	7.429710
    O	-6.164870	1.834770	-6.661070
    H	-6.582810	1.246680	-5.984260
    H	-5.270740	1.439040	-6.840740
    O	0.641290	-3.405520	-8.622440
    H	1.059230	-2.817420	-9.299240
    H	-0.252840	-3.009790	-8.442770
    O	7.764630	-3.405520	3.827120
    H	8.182570	-2.817420	3.150320
    H	6.870500	-3.009790	4.006800
    O	7.945980	-4.865110	0.577170
    H	8.690930	-4.834740	1.227350
    H	8.368130	-4.751070	-0.315650
    O	7.764630	-3.405520	-3.918240
    H	8.182570	-2.817420	-4.595040
    H	6.870500	-3.009790	-3.738560
    O	5.563530	-3.599870	-6.661070
    H	5.145590	-4.187970	-5.984260
    H	6.457660	-3.995600	-6.840740
    H	-4.900220	-0.850770	-7.264160
    H	-5.105640	-1.906570	-8.377470
    O	-5.523580	-1.570740	-7.538150
    O	-8.666010	3.194420	2.684320
    H	-8.894140	2.350000	2.218560
    H	-8.802770	3.875270	1.970420
    O	-5.625490	4.546740	-6.339840
    H	-5.934880	3.605690	-6.366790
    H	-4.646870	4.445050	-6.185320
    O	-5.523580	-1.570740	7.952570
    H	-4.900220	-0.850770	8.226560
    H	-5.105640	-1.906570	7.113250
    O	0.958470	1.834770	-9.702220
    H	0.540530	1.246680	-9.025420
    H	1.852600	1.439040	-9.881890
    O	-2.513520	1.029700	-9.558300
    H	-1.961230	0.330540	-9.988110
    H	-3.024280	1.455940	-10.297080
    O	4.113280	-2.600450	-8.766360
    H	3.560980	-1.901290	-8.336540
    H	4.624040	-3.026680	-8.027580
    O	7.225250	-6.117490	3.505900
    H	7.534640	-5.176440	3.532850
    H	6.246630	-6.015800	3.351380
    O	2.091540	-4.404940	8.973570
    H	2.643840	-5.104100	8.543760
    H	1.580780	-3.978700	8.234790
    O	5.660700	0.669480	-8.559420
    H	5.888830	1.513890	-8.093660
    H	5.797460	-0.011370	-7.845510
    O	7.225250	-6.117490	-4.239460
    H	7.534640	-5.176440	-4.212510
    H	6.246630	-6.015800	-4.393980
    O	9.214880	-4.404940	-1.812940
    H	9.767180	-5.104100	-2.242760
    H	8.704120	-3.978700	-2.551720
    O	-8.666010	3.194420	-5.061040
    H	-8.894140	2.350000	-5.526800
    H	-8.802770	3.875270	-5.774940
    O	1.497850	4.546740	-9.380990
    H	1.188460	3.605690	-9.407940
    H	2.476470	4.445050	-9.226470
    O	-3.010060	-2.600450	9.765510
    H	-3.562360	-1.901290	10.195330
    H	-2.499300	-3.026680	10.504290
    O	-7.615130	2.834190	6.724360
    H	-8.167420	3.533360	7.154170
    H	-7.104370	2.407960	7.463140
    O	3.340910	0.569530	-10.209340
    H	4.085860	0.599900	-9.559160
    H	3.763060	0.683570	-11.102170
    O	1.599760	-1.570740	-10.579300
    H	2.223120	-0.850770	-10.305310
    H	2.017700	-1.906570	-11.418620
    O	-6.164870	1.834770	8.829650
    H	-6.582810	1.246680	9.506460
    H	-5.270740	1.439040	8.649980
    O	5.563530	-3.599870	8.829650
    H	5.145590	-4.187970	9.506460
    H	6.457660	-3.995600	8.649980
    O	3.062400	-2.240220	10.429680
    H	2.834260	-3.084640	9.963920
    H	2.925640	-1.559370	9.715780
    O	-1.020430	-0.887900	-11.044050
    H	-1.329820	-1.828950	-11.070990
    H	-0.041810	-0.989590	-10.889530
    O	-10.128650	3.863900	-2.833940
    H	-9.505280	4.583870	-2.559960
    H	-9.710700	3.528070	-3.673260
    O	10.265760	-4.765170	2.227100
    H	10.493900	-3.920750	2.692860
    H	10.402520	-5.446020	2.941000
    O	3.142420	-4.765170	-10.222470
    H	3.370560	-3.920750	-9.756710
    H	3.279180	-5.446020	-9.508560
    O	7.945980	-4.865110	-7.168190
    H	8.690930	-4.834740	-6.518010
    H	8.368130	-4.751070	-8.061010
    O	9.214880	-4.404940	5.932420
    H	9.767180	-5.104100	5.502600
    H	8.704120	-3.978700	5.193640
    O	-10.128650	3.863900	4.911410
    H	-9.505280	4.583870	5.185400
    H	-9.710700	3.528070	4.072100
    O	-7.615130	2.834190	-8.766360
    H	-8.167420	3.533360	-8.336540
    H	-7.104370	2.407960	-8.027580
    O	-4.503160	-0.682850	11.251260
    H	-4.193770	0.258200	11.278200
    H	-5.481780	-0.581160	11.096740
    O	-0.491790	2.834190	-11.807510
    H	-1.044080	3.533360	-11.377700
    H	0.018970	2.407960	-11.068730
    O	0.822640	-4.865110	11.363680
    H	1.567590	-4.834740	12.013860
    H	1.244790	-4.751070	10.470860
    O	-3.963780	2.029120	-11.663590
    H	-3.545840	2.617220	-12.340400
    H	-4.857910	2.424850	-11.483920
    O	7.945980	-4.865110	8.322530
    H	8.690930	-4.834740	8.972710
    H	8.368130	-4.751070	7.429710
    O	-1.559810	-3.599870	11.870810
    H	-1.977750	-4.187970	12.547610
    H	-0.665680	-3.995600	11.691140
    O	5.382180	-2.140280	-11.156470
    H	4.637230	-2.170650	-11.806650
    H	4.960030	-2.254310	-10.263650
    O	10.265760	-4.765170	-5.518260
    H	10.493900	-3.920750	-5.052500
    H	10.402520	-5.446020	-4.804360
    O	7.123340	0.000000	-10.786510
    H	6.499970	-0.719980	-11.060500
    H	6.705390	0.335830	-9.947190
    O	11.728400	-5.434650	0.000000
    H	11.105040	-6.154620	-0.273990
    H	11.310460	-5.098820	0.839320
    O	4.605060	-5.434650	10.786520
    H	3.981700	-6.154620	10.512530
    H	4.187120	-5.098820	11.625840
    O	-6.346230	3.294370	-11.156470
    H	-7.091170	3.263990	-11.806650
    H	-6.768370	3.180330	-10.263650
    O	-1.462640	0.669480	-13.263620
    H	-1.234510	1.513890	-12.797860
    H	-1.325880	-0.011370	-12.549720
    O	4.609810	1.029700	-12.599450
    H	5.162110	0.330540	-13.029270
    H	4.099050	1.455940	-13.338230
    O	-8.666010	3.194420	10.429680
    H	-8.894140	2.350000	9.963920
    H	-8.802770	3.875270	9.715780
    O	9.214880	-4.404940	-9.558300
    H	9.767180	-5.104100	-9.988110
    H	8.704120	-3.978700	-10.297080
    O	-4.060940	-2.240220	13.470830
    H	-4.289080	-3.084640	13.005080
    H	-4.197700	-1.559370	12.756930
    O	7.764630	-3.405520	-11.663590
    H	8.182570	-2.817420	-12.340400
    H	6.870500	-3.009790	-11.483920
    O	-3.005310	3.863900	-13.620460
    H	-2.381940	4.583870	-13.346470
    H	-2.587370	3.528070	-14.459780
    O	0.777110	3.294370	-14.197620
    H	0.032170	3.263990	-14.847800
    H	0.354970	3.180330	-13.304800
    O	7.225250	-6.117490	11.251260
    H	7.534640	-5.176440	11.278200
    H	6.246630	-6.015800	11.096740
    O	11.728400	-5.434650	-7.745360
    H	11.105040	-6.154620	-8.019340
    H	11.310460	-5.098820	-6.906040
    O	11.728400	-5.434650	7.745360
    H	11.105040	-6.154620	7.471370
    H	11.310460	-5.098820	8.584680
    O	-10.128650	3.863900	-10.579300
    H	-9.505280	4.583870	-10.305310
    H	-9.710700	3.528070	-11.418620
    O	3.159560	2.029120	-14.704750
    H	3.577500	2.617220	-15.381550
    H	2.265430	2.424850	-14.525080
    O	6.102910	-0.887900	-14.085200
    H	5.793520	-1.828950	-14.112140
    H	7.081530	-0.989590	-13.930680
    O	0.101910	-6.117490	14.292410
    H	0.411300	-5.176440	14.319360
    H	-0.876710	-6.015800	14.137890
    O	-5.625490	4.546740	-14.085200
    H	-5.934880	3.605690	-14.112140
    H	-4.646870	4.445050	-13.930680
    O	5.660700	0.669480	-16.304770
    H	5.888830	1.513890	-15.839020
    H	5.797460	-0.011370	-15.590870
    O	10.265760	-4.765170	-13.263620
    H	10.493900	-3.920750	-12.797860
    H	10.402520	-5.446020	-12.549720
    O	1.497850	4.546740	-17.126350
    H	1.188460	3.605690	-17.153300
    H	2.476470	4.445050	-16.971830'''
    mol.atom = '\n'.join(mol.atom.split('\n')[1:nwater * 3 + 1])
    mol.basis = basis
    return mol


device = eval(os.getenv('DEFAULT_DEVICE', '0'))
for basis in ['ccpvdz', 'ccpvtz', 'ccpvqz']:
    for nwater in (1, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28,
                   32, 40, 48, 56, 64, 80, 96, 112, 128, 144):
        print("Nwater:", nwater, "basis:", basis)
        mol = get_mol(nwater, basis)
        mol.build()
        if device == 0:
            from byteqc.cuobc import scf
        else:
            from pyscf import scf
        rhf = scf.RHF(mol)
        rhf.chkfile = 'HF_chkfile_cpu.chk'
        rhf.max_cycle = 0
        rhf.scf()
        mol.verbose = 7
        rhf.verbose = 7
        if device == 0:
            start = time.time()
            e_corr, e_tot, rdm1 = cump2.DFMP2(mol, rhf, with_rdm1=True)
            t = time.time() - start
            print("GPU energy:", e_corr)
        else:
            mp = DFMP2(rhf)
            mp.max_memory = 240000
            _integral = mp.calculate_integrals_
            start = 0

            def integral(*args):
                global start
                start0 = time.time()
                _integral(*args)
                print("CPU(C) time for cderi_ovL_outcore  ",
                      time.time() - start0)
                start = time.time()
            mp.calculate_integrals_ = integral
            mp.run()
            rdm1_ref = mp.make_rdm1()
            t = time.time() - start
            print("CPU(C) time for mp2_get_corr  ", t)
            print("CPU energy:", mp.e_corr)
            print('GPU/CPU gamma point 1-RDM difference:', abs(rdm1 - rdm1_ref).sum())
        print("\n")
        if t > 10000:
            break
