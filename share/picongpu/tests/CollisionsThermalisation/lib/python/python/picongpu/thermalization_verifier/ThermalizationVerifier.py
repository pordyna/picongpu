from os import path
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as cs
import openpmd_api as api
import math
import textwrap

from is_close import is_close


class ThermalizationVerifier:
    """Verifies thermalization test output and generates reference output"""
    ELECTRON_MASS = cs.electron_mass
    ION_MASS = 10 * ELECTRON_MASS
    REFERENCE_FILE_NAME = "reference_output.npz"
    ION_DENSITY = 1.1e28
    ELECTRON_DENSITY = ION_DENSITY
    ION_CHARGE = 1
    INIT_TEMP_IONS = 1.8e-4 * cs.electron_mass * cs.speed_of_light**2  * 6.241509e18 / 1e3 # keV
    INIT_TEMP_ELECTRONS = 2e-4 * cs.electron_mass * cs.speed_of_light**2 * 6.241509e18 / 1e3 # keV

    def __init__(self, sim_output_path):
        self.series = api.Series(path.join(sim_output_path, 'simOutput/openPMD/simData_%T.h5'),
                                 api.Access_Type.read_only)
        self.unit_mass = self.series.iterations[0].get_attribute('unit_mass')
        self.e_T_mean = np.zeros(len(self.series.iterations), dtype=np.float64)
        self.i_T_mean = np.zeros(len(self.series.iterations), dtype=np.float64)
        self.dt = self.series.iterations[0].dt * self.series.iterations[0].time_unit_SI
        self.e_T_theory = None
        self.i_T_theory = None
        self.e_reference = None
        self.i_reference = None
        self.coulomb_log = None
        self.sim_output_path=sim_output_path

    def calculate_temperatures(self):
        """Calculates mean temperatures for electrons and ions for all time steps"""
        iterations = self.series.iterations
        for i in iterations:
            electrons = iterations[i].particles['e']
            ions = iterations[i].particles['i']
            e_p_m = {'x': None, 'y': None, 'z': None}
            i_p_m = {'x': None, 'y': None, 'z': None}
            e_p = {'x': None, 'y': None, 'z': None}
            i_p = {'x': None, 'y': None, 'z': None}

            for cor in ['x', 'y', 'z']:
                e_p_m[cor] = electrons['momentum'][cor]
                i_p_m[cor] = ions['momentum'][cor]
            e_w_m = electrons['weighting'][api.Mesh_Record_Component.SCALAR]
            i_w_m = ions['weighting'][api.Mesh_Record_Component.SCALAR]
            for cor in ['x', 'y', 'z']:
                e_p[cor] = e_p_m[cor][:]
                i_p[cor] = i_p_m[cor][:]
            e_w = e_w_m[:]
            i_w = i_w_m[:]

            self.series.flush()
            for cor in ['x', 'y', 'z']:
                assert np.all(np.isfinite(e_p[cor]))
                assert np.all(np.isfinite(i_p[cor]))
            e_v = {'x': None, 'y': None, 'z': None}
            i_v = {'x': None, 'y': None, 'z': None}

            for cor in ['x', 'y', 'z']:
                e_v[cor] = e_p[cor] / ((self.ELECTRON_MASS / self.unit_mass) * e_w)
                i_v[cor] = i_p[cor] / ((self.ION_MASS / self.unit_mass) * i_w)
            N_i = np.sum(i_w)

            i_vx0 = np.sum(i_v['x'] * i_w) / N_i
            self.i_T_mean[i] = np.sum((i_w * (i_v['x'] - i_vx0) ** 2)) / N_i

            i_vy0 = np.sum(i_v['y'] * i_w) / N_i
            self.i_T_mean[i] += np.sum((i_w * (i_v['y'] - i_vy0) ** 2)) / N_i

            i_vz0 = np.sum(i_v['z'] * i_w) / N_i
            self.i_T_mean[i] += np.sum((i_w * (i_v['z'] - i_vz0) ** 2)) / N_i

            N_e = np.sum(e_w)

            e_vx0 = np.sum(e_v['x'] * e_w) / N_e
            self.e_T_mean[i] = np.sum((e_w * (e_v['x'] - e_vx0) ** 2)) / N_e

            e_vy0 = np.sum(e_v['y'] * e_w) / N_e
            self.e_T_mean[i] += np.sum((e_w * (e_v['y'] - e_vy0) ** 2)) / N_e

            e_vz0 = np.sum(e_v['z'] * e_w) / N_e
            self.e_T_mean[i] += np.sum((e_w * (e_v['z'] - e_vz0) ** 2)) / N_e

        self.e_T_mean *= (2/3) * 0.5 * (self.ELECTRON_MASS / self.unit_mass) * self.series.iterations[0].get_attribute('unit_energy') * 6.241509e18 / 1e3
        self.i_T_mean *= (2/3) * 0.5 * (self.ION_MASS /self.unit_mass) *  self.series.iterations[0].get_attribute('unit_energy') * 6.241509e18 / 1e3

    def _calc_coulomb_log(self, temp_e, temp_i):
        # lambda_d_inv_sq = (1 / cs.epsilon_0**2 ) * (self.ELECTRON_DENSITY  * cs.elementary_charge**2 / temp_e + self.ION_DENSITY  * self.ION_CHARGE**2 * cs.elementary_charge**2 / temp_i)
        # b_0 = cs.elementary_charge**2 * self.ION_CHARGE / (cs.epsilon_0 * 4 *np.pi) / ((temp_e + temp_i)/2)
        # de_broglie_th_e = cs.h / np.sqrt(2 * np.pi * self.ELECTRON_MASS * temp_e)
        # de_broglie_th_i = cs.h / np.sqrt(2 * np.pi * self.ION_MASS * temp_i)
        # b_min = np.max([b_0, de_broglie_th_e /2 , de_broglie_th_i /2 ])
        # coulomb_log = (1/2) * np.log(1 + 1/(lambda_d_inv_sq * b_min**2))
        # #return max(2, coulomb_log)
        # return coulomb_log
        n_e_cgs = self.ELECTRON_DENSITY / 100**3
        temp_e_ev = temp_e * 1000
        return 24 - np.log(np.sqrt(n_e_cgs) / temp_e_ev)



    def calculate_theretical_values(self, coulomb_log=None):
        self.e_T_theory = np.empty_like(self.e_T_mean)
        self.i_T_theory = np.empty_like(self.i_T_mean)
        self.coulomb_log = np.empty(self.e_T_mean.size - 1)

        self.e_T_theory[0] = self.INIT_TEMP_ELECTRONS
        self.i_T_theory[0] = self.INIT_TEMP_IONS
        calc_log = False
        if coulomb_log is None:
            calc_log = True
        for ii in range(self.e_T_theory.size - 1):
            temp_e = self.e_T_theory[ii]
            temp_i = self.i_T_theory[ii]
            temp_e_joul = temp_e / (6.241509e18 / 1e3)
            temp_i_joul = temp_i / (6.241509e18 / 1e3)
            if calc_log:
                coulomb_log = self._calc_coulomb_log(temp_e, temp_i)
            self.coulomb_log[ii] = coulomb_log
            rate = ((2 / 3) * np.sqrt(2 / np.pi) * cs.elementary_charge**4
                    * self.ION_CHARGE**2 * np.sqrt(self.ION_MASS
                                                   * self.ELECTRON_MASS) * self.ION_DENSITY * coulomb_log
                    / (4 * np.pi * cs.epsilon_0**2 *
                       (self.ELECTRON_MASS*temp_e_joul + self.ION_MASS * temp_i_joul)**(3/2)))
            delta_temp = rate * (temp_e - temp_i) * self.dt
            #print(ii, rate, delta_temp, temp_e, temp_i, self.dt)
            self.e_T_theory[ii+1] = temp_e - delta_temp
            self.i_T_theory[ii+1] = temp_i + delta_temp


    def save_reference(self):
        """Save generated data as reference output"""
        # Avoid overwriting data
        if path.exists(self.REFERENCE_FILE_NAME):
            raise FileExistsError("File " + self.REFERENCE_FILE_NAME + " already exists.")
        reference = np.array([self.e_T_mean, self.i_T_mean])
        with open(self.REFERENCE_FILE_NAME, 'wb') as file:
            np.savez_compressed(file, e_T_mean=self.e_T_mean, i_T_mean=self.i_T_mean)

    def load_reference(self):
        """Load reference output"""
        with open(self.REFERENCE_FILE_NAME, 'rb') as file:
            reference_data = np.load(file)
            self.e_reference = reference_data["e_T_mean"]
            self.i_reference = reference_data["i_T_mean"]

    def compare(self, abs_tolerance, threshold, rel_tolerance):
        """Compare reference and calculated temperatures for electrons and ions"""
        test_result_e = is_close(self.e_reference, self.e_T_mean, abs_tolerance, threshold, rel_tolerance)
        test_result_i = is_close(self.i_reference, self.i_T_mean, abs_tolerance, threshold, rel_tolerance)
        return test_result_e, test_result_i

    def plot(self, to_file=False, file_name=None):
        f, (axTemp, axLog) = plt.subplots(2, sharex=True,gridspec_kw={'height_ratios': [3, 1]}, figsize=(10,8))
        times = np.array(self.series.iterations) * self.series.iterations[0].dt * self.series.iterations[0].time_unit_SI

        axTemp.plot(times, self.e_T_mean, label=r'$T_e$ sim')
        axTemp.plot(times, self.i_T_mean, label=r'$T_i$ sim')
        axTemp.plot(times, self.e_T_theory, label=r'$T_e$ theory')
        axTemp.plot(times, self.i_T_theory, label=r'$T_i$ theory')
        axTemp.plot(times, (self.e_T_mean + self.i_T_mean)/2, label=r'$(T_e + T_i)/2$ sim')

        axTemp.set_ylabel('T [keV]')


        axLog.plot(times[:-1], self.coulomb_log, label='coulomb log theory')
        axLog.set_ylabel(r'$\Lambda$')
        axLog.set_xlabel('t [fs]')

        axTemp.legend()
        axLog.legend()
        with open(Path(self.sim_output_path)/'input/cmakeFlagsSetup', 'r') as file:
            axTemp.set_title(file.readline() + '\n' +  textwrap.fill(file.readline(), 80))
        plt.tight_layout()
        if to_file:
            if file_name is None:
                file_name = 'thermalization_plot.png'
            f.savefig(file_name)
