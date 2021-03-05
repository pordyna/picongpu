from os import path
import numpy as np
import scipy.constants as cs
import openpmd_api as api

from is_close import is_close


class ThermalizationVerifier:
    """Verifies thermalization test output and generates reference output"""
    ELECTRON_MASS = cs.electron_mass
    ION_MASS = 10 * ELECTRON_MASS
    REFERENCE_FILE_NAME = "reference_output.npz"

    def __init__(self, sim_output_path):
        self.series = api.Series(
            path.join(sim_output_path, 'simOutput/openPMD/simData_%T.h5'),
            api.Access_Type.read_only)
        self.unit_mass = self.series.iterations[0].get_attribute('unit_mass')
        self.e_T_mean = np.zeros(len(self.series.iterations), dtype=np.float32)
        self.i_T_mean = np.zeros(len(self.series.iterations), dtype=np.float32)
        self.e_reference = None
        self.i_reference = None

    def calculate_temperatures(self):
        """Calculates mean temperatures for electrons and ions """
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
            e_v = {'x': None, 'y': None, 'z': None}
            i_v = {'x': None, 'y': None, 'z': None}

            for cor in ['x', 'y', 'z']:
                e_v[cor] = e_p[cor] / (
                        (self.ELECTRON_MASS / self.unit_mass) * e_w)
                i_v[cor] = i_p[cor] / ((self.ION_MASS / self.unit_mass) * i_w)
            n_i = np.sum(i_w)

            i_vx0 = np.sum(i_v['x'] * i_w) / n_i
            self.i_T_mean[i] = np.sum((i_w * (i_v['x'] - i_vx0) ** 2)) / n_i

            i_vy0 = np.sum(i_v['y'] * i_w) / n_i
            self.i_T_mean[i] += np.sum((i_w * (i_v['y'] - i_vy0) ** 2)) / n_i

            i_vz0 = np.sum(i_v['z'] * i_w) / n_i
            self.i_T_mean[i] += np.sum((i_w * (i_v['z'] - i_vz0) ** 2)) / n_i

            n_e = np.sum(e_w)

            e_vx0 = np.sum(e_v['x'] * e_w) / n_e
            self.e_T_mean[i] = np.sum((e_w * (e_v['x'] - e_vx0) ** 2)) / n_e

            e_vy0 = np.sum(e_v['y'] * e_w) / n_e
            self.e_T_mean[i] += np.sum((e_w * (e_v['y'] - e_vy0) ** 2)) / n_e

            e_vz0 = np.sum(e_v['z'] * e_w) / n_e
            self.e_T_mean[i] += np.sum((e_w * (e_v['z'] - e_vz0) ** 2)) / n_e

    def save_reference(self):
        """Save generated data as reference output"""
        # Avoid overwriting data
        if path.exists(self.REFERENCE_FILE_NAME):
            raise FileExistsError(
                "File " + self.REFERENCE_FILE_NAME + " already exists.")
        with open(self.REFERENCE_FILE_NAME, 'wb') as file:
            np.savez_compressed(file, e_T_mean=self.e_T_mean,
                                i_T_mean=self.i_T_mean)

    def load_reference(self):
        """Load reference output"""
        reference_data = np.load(self.REFERENCE_FILE_NAME)
        self.e_reference = reference_data["e_T_mean"]
        self.i_reference = reference_data["i_T_mean"]

    def compare(self, abs_tolerance, threshold, rel_tolerance):
        """Compare reference and calculated temperatures"""
        test_result_e = is_close(self.e_reference, self.e_T_mean,
                                 abs_tolerance, threshold, rel_tolerance)
        test_result_i = is_close(self.i_reference, self.i_T_mean,
                                 abs_tolerance, threshold, rel_tolerance)
        return test_result_e, test_result_i
