import numpy as np
from pandas import DataFrame

from nandc_forward import calc_ray_paras


class DataIN():
    '''
    class DataIn contains information about events, stations and ray parameters
    :param data: stations locations and observations [stid, stlo, stla, tau1, pol, tau2, amp, tau3, spr]
    :type data: pd.DataFrame
    :param evinfo: event information, contains evid, evlo, evla, evdp and local magnitude (if has)
    :type evinfo: list
    :param vmodel_TauPy: TauPy velocity model
    :type vmodel_TauPy: TauPyModel
    :param vmodel_DataFrame: velocity model as a DataFrame
    :type vmodel_DataFrame: DataFrame
    '''

    @staticmethod
    def _project_beachball_angles(azimuths, takeoff_angles):
        azimuths = np.asarray(azimuths, dtype=float)
        takeoff_angles = np.asarray(takeoff_angles, dtype=float)

        valid = np.isfinite(azimuths) & np.isfinite(takeoff_angles)
        azimuths = azimuths[valid]
        takeoff_angles = takeoff_angles[valid]
        if azimuths.size == 0:
            return azimuths, takeoff_angles

        upward = takeoff_angles > 90.0
        takeoff_angles = np.where(upward, 180.0 - takeoff_angles, takeoff_angles)
        azimuths = np.mod(azimuths + np.where(upward, 180.0, 0.0), 360.0)

        valid = (takeoff_angles >= 0.0) & (takeoff_angles <= 90.0)
        return azimuths[valid], takeoff_angles[valid]

    @staticmethod
    def _max_azimuth_gap(azimuths):
        azimuths = np.asarray(azimuths, dtype=float)
        azimuths = azimuths[np.isfinite(azimuths)]
        if azimuths.size == 0:
            return np.nan

        azimuths = np.sort(np.mod(azimuths, 360.0))
        if azimuths.size == 1:
            return 360.0

        gaps = np.diff(azimuths)
        wrap_gap = azimuths[0] + 360.0 - azimuths[-1]
        return float(np.max(np.append(gaps, wrap_gap)))

    @staticmethod
    def _max_takeoff_gap(takeoff_angles):
        takeoff_angles = np.asarray(takeoff_angles, dtype=float)
        takeoff_angles = takeoff_angles[np.isfinite(takeoff_angles)]
        takeoff_angles = takeoff_angles[(takeoff_angles >= 0.0) & (takeoff_angles <= 90.0)]
        if takeoff_angles.size == 0:
            return np.nan

        takeoff_angles = np.sort(takeoff_angles)
        if takeoff_angles.size == 1:
            return float(max(takeoff_angles[0], 90.0 - takeoff_angles[0]))

        gaps = np.diff(takeoff_angles)
        gaps = np.concatenate((
            [takeoff_angles[0]],
            gaps,
            [90.0 - takeoff_angles[-1]]
        ))
        return float(np.max(gaps))

    @classmethod
    def _calc_projected_gaps(cls, azimuths, takeoff_angles):
        azimuths, takeoff_angles = cls._project_beachball_angles(azimuths, takeoff_angles)
        return cls._max_azimuth_gap(azimuths), cls._max_takeoff_gap(takeoff_angles)

    def __init__(self, data, evinfo, vmodel_TauPy, vmodel_DataFrame, alpha, beta, rho):

        if type(data) is not DataFrame:
            raise TypeError("Type of input data should be DataFrame.")

        # events info
        if len(evinfo) == 5:
            self.evid, self.evlo, self.evla, self.evdp, self.evmag = evinfo[0], evinfo[1], evinfo[2], evinfo[3], evinfo[4]
        else:
            self.evid, self.evlo, self.evla, self.evdp = evinfo[0], evinfo[1], evinfo[2], evinfo[3]

        # calculate ray parameters for further inversion
        ray = calc_ray_paras(
            self.evlo, self.evla, self.evdp, data.iloc[:, 0:4],
            vmodel_TauPy, vmodel_DataFrame, alpha, beta, rho
        )  # DataFrame
        ray_array = ray.to_numpy(dtype='float')  # should not only have one station numpy array N*7

        # remove stations whose ray parameters cannot be calculated
        zero_tko = np.where(ray_array[:, 1] == -12345)[0]
        data = data.drop(index=zero_tko).reset_index(drop=True)
        self.ray = ray.drop(index=zero_tko).reset_index(drop=True)
        self.ray_array = np.delete(ray_array, zero_tko, axis=0)

        # stations info and parameters
        self.stations = data.iloc[:, 0:4]  # DataFrame

        # weight and observations
        self.tau1_raw = data['tau1'].to_numpy(dtype=float)
        self.tau2_raw = data['tau2'].to_numpy(dtype=float)
        self.tau3_raw = data['tau3'].to_numpy(dtype=float)

        self.pol_raw = data['pol'].to_numpy(dtype=float)
        self.amp_raw = data['amp'].to_numpy(dtype=float)
        self.spr_raw = data['spr'].to_numpy(dtype=float)

        # remove None (marked as zero) observations
        self.zero_pol_index = np.where(self.pol_raw == 0)[0]
        self.zero_amp_index = np.where(self.amp_raw == 0)[0]
        self.zero_spr_index = np.where(self.spr_raw == 0)[0]

        self.tau1 = np.delete(self.tau1_raw, self.zero_pol_index)
        self.tau2 = np.delete(self.tau2_raw, self.zero_amp_index)
        self.tau3 = np.delete(self.tau3_raw, self.zero_spr_index)

        self.pol = np.delete(self.pol_raw, self.zero_pol_index)
        self.amp = np.delete(self.amp_raw, self.zero_amp_index)
        self.spr = np.delete(self.spr_raw, self.zero_spr_index)

        ray_azi = self.ray['azi'].to_numpy(dtype=float)
        ray_tko = self.ray['tko'].to_numpy(dtype=float)

        self.pol_max_azi_gap, self.pol_max_takeoff_gap = self._calc_projected_gaps(
            np.delete(ray_azi, self.zero_pol_index),
            np.delete(ray_tko, self.zero_pol_index)
        )
        self.amp_max_azi_gap, self.amp_max_takeoff_gap = self._calc_projected_gaps(
            np.delete(ray_azi, self.zero_amp_index),
            np.delete(ray_tko, self.zero_amp_index)
        )
        self.spr_max_azi_gap, self.spr_max_takeoff_gap = self._calc_projected_gaps(
            np.delete(ray_azi, self.zero_spr_index),
            np.delete(ray_tko, self.zero_spr_index)
        )

        self.gap_stats = {
            'pol': {'max_azi_gap': self.pol_max_azi_gap, 'max_takeoff_gap': self.pol_max_takeoff_gap},
            'amp': {'max_azi_gap': self.amp_max_azi_gap, 'max_takeoff_gap': self.amp_max_takeoff_gap},
            'spr': {'max_azi_gap': self.spr_max_azi_gap, 'max_takeoff_gap': self.spr_max_takeoff_gap}
        }
