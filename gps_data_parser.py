# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 22:33:23 2020

@author: ryan4
"""

import numpy as np
import scipy.linalg as la

MU = 3.986005 * 10**14  # m^3/s^2
SPEED_OF_LIGHT = 2.99792458 * 10**8  # m/s
EARTH_ROT_RATE = 7.2921151467 * 10**-5  # rad/s
PI = 3.1415926535898
FLATTENING = 1 / 298.257223563
ECCENTRICITY = np.sqrt(FLATTENING * (2 - FLATTENING))
EQ_RAD = 6378137  # m
POL_RAD = EQ_RAD * (1 - FLATTENING)

def parse_icp(r_file, times):
    max_lines = 0
    with open(r_file, 'r') as fin:
        for line in fin:
            if line.strip():
                max_lines = max_lines + 1

    pranges = -np.ones(max_lines)
    trans_times = np.ones(max_lines)
    with open(r_file, 'r') as fin:
        ii = 0
        for line in fin:
            if line.strip():
                cols = line.split()
                pranges[ii] = float(cols[7])
                trans_times[ii] = float(cols[0])
                if times.size == 0:
                    times = np.array([trans_times[ii]])
                elif abs(times - trans_times[ii]).min() > 0.2:
                    times = np.append(times, trans_times[ii])
                ii = ii + 1

    times.sort()
    return (pranges, trans_times), times


def parse_rinex_v2(r_file):
    with open(r_file, 'r') as fin:
        header_finished = False
        nav_line = 0
        max_nav_lines = 8
        nav_msgs = {}
        for line in fin:
            if header_finished:
                if nav_line == 0:
                    prn = line[0:2].strip()
                    year = float(line[3:5])
                    month = float(line[6:8])
                    day = float(line[9:10])
                    hour = float(line[11:14])
                    minute = float(line[14:17])
                    second = float(line[17:22])
                    clock_bias = float(line[22:37]) * 10**float(line[38:41])
                    clock_drift = float(line[41:56]) * 10**float(line[57:60])
                    clock_drift_rate = float(line[60:75]) \
                        * 10**float(line[76:79])
                elif nav_line == 1:
                    iode = float(line[4:18]) * 10**float(line[19:22])
                    crs = float(line[22:37]) * 10**float(line[38:41])
                    delta_n = float(line[41:56]) * 10**float(line[57:60])
                    m0 = float(line[60:75]) * 10**float(line[76:79])
                elif nav_line == 2:
                    cuc = float(line[4:18]) * 10**float(line[19:22])
                    eccentricity = float(line[22:37]) * 10**float(line[38:41])
                    cus = float(line[41:56]) * 10**float(line[57:60])
                    sqrt_a = float(line[60:75]) * 10**float(line[76:79])
                elif nav_line == 3:
                    toe = float(line[4:18]) * 10**float(line[19:22])
                    cic = float(line[22:37]) * 10**float(line[38:41])
                    long_ascend_node = float(line[41:56]) \
                        * 10**float(line[57:60])
                    cis = float(line[60:75]) * 10**float(line[76:79])
                elif nav_line == 4:
                    i0 = float(line[4:18]) * 10**float(line[19:22])
                    crc = float(line[22:37]) * 10**float(line[38:41])
                    arg_peri = float(line[41:56]) * 10**float(line[57:60])
                    omega_dot = float(line[60:75]) * 10**float(line[76:79])
                elif nav_line == 5:
                    idot = float(line[4:18]) * 10**float(line[19:22])
                    codes = float(line[22:37]) * 10**float(line[38:41])
                    gps_week = float(line[41:56]) * 10**float(line[57:60])
                    p_dta_flag = float(line[60:75]) * 10**float(line[76:79])
                elif nav_line == 6:
                    sv_acc = float(line[4:18]) * 10**float(line[19:22])
                    sv_health = float(line[22:37]) * 10**float(line[38:41])
                    tgd = float(line[41:56]) * 10**float(line[57:60])
                    iodc = float(line[60:75]) * 10**float(line[76:79])
                elif nav_line == 7:
                    trans_time = float(line[4:18]) * 10**float(line[19:22])
                    fit_int = float(line[22:37]) * 10**float(line[38:41])
                nav_line += 1

                if nav_line >= max_nav_lines:
                    nav_line = 0
                    if prn in nav_msgs:
                        # print("Adding to PRN: {p:s}".format(p=prn))
                        if prn == "2":
                            tmp = 1
                        nav_msgs[prn].add_vals(sqrt_a=sqrt_a,
                                               eccentricity=eccentricity,
                                               i0=i0,
                                               long_ascend_node=long_ascend_node,
                                               arg_peri=arg_peri, m0=m0,
                                               idot=idot, omega_dot=omega_dot,
                                               delta_n=delta_n,
                                               cuc=cuc, cus=cus, crc=crc,
                                               crs=crs, cic=cic, cis=cis,
                                               toe=toe, time=trans_time,
                                               clock_bias=clock_bias,
                                               clock_drift=clock_drift,
                                               clock_drift_rate=clock_drift_rate)
                    else:
                        # print("Found PRN: {p:s}".format(p=prn))
                        nav_msg = \
                            rinex_nav_msg(prn=prn,
                                          sqrt_a=sqrt_a,
                                          eccentricity=eccentricity,
                                          i0=i0,
                                          long_ascend_node=long_ascend_node,
                                          arg_peri=arg_peri, m0=m0,
                                          idot=idot, omega_dot=omega_dot,
                                          delta_n=delta_n, cuc=cuc, cus=cus,
                                          crc=crc, crs=crs, cic=cic, cis=cis,
                                          toe=toe, time=trans_time,
                                          clock_bias=clock_bias,
                                          clock_drift=clock_drift,
                                          clock_drift_rate=clock_drift_rate)
                        nav_msgs[prn] = nav_msg

            elif "end of header" in line.lower():
                header_finished = True
                continue

    return nav_msgs


class rinex_nav_msg(object):
    def __init__(self, **kwargs):
        self.PRN = kwargs['prn']
        self.sqrt_a = np.array([kwargs['sqrt_a']])
        self.eccentricity = np.array([kwargs['eccentricity']])
        self.i0 = np.array([kwargs['i0']])
        self.long_ascend_node = np.array([kwargs['long_ascend_node']])
        self.arg_peri = np.array([kwargs['arg_peri']])
        self.m0 = np.array([kwargs['m0']])
        self.idot = np.array([kwargs['idot']])
        self.omega_dot = np.array([kwargs['omega_dot']])
        self.delta_n = np.array([kwargs['delta_n']])
        self.cos_cor_arg_lat = np.array([kwargs['cuc']])
        self.sin_cor_arg_lat = np.array([kwargs['cus']])
        self.cos_cor_orb_rad = np.array([kwargs['crc']])
        self.sin_cor_orb_rad = np.array([kwargs['crs']])
        self.cos_cor_inc_ang = np.array([kwargs['cic']])
        self.sin_cor_inc_ang = np.array([kwargs['cis']])
        self.toe = np.array([kwargs['toe']])
        self.tol = kwargs.get('tol', 1 * 10**-8)
        self.trans_time = np.array([kwargs['time']])
        self.clock_bias = np.array([kwargs['clock_bias']])
        self.clock_drift = np.array([kwargs['clock_drift']])
        self.clock_drift_rate = np.array([kwargs['clock_drift_rate']])

    def add_vals(self, **kwargs):
        self.sqrt_a = np.hstack((self.sqrt_a, np.array([kwargs['sqrt_a']])))
        self.eccentricity = np.hstack((self.eccentricity,
                                       np.array([kwargs['eccentricity']])))
        self.i0 = np.hstack((self.i0, np.array([kwargs['i0']])))
        self.long_ascend_node = \
            np.hstack((self.long_ascend_node,
                      np.array([kwargs['long_ascend_node']])))
        self.arg_peri = np.hstack((self.arg_peri,
                                   np.array([kwargs['arg_peri']])))
        self.m0 = np.hstack((self.m0, np.array([kwargs['m0']])))
        self.idot = np.hstack((self.idot, np.array([kwargs['idot']])))
        self.omega_dot = np.hstack((self.omega_dot,
                                   np.array([kwargs['omega_dot']])))
        self.delta_n = np.hstack((self.delta_n, np.array([kwargs['delta_n']])))
        self.cos_cor_arg_lat = np.hstack((self.cos_cor_arg_lat,
                                          np.array([kwargs['cuc']])))
        self.sin_cor_arg_lat = np.hstack((self.sin_cor_arg_lat,
                                          np.array([kwargs['cus']])))
        self.cos_cor_orb_rad = np.hstack((self.cos_cor_orb_rad,
                                          np.array([kwargs['crc']])))
        self.sin_cor_orb_rad = np.hstack((self.sin_cor_orb_rad,
                                          np.array([kwargs['crs']])))
        self.cos_cor_inc_ang = np.hstack((self.cos_cor_inc_ang,
                                          np.array([kwargs['cic']])))
        self.sin_cor_inc_ang = np.hstack((self.sin_cor_inc_ang,
                                          np.array([kwargs['cis']])))
        self.toe = np.hstack((self.toe, [kwargs['toe']]))
        self.trans_time = np.hstack([self.trans_time, [kwargs['time']]])
        self.clock_bias = np.hstack([self.clock_bias, [kwargs['clock_bias']]])
        self.clock_drift = np.hstack([self.clock_bias,
                                      [kwargs['clock_drift']]])
        self.clock_drift_rate = np.hstack([self.clock_bias,
                                           [kwargs['clock_drift_rate']]])
        self.ecc_anom = np.inf * np.ones(self.toe.size)

    def calculate_orbit(self, trans_time=None):
        if trans_time is None:
            trans_time = self.toe
        time_len = trans_time.size
        pos_ECEF = np.zeros((time_len, 3))
        for jj in range(0, time_len):
            # find index of last known values based on desired "transmit time"
            # note this assumes things are sorted in increasing order
            ii = np.argmin(np.abs(self.toe - trans_time[jj]))  # 0
#            for kk in range(0, self.toe.size):
#                last_ind = (kk == self.toe.size - 1)
#                if trans_time[jj] <= self.toe[kk] or last_ind:
#                    ii = kk
#                    break
            semimajor = self.sqrt_a[ii]**2
            mean_motion = np.sqrt(MU / semimajor**3) + self.delta_n[ii]
            tk = trans_time[jj] - self.toe[ii]
            if tk > 302400:
                tk = tk - 604800
            elif tk < -302400:
                tk = tk + 604800

            mean_anom = self.m0[ii] + mean_motion * tk
            ecc_anom = mean_anom
            ratio = 0
            first_pass = True
            while np.abs(ratio) > self.tol or first_pass:
                first_pass = False
                ecc_anom = ecc_anom - ratio
                err = ecc_anom - self.eccentricity[ii] * np.sin(ecc_anom) \
                    - mean_anom
                der = 1 - self.eccentricity[ii] * np.cos(ecc_anom)
                ratio = err / der
            if np.isinf(self.ecc_anom[ii]):
                self.ecc_anom[ii] = ecc_anom
            tmp = 1 - self.eccentricity[ii] * np.cos(ecc_anom)
            cos_true_anom = (np.cos(ecc_anom) - self.eccentricity[ii]) \
                / tmp
            sin_true_anom = np.sqrt(1 - self.eccentricity[ii]**2) \
                * np.sin(ecc_anom) / tmp
            true_anom = np.arctan2(sin_true_anom, cos_true_anom)
            arg_lat = true_anom + self.arg_peri[ii]
            cor_arg_lat = arg_lat + self.sin_cor_arg_lat[ii] \
                * np.sin(2 * arg_lat) + self.cos_cor_arg_lat[ii] \
                * np.cos(2 * arg_lat)
            rad = semimajor * (1 - self.eccentricity[ii]
                               * np.cos(ecc_anom)) \
                + self.sin_cor_orb_rad[ii] * np.sin(2 * arg_lat) \
                + self.cos_cor_orb_rad[ii] * np.cos(2 * arg_lat)
            cor_inc = self.i0[ii] + self.idot[ii] * tk \
                + self.sin_cor_inc_ang[ii] * np.sin(2 * arg_lat) \
                + self.cos_cor_inc_ang[ii] * np.cos(2 * arg_lat)
            cor_long_ascend = self.long_ascend_node[ii] \
                + (self.omega_dot[ii] - EARTH_ROT_RATE) * tk \
                - EARTH_ROT_RATE * self.toe[ii]
            xp = rad * np.cos(cor_arg_lat)
            yp = rad * np.sin(cor_arg_lat)
            c_om = np.cos(cor_long_ascend)
            s_om = np.sin(cor_long_ascend)
            ci = np.cos(cor_inc)
            si = np.sin(cor_inc)
            x = xp * c_om - yp * ci * s_om
            y = xp * s_om + yp * ci * c_om
            z = yp * si
            pos_ECEF[jj, :] = np.array([x, y, z])

        return pos_ECEF


def ecef_to_lla(xyz):
    lon = np.arctan2(xyz[1], xyz[0])
    p = np.sqrt(xyz[0]**2 + xyz[1]**2)
    E = np.sqrt(EQ_RAD**2 - POL_RAD**2)
    F = 54 * (POL_RAD * xyz[2])**2
    G = p**2 + (1 - ECCENTRICITY**2) \
        * (xyz[2]**2) - (ECCENTRICITY * E)**2
    c = ECCENTRICITY**4 * F * p**2 / G**3
    s = (1 + c + np.sqrt(c**2 + 2 * c))**(1 / 3)
    P = (F / (3 * G**2)) / (s + 1 / s + 1)**2
    Q = np.sqrt(1 + 2 * ECCENTRICITY**4 * P)
    k1 = -P * ECCENTRICITY**2 * p / (1 + Q)
    k2 = 0.5 * EQ_RAD**2 * (1 + 1 / Q)
    k3 = -P * (1 - ECCENTRICITY**2) * xyz[2]**2 / (Q * (1 + Q))
    k4 = -0.5 * P * p**2
    k5 = p - ECCENTRICITY**2 * (k1 + np.sqrt(k2 + k3 + k4))
    U = np.sqrt(k5**2 + xyz[2]**2)
    V = np.sqrt(k5**2 + (1 - ECCENTRICITY**2) * xyz[2]**2)
    alt = U * (1 - POL_RAD**2 / (EQ_RAD * V))
    z0 = POL_RAD**2 * xyz[2] / (EQ_RAD * V)
    ep = EQ_RAD / POL_RAD * ECCENTRICITY
    lat = np.arctan((xyz[2] + z0 * ep**2) / p)
    return lat, lon, alt