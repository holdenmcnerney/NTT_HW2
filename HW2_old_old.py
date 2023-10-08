#!/bin/python3

import warnings 
import numpy as np

import gps_data_parser as gdp

warnings.simplefilter('error')

def ecef_base_avg(filepath: str):

    # "./data_base/ecef_rx0.txt"
    data = np.loadtxt(filepath)
    avg_x = np.average(data[:, 2])
    avg_y = np.average(data[:, 3])
    avg_z = np.average(data[:, 4])

    return avg_x, avg_y, avg_z

def los_vectors(base_location: np.array, rinex_data: dict, \
                    start_time: float, end_time: float, \
                    sv_list: list):

    icp_data_struct = {}

    for sv in sv_list:

        icp_base = gdp.parse_icp("./data_base/icp_sat" + str(sv) + ".txt", \
                                np.array([start_time,end_time]))
        pseudorange = icp_base[0][0]
        trans_times = icp_base[0][1]
        orbit_xyz = rinex_data[str(sv)].calculate_orbit(trans_times)
        los_vector = np.array([0, 0, 0])
        elevation_vec = np.array(0)

        for orbits in orbit_xyz:
            
            distance = np.linalg.norm(orbits - base_location)
            los_vec = orbits / distance
            if los_vec[2] > 1:
                elevation = np.arcsin(1 - abs(1 - los_vec[2]))
            else:
                elevation = np.arcsin(los_vec[2])
            los_vector = np.vstack((los_vector, los_vec))
            elevation_vec = np.vstack((elevation_vec, elevation))

        los_vector = np.delete(los_vector, 0, axis=0)
        elevation_vec = np.delete(elevation_vec, 0, axis=0)

        icp_array = np.column_stack([pseudorange, trans_times, orbit_xyz, \
                                     los_vector, elevation_vec])

        icp_data_struct[sv] = icp_array

    return icp_data_struct

def rover_icp_load(rinex_data: dict, \
                    start_time: float, end_time: float, \
                    sv_list: list):

    icp_data_struct = {}

    for sv in sv_list:

        icp_base = gdp.parse_icp("./data_rover/icp_sat" + str(sv) + ".txt", \
                                np.array([start_time,end_time]))
        pseudorange = icp_base[0][0]
        trans_times = icp_base[0][1]
        orbit_xyz = rinex_data[str(sv)].calculate_orbit(trans_times)

        icp_array = np.column_stack([pseudorange, trans_times, orbit_xyz])

        icp_data_struct[sv] = icp_array

    return icp_data_struct

def ref_sv(icp_data: dict, sv_list: list):

    ref_sv = np.array([0, 0, 0])

    for time in icp_data[12][:, 1]:
        
        max_elevation = 0
        max_sv = 0
        for sv in sv_list:

            if np.where(icp_data[sv][:, 1] == time):

                idx = np.where(icp_data[sv][:, 1] == time)
                elevation = icp_data[sv][idx, 8]

                if np.size(icp_data[sv][idx, 8]) != 0 and \
                    icp_data[sv][idx, 8] > max_elevation:

                    max_elevation = elevation
                    max_sv = sv

        current_ref = np.array([time, max_sv, max_elevation[0][0]])
        ref_sv = np.vstack((ref_sv, current_ref))

    ref_sv = np.delete(ref_sv, 0, axis=0)

    return ref_sv

def calc_num_sv(icp_data_rover: dict, time: float, sv_list: list):
    
    num_sv = 0

    for sv in sv_list:

        # if time in icp_data_rover[sv][:, 1]:
        if abs(icp_data_rover[sv][:, 1] - time).min() < 0.001:

            num_sv += 1

    return num_sv

def dd_pseudo_position(icp_data_rover: dict, ref_svs: np.array, sv_list: list):

    for time in ref_svs[:, 0]:

        if calc_num_sv(icp_data_rover, time, sv_list) >= 4:
            
            a = 1

    return 0

def main():

    start_time = 417000
    end_time = 418000

    avg_base = ecef_base_avg("./data_base/ecef_rx0.txt")

    sv_list = [2, 4, 5, 9, 10, 12, 17, 23, 25]

    rinex_data = gdp.parse_rinex_v2("./brdc2930.11n")

    icp_data_base = los_vectors(avg_base, rinex_data, start_time, \
                                end_time, sv_list)
    
    icp_data_rover = rover_icp_load(rinex_data, start_time, end_time, sv_list)

    ref_svs = ref_sv(icp_data_base, sv_list)

    print(np.unique(ref_svs[:, 1]))

    # print(dd_pseudo_position(icp_data_rover, ref_svs, sv_list))
    
    pass

if __name__ == "__main__":
    main()