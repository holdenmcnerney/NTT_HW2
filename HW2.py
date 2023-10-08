#!/bin/python3

import navpy 
import numpy as np
import numpy.linalg as npl
import gps_data_parser as gdp

def ecef_base_avg(filepath: str):

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

        icp_base = gdp.parse_icp(f"./data_base/icp_sat{sv}.txt", \
                                np.array([start_time,end_time]))
        pseudorange = icp_base[0][0]
        trans_times = icp_base[0][1]
        orbit_ecef = rinex_data[str(sv)].calculate_orbit(trans_times)
        orbit_ned = navpy.ecef2ned(orbit_ecef, base_location[0], \
                                   base_location[1], base_location[2], \
                                    latlon_unit='rad', alt_unit='m', \
                                    model='wgs84')
        los_vector = np.array([0, 0, 0])
        elevation_vec = np.array(0)

        for orbits in orbit_ned:
            
            distance = npl.norm(orbits)
            los_vec = orbits / distance
            elevation = np.arcsin(-los_vec[2])
            los_vector = np.vstack((los_vector, los_vec))
            elevation_vec = np.vstack((elevation_vec, elevation))

        los_vector = np.delete(los_vector, 0, axis=0)
        elevation_vec = np.delete(elevation_vec, 0, axis=0)
        icp_array = np.column_stack([pseudorange, trans_times, orbit_ned, \
                                     los_vector, elevation_vec])
        icp_data_struct[sv] = icp_array

    return icp_data_struct

def rover_icp_load(base_location: np.array, rinex_data: dict, \
                    start_time: float, end_time: float, \
                    sv_list: list):

    icp_data_struct = {}

    for sv in sv_list:

        icp_base = gdp.parse_icp(f"./data_rover/icp_sat{sv}.txt", \
                                np.array([start_time,end_time]))
        pseudorange = icp_base[0][0]
        trans_times = icp_base[0][1]
        orbit_ecef = rinex_data[str(sv)].calculate_orbit(trans_times)
        orbit_ned = navpy.ecef2ned(orbit_ecef, base_location[0], \
                                   base_location[1], base_location[2], \
                                    latlon_unit='rad', alt_unit='m', \
                                    model='wgs84')
        icp_array = np.column_stack([pseudorange, trans_times, orbit_ned])
        icp_data_struct[sv] = icp_array

    return icp_data_struct

def ref_sv(icp_data: dict, sv_list: list):

    ref_sv = np.array([0, 0, 0])

    for time in icp_data[12][:, 1]:
        
        max_elevation = 0
        max_sv = 0
        for sv in sv_list:

            idx = np.where(icp_data[sv][:, 1] == time)
            elevation = icp_data[sv][idx, 8]

            if np.size(icp_data[sv][idx, 8]) != 0 and \
                icp_data[sv][idx, 8] > max_elevation:

                max_elevation = elevation[0][0]
                max_sv = sv

        current_ref = np.array([time, max_sv, max_elevation])
        ref_sv = np.vstack((ref_sv, current_ref))

    ref_sv = np.delete(ref_sv, 0, axis=0)

    return ref_sv

def find_vis_sv(icp_data_rover: dict, time: float, sv_list: list):
    
    num_sv = 0

    visible_sv = []

    for sv in sv_list:

        if abs(icp_data_rover[sv][:, 1] - time).min() < 0.001:

            visible_sv.append(sv)

    return visible_sv

def guass_newton(ref_loc: np.array, pseudorange: np.array, \
                 sv_loc: np.array):

    beta = [0, 0, 0]
    delta_k = np.array([0, 0, 0])
    first = True

    while (first or (npl.norm(delta_k) / npl.norm(beta)) > 0.001):

        first = False

        f_x_b_vec = np.array([0])
        jacobian_vec = np.array([0, 0, 0])

        for sv in sv_loc:

            f_x_b = (npl.norm(beta - sv) - npl.norm(sv)) \
                    - (npl.norm(beta - ref_loc) - npl.norm(ref_loc))
            jacobian = (beta - sv) / npl.norm(beta - sv) \
                    - (beta - ref_loc) / npl.norm(beta - ref_loc)
            f_x_b_vec = np.vstack((f_x_b_vec, f_x_b))
            jacobian_vec = np.vstack((jacobian_vec, jacobian))

        f_x_b_vec = np.delete(f_x_b_vec, 0, axis=0)
        jacobian_vec = np.delete(jacobian_vec, 0, axis=0)
        delta_k = npl.inv(np.transpose(jacobian_vec) @ jacobian_vec) \
                    @ np.transpose(jacobian_vec) @ (pseudorange - f_x_b_vec)
        beta += np.transpose(delta_k)

        delta =(npl.norm(delta_k) / npl.norm(beta))

    return beta

def dd_pseudo_position(icp_data_rover: dict, icp_data_base: np.array, \
                        ref_svs: np.array, sv_list: list):

    rover_loc_hist = np.array([0, 0, 0])
    time_hist = np.array([0])

    for time in ref_svs[:, 0]:

        visible_sv = find_vis_sv(icp_data_rover, time, sv_list)
        ref_idx = np.where(ref_svs[:, 0] == time)[0][0]
        ref_loc_idx = np.absolute(icp_data_rover[ref_svs[ref_idx, 1]][:, 1] \
                                       - time).argmin()
        ref_loc_ned = icp_data_rover[ref_svs[ref_idx, 1]][ref_loc_idx, 2:5]
        pseudorange_vec = np.array([0])
        sv_loc_vec = np.array([0, 0, 0])

        if len(visible_sv) >= 4:

            for sv in visible_sv:

                if sv is not int(ref_svs[ref_idx, 1]):
                        
                    current_sv_rover_idx = np.absolute(icp_data_rover[sv][:, 1] \
                                                - time).argmin()
                    current_sv_base_idx = np.absolute(icp_data_base[sv][:, 1] \
                                                - time).argmin()
                    pseudorange_rover = icp_data_rover[sv][current_sv_rover_idx, 0]
                    pseudorange_rover_ref = icp_data_rover[ref_svs[ref_idx, 1]][current_sv_rover_idx, 0]
                    pseuorange_base = icp_data_base[sv][current_sv_base_idx, 0]
                    pseudorange_base_ref = icp_data_base[ref_svs[ref_idx, 1]][current_sv_base_idx, 0]
                    double_diff_pseudo = (pseudorange_rover - pseuorange_base) \
                                        - (pseudorange_rover_ref - pseudorange_base_ref)
                    sv_loc_ned = icp_data_rover[sv][current_sv_rover_idx, 2:5]
                    pseudorange_vec = np.vstack((pseudorange_vec,double_diff_pseudo))
                    sv_loc_vec = np.vstack((sv_loc_vec, sv_loc_ned))

        
            pseudorange_vec = np.delete(pseudorange_vec, 0, axis=0)
            sv_loc_vec = np.delete(sv_loc_vec, 0, axis=0)
            rover_loc = guass_newton(ref_loc_ned, pseudorange_vec, sv_loc_vec)
            rover_loc_hist = np.vstack((rover_loc_hist, rover_loc))
            time_hist = np.vstack((time_hist, time))

    rover_loc_hist = np.delete(rover_loc_hist, 0, axis=0)
    time_hist = np.delete(time_hist, 0, axis=0)

    return time_hist, rover_loc_hist

def main():

    start_time = 417000
    end_time = 418000

    avg_base_ecef = ecef_base_avg("./data_base/ecef_rx0.txt")
    avg_base_lla = gdp.ecef_to_lla(avg_base_ecef)
    sv_list = [2, 4, 5, 9, 10, 12, 17, 23, 25]
    rinex_data = gdp.parse_rinex_v2("./brdc2930.11n")
    icp_data_base = los_vectors(avg_base_lla, rinex_data, \
                                start_time, end_time, sv_list)
    icp_data_rover = rover_icp_load(avg_base_lla, rinex_data, \
                                    start_time, end_time, sv_list)
    ref_svs = ref_sv(icp_data_base, sv_list)
    time, rover_loc = dd_pseudo_position(icp_data_rover, icp_data_base, \
                                          ref_svs, sv_list)
    
    pass

if __name__ == "__main__":
    main()