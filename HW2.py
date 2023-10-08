#!/bin/python3

import navpy 
import numpy as np
import matplotlib as mpl
import numpy.linalg as npl
import matplotlib.pyplot as plt
import gps_data_parser as gdp

# Matplotlib global params
plt.rcParams['axes.grid'] = True

def ecef_base_avg(filepath: str):
    '''
    Opens base station ecef file and finds average x, y, and z
    '''
    
    data = np.loadtxt(filepath)
    avg_x = np.average(data[:, 2])
    avg_y = np.average(data[:, 3])
    avg_z = np.average(data[:, 4])

    return avg_x, avg_y, avg_z

def los_vectors(base_location: np.array, rinex_data: dict, \
                    start_time: float, end_time: float, \
                    sv_list: list):
    '''
    Reads in base station icp files, creates dictionary with key of PRN and
    value of an array containing pseudorange, time, sv location in NED, LOS 
    vectors from base station to sv, and the corresponding elevation angle
    
    Returns dictionary of PRN keys and corresponding value arrays
    '''

    # Initialize output dictionary
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
    '''
    Reads in rover icp files, creates dictionary with key of PRN and
    value of an array containing pseudorange, time, sv location in NED

    Returns dictionary of PRN keys and corresponding value arrays
    '''

    # Initialize output dictionary
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
    '''
    Finds the max elevation for base station to sv for all timesteps
    Returns array of time, the sv with the largest elevation, and the elevation
    '''

    # Initialize output array
    ref_sv = np.array([0, 0, 0])

    # Used sv 12 since it was the longest file
    for time in icp_data[12][:, 1]:
        
        max_elevation = 0
        max_elev_sv = 0
        for sv in sv_list:

            # Find where/if the time is in the sv dictionary value array
            idx = np.where(icp_data[sv][:, 1] == time)
            elevation = icp_data[sv][idx, 8]

            if np.size(idx) != 0 and icp_data[sv][idx, 8] > max_elevation:

                max_elevation = elevation[0][0]
                max_elev_sv = sv

        current_ref = np.array([time, max_elev_sv, max_elevation])
        ref_sv = np.vstack((ref_sv, current_ref))

    ref_sv = np.delete(ref_sv, 0, axis=0)

    return ref_sv

def find_vis_sv(icp_data_rover: dict, icp_data_base: np.array, \
                 time: float, sv_list: list):
    '''
    Generates the list of svs that both the base station and the rover have 
    data for at a timestamp

    Returns a list of visible svs
    '''
    
    # Initialize output list
    visible_sv = []

    for sv in sv_list:

        # Checks to see if the time can be found for an sv at timestamp
        if abs(icp_data_rover[sv][:, 1] - time).min() < 0.001 and \
            abs(icp_data_base[sv][:, 1] - time).min() < 0.001:

            visible_sv.append(sv)

    return visible_sv

def guass_newton(ref_loc: np.array, pseudorange: np.array, \
                 sv_loc: np.array):
    '''
    Solves a NLS problem with GNA at timestamp
    Returns the optimal value and two dilutions of precision
    '''
    
    beta = [0, 0, 0]
    delta_k = np.array([0, 0, 0])
    first = True

    # Used first flag to enter while loop without generating numpy warning
    while (first or (npl.norm(delta_k) / npl.norm(beta)) >  0.0001):

        first = False

        f_x_b_vec = np.array([0])
        jacobian_vec = np.array([0, 0, 0])

        for sv in sv_loc:

            # Builds f_x_b and jacobian for timestamp
            f_x_b = (npl.norm(beta - sv) - npl.norm(sv)) \
                    - (npl.norm(beta - ref_loc) - npl.norm(ref_loc))
            jacobian = (beta - sv) / npl.norm(beta - sv) \
                    - (beta - ref_loc) / npl.norm(beta - ref_loc)
            f_x_b_vec = np.vstack((f_x_b_vec, f_x_b))
            jacobian_vec = np.vstack((jacobian_vec, jacobian))

        f_x_b_vec = np.delete(f_x_b_vec, 0, axis=0)
        jacobian_vec = np.delete(jacobian_vec, 0, axis=0)
        cov_beta = npl.inv(np.transpose(jacobian_vec) @ jacobian_vec)
        delta_k = cov_beta @ np.transpose(jacobian_vec) @ (pseudorange - f_x_b_vec)
        beta += np.transpose(delta_k)
        vdop = np.sqrt(cov_beta[2][2])
        hdop = np.sqrt(cov_beta[0][0] + cov_beta[1][1])

        delta =(npl.norm(delta_k) / npl.norm(beta))

    return beta, vdop, hdop

def dd_pseudo_position(icp_data_rover: dict, icp_data_base: np.array, \
                        ref_svs: np.array, sv_list: list):
    '''
    Calculates rover location in NED frame using double differenced pseudorange
    equation

    Returns time, time history of rover location calculated with GNA, time 
    history of VDOP, and time history of HDOP
    '''
    
    # Initialize output arrays
    rover_loc_hist = np.array([0, 0, 0])
    time_hist = np.array([0])
    vdop_hist = np.array([0])
    hdop_hist = np.array([0])

    for time in ref_svs[:, 0]:

        visible_sv = find_vis_sv(icp_data_rover, icp_data_base, time, sv_list)
        ref_idx = np.where(ref_svs[:, 0] == time)[0][0]
        ref_sv_num = ref_svs[ref_idx, 1]
        ref_loc_rover_idx = np.absolute(icp_data_rover[ref_sv_num][:, 1] \
                                       - time).argmin()
        ref_loc_base_idx = np.absolute(icp_data_base[ref_sv_num][:, 1] \
                                       - time).argmin()
        ref_loc_ned = icp_data_rover[ref_sv_num][ref_loc_rover_idx, 2:5]
        pseudorange_base_ref = icp_data_base[ref_sv_num][ref_loc_base_idx, 0]
        
        pseudorange_vec = np.array([0])
        sv_loc_vec = np.array([0, 0, 0])

        if len(visible_sv) >= 4:

            for sv in visible_sv:

                if sv is not int(ref_sv_num):
                    
                    # Find indexes and pseudoranges of sv at timestamp and stack
                    # pseudorange and sv location (in NED) arrays 
                    current_sv_rover_idx = np.absolute(icp_data_rover[sv][:, 1] \
                                                - time).argmin()
                    current_sv_base_idx = np.absolute(icp_data_base[sv][:, 1] \
                                                - time).argmin()
                    pseudorange_rover = icp_data_rover[sv][current_sv_rover_idx, 0]
                    pseudorange_rover_ref = icp_data_rover[ref_sv_num][ref_loc_rover_idx, 0]
                    pseudorange_base = icp_data_base[sv][current_sv_base_idx, 0]
                    double_diff_pseudo = (pseudorange_rover - pseudorange_base) \
                                        - (pseudorange_rover_ref - pseudorange_base_ref)
                    sv_loc_ned = icp_data_rover[sv][current_sv_rover_idx, 2:5]
                    pseudorange_vec = np.vstack((pseudorange_vec,double_diff_pseudo))
                    sv_loc_vec = np.vstack((sv_loc_vec, sv_loc_ned))

            pseudorange_vec = np.delete(pseudorange_vec, 0, axis=0)
            sv_loc_vec = np.delete(sv_loc_vec, 0, axis=0)
            # Calculate rover location at timestamp with inputs from above
            rover_loc, vdop, hdop = guass_newton(ref_loc_ned, pseudorange_vec, sv_loc_vec)
            rover_loc_hist = np.vstack((rover_loc_hist, rover_loc))
            time_hist = np.vstack((time_hist, time))
            vdop_hist = np.vstack((vdop_hist, vdop))
            hdop_hist = np.vstack((hdop_hist, hdop))


    rover_loc_hist = np.delete(rover_loc_hist, 0, axis=0)
    time_hist = np.delete(time_hist, 0, axis=0)
    vdop_hist = np.delete(vdop_hist, 0, axis=0)
    hdop_hist = np.delete(hdop_hist, 0, axis=0)

    return time_hist, rover_loc_hist, vdop_hist, hdop_hist

def main():

    # Rounded time bounds from ICP files used in calculate_orbit() method
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
    time, rover_loc, vdop, hdop = dd_pseudo_position(icp_data_rover, icp_data_base, \
                                          ref_svs, sv_list)
    
    # Plot results in 2 figures with 2 subplots each
    fig1, ax1 = plt.subplots(1, 2)
    ax1[0].set_title('Position NED')
    ax1[0].set_xlabel(r'East ($m$)')
    ax1[0].set_ylabel(r'North ($m$)')
    ax1[0].plot(rover_loc[:, 1], rover_loc[:, 0])

    ax1[1].set_title('Height NED')
    ax1[1].set_ylabel(r'Height ($m$)')
    ax1[1].set_xlabel(r'Time ($s$)')
    ax1[1].plot(time - time[0], rover_loc[:, 2])
    
    fig2, ax2 = plt.subplots(1, 2)
    ax2[0].set_title('VDOP')
    ax2[0].set_ylabel(r'VDOP')
    ax2[0].set_xlabel(r'Time ($s$)')
    ax2[0].plot(time - time[0], vdop)

    ax2[1].set_title('HDOP')
    ax2[1].set_ylabel(r'HDOP')
    ax2[1].set_xlabel(r'Time ($s$)')
    ax2[1].plot(time - time[0], hdop)
    plt.show()

if __name__ == "__main__":

    main()