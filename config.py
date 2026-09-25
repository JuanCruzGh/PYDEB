# -*- coding: utf-8 -*-
"""
Created on Wed May 20 17:41:22 2026

@author: ThinkPad
"""
# =============================================================================
# UNIVERSAL CONSTANTS
# =============================================================================

CONSTANTS = {

    "g"      : 9.81,
    "k_vk"   : 0.41,
    "sigma"  : 5.67e-8,
    "Rgas"   : 8.31447,
    "Mair"   : 0.0289644,
    "p0"     : 101325,
    "T0"     : 288.15,
    "Lapse"  : 0.0065,
    "L_v"    : 2476000,
    "L_f"    : 334000,
    "L_s"   : 2.83e6,  
    "rho_w"  : 999.7,
    "c_w"    : 4181.3,
    "c_ad"   : 1005,
    "rho_i"  : 915,
    "T_f"    : 0.0,

}

# =============================================================================
# MODEL SETTINGS
# =============================================================================

MODEL = {

    "timestep" : 3600,
    "z_a"      : 2.0,
    "range_"   : 0.5,
    "N"        : 10,

}

# =============================================================================
# SNOW PARAMETERS
# =============================================================================

SNOW = {

    "z_0_s"     : 0.00231,
    "albedo_s"  : 0.52,
    "epsilon_s" : 0.97,

}

# =============================================================================
# SITE PARAMETERS
# =============================================================================

SITES = {

    "4400_DC": {

        "meteo_file" : "met_data_4400_DC.csv",

        # Debris thickness
        "d" : 0.5,

        # Debris parameters
        "z_0_d"     : 0.0123, 
        "k_d"       : 0.978,
        "rho_d"     : 2685,
        "c_d"       : 816,
        "epsilon_d" : 0.94,
        "albedo_d"  : 0.13,

    },

    "4500_T": {

        "meteo_file" : "met_data_4500_T.csv",

        "d" : 0.09,

        "z_0_d"     : 0.03,
        "k_d"       : 0.954, # 0.67 Conway & Rasmussen (2000) method; ref value 0.96,
        "rho_d"     : 2685,
        "c_d"       : 816,
        "epsilon_d" : 0.94,
        "albedo_d"  : 0.13,

    },

    "4500_I": {

        "meteo_file" : "met_data_4500_I.csv",

        "d" : 0.0,

        "z_0_d"     : 0.0405 ,  
        "epsilon_i" : 0.97,
        "albedo_i"  :   0.34, 
        
    },


    "4500_DC": {

        "meteo_file" : "met_data_4500_DC.csv",
        
        # Debris thickness
        "d" : 0.68,

        # Debris parameters
        # "z_0_d"     : 0.007, # 0.013-0.075; REF 0.016
        # "k_d"       : 1.38, # 0.81 Conway & Rasmussen (2000) method; ref value 0.96,
        "z_0_d"     : 0.0052, # 0.013-0.075; REF 0.016
        "k_d"       : 1.423, # 0.81 Conway & Rasmussen (2000) method; ref value 0.96,
        "rho_d"     : 2685,   #1496,
        "c_d"       : 816,
        "epsilon_d" : 0.94,
        "albedo_d"  : 0.13,

    },

    "4600_T": {

        "meteo_file" : "met_data_4600_T.csv",

        "d" : 0.06,

        "z_0_d"     : 0.09,
        "k_d"       : 1.003, # 0.67 Conway & Rasmussen (2000) method; ref value 0.96,
        "rho_d"     : 2685,
        "c_d"       : 816,
        "epsilon_d" : 0.94,
        "albedo_d"  : 0.13,

    },

    "4600_I": {

        "meteo_file" : "met_data_4600_I.csv",

        "d" : 0.0,

        "z_0_d"     : 0.0356,  
        "epsilon_i" : 0.97,
        "albedo_i"  :   0.34, 

    },

}