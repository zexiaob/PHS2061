#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:15:06 2024

@author: jinzeyuan
"""
import numpy as np
import monashspa.PHS2061 as spa
import matplotlib.pyplot as plt
import pandas as pd

#all in V
RT_V_Vs = [-0.001,
        0.100,
        0.201,
        0.301,
        0.400,
        0.500,
        0.599,
        0.697,
        0.794,
        0.902,
        0.997,
        1.103,
        1.199,
        1.297,
        1.396,
        1.503,
        1.600,
        1.705,
        1.805,
        1.902,
        1.999,
        2.07,
        2.17,
        2.30,
        2.40,
        2.50,
        2.60,
        2.70,
        2.80,
        2.90,
        3.00,
        3.10,
        3.20,
        3.30,
        3.40,
        3.50,
        4.00,
        4.50,
        5.00,
        5.50,
        6.00,
        6.50,
        7.00,
        7.50,
        8.00,
        8.50,
        9.00,
        9.50,
        10.00,
        10.50,
        11.00,
        11.50,
        11.51,
        11.52,
        11.53,
        11.54,
        11.55,
        11.56,
        11.57,
        11.60,
        12.00]

RT_Vs = [0.000,
      0.000,
      0.000,
      0.001,
      0.003,
      0.005,
      0.007,
      0.010,
      0.012,
      0.016,
      0.019,
      0.022,
      0.026,
      0.029,
      0.033,
      0.037,
      0.041,
      0.045,
      0.049,
      0.054,
      0.058,
      0.062,
      0.066,
      0.072,
      0.077,
      0.081,
      0.086,
      0.090,
      0.095,
      0.100,
      0.104,
      0.108,
      0.113,
      0.118,
      0.122,
      0.126,
      0.148,
      0.171,
      0.196,
      0.221,
      0.247,
      0.274,
      0.302,
      0.332,
      0.363,
      0.395,
      0.428,
      0.462,
      0.497,
      0.532,
      0.566,
      0.590,
      0.594,
      0.595,
      0.597,
      0.598,
      0.599,
      0.610,
      0.620,
      0.610,
      1.89]

RT_Vp = [0.000,
      0.000,
      0.000,
      0.001,
      0.002,
      0.004,
      0.006,
      0.008,
      0.010,
      0.012,
      0.013,
      0.014,
      0.014,
      0.015,
      0.015,
      0.015,
      0.014,
      0.014,
      0.014,
      0.013,
      0.013,
      0.013,
      0.012,
      0.012,
      0.011,
      0.011,
      0.011,
      0.010,
      0.010,
      0.010,
      0.010,
      0.010,
      0.010,
      0.009,
      0.009,
      0.0097,
      0.0094,
      0.0092,
      0.0093,
      0.0096,
      0.0103,
      0.0112,
      0.0123,
      0.0137,
      0.0155,
      0.0179,
      0.0208,
      0.0248,
      0.0299,
      0.0354,
      0.0427,
      0.0635,
      0.0641,
      0.0652,
      0.0664,
      0.0677,
      0.0689,
      0.1455,
      0.1862,
      0.196,
      1.813]

RT_V_Vs = np.array(RT_V_Vs)
RT_Vs = np.array(RT_Vs)
RT_Vp = np.array(RT_Vp)


LT_V_Vs = [-0.01,
           0.099,
           0.198,
           0.297,
           0.396,
           0.506,
           0.604,
           0.702,
           0.798,
           0.903,
           1.000,
           1.096,
           1.202,
           1.300,
           1.398,
           1.504,
           1.600,
           1.700,
           1.800,
           1.900,
           1.997,
           2.00,
           2.10,
           2.20,
           2.30,
           2.40,
           2.50,
           2.60,
           2.70,
           2.80,
           2.90,
           3.00,
           3.10,
           3.20,
           3.30,
           3.40,
           3.50,
           4.00,
           4.50,
           5.00,
           5.50,
           6.00,
           6.50,
           7.00,
           7.50
           ]

LT_Vs = [0.0006,
         0.0015,
         0.0031,
         0.0051,
         0.0075,
         0.0104,
         0.0133,
         0.0163,
         0.0194,
         0.0231,
         0.0265,
         0.0301,
         0.0341,
         0.0380,
         0.0420,
         0.0465,
         0.0506,
         0.0548,
         0.0598,
         0.0637,
         0.0681,
         0.0695,
         0.0741,
         0.0791,
         0.0838,
         0.0890,
         0.0938,
         0.0991,
         0.1044,
         0.1092,
         0.1144,
         0.1198,
         0.1247,
         0.1300,
         0.1355,
         0.1404,
         0.1458,
         0.1730,
         0.202,
         0.233,
         0.267,
         0.303,
         0.344,
         0.385,
         0.427]

LT_Vp = [0.0016,
         0.0032,
         0.0051,
         0.0066,
         0.0082,
         0.0103,
         0.0124,
         0.0145,
         0.0168,
         0.0194,
         0.0219,
         0.0244,
         0.0272,
         0.0300,
         0.0328,
         0.0359,
         0.0388,
         0.0417,
         0.0450,
         0.0476,
         0.0506,
         0.0515,
         0.0544,
         0.0577,
         0.0607,
         0.0640,
         0.0672,
         0.0705,
         0.0739,
         0.0769,
         0.0803,
         0.0837,
         0.0869,
         0.0903,
         0.0938,
         0.0971,
         0.1006,
         0.1180,
         0.1364,
         0.1569,
         0.1796,
         0.203,
         0.228,
         0.255,
         0.295]


LT_V_Vs = np.array(LT_V_Vs)
LT_Vs = np.array(LT_Vs)
LT_Vp = np.array(LT_Vp)

plt.figure(1)
plt.plot(RT_V_Vs,RT_Vp,ls = "none", marker = "x", label="room temptaure")
plt.plot(LT_V_Vs,LT_Vp,ls = "none", marker = "x", label="low temptaure")
plt.xlabel("V-Vs[V]")
plt.ylabel("Vp[V]")
plt.legend()
plt.show

def calculate_uncertainty_precision(values):
    uncertainties = []
    for value in values:
        if value == 0:
            uncertainty = 0.005  # 对于值为0的情况，设置一个小的不确定度
        else:
            str_value = str(value)
            if '.' in str_value:
                decimal_part = str_value.split('.')[1]
                if 'e' in decimal_part:  # 检查科学计数法
                    decimal_part = decimal_part.split('e')[0]
                uncertainty_position = len(decimal_part) + 1  # 确定不确定度的位置
                uncertainty = float('0.' + '0'*(uncertainty_position-1) + '5')
            else:
                uncertainty = 0.5  # 对于整数，设定默认不确定度为0.5
        uncertainties.append(uncertainty)
    return uncertainties

# 计算每个数组的不确定度
RT_V_Vs_unc = calculate_uncertainty_precision(RT_V_Vs)
RT_Vs_unc = calculate_uncertainty_precision(RT_Vs)
RT_Vp_unc = calculate_uncertainty_precision(RT_Vp)
LT_V_Vs_unc = calculate_uncertainty_precision(LT_V_Vs)
LT_Vs_unc = calculate_uncertainty_precision(LT_Vs)
LT_Vp_unc = calculate_uncertainty_precision(LT_Vp)

# 创建DataFrame
RT_data = {
    "RT_V_Vs[V]": RT_V_Vs,
    "RT_V_Vs_unc[V]": RT_V_Vs_unc,
    "RT_Vs[V]": RT_Vs,
    "RT_Vs_unc[V]": RT_Vs_unc,
    "RT_Vp[V]": RT_Vp,
    "RT_Vp_unc[V]": RT_Vp_unc
}

LT_data = {
    "LT_V_Vs[V]": LT_V_Vs,
    "LT_V_Vs_unc[V]": LT_V_Vs_unc,
    "LT_Vs[V]": LT_Vs,
    "LT_Vs_unc[V]": LT_Vs_unc,
    "LT_Vp[V]": LT_Vp,
    "LT_Vp_unc[V]": LT_Vp_unc
    }

RT_df = pd.DataFrame(RT_data)
LT_df = pd.DataFrame(LT_data)
RT_df.to_csv("RT_DATA.csv")
LT_df.to_csv("LT_DATA.csv")

