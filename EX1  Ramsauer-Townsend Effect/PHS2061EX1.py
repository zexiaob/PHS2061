#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:15:06 2024

@author: jinzeyuan
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sp

"""
If you need a csv to put into the googledoc
change "need_csv" to True
The csv will be saved to the same path as this program

some of uncertainties is wrong, pls double check!!!
some of uncertainties is wrong, pls double check!!!
some of uncertainties is wrong, pls double check!!!
(Because uncertainty is calculated using the number of decimal places in the data)

sorry for that I left some chinese in it, if u want to read it, pls use googletranslate
"""
need_csv = False

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


df_LT = pd.read_csv('LT_DS.csv')
df_RT = pd.read_csv('RT_DS.csv')

df_u_LT_V_Vs = df_LT['V-Vs uncertainty']
df_u_LT_Vs = df_LT['Vs uncertainty']
df_u_LT_Vp = df_LT['Vp uncertainty']

df_u_RT_V_Vs = df_RT['V-Vs uncertainty']
df_u_RT_Vs = df_RT['Vs uncertainty']
df_u_RT_Vp = df_RT['Vp uncertainty']

u_LT_V_Vs = df_u_LT_V_Vs.to_numpy()
u_LT_Vs = df_u_LT_Vs.to_numpy()
u_LT_Vp = df_u_LT_Vp.to_numpy()

u_RT_V_Vs = df_u_RT_V_Vs.to_numpy()
u_RT_Vs = df_u_RT_Vs.to_numpy()
u_RT_Vp = df_u_RT_Vp.to_numpy()

LT_V_Vs = np.array(LT_V_Vs)
LT_Vs = np.array(LT_Vs)
LT_Vp = np.array(LT_Vp)

LT_Ip = LT_Vp/10 # in mA
RT_Ip = RT_Vp/10

plt.figure(1)
plt.figure(figsize=(10,6))
plt.errorbar(RT_V_Vs,RT_Ip,yerr = (u_RT_Vp/10),xerr = u_RT_V_Vs,ls = "--", marker = "x", color = "black", label="With Xe gas")
plt.errorbar(LT_V_Vs,LT_Ip,yerr = (u_LT_Vp/10),xerr = u_LT_V_Vs,ls = "--", marker = "x", color = "red", label="Xe frozen out")
plt.title("Variation of plate current(Ip) with Accelerating Voltage")
plt.xlabel("V-Vs[V]")
plt.ylabel("Ip[mA]")
plt.legend()
plt.show

plt.figure(1)
plt.figure(figsize=(10,6))
plt.errorbar(RT_V_Vs,RT_Ip,yerr = (u_RT_Vp/10),xerr = u_RT_V_Vs,ls = "none", marker = "x", color = "black", label="With Xe gas")
plt.errorbar(LT_V_Vs,LT_Ip,yerr = (u_LT_Vp/10),xerr = u_LT_V_Vs,ls = "none", marker = "x", color = "red", label="Xe frozen out")
plt.xlim(0,3.5)
plt.ylim(0,0.01)
plt.hlines(0.0015, 0, 3.5, color = "black", ls="--")
plt.title("Variation of plate current(Ip) with Accelerating Voltage")
plt.xlabel("V-Vs[V]")
plt.ylabel("Ip[mA]")
plt.legend()
plt.show


# 截断RT_V_Vs和LT_V_Vs到3.5V之前
RT_V_Vs_truncated = RT_V_Vs[RT_V_Vs <= 3.5]
LT_V_Vs_truncated = LT_V_Vs[LT_V_Vs <= 3.5]

# 截断RT_Vp和LT_Vp以匹配上述截断
RT_Vp_truncated = RT_Vp[:len(RT_V_Vs_truncated)]
LT_Vp_truncated = LT_Vp[:len(LT_V_Vs_truncated)]

RT_Vs_truncated = RT_Vs[:len(RT_V_Vs_truncated)]
LT_Vs_truncated = LT_Vs[:len(LT_V_Vs_truncated)]

u_RT_Vp_truncated = u_RT_Vp[:len(RT_V_Vs_truncated)]
u_LT_Vp_truncated = u_LT_Vp[:len(LT_V_Vs_truncated)]
u_LT_V_Vs_truncated = u_LT_V_Vs[:len(LT_V_Vs_truncated)]

RT_V_Vs_truncated_extension = RT_V_Vs[RT_V_Vs <= 6]
LT_V_Vs_truncated_extension = LT_V_Vs[LT_V_Vs <= 6]

RT_Vp_truncated_extension = RT_Vp[:len(RT_V_Vs_truncated_extension)]
LT_Vp_truncated_extension = LT_Vp[:len(LT_V_Vs_truncated_extension)]

RT_Vs_truncated_extension = RT_Vs[:len(RT_V_Vs_truncated_extension)]
LT_Vs_truncated_extension = LT_Vs[:len(LT_V_Vs_truncated_extension)]

u_RT_Vp_truncated_extension = u_RT_Vp[:len(RT_V_Vs_truncated_extension)]
u_LT_Vp_truncated_extension = u_LT_Vp[:len(LT_V_Vs_truncated_extension)]

u_RT_Vs_truncated_extension = u_RT_Vs[:len(RT_V_Vs_truncated_extension)]
u_LT_Vs_truncated_extension = u_LT_Vs[:len(LT_V_Vs_truncated_extension)]

u_LT_V_Vs_truncated_extension = u_RT_V_Vs[:len(RT_V_Vs_truncated_extension)]
u_RT_V_Vs_truncated_extension = u_RT_V_Vs[:len(RT_V_Vs_truncated_extension)]

T = RT_Vp_truncated/LT_Vp_truncated
u_T = np.sqrt(((u_RT_Vp_truncated/LT_Vp_truncated)**2)+(((-RT_Vp_truncated)*(LT_Vp_truncated**(-2))*u_LT_Vp_truncated)**2))
print(u_T)
plt.figure(2)
plt.figure(figsize=(10,6))
plt.errorbar(LT_V_Vs_truncated,T,yerr = u_T,xerr = u_LT_V_Vs_truncated ,ls = "none", marker = "x", color = "black")
plt.xlabel("V-Vs[V]")
plt.ylabel("transmission probability")
plt.title("Variation of Electron Transmission Probability(T) with Accelerating Voltage")
plt.vlines(1, -0.3, 0.7, color = "red", ls="--")
plt.hlines(0.618,0,3.5, color = "black", ls="--")
plt.vlines(0.903, -0.3, 0.7, color = "black", ls="--")
plt.show()


𝜎 = -np.log(T)
u_𝜎 = np.abs(u_T/T)

plt.figure(3)
plt.figure(figsize=(10,6))
plt.errorbar(LT_V_Vs_truncated,𝜎,xerr = u_LT_V_Vs_truncated,yerr=u_𝜎,ls = "none", marker = "x", color = "black")
plt.xlabel("V-Vs[V]")
plt.ylabel("scattering cross-section")
plt.title("Evolution of Scattering Cross-Section ($\sigma$) with Accelerating Voltage")
plt.show()


T_extension = (RT_Vp_truncated_extension*LT_Vs_truncated_extension)/(LT_Vp_truncated_extension*RT_Vs_truncated_extension)

a = RT_Vp_truncated_extension
b = LT_Vs_truncated_extension
c = LT_Vp_truncated_extension
d = RT_Vs_truncated_extension
ua = u_RT_Vp_truncated_extension
ub = u_LT_Vs_truncated_extension
uc = u_LT_Vp_truncated_extension
ud = u_RT_Vs_truncated_extension

ab = a*b
cd = c*d
c2d = (c**2)*d
cd2 = c*(d**2)

一 = b / cd
二 = a / cd
三 = ab / c2d
四 = ab / cd2

五 = 一*ua
六 = 二*ub
七 = 三*uc
八 = 四*ud

一 = 五**2
二 = 六**2
三 = 七**2
四 = 八**2

u_T_extension = np.sqrt(一+二+三+四)


plt.figure(4)
plt.figure(figsize=(10,6))
plt.errorbar(LT_V_Vs_truncated_extension,T_extension,xerr = u_LT_V_Vs_truncated_extension, yerr = u_T_extension,ls = "none", marker = "x", color = "black", label = "T consider the tube geometry")
plt.errorbar(LT_V_Vs_truncated,T,yerr = u_T,xerr = u_LT_V_Vs_truncated ,ls = "none", marker = "x", color = "r", label = "T without consider the tube geometry")
plt.xlabel("V-Vs[V]")
plt.ylabel("transmission probability")
plt.title("Variation of Electron Transmission Probability(T) with Accelerating Voltage(For extension)")
plt.legend()
plt.show()

𝜎_extension = -np.log(T_extension)
u_𝜎_extension = np.abs(u_T_extension/T_extension)

plt.figure(5)
plt.figure(figsize=(10,6))
plt.errorbar(LT_V_Vs_truncated_extension,𝜎_extension,xerr = u_LT_V_Vs_truncated_extension, yerr = u_𝜎_extension,ls = "none", marker = "x", color = "black", label = "$\sigma$ consider the tube geometry")
plt.errorbar(LT_V_Vs_truncated,𝜎,xerr = u_LT_V_Vs_truncated,yerr=u_𝜎,ls = "none", marker = "x", color = "r", label = "$\sigma$ without consider the tube geometry")
plt.xlabel("V-Vs[V]")
plt.ylabel("scattering cross-section")
plt.title("Evolution of Scattering Cross-Section ($\sigma$) with Accelerating Voltage(For extension)")
plt.legend()
plt.show()

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

if need_csv:
    RT_df.to_csv("RT_DATA.csv")
    LT_df.to_csv("LT_DATA.csv")

