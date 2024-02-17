import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import matplotlib.ticker as mtick
import os
import warnings
import torch.nn as nn
import torch
from scipy import stats
from scipy.stats import chi2
from scipy.stats import norm

def calculate_volt_modepi(volt_all):
    """
    Simplified function to calculate volt_modepi and volt_di
    """
    volt_mode = volt_all.mean(axis=1)
    volt_std = volt_all.std(axis=1)
    volt_lamda = 1 / volt_std

    volt_pi1 = (1 / (2 * np.pi * volt_all.pow(3))).mul(volt_lamda, axis=0).pow(0.5)
    volt_pi2 = (((-1) * ((volt_all.sub(volt_mode, axis=0))).pow(2).mul(volt_lamda, axis=0)) / (2 * volt_all.mul(volt_mode.pow(2), axis=0))).apply(np.exp)
    volt_pi = volt_pi1 * volt_pi2

    volt_modepi = ((volt_pi * volt_all).sum(axis=1)) / (volt_pi.sum(axis=1))
    volt_di = volt_all.sub(volt_modepi, axis=0)

    return volt_modepi, volt_di


def solvers(volt_modepi, volt_di, volt_all, soc, b, current, temp_avg):
    num = volt_all.shape[1]
    P = np.eye(2)
    Pi = np.zeros((volt_all.shape[0], num))
    Pi[0, 0:num] = 1000 * np.ones((1, num))
    Q = np.array([[1e-5, 0], [0, 1e-5]])
    R = 0.1
    Qi = 1e-4
    Ri = 2

    RO = 1e-3
    RP = 5.2203e-4
    CP = 5e3
    # DRi = [3.41688e-6, -6.19014e-6, -1.11437e-5, -3.64918e-06]
    DRi = 3 * 10 ** (-6) * np.ones(volt_all.shape[1])
    t = RP * CP
    SOC = np.zeros(volt_all.shape[0])
    smin = np.zeros(volt_all.shape[0])
    RH = np.zeros(volt_all.shape[0])

    SOC[0] = 0.01 * soc[0]
    smin[0] = 0.01 * soc[0] - 0.001
    RH[0] = 0.01 * soc[0]

    x = np.zeros((2, volt_all.shape[0]))
    xpre = np.zeros((2, volt_all.shape[0] + 1))
    x[:, 0] = np.array([0, soc[0]]).T
    xi = np.zeros((volt_all.shape[0], num))
    Xi = np.zeros((volt_all.shape[0], num))
    Xi[0, 0:num] = SOC[0] * np.ones((1, num))
    Xipre = np.zeros((volt_all.shape[0] + 1, num))

    A = np.array([[np.exp(-1 / t), 0], [0, 1]])
    B = np.array([(1 - np.exp(-1 / t)) * RP, -1 / (23 * 3600)])
    SOCi = np.zeros((volt_all.shape[0], num))
    SOCi = SOC[0] * np.ones((1, num))
    I = current
    temp = temp_avg
    OCV = [0] * volt_all.shape[0]
    OCVi = np.zeros((volt_all.shape[0], num))
    DUi = np.zeros((volt_all.shape[0], num))
    Ui = np.zeros((volt_all.shape[0], num))
    Uipre = np.zeros((volt_all.shape[0] + 1, num))
    ei = np.zeros((volt_all.shape[0], num))
    DUt = np.array(volt_di)
    C = np.zeros((volt_all.shape[0], 2))
    U = [0] * volt_all.shape[0]
    Upre = [0] * (volt_all.shape[0] + 1)
    Spre = [0] * (volt_all.shape[0] + 1)
    # Ut = volt_mode
    Ut = volt_modepi
    V = [0] * volt_all.shape[0]
    e = [0] * volt_all.shape[0]

    for k in range(1, volt_all.shape[0] - 1):
        # for k in range(1, 112):
        x[:, k] = np.dot(A, x[:, k - 1]) + B * I.iloc[k - 1]
        # print(temp.iloc[k])
        if x[1, k] > 1:
            x[1, k] = 1
        OCV[k] = b[17] * x[1, k] ** 17 + b[16] * x[1, k] ** 16 + b[15] * x[1, k] ** 15 + b[14] * x[1, k] ** 14 + b[13] * \
                 x[1, k] ** 13 + \
                 b[12] * x[1, k] ** 12 + b[11] * x[1, k] ** 11 + b[10] * x[1, k] ** 10 + b[9] * x[1, k] ** 9 + b[8] * x[
                     1, k] ** 8 + \
                 b[7] * x[1, k] ** 7 + b[6] * x[1, k] ** 6 + b[5] * x[1, k] ** 5 + b[4] * x[1, k] ** 4 + b[3] * x[
                     1, k] ** 3 + \
                 b[2] * x[1, k] ** 2 + b[1] * x[1, k] + b[0] + b[18] * temp.iloc[k] + b[19] * temp.iloc[k] ** 2 + b[
                     20] * temp.iloc[k] ** 3
        C[k, :] = [-1, b[17] * x[1, k] ** 16 * 17 + b[16] * x[1, k] ** 15 * 16 + b[15] * x[1, k] ** 14 * 15 + b[14] * x[
            1, k] ** 13 * 14 +
                   b[13] * x[1, k] ** 12 * 13 + b[12] * x[1, k] ** 11 * 12 + b[11] * x[1, k] ** 10 * 11 + b[10] * x[
                       1, k] ** 9 * 10 +
                   b[9] * x[1, k] ** 8 * 9 + b[8] * x[1, k] ** 7 * 8 + b[7] * 7 * x[1, k] ** 6 + b[6] * 6 * x[
                       1, k] ** 5 +
                   b[5] * 5 * x[1, k] ** 4 + b[4] * 4 * x[1, k] ** 3 + 3 * b[3] * x[1, k] ** 2 + 2 * b[2] * x[1, k] + b[
                       1]]
        U[k] = OCV[k] - x[0, k] - RO * I.iloc[k]
        e[k] = Ut.iloc[k] - U[k]
        rou = 1.2
        beta = 0.15
        gamma = 1.5
        if k == 2:
            V[k] = e[k] * e[k].T
        else:
            V[k] = (rou * V[k - 1] + e[k] * e[k].T) / (1 + rou)
        N = V[k] - beta * R - np.dot(np.dot(C[k, :], Q), (np.dot(C[k, :], Q)).T)
        M = np.dot(np.dot(C[k, :], A), np.dot(np.dot(P, A.T), np.dot(C[k, :], A).T))
        Ek = N / M
        if gamma * Ek > 1 and gamma * Ek < 1.5:
            lambda_k = gamma * Ek
        elif gamma * Ek >= 1.5:
            lambda_k = 1.5
        else:
            lambda_k = 1
        P = lambda_k * (A.dot(P)).dot(A.T) + Q
        K = np.dot(P, C[k, :].T) / (np.dot(C[k, :], np.dot(P, C[k, :].T)) + R)
        x[:, k] = x[:, k] + np.dot(K, e[k])
        P = P - np.dot(np.dot(K, C[k, :]), P)
        xpre[:, k + 1] = np.dot(A, x[:, k]) + B * I.iloc[k]
        Spre[k + 1] = soc[k] + B[1] * I.iloc[k]
        Upre[k + 1] = OCV[k] - xpre[0, k + 1] - RO * I.iloc[k + 1]

        for j in range(volt_all.shape[1]):
            xi[k, j] = xi[k - 1, j]
            Pi[k, j] = Pi[k - 1, j] + Qi
            OCVi[k, j] = b[17] * (xi[k, j] + x[1, k]) ** 17 + b[16] * (xi[k, j] + x[1, k]) ** 16 + b[15] * (
                        xi[k, j] + x[1, k]) ** 15 + b[14] * (xi[k, j] + x[1, k]) ** 14 + b[13] * (
                                     xi[k, j] + x[1, k]) ** 13 + b[12] * (xi[k, j] + x[1, k]) ** 12 + b[11] * (
                                     xi[k, j] + x[1, k]) ** 11 + b[10] * (xi[k, j] + x[1, k]) ** 10 + b[9] * (
                                     xi[k, j] + x[1, k]) ** 9 + b[8] * (xi[k, j] + x[1, k]) ** 8 + b[7] * (
                                     xi[k, j] + x[1, k]) ** 7 + b[6] * (xi[k, j] + x[1, k]) ** 6 + b[5] * (
                                     xi[k, j] + x[1, k]) ** 5 + b[4] * (xi[k, j] + x[1, k]) ** 4 + b[3] * (
                                     xi[k, j] + x[1, k]) ** 3 + b[2] * (xi[k, j] + x[1, k]) ** 2 + b[1] * (
                                     xi[k, j] + x[1, k]) + b[0] + b[18] * temp[k] + b[19] * temp[k] ** 2 + b[20] * temp[
                             k] ** 3;
            if OCVi[k, j] > OCV[k] + 0.1:
                OCVi[k, j] = OCV[k] + 0.1
            Ci = b[17] * (xi[k, j] + x[1, k]) ** 16 * 17 + b[16] * (xi[k, j] + x[1, k]) ** 15 * 16 + b[15] * (
                        xi[k, j] + x[1, k]) ** 14 * 15 + b[14] * (xi[k, j] + x[1, k]) ** 13 * 14 + b[13] * (
                             xi[k, j] + x[1, k]) ** 12 * 13 + b[12] * (xi[k, j] + x[1, k]) ** 11 * 12 + b[11] * (
                             xi[k, j] + x[1, k]) ** 10 * 11 + b[10] * (xi[k, j] + x[1, k]) ** 9 * 10 + b[9] * (
                             xi[k, j] + x[1, k]) ** 8 * 9 + b[8] * (xi[k, j] + x[1, k]) ** 7 * 8 + b[7] * 7 * (
                             xi[k, j] + x[1, k]) ** 6 + b[6] * 6 * (xi[k, j] + x[1, k]) ** 5 + b[5] * 5 * (
                             xi[k, j] + x[1, k]) ** 4 + b[4] * 4 * (xi[k, j] + x[1, k]) ** 3 + 3 * b[3] * (
                             xi[k, j] + x[1, k]) ** 2 + 2 * b[2] * (xi[k, j] + x[1, k]) + b[1]
            DUi[k, j] = OCVi[k, j] - OCV[k] - I[k] * DRi[j]
            Ui[k, j] = U[k] + DUi[k, j]
            Uipre[k + 1, j] = Upre[k + 1] + DUi[k, j]
            ei[k, j] = DUt[k, j] - DUi[k, j]
            Ki = (Pi[k, j] * Ci.T) / ((Ci * Pi[k, j] * Ci.T + Ri))
            xi[k, j] = xi[k, j] + Ki * ei[k, j]
            # Xi[k,j]=x[1,k].T+xi[k,j]
            Xi[k, j] = soc[k].T + xi[k, j]
            Xipre[k + 1, j] = xpre[1, k + 1].T + xi[k, j]
            Pi[k, j] = Pi[k, j] - Ki * Ci * Pi[k, j]
            # deta1[k,j]=1000*(Ui[k,j]-Utt[k,j])
            # deta4[k,j]=100*(Xi[k,j]-SOCi[k,j])
    xipre = np.zeros((xi.shape[0], xi.shape[1]))
    xipre[1:, :] = xi[:xi.shape[0] - 1, :]
    return xipre, xpre, Spre, Upre, Xipre, x, DUi, Xi


def custom_activation(x):
    return 2.5 + 1.8 * torch.sigmoid(x)


def PCA(data, l1, l2):
    # Data standardization
    data_mean = np.mean(data, 0)
    data_std = np.std(data, 0)
    data_nor = (data - data_mean) / data_std
    # Calculate covariance matrix for standardized data
    X = np.cov(data_nor.T)
    # Calculate singular values for covariance matrix
    P, v, P_t = np.linalg.svd(X)  # This function returns three values u s v
    v_ratio = np.cumsum(v) / np.sum(v)
    # Find the index of eigenvalues with a cumulative ratio greater than 0.95
    k = np.where(v_ratio > 0.95)[0]
    # New principal components
    p_k = P[:, :k[0]]
    v_I = np.diag(1 / v[:k[0]])
    # T2 statistic threshold calculation
    coe = k[0] * (np.shape(data)[0] - 1) * (np.shape(data)[0] + 1) / \
        ((np.shape(data)[0] - k[0]) * np.shape(data)[0])
    T_95_limit = coe * stats.f.ppf(0.95, k[0], (np.shape(data)[0] - k[0]))
    T_99_limit = coe * stats.f.ppf(l1, k[0], (np.shape(data)[0] - k[0]))
    # SPE statistic threshold calculation
    O1 = np.sum((v[k[0]:]) ** 1)
    O2 = np.sum((v[k[0]:]) ** 2)
    O3 = np.sum((v[k[0]:]) ** 3)
    h0 = 1 - (2 * O1 * O3) / (3 * (O2 ** 2))
    c_95 = norm.ppf(0.95)
    c_99 = norm.ppf(l2)
    SPE_95_limit = O1 * ((h0 * c_95 * ((2 * O2) ** 0.5) /
                         O1 + 1 + O2 * h0 * (h0 - 1) / (O1 ** 2)) ** (1 / h0))
    SPE_99_limit = O1 * ((h0 * c_99 * ((2 * O2) ** 0.5) /
                         O1 + 1 + O2 * h0 * (h0 - 1) / (O1 ** 2)) ** (1 / h0))
    return v_I, v, v_ratio, p_k, data_mean, data_std, T_95_limit, T_99_limit, SPE_95_limit, SPE_99_limit, P, k, P_t, X, data_nor


def SPE(data_in, data_mean, data_std, p_k):
    test_data_nor = ((data_in - data_mean) / data_std).reshape(len(data_in), 1)
    I = np.eye(len(data_in))
    Q_count = np.dot(np.dot((I - np.dot(p_k, p_k.T)), test_data_nor).T,
                     np.dot((I - np.dot(p_k, p_k.T)), test_data_nor))
    return Q_count
def cont(X_col, data_mean, data_std, X_test, P, num_pc, lamda, T2UCL1):
    X_test = ((X_test - data_mean) / data_std)
    S = np.dot(X_test, P[:, :num_pc])
    r = []
    ee = (T2UCL1 / num_pc)
    for i in range(num_pc):
        aa = S[i] * S[i]
        a = aa / lamda[i, i]
        if a > ee:
            r = np.append(r, i)
    cont = np.zeros((len(r), X_col))
    for i in range(len(r)):
        for j in range(X_col):
            cont[i, j] = np.abs(S[i] / lamda[i, i] * P[j, i] * X_test[j])
    contT = np.zeros(X_col)
    for j in range(X_col):
        contT[j] = np.sum(cont[:, j])
    I = np.eye((np.dot(P, P.T)).shape[0], (np.dot(P, P.T)).shape[1])
    e = np.dot(X_test, (I - np.dot(P, P.T)))
    contQ = np.square(e)
    return contT, contQ
def Ratio_cu(x):
    sum = np.sum(x)
    for i in range(x.shape[0]):
        x[i] = x[i] / sum
    return x
def SlidingAverage_list(s, n):
    mean = []
    if len(s) > n:
        for m in range (n):
            mean.append(np.mean(s[:n]))
        # mean = s[:n].tolist()
        for i in range(n, len(s)):
            select_s = s[i - n: i ]
            mean_s = np.mean(select_s)
            mean.append(mean_s)
    else:
        mean = s.tolist()
    return mean

def SlidingAverage(s, n):
    mean = []
    if len(s) > n:
        for m in range (n):
            mean.append(np.mean(s[:n]))
        for i in range(n, len(s)):
            select_s = s[i - n: i ]
            mean_s = np.mean(select_s)
            mean.append(mean_s)
    else:
        mean = s.tolist()
    return mean
def DiagnosisFeature(ERRORU,ERRORX):
    ERRORUm = pd.DataFrame(ERRORU).max(axis=1)
    ERRORXm = pd.DataFrame(ERRORX).max(axis=1)
    meanu = ERRORU.mean(axis=1)
    meanx = ERRORX.mean(axis=1)
    Z_U = (pd.DataFrame(ERRORU).sub(meanu, axis=0).div(ERRORU.std(axis=1), axis=0)).max(axis=1)
    Z_X = (pd.DataFrame(ERRORX).sub(meanx, axis=0).div(ERRORX.std(axis=1), axis=0)).max(axis=1)
    second_largest_ERRORU = pd.Series(np.apply_along_axis(lambda row: np.partition(row, -2)[-2], axis=1, arr=ERRORU))
    max_diff_ERRORU = (ERRORUm - second_largest_ERRORU).div(ERRORU.std(axis=1), axis=0)
    second_largest_ERRORX = pd.Series(np.apply_along_axis(lambda row: np.partition(row, -2)[-2], axis=1, arr=ERRORX))
    max_diff_ERRORX = (ERRORXm - second_largest_ERRORX).div(ERRORX.std(axis=1), axis=0)
    alpha = 0.2
    Z_U_smoothed = ERRORUm.ewm(alpha=alpha).mean()
    Z_X_smoothed = ERRORXm.ewm(alpha=alpha).mean()
    max_diff_ERRORX = pd.Series(SlidingAverage_list(max_diff_ERRORX , 100))
    Z_X = pd.Series(SlidingAverage_list(Z_X , 100))
    Z_X_smoothed = pd.Series(SlidingAverage_list(Z_X_smoothed , 100))
    df_data = pd.concat([max_diff_ERRORU,max_diff_ERRORX,Z_U,Z_X,Z_U_smoothed,Z_X_smoothed], axis=1)
    return df_data

def ClassifyFeature(temp_max,temp_avg,CONTN,insulation_resistance,threshold1,fai):
    reversed_fai = np.flip(fai[:f_time])
    index_array = np.where(reversed_fai < threshold1)[0]
    index = index_array[0] if index_array.size > 0 else None
    original_index = f_time - index - 1
    temp_dif = temp_max - temp_avg
    features_array = np.empty((0, CONTN.shape[1] + 5))
    for ttime in range(original_index + 1, f_time + 1, 1):
        Feature1 = CONTN[ttime, :]
        f1 = max(fai)
        max_erroru_column = np.argmax(ERRORU[ttime, :])
        max_count_U = np.sum(np.argmax(ERRORU[ttime - 3000:ttime, :], axis=1) == max_erroru_column)
        f2 = max_count_U / 3000
        f3 = temp_dif[f_time - 50:f_time].max()
        f4 = insulation_resistance.iloc[ttime-1000:ttime].min()
        f5 = volt_all.iloc[ttime-100:ttime,:].min().min()
        new_features = np.concatenate((Feature1, np.array([f1, f2, f3, f4, f5])))  
        features_array = np.vstack((features_array, new_features))  
    vin_feature = pd.DataFrame(features_array)
    return vin_feature
