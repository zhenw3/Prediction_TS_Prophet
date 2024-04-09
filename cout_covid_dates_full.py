import pandas as pd
import os
import numpy as np
from prophet import Prophet
import time
from datetime import datetime
import matplotlib.pyplot as plt
from utils import days_between, if_between, date_add, get_weekday, save_df2excel, generate_cn_holidays
from new_prophet import prophet_obj


import warnings
warnings.filterwarnings("ignore")


######################################################## prophet有关参数 ########################################################
def chg_pts_generate():
    chg_pts_setting = ['2017-01-01',
                       '2017-07-01',
                       '2018-01-01',
                       '2018-07-01',
                       '2019-01-01',
                       '2019-07-01',
                       '2020-01-25','2020-03-01','2020-04-01','2020-05-01','2020-06-01','2020-07-01',
                       '2021-01-01','2021-07-01','2022-12-01']
    return chg_pts_setting


def holiday_df_generate():
    ### 输出：节假日
    return generate_cn_holidays()


###################################################### 计算剔除极端值的标准差 #####################################################
### 输出：某个list剔除前后10%极端值后的结果
def lst_percentle(value_lst, p):
    return np.percentile(value_lst, p)
    
def exclude_extreme(value_lst, p):
    lb = lst_percentle(value_lst, p)
    ub = lst_percentle(value_lst, 1 - p)
    return [i for i in value_lst if i <= ub and i >= lb]

def lst_std(value_lst, p = 0.05):
    new_lst = exclude_extreme(value_lst, p)
    return np.std(new_lst)


######################################################## 识别结果整理输出 ########################################################
def idx_pair_2_dates(idx_pair_lst, df):
    ### Step4 输入：异常起止序号  输出：序号对应起止日期
    dates_lst = df['ds'].tolist()
    return [(dates_lst[i[0]], dates_lst[i[1]]) for i in idx_pair_lst]

def dates_pair_2_holidays(dates_pair):
    ### Step5 输入：异常起止日期  输出：Prophet节假日格式异常日期
    covid_lst = []
    ds_lst = []
    lower_lst = []
    upper_lst = []
    for i in range(0, len(dates_pair)):
        covid_lst.append('covid_'+str(i))
        ds_lst.append(dates_pair[i][0])
        lower_lst.append(0)
        upper_lst.append(days_between(dates_pair[i][0], dates_pair[i][1]))
    return pd.DataFrame({'holiday': covid_lst,
                         'ds': ds_lst,
                         'lower_window': lower_lst,
                         'upper_window': upper_lst})


######################################################### 异常识别基本模块 ########################################################
def adjunct_lst(idx_lst):
    ### 输入：[2,10,11]  输出：[1,3,9,12]
    return list(set(([j for j in [i+1 for i in idx_lst] + [i-1 for i in idx_lst] if j not in idx_lst])))


def abnormal_idx(value_lst, thres_rate = 1.6, min_cont_days = 7, abnormal_idxes = []):
    ### Step3  输出新的异常时间坐标，是否相邻扩展，是否新时间段
    ### -------- 参数 -------- ###
    ### value_lst: 残差， thrs_rate: 标准化残差的阈值
    ### min_cont_days: 最小连续异常天数
    ### abnormal_idxes: 已识别异常index上下界
    ### -------- 参数 -------- ###
    ### 返回剔除极端值后的残差  标准差  和  判定为极端值的阈值
    var_std = lst_std(
        [value_lst[i] for i in range(0, len(value_lst)) if i not in abnormal_idxes], 
    0.1)
    threshold = thres_rate * var_std
    #print(abnormal_idxes)
    ### adjunct index of abnormal_idxes
    nbr_idx = adjunct_lst(abnormal_idxes)
    
    ### 连续超过阈值的天数
    ### 累计差异大小
    cont_days = []
    diff_lst = []
    cum_diff = 0
    cum_days = 0
    
    for i in range(0, len(value_lst)):
        if value_lst[i] > threshold:
            cum_days += 1
            cum_diff += value_lst[i]
        else:
            cum_days = 0
            cum_diff = 0
        
        cont_days.append(cum_days)
        diff_lst.append(cum_diff)
    
    ### 从右往左最大累计超过阈值天数
    right_idx = len(value_lst) - 1
    idx_pair = []
    diff_rnk = []
    
    if_nbr = 0
    if_new_area = 0
    
    while right_idx >=0:
        if cont_days[right_idx] >= 1:
            ### 当天是异常值
            if (right_idx in nbr_idx) or ((right_idx - cont_days[right_idx] + 1) in nbr_idx):
                ### 若和已有idx相邻，加入
                idx_pair.append((right_idx - cont_days[right_idx] + 1, right_idx, 0)) ### 0表示是因为相邻被加入
                right_idx = right_idx - max(cont_days[right_idx] - 1, 1)
                if_nbr = 1
                
            elif cont_days[right_idx] >= min_cont_days:
                ### 若不相邻，但连续异常超过7天
                idx_pair.append((right_idx - cont_days[right_idx] + 1, right_idx, diff_lst[right_idx])) ### 差异大小
                diff_rnk.append(diff_lst[right_idx])
                right_idx = right_idx - max(cont_days[right_idx] - 1, 1)
                if_new_area = 1

            else:
                right_idx = right_idx - 1
        else:
            ### 当天不是异常值
            right_idx = right_idx - 1
    
    
    diff_rnk.sort(reverse=True)
    ### 只选择和已识别异常值相邻的  或  “最大”的异常值区域
    idx_pair = [i for i in idx_pair if (i[2]==0 or i[2] in diff_rnk[0:1])]
    #print(idx_pair)
    #print(idx_lst_2_pairs(abnormal_idxes))
    # print(idx_lst_2_pairs(abnormal_idxes))
    # print(idx_pair)
    # print('if_nbr = {0}  if_new_area = {1}'.format(if_nbr, if_new_area))
    res = idx_pair_2_lst(idx_lst_2_pairs(abnormal_idxes) + idx_pair)
    #print(res)
    return res, if_nbr, if_new_area


def idx_lst_2_pairs(idx_lst):
    ### idx_pair_2_lst 的逆过程
    idx_lst.sort()
    tmp = []
    left = 0
    right = 0
    while (right < len(idx_lst)):
        if left + 1 == len(idx_lst):
            tmp.append((idx_lst[left], idx_lst[left]))
            break
        elif right == len(idx_lst) - 1:
            tmp.append((idx_lst[left], idx_lst[right]))
            break
        elif idx_lst[right + 1] - idx_lst[right] == 1:
            right += 1
        else:
            tmp.append((idx_lst[left], idx_lst[right]))
            left = right + 1
            right = right + 1
    return tmp

def idx_pair_2_lst(pair_lst):
    ### 输入：[(1,4),(10,12)]  输出：[1,2,3,4,10,11,12]
    tmp = []
    for i in pair_lst:
        tmp = tmp + list(range(i[0], i[1]+1))
    tmp = list(set(tmp))
    tmp.sort()
    return tmp


def holidays_2_idx_lst(covid_df, dates_lst):
    ### step2 输入：节假日（疫情）及其上下界   输出：(上界,下界) 及其对应的ds序列的坐标对
    idx_lst = []
    if covid_df is not None:
        if covid_df.shape[0] == 0:
            return []
        else:
            for index, row in covid_df.iterrows():
                
                lower_dt = date_add(row['ds'], row['lower_window'])
                upper_dt = date_add(row['ds'], row['upper_window'])
                lower_idx = dates_lst.index(lower_dt)
                upper_idx = dates_lst.index(upper_dt)
                idx_lst.append((lower_idx, upper_idx))
                #print(idx_lst)
            return idx_pair_2_lst(idx_lst)
    else:
        return []


def residual_series(df,detect_start_date,cv_list,covid_dates = None):
    ### step1 输出：预测和实际的gap
    holidays_summary = holiday_df_generate()
    holidays_summary = pd.concat([holidays_summary, covid_dates])

    pf_obj =  prophet_obj(
                    model_df = df, ## 数据df
                    pf_params = {
                        'seasonality_mode': 'additive',
                        'interval_width': 0.8,
                        'holidays': holidays_summary,
                        'daily_seasonality': False,
                        'changepoints':chg_pts_generate()
                    },
                    default_training_start_dt = '2017-01-01', 
                    default_training_end_dt = '2030-01-01',
                    extra_reg_lst = cv_list, ## 回归变量名
                    if_log = True ## y是否为取log
                    )
    
    ### 预测
    fcst_df_tmp = pf_obj.predict(
            jump_off_date = df.ds.max(),
            fcst_stt_date = date_add(detect_start_date,1),
            fcst_end_date = df.ds.max(),
            if_train_again = True,
            if_plot = False,
            default_reg = 'Auto'
            )[['ds','yhat','yhat_lower','yhat_upper']].copy()


    fcst_df_tmp = fcst_df_tmp.loc[
                                    fcst_df_tmp['ds'].apply(lambda x: 
                                                            (
                                                                if_between(x, detect_start_date, df.ds.max()) 
                                                            ))
                                 ]
    
    fcst_df_tmp = df[['ds','y']].merge(fcst_df_tmp, on = ['ds'], how = 'right')
    
    fcst_df_tmp['residual'] = fcst_df_tmp['yhat']/fcst_df_tmp['y'] - 1
    
    return fcst_df_tmp


######################################################### 单个异常时段识别 ########################################################
def cout_covid_dates(df, detect_start_date,cv_list, covid_dates = None,
                     thres_rate = 1.6, min_cont_days = 7):
    ### 输出：单个疫情时间段
    resid_df = residual_series(df,detect_start_date,cv_list, covid_dates)
    
    covid_idx = holidays_2_idx_lst(covid_dates, resid_df['ds'].tolist())
    
    covid_idx, if_nbr, if_new_area = abnormal_idx(resid_df['residual'].tolist(), thres_rate, min_cont_days, covid_idx)
    
    dates_pair = idx_pair_2_dates(idx_lst_2_pairs(covid_idx), resid_df)
    
    covid_holi_df = dates_pair_2_holidays(dates_pair)
    
    return covid_holi_df, if_nbr, if_new_area


######################################################### 循环输出多段异常 ########################################################
def cout_covid_dates_full(df, detect_start_date,cv_list, covid_dates = None,
                     thres_rate = 1.6, min_cont_days = 7, iter_ = 10):
    ### 主要函数，输出：循环多次输出多个疫情时间段（若有）
    print('----------------------- detecting -----------------------')
    areas_detected = 0
    covid_holi_df = covid_dates
    
    while areas_detected <= iter_ :
        print('------ iteration {} ------'.format(areas_detected + 1))
        covid_holi_df, if_nbr, if_new_area = cout_covid_dates(df, detect_start_date, cv_list, covid_dates = covid_holi_df)
        # print(covid_holi_df)
        # print(covid_holi_df)
        if if_new_area:
            areas_detected += 1
        else:
            break
    return covid_holi_df




