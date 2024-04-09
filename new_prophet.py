import pandas as pd
import os
import numpy as np
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
import time
import math
from datetime import datetime
import matplotlib.pyplot as plt
from utils import days_between, if_between, nvl, date_add
from tqdm import trange

import warnings
warnings.filterwarnings("ignore")



########################################################################################### 其他相关函数
def stan_init(m):
    """Retrieve parameters from a trained model.
    
    Retrieve parameters from a trained model in the format
    used to initialize a new Stan model.
    
    Parameters
    ----------
    m: A trained model of the Prophet class.
    
    Returns
    -------
    A Dictionary containing retrieved parameters of m.
    
    """
    res = {}
    for pname in ['k', 'm', 'sigma_obs']:
        res[pname] = m.params[pname][0][0]
    for pname in ['delta', 'beta']:
        res[pname] = m.params[pname][0]
    return res

class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])

############################################################# Prophet类，方便后续使用 #############################################################
class prophet_obj(object):
    ################################ 初始化 ################################
    def __init__(
                self,
                model_df, ## 数据df: must contain y and yhat
                pf_params,
                ### prophet 模型常用参数设置：
                ### seasonality_mode: 'additive'、'multiplicative'
                ### holidays：节假日，df格式
                ### holidays_prior_scale：节假日影响大小，10，一般不会过拟合
                ### changepoint_range：用（1 - changepoint_range）的数据来估计近期趋势，0.8
                ### changepoints：a list of 趋势变化节点
                ### changepoint_prior_scale：自动拟合趋势的复杂度（changepoints才有效），0.05
                ### seasonality_prior_scale：季节性拟合复杂度，10，一般不会过拟合
                ### interval_width：置信区间，0.8
                ### growth：'linear'、'losgistic'
                default_training_start_dt = '2000-01-01', ## 训练集起始日期
                default_training_end_dt = '2030-01-01', ## 训练集结束日期
                if_log = False, ## y是否先取log，再建模
                extra_reg_lst = [],
                ):
        
        self.model_df = model_df.copy()
        self.pf_params = pf_params
        self.default_training_start_dt = default_training_start_dt
        self.default_training_end_dt = default_training_end_dt
        self.if_log = if_log
        self.extra_reg_lst = extra_reg_lst
        if if_log:
            self.model_df['y_orig'] = self.model_df['y']
            self.model_df['y'] = self.model_df['y'].apply(lambda x: np.log(x))
        self.pm = None
        self.trained_times = 0 
    

    ################################ 模型初始化 ################################    
    def init_model(self):
        with suppress_stdout_stderr():
            self.pm = Prophet(**self.pf_params)
        if len(self.extra_reg_lst)>=1:
            for i in self.extra_reg_lst:
                self.pm.add_regressor(i)
        
    
    ################################ 模型训练 ################################
    def train_model(
                self,
                training_start_dt = None, ## 训练集开始日期
                training_end_dt = None, ## 训练集结束日期
                ):

        training_start_dt = nvl(training_start_dt, self.default_training_start_dt)
        training_end_dt = nvl(training_end_dt, self.default_training_end_dt)
        
        df_train = self.model_df.loc[self.model_df['ds'].apply(lambda x: if_between(x, training_start_dt, training_end_dt))]
        
        ################# 如果是手动设置趋势变化节点，则限制训练集内最后一个趋势变化节点要在至少30日以前，保证最近趋势估计有至少30+样本 #################
        if 'changepoints' in self.pf_params:
            if len(self.pf_params['changepoints'])>0:
                chg_pts_for_training = [i for i in self.pf_params['changepoints'] if days_between(i, training_end_dt) > 30]
                chg_pts_for_training = [i for i in chg_pts_for_training if days_between(training_start_dt, i) >= 0]
                self.pf_params['changepoints'] = chg_pts_for_training
            else:
                chg_pts_for_training = []
        ##################################################################################################################################
        
        ### Prophet doesn't allow train multiple times
        self.init_model()

        self.pm.fit(df_train)
        self.trained_times += 1
        
        # return self.pm.copy()
    
    ###################### 如果有协变量，计算协变量历史均值 ######################
    def get_default_reg(
                self,
                jump_off_date,
                period = 60 ## 默认计算前60天均值
                ):
        train_len = max(period, 1)
        # print('---------Default reg calc length = {}----------'.format(train_len))
        # print('---------Default jump off date = {}----------'.format(jump_off_date))
        res = (
            self.model_df.loc[
                self.model_df['ds'].apply(
                    lambda x: if_between(x, date_add(jump_off_date, - train_len), jump_off_date)
                )
            ][self.extra_reg_lst]
        ).mean()
        res = res.tolist()
        
        return res

    ################################ 模型预测 ################################
    def predict(
                self,
                training_start_dt = None, ## 训练集开始日期
                jump_off_date = None, ## 训练集结束日期
                fcst_stt_date = None, ## 预测开始日期
                fcst_end_date = None, ## 预测结束日期
                default_reg = [0, 0], ## 如果协变量为控制，则fillna，手动设置，或自动根据历史平均计算
                if_train_again = True, ## 否：重新训练模型；是：用已有模型预测
                if_keep_hist = True, ## 是否保留训练数据
                if_plot = False, ## 是否画prophet自带时序拆解图
                ):

        training_start_dt = nvl(training_start_dt, self.default_training_start_dt)
        jump_off_date = nvl(jump_off_date, self.default_training_end_dt)
        fcst_stt_date = nvl(fcst_stt_date, date_add(jump_off_date, 1))
        fcst_end_date = nvl(fcst_end_date, date_add(jump_off_date, 30))
        
        ################################ 是否重新训练 ################################
        if if_train_again or (self.trained_times == 0):
            self.train_model(
                training_start_dt = training_start_dt, ## 训练集开始日期
                training_end_dt = jump_off_date, ## 训练集结束日期
            )
        else:
            print('\n------------- Using Existing Model for Forecasting -------------\n')
        ################################ 是否重新训练 ################################

        
        ################################### 预测 ####################################
        forecast_len = days_between(jump_off_date, fcst_end_date) ## 从 jump off date 到 end date 的长度
        future_dts = self.pm.make_future_dataframe(periods = forecast_len)
        future_dts['ds'] = future_dts['ds'].apply(lambda x: str(x)[0:10])
        future_dts = (
            future_dts.merge(
                self.model_df[['ds'] + self.extra_reg_lst],
                on = ['ds'],
                how = 'left'
            )
        )
        if (default_reg == 'Auto') or (default_reg is None):
            default_reg = self.get_default_reg(jump_off_date = jump_off_date)
        if isinstance(default_reg, int):
            default_reg = self.get_default_reg(period = default_reg, jump_off_date = jump_off_date)
        
        for i in range(len(self.extra_reg_lst)):
            future_dts[self.extra_reg_lst[i]] = future_dts[self.extra_reg_lst[i]].fillna(default_reg[i]) ## 若为oot，则填补回归变量

        forecast = self.pm.predict(future_dts)
        
        forecast['ds'] = forecast['ds'].apply(lambda x: str(x)[0:10])

        ### 如果取log则exp
        if self.if_log:
            forecast[[col for col in forecast.columns if col != 'ds']] = forecast[[col for col in forecast.columns if col != 'ds']].applymap(lambda x: np.exp(x))
        
        ### 和真实y关联
        if self.if_log:    
            forecast = forecast.merge(self.model_df[['ds', 'y_orig']], on = ['ds'], how = 'left')
            forecast = forecast.rename(columns = {'y_orig': 'y'})
        else: 
            forecast = forecast.merge(self.model_df[['ds', 'y']], on = ['ds'], how = 'left')


        ################################### 预测 ####################################

        ### 标识样本
        forecast['if_train'] = forecast['ds'].apply(lambda x: 1 if if_between(x, training_start_dt, jump_off_date) else 0)
        forecast['if_pred'] = forecast['ds'].apply(lambda x: 1 if if_between(x, fcst_stt_date, fcst_end_date) else 0)

        ### 是否保留历史数据
        if if_keep_hist:
            ### 保留历史数据
            if type(if_keep_hist) is int:
                forecast = forecast.loc[forecast['ds'] >= date_add(jump_off_date, if_keep_hist)]
            else:
                pass
        else:
            ### 去除历史数据
            forecast = forecast.loc[forecast['ds'] >= fcst_stt_date]


        self.forecast = forecast.copy()

        if if_plot:
            self.plot_ts_components()

        return forecast
        
    
    ################################ 画时序拆解图 ################################
    def plot_ts_components(self):
        self.forecast['ds'] = self.forecast['ds'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
        fig = self.pm.plot(self.forecast)
        a = add_changepoints_to_plot(fig.gca(), self.pm, self.forecast)
        fig2 = self.pm.plot_components(self.forecast)
        self.forecast['ds'] = self.forecast['ds'].apply(lambda x: str(x)[0:10])


    