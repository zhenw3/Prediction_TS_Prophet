import pandas as pd
import os
import numpy as np
import time
import math
from datetime import datetime
import matplotlib.pyplot as plt
from utils import days_between, if_between, date_add, generate_cn_holidays, generate_dates_info, nvl, get_weekday
import gc

import warnings
warnings.filterwarnings("ignore")

gc.collect()


def get_week_adj_param(df, train_indicator = 'if_train', decay = 0.95):
	##### 输入 #####
	## 待校正df
	## train_indicator：标志哪一段是训练集
	## decay：时间加权衰减系数，给近期数据更高权重。越小，衰减越快
	##### 输出 #####
	## 近半年平均周日比非周日高估多少
	if train_indicator in df.columns:
		df = df.loc[df[train_indicator]==1].copy()
	else:
		pass
	df = df.sort_values(by = ['ds'], ascending = True)
	df = df.iloc[-182:].copy()
	df['week'] = [i//7 for i in range(df.shape[0], 0, -1)]
	res = df.groupby(['week'], as_index = False).apply(lambda x: pd.Series({
		'if_fes_covid': (x['if_fes'].sum() + x['if_covid'].sum()),
		'sunday_error': (x['yhat']*(x['weekday']==7)).sum()/(x['y']*(x['weekday']==7)).sum() - 1,
		'weekday_error': (x['yhat']*(x['weekday']!=7)).sum()/(x['y']*(x['weekday']!=7)).sum() - 1,
	}))
	res = res.query("if_fes_covid == 0")
	if res.shape[0] < 4:
		return 0 ### 如果小于4周数据，则代表样本量不够，不进行校准
	else:
		res['weight'] = res['week'].apply(lambda x: decay**x)
		res['weight'] = res['weight']/(res['weight'].sum()) ## 保证相加 = 1
		week_adj_param = (res['weight']*(res['sunday_error'] - res['weekday_error'])).sum()
		# print(week_adj_param)
		
		return week_adj_param ## 周日相比平时高估多少


def adj_weekdays(var_lst, weekdays, adj_factors):
	### var_lst: 待校正list
	### weekdays: 对应的星期
	### adj_factors: 表示周一到周六 应该相对 比周日增加的%
	# adj_factors = max(min(adj_factors, 0.15), 0) ### adj_factors 在 0 和 0.15 之间
	
	sunday_sum = sum([var_lst[i] if weekdays[i] == 7 else 0 for i in range(len(weekdays))])
	other_sum = sum([var_lst[i] if weekdays[i] != 7 else 0 for i in range(len(weekdays))])
	
	#s * (1 + x) + o * (1 + x + adj) = s + o
	#sx + o(x + adj) = 0
	#(s + o)x = -o adj
	#x = -(adj*o)/(s+o)
	s_adj = -(adj_factors * other_sum)/(other_sum + sunday_sum)
	o_adj = (adj_factors * sunday_sum)/(other_sum + sunday_sum)
	
	return [var_lst[i]*(1+s_adj) if weekdays[i] == 7 else var_lst[i]*(1+o_adj) for i in range(len(weekdays))]


def df_join_holiday(df, holiday_df):
	pred_df = df.copy()

	holiday_df = nvl(holiday_df, generate_cn_holidays())

	t_holiday_df = generate_dates_info(holiday_df)[['ds', 'if_fes', 'if_covid', 'weekday', 'if_week_fes']]

	### 预测值和真实值df，关联日期相关信息
	pred_df = pred_df.merge(t_holiday_df, on = ['ds'], how = 'left')

	return pred_df



class ratio_calib(object):
	def __init__(
				self,
				adj_level = 'm',
				if_adj_level = True, 
				if_adj_tradition = True, 
				if_adj_tradition_judge = True,
				week_adj = None, 
				level_adj_param = None
				):
		
		self.adj_level = adj_level
		self.if_adj_level = if_adj_level
		self.if_adj_tradition = if_adj_tradition
		self.if_adj_tradition_judge = if_adj_tradition_judge
		self.week_adj = week_adj
		self.level_adj_param = level_adj_param


	def get_pre_this_mth_dts(self, jumpoff_dt, start_dt, end_dt):
		### get dates 28 days before jumpoff and dates between start and end --  for the past 5 years
		pre_mth_dates = [str(i)[0:10] for i in pd.date_range(start = date_add(jumpoff_dt, -27), end = jumpoff_dt)]\
		+ [str(i)[0:10] for i in pd.date_range(start = date_add(jumpoff_dt,-365-27), end = date_add(jumpoff_dt, -365))]\
		+ [str(i)[0:10] for i in pd.date_range(start = date_add(jumpoff_dt,-365*2-27), end = date_add(jumpoff_dt, -365*2))]\
		+ [str(i)[0:10] for i in pd.date_range(start = date_add(jumpoff_dt,-365*3-27), end = date_add(jumpoff_dt, -365*3))]\
		+ [str(i)[0:10] for i in pd.date_range(start = date_add(jumpoff_dt,-365*4-27), end = date_add(jumpoff_dt, -365*4))]\
		+ [str(i)[0:10] for i in pd.date_range(start = date_add(jumpoff_dt,-365*5-27), end = date_add(jumpoff_dt, -365*5))]
		
		pre_mth_year = [0]*28 + [1]*28 + [2]*28 + [3]*28 + [4]*28 + [5]*28
		week = [4]*7 + [3]*7 + [2]*7 + [1]*7

		this_mth_dates = [str(i)[0:10] for i in pd.date_range(start = date_add(start_dt, 0), end = date_add(end_dt, 0))]\
		+ [str(i)[0:10] for i in pd.date_range(start = date_add(start_dt, -365), end = date_add(end_dt, -365))]\
		+ [str(i)[0:10] for i in pd.date_range(start = date_add(start_dt, -365*2), end = date_add(end_dt, -365*2))]\
		+ [str(i)[0:10] for i in pd.date_range(start = date_add(start_dt, -365*3), end = date_add(end_dt, -365*3))]\
		+ [str(i)[0:10] for i in pd.date_range(start = date_add(start_dt, -365*4), end = date_add(end_dt, -365*4))]\
		+ [str(i)[0:10] for i in pd.date_range(start = date_add(start_dt, -365*5), end = date_add(end_dt, -365*5))]
		
		len_st_ed = days_between(start_dt, end_dt) + 1
		
		this_mth_year = [0]*len_st_ed + [1]*len_st_ed + [2]*len_st_ed + [3]*len_st_ed + [4]*len_st_ed + [5]*len_st_ed
		
		return pd.DataFrame({'ds': pre_mth_dates, 'pre_mth_year': pre_mth_year, 'week': week * 6}), pd.DataFrame({'ds': this_mth_dates, 'this_mth_year': this_mth_year})


	def get_mthly_avg_info(self, df):
		jumpoff_dt = df.query("if_train == 1")['ds'].max()
		start_dt = df.query("if_pred == 1")['ds'].min()
		end_dt = df.query("if_pred == 1")['ds'].max()

		pre_mth_dates, this_mth_dates = self.get_pre_this_mth_dts(jumpoff_dt, start_dt, end_dt)
		#print(pre_mth_dates)
		#print(this_mth_dates)
		# df = df.loc[df['ds']<=jumpoff_dt]
		df = df.merge(pre_mth_dates, how = 'left', on = ['ds'])
		df = df.merge(this_mth_dates, how = 'left', on = ['ds'])
		
		#print(df.tail(40))

		pre_mth_info = df.loc[~df['pre_mth_year'].isnull()].groupby(['pre_mth_year'], as_index = False).apply(lambda x: pd.Series({
						"7d_ma_max": (x["y"].rolling(window=7, min_periods=7).mean()).max(),
						"7d_ma_min": (x["y"].rolling(window=7, min_periods=7).mean()).min(),
						"7d_ma_range": (x["y"].rolling(window=7, min_periods=7).mean()).max() - \
						(x["y"].rolling(window=7, min_periods=7).mean()).min(),
						"week1_error": (x['y'] * (x['week'].apply(lambda x: 1 if x==1 else np.nan))).mean() 
										- (x['yhat'] * (x['week'].apply(lambda x: 1 if x==1 else np.nan))).mean(),
						"week2_error": (x['y'] * (x['week'].apply(lambda x: 1 if x==2 else np.nan))).mean() 
										- (x['yhat'] * (x['week'].apply(lambda x: 1 if x==2 else np.nan))).mean(),
						"week3_error": (x['y'] * (x['week'].apply(lambda x: 1 if x==3 else np.nan))).mean() 
										- (x['yhat'] * (x['week'].apply(lambda x: 1 if x==3 else np.nan))).mean(),
						"week4_error": (x['y'] * (x['week'].apply(lambda x: 1 if x==4 else np.nan))).mean() 
										- (x['yhat'] * (x['week'].apply(lambda x: 1 if x==4 else np.nan))).mean(),
		}))
		
		df['y'] = df['y'].fillna(0)
		this_mth_info = df.loc[~df['this_mth_year'].isnull()].groupby(['this_mth_year'], as_index = False).apply(lambda x: pd.Series({
						"7d_ma_max": ((x['yhat']*(x['this_mth_year']==0) + x['y']*(x['this_mth_year']!=0)).rolling(window=7, min_periods=7).mean()).max(),
						"7d_ma_min": ((x['yhat']*(x['this_mth_year']==0) + x['y']*(x['this_mth_year']!=0)).rolling(window=7, min_periods=7).mean()).min(),
						"7d_ma_range": ((x['yhat']*(x['this_mth_year']==0) + x['y']*(x['this_mth_year']!=0)).rolling(window=7, min_periods=7).mean()).max() - \
						((x['yhat']*(x['this_mth_year']==0) + x['y']*(x['this_mth_year']!=0)).rolling(window=7, min_periods=7).mean()).min(),
		}))
		return pre_mth_info, this_mth_info

	def adj_fcst_level_rr(self, pre_mth_info):
		### 若存在系统性高估/低估，修正
		w1e = pre_mth_info.loc[pre_mth_info['pre_mth_year']==0]['week1_error'].iloc[0]
		w2e = pre_mth_info.loc[pre_mth_info['pre_mth_year']==0]['week2_error'].iloc[0]
		w3e = pre_mth_info.loc[pre_mth_info['pre_mth_year']==0]['week3_error'].iloc[0]
		w4e = pre_mth_info.loc[pre_mth_info['pre_mth_year']==0]['week4_error'].iloc[0]
		
		value_lst = [w1e, w2e, w3e, w4e]
		sign_lst = [int(w1e/abs(w1e)), int(w2e/abs(w2e)), int(w3e/abs(w3e)), int(w4e/abs(w4e))]
		
		if sign_lst == [1,1,1,1]:
			return (w1e + w2e + w3e + w4e)/4
		if (sign_lst == [1,1,-1,1]) or (sign_lst == [1,-1,1,1]) or (sign_lst == [1,1,1,-1]):
			return sum([value_lst[i] for i in range(len(value_lst)) if sign_lst[i]==1])/sum([1 for i in range(len(value_lst)) if sign_lst[i]==1])*0.8
		if (sign_lst == [-1,-1,1,-1]) or (sign_lst == [-1,1,-1,-1]) or (sign_lst == [-1,-1,-1,1]):
			return sum([value_lst[i] for i in range(len(value_lst)) if sign_lst[i]==-1])/sum([1 for i in range(len(value_lst)) if sign_lst[i]==-1])*0.8
		if (sign_lst == [1,-1,-1,-1]) or (sign_lst == [-1,1,1,1]):
			return sum(value_lst[1:])/3*0.5
		else:
			return 0

	def judge_if_adj_tradition(self, pre_mth_info):
		this_year_max = pre_mth_info.loc[pre_mth_info['pre_mth_year']==0]['7d_ma_max'].iloc[0]
		this_year_min = pre_mth_info.loc[pre_mth_info['pre_mth_year']==0]['7d_ma_min'].iloc[0]
		this_year_range = pre_mth_info.loc[pre_mth_info['pre_mth_year']==0]['7d_ma_range'].iloc[0]
		
		if_adj_max = True
		if_adj_min = True
		if_adj_range = True
		
		if pre_mth_info['7d_ma_max'].max() == this_year_max:
			### 今年创新高
			if_adj_max = False
		if pre_mth_info['7d_ma_min'].min() == this_year_min:
			### 今年创新低
			if_adj_min = False
		if pre_mth_info['7d_ma_range'].max() == this_year_range:
			### 今年范围创新高
			if_adj_range = False
		return if_adj_max, if_adj_min, if_adj_range

	def get_fcst_period_min_max_range(self, this_mth_info):
		pred_max = this_mth_info.loc[this_mth_info['this_mth_year']==0]['7d_ma_max'].iloc[0]
		pred_min = this_mth_info.loc[this_mth_info['this_mth_year']==0]['7d_ma_min'].iloc[0]
		pred_range = this_mth_info.loc[this_mth_info['this_mth_year']==0]['7d_ma_range'].iloc[0]
		hist_max = this_mth_info.loc[this_mth_info['this_mth_year']!=0]['7d_ma_max'].max()
		hist_min = this_mth_info.loc[this_mth_info['this_mth_year']!=0]['7d_ma_min'].min()
		hist_range = this_mth_info.loc[this_mth_info['this_mth_year']!=0]['7d_ma_range'].max()
		return pred_max, pred_min, pred_range, hist_max, hist_min, hist_range

	
	def adj_coef(self, pred_max, pred_min, hist_max, hist_min, hist_range_max,
				if_adj_max = True,
				if_adj_min = True,
				if_adj_range = True):
		m = pred_max
		n = pred_min
		
		if pred_max > hist_max:
			## 调整上界
			if if_adj_max:
				m = hist_max
			else:
				pass
			if ((pred_max - pred_min) > hist_range_max) and (if_adj_range):
				## 调整范围
				n = m - hist_range_max
			else:
				n = m - (pred_max - pred_min)
		elif pred_min < hist_min:
			## 调整下界
			if if_adj_min:
				n = hist_min
			else:
				pass
			if ((pred_max - pred_min) > hist_range_max) and (if_adj_range):
				## 调整范围
				m = n + hist_range_max
			else:
				m = n + (pred_max - pred_min)
		else:
			pass

		return (m - n)/(pred_max - pred_min), (n*pred_max - m*pred_min)/(pred_max - pred_min)


	def adj_coef_rr(self, 
		   df, 
		   adj_level = None,
		   if_adj_level = True, 
		   if_adj_tradition = True, 
		   if_adj_tradition_judge = True
		   ):
		adj_level = nvl(adj_level, self.adj_level)
		jumpoff_dt = df.query("if_train == 1")['ds'].max()
		start_dt = df.query("if_pred == 1")['ds'].min()
		end_dt = df.query("if_pred == 1")['ds'].max()
		
		a = 1 
		b = 0 

		pre_mth_info, this_mth_info = self.get_mthly_avg_info(df)
		pred_max, pred_min, pred_range, hist_max, hist_min, hist_range = self.get_fcst_period_min_max_range(this_mth_info)
		level_shift_param, if_adj_max, if_adj_min, if_adj_range = 0, 0, 0, 0

		### level校准
		if if_adj_level:
			level_shift_param = self.adj_fcst_level_rr(pre_mth_info)
			self.orig_level_shift_param = level_shift_param
			#print('11111111111111111111 {} 1111111111111111111'.format(level_shift_param))
			### 根据校准强度调整
			if adj_level == 'h':
				level_shift_param = level_shift_param * 0.9
			elif adj_level == 'm':
				level_shift_param = level_shift_param * 0.5
			elif adj_level == 'l':
				level_shift_param = level_shift_param * 0.2
			elif adj_level == 'n':
				level_shift_param = level_shift_param * 0
			else:
				level_shift_param = level_shift_param * 0
		### 范围校准
		if if_adj_tradition:
			#print(this_mth_info)
			#print('222222222222222 {0} {1} {2} {3} {4} {5} 2222222222222'.format(pred_max, pred_min, pred_range, hist_max, hist_min, hist_range))
			if_adj_max, if_adj_min, if_adj_range = self.judge_if_adj_tradition(pre_mth_info)
		## print('333333333333333 {}')

		if if_adj_level:
			b = level_shift_param
			pred_max = pred_max + level_shift_param
			pred_min = pred_min + level_shift_param

		if if_adj_tradition:
			if if_adj_tradition_judge:
				### a斜率，b截距
				a1, b1 = self.adj_coef(pred_max, pred_min, hist_max, hist_min, hist_range,
									 if_adj_max, if_adj_min, if_adj_range)
			else:
				a1, b1 = self.adj_coef(pred_max, pred_min, hist_max, hist_min, hist_range)
			a = a*a1
			b = b*a1 + b1

		return a, b

	def adj_fcst(self, 
				df,  ### 需要校准的预测结果
				holiday_df = None,  ### 节假日信息
				y_adj = 'y_adj', y_real = 'y', y_fcst = 'yhat', 
				adj_level = None,
				if_adj_level = None, 
				if_adj_tradition = None, 
				if_adj_tradition_judge = None,
				week_adj = None, 
				level_adj_param = None
				):
		### 待校准结果-关联节假日
		df = df_join_holiday(df, holiday_df).copy()
		
		### 校准配置
		adj_level = nvl(adj_level, self.adj_level)
		if_adj_level = nvl(if_adj_level, self.if_adj_level)
		if_adj_tradition = nvl(if_adj_tradition, self.if_adj_tradition)
		if_adj_tradition_judge = nvl(if_adj_tradition_judge, self.if_adj_tradition_judge)

		### 星期校准系数
		week_adj = nvl(week_adj, self.week_adj)
		if week_adj == False:
			week_adj_param = 0 
		else:
			week_adj_param = nvl(week_adj, get_week_adj_param(df))
		self.week_adj_param = week_adj_param

		### 均值校准系数
		if level_adj_param is not None:
			a = 1 
			b = level_adj_param
		else:
			a, b = self.adj_coef_rr(df, adj_level, if_adj_level, if_adj_tradition, if_adj_tradition_judge)
		self.a, self.b = a, b
		adj_lst = df.loc[df['if_pred'] == 1][y_fcst].tolist()
		
		df[y_adj] = df[y_fcst]
		df.loc[df['if_pred'] == 1, y_adj] = [a*i + b for i in adj_lst]

		### 星期校准：只在非节假日周校准
		var_lst = df.loc[(df['if_pred'] == 1) & (df['if_week_fes']==0)][y_adj].tolist()
		weekdays = df.loc[(df['if_pred'] == 1) & (df['if_week_fes']==0)]['weekday'].tolist()
		adj_lst2 = adj_weekdays(var_lst, weekdays, week_adj_param)
		df.loc[(df['if_pred'] == 1) & (df['if_week_fes']==0), y_adj] = adj_lst2

		return df



class cnt_calib(object):
	def __init__(
				self,
				adj_level = 'm',
				week_adj = None,
				level_adj_param = None,
				decays = 0.8
				):
		
		self.adj_level = adj_level
		self.week_adj = week_adj
		self.level_adj_param = level_adj_param
		self.decays = decays

	### 输出level调整系数
	def level_shift_param(self, df, y_real = 'y', y_fcst = 'yhat', decays = 0.8, adj_level = 'm'):
		### 如果有关于疫情的标识
		df = df.query("if_covid == 0").copy()
		
		# decay_rates = 1/(decays + decays**2 + decays**3 + decays**4)
		if df.shape[0] >= 7:
			error_df = df.groupby(['weeks'], as_index = False).apply(lambda x: pd.Series({
				'days': x['ds'].count(),
				'actual': x[y_real].sum(),
				'pred': x[y_fcst].sum(),
				'error': x[y_real].sum()/x[y_fcst].sum() - 1
			}))

			# error_df['weight'] = (error_df['weeks'].apply(lambda x: (decays**x) * decay_rates))* (error_df['days'].apply(lambda x: x/7))
			error_df['weight'] = (error_df['weeks'].apply(lambda x: (decays**x))) * error_df['actual']
			error_df['weight'] = error_df['weight']/(error_df['weight'].sum()) ### 归一化
			error_df['weight'] = error_df['weight'] * (error_df['days'].apply(lambda x: x/7))

			adj_factor_orig = (error_df['weight'] * error_df['error']).sum() 

			confidence = 0.1

			adj_descent = []
			if adj_level == 'h': 
				adj_descent = [1, 1, 1, 0.95, 0.9]
			elif adj_level == 'm':
				adj_descent = [1, 0.9, 0.8, 0.6, 0.4]
			else:
				adj_descent = [0.9, 0.7, 0.5, 0.3, 0.1]

			if error_df.shape[0]<=1:
				pass
			else:
				width = error_df['error'].max() - error_df['error'].min()
				ave = abs(error_df['error'].mean())
				if width < ave:
					confidence = adj_descent[0] * error_df.shape[0]/4
				elif width < 1.5 * ave:
					confidence = adj_descent[1] * error_df.shape[0]/4
				elif width < 2 * ave:
					confidence = adj_descent[2] * error_df.shape[0]/4
				elif width < 3 * ave:
					confidence = adj_descent[3] * error_df.shape[0]/4
				else:
					confidence = adj_descent[4] * error_df.shape[0]/4

			return adj_factor_orig * confidence
		
		else:
			return 0


	### 输入：预测序列 + 预测序列前后30天（如有）
	### 输出：level shift系数，星期调整系数
	def get_adj_params(self, df, y_real = 'y', y_fcst = 'yhat', train_indicator = 'if_train', decays = 0.8, if_post = False, adj_level = 'm'):
		### df应包括：ds日期，weekday星期，y实际，y预测，adj_indicator需要调整的序列标识
		### 是否节假日indicator，是否covid indicator
		jumpoff_dt = df.loc[df[train_indicator]==1]['ds'].max()
		
		### 当前仅用前28天数据
		pre_df = df.loc[df['ds'].apply(lambda x: if_between(x, date_add(jumpoff_dt, -27), jumpoff_dt))].copy()
		pre_df['days'] = np.arange(pre_df.shape[0], 0, -1)
		pre_df['weeks'] = pre_df['days'].apply(lambda x: int((x-1)/7) + 1)
		
		level_adj_pre = self.level_shift_param(pre_df, y_real, y_fcst, decays, adj_level)
		level_adj_post = level_adj_pre
		
		############ 疑似无用 ############
		if if_post:
			post_df = fcst.loc[fcst['ds'].apply(lambda x: if_between(x, date_add(fcst_end, 1), 
																	 date_add(fcst_start, 28)))].copy()
			post_df['days'] = np.arange(1, post_df.shape[0]+1, 1)
			post_df['weeks'] = post_df['days'].apply(lambda x: int((x-1)/7) + 1)
		############ 疑似无用 ############
		
		return level_adj_pre, level_adj_post


	def adj_fcst_level(self, var_lst, start_adj, end_adj):
		adj_var = []
		lst_len = len(var_lst)
		for i in range(0, lst_len):
			adj_param = 1 + start_adj + ((end_adj - start_adj)/(lst_len - 1)) * i
			adj_var.append(var_lst[i] * adj_param)
		return adj_var


	### 最终函数
	def adj_fcst(self, 
				df, ### 需要校准的预测结果
				holiday_df = None, ### 节假日信息
				y_adj = 'y_adj', y_real = 'y', y_fcst = 'yhat', ### 要校准的是哪些列，结果存到y_adj
				adj_indicator = None, train_indicator = None, ### 标志哪些是训练集，哪些是验证集
				level_adj_param = None,
				decays = 0.8, ### 时间权重
				week_adj = None, ### 星期校准系数
				adj_level = None, ### h、m、l几档
				if_post = False):
		
		### 待校准结果-关联节假日
		df = df_join_holiday(df, holiday_df).copy()
		### 列名
		adj_indicator = nvl(adj_indicator, 'if_pred')
		train_indicator = nvl(train_indicator, 'if_train')

		### 是否有输入的均值校准系数
		level_adj_param = nvl(level_adj_param, self.level_adj_param)

		### 时间衰减系数
		decays = nvl(decays, self.decays)
		
		### 星期校准系数
		week_adj = nvl(week_adj, self.week_adj)
		if week_adj == False:
			week_adj_param = 0 
		else:
			week_adj_param = nvl(week_adj, get_week_adj_param(df, train_indicator))
		self.week_adj_param = week_adj_param

		### 均值校准强度
		adj_level = nvl(nvl(adj_level, self.adj_level), 'l')

		df = df[['ds', 'yhat', 'yhat_upper', 'yhat_lower', 'y', 'weekday', 'if_fes', 'if_covid', 'if_week_fes', adj_indicator, train_indicator]]
		
		### 均值校准系数
		if level_adj_param is None:
			start_adj, end_adj = self.get_adj_params(df, y_real, y_fcst, train_indicator, decays, if_post, adj_level)
			level_adj_param = start_adj
			self.level_adj_param = level_adj_param
		else:
			start_adj = level_adj_param
			end_adj = level_adj_param
			self.level_adj_param = level_adj_param
		

		### 初始化校准列
		df[y_adj] = df[y_fcst]

		### 均值校准
		var_lst = df.loc[(df[adj_indicator] == 1)][y_fcst].tolist()
		adj_lst = self.adj_fcst_level(var_lst, start_adj, end_adj)
		df.loc[df[adj_indicator] == 1, y_adj] = adj_lst

		### 星期校准：只在非节假日周校准
		var_lst = df.loc[(df[adj_indicator] == 1) & (df['if_week_fes']==0)][y_adj].tolist()
		weekdays = df.loc[(df[adj_indicator] == 1) & (df['if_week_fes']==0)]['weekday'].tolist()
		adj_lst = adj_weekdays(var_lst, weekdays, week_adj_param)
		df.loc[(df[adj_indicator] == 1) & (df['if_week_fes']==0), y_adj] = adj_lst

		return df 





