from datetime import datetime
import pandas as pd
import xlsxwriter
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties
plt.rcParams['font.family'] = ['Arial Unicode MS']

# def __init__(
#             self,
#             growth='linear',
#             changepoints=None,
#             n_changepoints=25,
#             changepoint_range=0.8,
#             yearly_seasonality='auto',
#             weekly_seasonality='auto',
#             daily_seasonality='auto',
#             holidays=None,
#             seasonality_mode='additive',
#             seasonality_prior_scale=10.0,
#             holidays_prior_scale=10.0,
#             changepoint_prior_scale=0.05,
#             mcmc_samples=0,
#             interval_width=0.80,
#             uncertainty_samples=1000,
#             stan_backend=None
#     )

########################################################################################### 其他函数
def nvl(value, replace_value):
    if value is None:
        return replace_value
    else:
        return value


def save_df2excel(filename,sheetname,dataframe):
    ### save an pd.df to an excel worksheet
    with pd.ExcelWriter(filename, engine='openpyxl', mode='a') as writer: 
        workBook = writer.book
        try:
            workBook.remove(workBook[sheetname])
        except:
            workbook = xlsxwriter.Workbook(filename)
            worksheet = workbook.add_worksheet(sheetname)
        finally:
            dataframe.to_excel(writer, sheet_name=sheetname,index=False)
            writer.save()


def plot_fcst(df_plot, city_name = None, if_save_imag = False):
    fig = plt.figure(figsize=(8, 5))
    
    df_plot['date'] = df_plot['ds'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
    df_plot = df_plot.rename(columns={'yhat':'y_pred','y':'y_true'})
    df_plot.index = df_plot['date']

    plt.plot(df_plot.date, df_plot['y_pred'],"r--",label='y_pred')
    plt.plot(df_plot.date, df_plot['y_true'],"b",label='y_true')
    if city_name is not None:
        plt.title('{}'.format(city_name))
    plt.grid("on")
    fig.autofmt_xdate()
    
    plt.legend()
    if if_save_imag:
        plt.savefig(if_save_imag)
    plt.show()


########################################################################################### 日期相关函数
def get_next_month(cal_dt):
    """
    返回cal_dt(YYYY-MM-DD)的下一个月
    """
    dt = str(cal_dt)
    year = int(dt[0:4])
    month = int(dt[5:7])
    if month == 12:
        return str(year + 1) + '-' + '01'
    else:
        return str(year * 100 + month + 1)[0:4] + '-' + str(year * 100 + month + 1)[4:6]

def get_next_month_first_day(cal_dt):
    """
    返回cal_dt(YYYY-MM-DD)的下一个月的第一天
    """
    return get_next_month(cal_dt) + '-01'

def get_next_month_last_day(cal_dt):
    """
    返回cal_dt(YYYY-MM-DD)的下一个月的最后一天
    """
    next_mth_month = int(get_next_month(cal_dt)[5:7])
    next_mth_year = int(get_next_month(cal_dt)[0:4])
    if next_mth_month in [1, 3, 5, 7, 8, 10, 12]:
        return get_next_month(cal_dt) + '-31'
    elif next_mth_month == 2:
        if next_mth_year % 100 == 0:
            if next_mth_year % 400 == 0:
                return get_next_month(cal_dt) + '-29'
            else:
                return get_next_month(cal_dt) + '-28'
        elif next_mth_year % 4 == 0:
            return get_next_month(cal_dt) + '-29'
        else:
            return get_next_month(cal_dt) + '-28'
    else:
        return get_next_month(cal_dt) + '-30'

def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return ((d2 - d1).days)


def if_between(date, d1 = '2000-01-01', d2 = '2030-01-01'):
    ## 默认为在d1和d2之间
    if days_between(d1, date)>=0:
        if days_between(date, d2)>=0:
            return True
        else:
            return False
    else:
        return False

def date_add(dt, days_delta):
    import datetime
    sdate  = datetime.datetime.strptime(dt,'%Y-%m-%d')
    delta7 = datetime.timedelta(days=days_delta)
    edate  = sdate + delta7
    return str(edate)[0:10]


def get_current_date():
    ## 返回YYYY-MM-DD格式的今日日期，string格式
    from datetime import date
    current_date = str(date.today())[0:10]
    return current_date


def get_weekday(dt):
    return datetime.strptime(dt,'%Y-%m-%d').weekday() + 1


########################################################################################### 节假日
def generate_cn_holidays():
    ## 春节
    spring_fest = pd.DataFrame({
      'holiday': 'lunar_new_year',
      'ds': pd.to_datetime(['2010-02-14', '2011-02-03', '2012-01-23','2013-02-10', '2014-01-31',
                           '2015-02-19', '2016-02-08', '2017-01-28', '2018-02-16', '2019-02-05',
                           '2020-01-25', '2021-02-12', '2022-02-01', '2023-01-22', '2024-02-10']),
      'lower_window': -18,
      'upper_window': 30,
    })

    ## 国庆
    national_day = pd.DataFrame({
      'holiday': 'national_celebration',
      'ds': pd.to_datetime(['2010-10-01', '2011-10-01', '2012-10-01','2013-10-01', '2014-10-01',
                           '2015-10-01', '2016-10-01', '2017-10-01', '2018-10-01', '2019-10-01',
                           '2020-10-01', '2021-10-01', '2022-10-01', '2023-09-29', '2024-10-01']),
      'lower_window': -3,
      'upper_window': 7,
    })

    ## 元旦
    yuandan = pd.DataFrame({
      'holiday': 'yuandan',
      'ds': pd.to_datetime(['2010-01-01', '2011-01-01', '2012-01-01','2013-01-01', '2014-01-01',
                           '2015-01-01', '2016-01-01', '2017-01-01', '2017-12-30', '2018-12-30',
                           '2020-01-01', '2021-01-01', '2022-01-01', '2022-12-31', '2023-12-30',
                           '2025-01-01'
                           ]),
      'lower_window': -3,
      'upper_window': 2,
    })

    ## 疫情
    pandamic = pd.DataFrame({
      'holiday': 'pandamic',
      'ds': pd.to_datetime(['2020-01-25']),
      'lower_window': 0,
      'upper_window': 120,
    })

    # ## 疫情后恢复
    # pandamic_recovery = pd.DataFrame({
    #   'holiday': 'pandamic_recovery',
    #   'ds': pd.to_datetime(['2020-03-25']),
    #   'lower_window': 0,
    #   'upper_window': 60,
    # })

    ### qingming:3; duanwu:3; zhongqiu:3
    ## 清明
    qingming = pd.DataFrame({
      'holiday': 'qingming',
      'ds': pd.to_datetime(['2010-04-03', '2011-04-03', '2012-04-02','2013-04-04', '2014-04-05',
                           '2015-04-04', '2016-04-04', '2017-04-03', '2018-04-05', '2019-04-05',
                           '2020-04-04', '2021-04-03', '2022-04-03', '2023-04-05']),
      'lower_window': -1,
      'upper_window': 2,
    })

    ## 劳动，2019年只在5.1休一天
    labor = pd.DataFrame({
      'holiday': 'labor',
      'ds': pd.to_datetime(['2010-05-01', '2011-04-30', '2012-04-29','2013-04-29', '2014-05-01',
                           '2015-05-01', '2016-05-01', '2017-05-01', '2018-04-29', '2019-05-01', 
                           '2020-05-01', '2021-05-01', '2022-04-30', '2023-04-29']),
      'lower_window': -3,
      'upper_window': 3,
    })

    ## 端午
    duanwu = pd.DataFrame({
      'holiday': 'duanwu',
      'ds': pd.to_datetime(['2010-06-14', '2011-06-04', '2012-06-22','2013-06-10', '2014-05-31',
                           '2015-06-20', '2016-06-09', '2017-05-28', '2018-06-16', '2019-06-07',
                           '2020-06-25', '2021-06-12', '2022-06-03', '2023-06-22']),
      'lower_window': -3,
      'upper_window': 2,
    })

    ## 中秋
    # 2012、2017、2020、2023年中秋和国庆重叠，2015只放一天
    zhongqiu = pd.DataFrame({
      'holiday': 'zhongqiu',
      'ds': pd.to_datetime(['2010-09-22', '2011-09-10',               '2013-09-19', '2014-09-06',
                                         '2016-09-15',               '2018-09-22', '2019-09-13',
                                         '2021-09-19', '2022-09-10'                           ]),
      'lower_window': -1,
      'upper_window': 2,
    })

    holidays_summary = pd.concat((spring_fest, national_day, yuandan,
                                  pandamic, 
                                  #pandamic_recovery,
                                  qingming, labor, duanwu, zhongqiu))
    return holidays_summary

def transform_holiday_df(df):
    holiday_name = df['holiday'].iloc[0]
    ds_lst = []
    nbr_lst = []

    for index, row in df.iterrows():
        start_ds = date_add(str(row['ds'])[0:10], row['lower_window'])
        end_ds = date_add(str(row['ds'])[0:10], row['upper_window'])
        ds_lst_tmp = pd.date_range(start=start_ds, end= end_ds, freq = 'd')
        ds_lst_tmp = [str(i)[0:10] for i in ds_lst_tmp]
        nbr_lst_tmp = range(0, len(ds_lst_tmp))
        nbr_lst_tmp = [i + row['lower_window'] for i in nbr_lst_tmp]
        nbr_lst_tmp = [(i+1) if i >= 0 else i for i in nbr_lst_tmp]

        ds_lst += ds_lst_tmp
        nbr_lst += nbr_lst_tmp

    trans_df = pd.DataFrame({
        'ds': ds_lst,
        holiday_name: nbr_lst
        })
    return trans_df.drop_duplicates(subset='ds', keep='first')



def generate_dates_info(holiday_df = None):
    ds_lst = pd.date_range(start='20100101',end='20250101', freq = 'd')
    ds_lst = [str(i)[0:10] for i in ds_lst]
    dates_info = pd.DataFrame({'ds': ds_lst})

    holiday_df = (nvl(holiday_df, generate_cn_holidays())).copy()
    holiday_lst = [i for i in holiday_df['holiday'].unique() if ('covid' not in i.lower() and 'pandamic' not in i.lower())]
    covid_lst = [i for i in holiday_df['holiday'].unique() if ('covid' in i.lower() or 'pandamic' in i.lower())]

    ### 节假日
    for i in holiday_lst:
        trans_holiday_df = transform_holiday_df(holiday_df.query("holiday == @i"))
        dates_info = dates_info.merge(trans_holiday_df, on = ['ds'], how = 'left')

    ### 疫情
    covid_df = holiday_df.query("holiday in @covid_lst").copy()
    if covid_df.shape[0]>0:
        covid_df['holiday'] = 'covid'
        trans_covid_df = transform_holiday_df(covid_df)
        dates_info = dates_info.merge(trans_covid_df, on = ['ds'], how = 'left')
    else:
        dates_info['covid'] = 0

    ###### 节假日
    dates_info['if_holiday'] = dates_info[holiday_lst].sum(axis = 1)
    dates_info['if_fes'] = dates_info['if_holiday'].apply(lambda x: 1 if (x<0 or x>0) else 0)
    ### 节假日前：高峰期 // 节假日中：低谷期
    dates_info['pre_holiday'] = dates_info['if_holiday'].apply(lambda x: 1 if x < 0 else 0)
    dates_info['post_holiday'] = dates_info['if_holiday'].apply(lambda x: 1 if x > 0 else 0)
    ### 是否节假日
    dates_info['if_holiday'] = dates_info['if_holiday'].apply(lambda x: 1 if (x > 0 or x < 0) else 0)
    ###### 是否疫情
    dates_info['if_covid'] = dates_info['covid'].apply(lambda x: 1 if (x > 0 or x < 0) else 0)
    ### normal days：非节假日，非疫情
    dates_info['normal_day'] = dates_info.apply(lambda x: 1 if (x['if_holiday'] == 0 and x['if_covid'] == 0) else 0, axis = 1)
    ###### 星期
    dates_info['weekday'] = dates_info['ds'].apply(lambda x: get_weekday(x))
    dates_info['week_rank'] = (dates_info['weekday']==1).cumsum()
    dates_info = (
        dates_info.
        merge(
            dates_info.groupby(['week_rank'], as_index = False).agg(if_week_fes = ('if_fes', 'sum')),
            on = ['week_rank'],
            how = 'left'
        )
    )
    dates_info['if_week_fes'] = dates_info['if_week_fes'].apply(lambda x: 1 if x>0 else 0) ## 当周是否包含节假日
    ### 兼容老版本
    dates_info['Dates'] = dates_info['ds']
    
    dates_info = dates_info.fillna(0)
    return dates_info




########################################################################################### 回测df
def generate_backtest_df(
    jumpoff_date_lst,
    start_date_lst = None,
    end_date_lst = None,
    gap_days = 1,
    horizon = 30
    ):
    if start_date_lst is None:
        start_date_lst = [date_add(i, gap_days) for i in jumpoff_date_lst]
    if end_date_lst is None:
        end_date_lst = [date_add(i, horizon - 1) for i in start_date_lst]

    return pd.DataFrame({'jump_off_dt': jumpoff_date_lst, 'fcst_stt_dt': start_date_lst, 'fcst_end_dt': end_date_lst})


