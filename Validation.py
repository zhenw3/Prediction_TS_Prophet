
from __future__ import absolute_import, division, print_function

import logging
from tqdm.auto import tqdm
from copy import deepcopy
import concurrent.futures

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from datetime import datetime

from utils import date_add, generate_dates_info

logger = logging.getLogger('prophet')


def generate_cutoffs(df, horizon, initial, period):
    """Generate cutoff dates

    Parameters
    ----------
    df: pd.DataFrame with historical data.
    horizon: pd.Timedelta forecast horizon.
    initial: pd.Timedelta window of the initial forecast period.
    period: pd.Timedelta simulated forecasts are done with this period.

    Returns
    -------
    list of pd.Timestamp
    """
    # Last cutoff is 'latest date in data - horizon' date
    cutoff = df['ds'].max() - horizon
    if cutoff < df['ds'].min():
        raise ValueError('Less data than horizon.')
    result = [cutoff]
    while result[-1] >= min(df['ds']) + initial:
        cutoff -= period
        # If data does not exist in data range (cutoff, cutoff + horizon]
        if not (((df['ds'] > cutoff) & (df['ds'] <= cutoff + horizon)).any()):
            # Next cutoff point is 'last date before cutoff in data - horizon'
            if cutoff > df['ds'].min():
                closest_date = df[df['ds'] <= cutoff].max()['ds']
                cutoff = closest_date - horizon
            # else no data left, leave cutoff as is, it will be dropped.
        result.append(cutoff)
    result = result[:-1]
    if len(result) == 0:
        raise ValueError(
            'Less data than horizon after initial window. '
            'Make horizon or initial shorter.'
        )
    logger.info('Making {} forecasts with cutoffs between {} and {}'.format(
        len(result), result[-1], result[0]
    ))
    return list(reversed(result))


def cross_validation(model, cv_df, calib = None, parallel=None, disable_tqdm=False):
    """Cross-Validation for time series.

    Computes forecasts from historical cutoff points, which user can input.
    If not provided, begins from (end - horizon) and works backwards, making
    cutoffs with a spacing of period until initial is reached.

    When period is equal to the time interval of the data, this is the
    technique described in https://robjhyndman.com/hyndsight/tscv/ .

    Parameters
    ----------
    model: Prophet class object. Fitted or Unfitted Prophet model.
    cv_df: cutoff dates(jumpoff date), forecast start dates and forecast end dates to validate training
    calib: calibration object that adjust model prediction

    parallel : {None, 'processes', 'threads', 'dask', object}
        How to parallelize the forecast computation. By default no parallelism
        is used.

        * None : No parallelism.
        * 'processes' : Parallelize with concurrent.futures.ProcessPoolExectuor.
        * 'threads' : Parallelize with concurrent.futures.ThreadPoolExecutor.
            Note that some operations currently hold Python's Global Interpreter
            Lock, so parallelizing with threads may be slower than training
            sequentially.
        * 'dask': Parallelize with Dask.
           This requires that a dask.distributed Client be created.
        * object : Any instance with a `.map` method. This method will
          be called with :func:`single_cutoff_forecast` and a sequence of
          iterables where each element is the tuple of arguments to pass to
          :func:`single_cutoff_forecast`

          .. code-block::

             class MyBackend:
                 def map(self, func, *iterables):
                     results = [
                        func(*args)
                        for args in zip(*iterables)
                     ]
                     return results
                     
    disable_tqdm: if True it disables the progress bar that would otherwise show up when parallel=None
    extra_output_columns: A String or List of Strings e.g. 'trend' or ['trend'].
         Additional columns to 'yhat' and 'ds' to be returned in output.

    Returns
    -------
    A pd.DataFrame with the forecast, actual value and cutoff.
    """
    
    cv_df = cv_df.drop_duplicates(subset='jump_off_dt', keep='first')

    cutoffs = cv_df['jump_off_dt'].tolist()
    start_dts = cv_df['fcst_stt_dt'].tolist()
    end_dts = cv_df['fcst_end_dt'].tolist()

    print("Validating {} Cutoff points".format(len(cutoffs)))
    print(cutoffs)

    if parallel:
        valid = {"threads", "processes", "dask"}

        if parallel == "threads":
            pool = concurrent.futures.ThreadPoolExecutor()
        elif parallel == "processes":
            pool = concurrent.futures.ProcessPoolExecutor()
        elif parallel == "dask":
            try:
                from dask.distributed import get_client
            except ImportError as e:
                raise ImportError("parallel='dask' requires the optional "
                                  "dependency dask.") from e
            pool = get_client()
            # delay df and model to avoid large objects in task graph.
            df, model = pool.scatter([df, model])
        elif hasattr(parallel, "map"):
            pool = parallel
        else:
            msg = ("'parallel' should be one of {} for an instance with a "
                   "'map' method".format(', '.join(valid)))
            raise ValueError(msg)

        iterables = ((model, calib, cutoff, start_dt, end_dt)
                     for cutoff, start_dt, end_dt in zip(cutoffs, start_dts, end_dts))
        iterables = zip(*iterables)

        logger.info("Applying in parallel with %s", pool)
        predicts = pool.map(single_cutoff_forecast, *iterables)
        if parallel == "dask":
            # convert Futures to DataFrames
            predicts = pool.gather(predicts)

    else:
        predicts = [
            single_cutoff_forecast(model, calib, cutoff, start_dt, end_dt) 
            for cutoff, start_dt, end_dt in (tqdm(zip(cutoffs, start_dts, end_dts)) if not disable_tqdm else zip(cutoffs, start_dts, end_dts))
        ]

    # Combine all predicted pd.DataFrame into one pd.DataFrame
    return pd.concat(predicts, axis=0).reset_index(drop=True)


def single_cutoff_forecast(model, calib, cutoff, start_dt, end_dt):
    """Forecast for single cutoff. Used in cross validation function
    when evaluating for multiple cutoffs either sequentially or in parallel .

    Parameters
    ----------
    df: pd.DataFrame.
        DataFrame with history to be used for single
        cutoff forecast.
    model: Prophet model object.
    cutoff: pd.Timestamp cutoff date.
        Simulated Forecast will start from this date.
    horizon: pd.Timedelta forecast horizon.
    predict_columns: List of strings e.g. ['ds', 'yhat'].
        Columns with date and forecast to be returned in output.

    Returns
    -------
    A pd.DataFrame with the forecast, actual value and cutoff.

    """

    # Generate new object with copying fitting options
    m = deepcopy(model)

    # forecast
    pred_df = m.predict(
        jump_off_date = cutoff, ## 训练集结束日期
        fcst_stt_date = start_dt, ## 预测开始日期
        fcst_end_date = end_dt, ## 预测结束日期
        if_train_again = True, ## 否：重新训练模型；是：用已有模型预测
        if_keep_hist = True, ## 是否保留训练数据
    )[['ds', 'y', 'yhat', 'yhat_upper', 'yhat_lower', 'if_train', 'if_pred']]
    
    # adjust 
    if calib is None:
        pass
    else:
        c = deepcopy(calib)
        try:
            print("Access holidays")
            holidays = model.pf_params['holidays']
        except:
            print("No holidays info!")
            holidays = None
        pred_df = c.adj_fcst(pred_df, holidays)
        pred_df = pred_df[['ds', 'y', 'yhat', 'yhat_upper', 'yhat_lower', 'if_train', 'if_pred', 'y_adj']]
        pred_df = pred_df.rename(columns = {'yhat': 'yhat_orig', 'y_adj': 'yhat'})

    # truncate data
    pred_df = pred_df.loc[pred_df['ds'] >= date_add(cutoff, -30)]

    # add cutoff info
    pred_df['cutoff'] = cutoff

    return pred_df


def get_cv_stats(cv_res):
    
    df = cv_res.query("if_pred == 1").copy()
    ## 关联通用节假日信息
    df = (
        df.merge(
            generate_dates_info()[['ds', 'pre_holiday', 'post_holiday', 'normal_day']],
            on = ['ds'],
            how = 'left'
            )
        )
    
    jump_off_lst = sorted((df.cutoff).unique())

    y_stats_df, mth_avg_pred_lst, mth_avg_true_lst, daily_mape, pct_monthly_error, pre_holiday_mape, holiday_mape, normal_mape = [], [], [], [], [], [], [], []

    for jp_dt in jump_off_lst:
        y_stats = df.query("cutoff == @jp_dt")
        y_stats_df.append(y_stats)
        
        ### 天粒度error
        y_stats['pct_error'] = y_stats['yhat']/(y_stats['y'].apply(lambda x: 0.01 if x==0 else x)) - 1

        ## 月均
        mth_avg_pred_lst.append(y_stats['yhat'].mean())
        mth_avg_true_lst.append(y_stats['y'].mean())
        
        ### MAPE
        daily_mape.append((abs(y_stats['pct_error'])).mean())
        pre_holiday_mape.append(
            (abs(y_stats['pct_error']) * y_stats['pre_holiday']).sum()
            /y_stats['pre_holiday'].sum()
        )
        holiday_mape.append(
            (abs(y_stats['pct_error']) * y_stats['post_holiday']).sum()
            /y_stats['post_holiday'].sum()
        )
        normal_mape.append(
            (abs(y_stats['pct_error']) * y_stats['normal_day']).sum()
            /y_stats['normal_day'].sum()
        )

        ### 月均误差
        pct_monthly_error.append((y_stats['yhat'].mean())/(y_stats['y'].mean()) - 1)

    result_summary = pd.DataFrame({
        'jump_off_date': jump_off_lst,
        'daily_mape': daily_mape,
        'pre_holiday_mape': pre_holiday_mape,
        'holiday_mape': holiday_mape,
        'normal_mape': normal_mape,
        'pct_monthly_error': pct_monthly_error,
        'daily_avg_pred': mth_avg_pred_lst,
        'daily_avg_true': mth_avg_true_lst,
        'result_df': y_stats_df,
    })
    
    return result_summary

def display_cv_result(result_summary, if_save_imag = False):

    print(result_summary[[i for i in result_summary.columns if i != 'result_df']])
    print('\n Monthly MAPE: {0}'.format(
        round(result_summary['pct_monthly_error'].apply(lambda x: abs(x)).mean(),  5)
    ))
    print('\n Daily MAPE Mean: {0}'.format(
        round(result_summary['daily_mape'].mean(),  5)
    ))
    
    mth_plot_lst = result_summary['jump_off_date'].tolist()
    
    #result_summary = self.result_summary
    
    fig = plt.figure(figsize=(14, math.ceil(len(mth_plot_lst)/2)*3))
    for i in range(len(mth_plot_lst)):
        mth_plot = mth_plot_lst[i]

        df_plot = result_summary.loc[result_summary['jump_off_date']==mth_plot]['result_df'].iloc[0]
        df_plot.index = df_plot.ds.apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
        
        df_plot = df_plot.rename(columns = {'y':'y_true','yhat':'y_pred'})
        #df_plot.index = df_plot.ds

        plt.subplot(math.ceil(len(mth_plot_lst)/2),2,(i+1))
        plt.plot(df_plot['y_pred'],"r--",label='y_pred')
        plt.plot(df_plot['y_true'],"b",label='y_true')
        plt.title('{}'.format(mth_plot))
        plt.grid("on")
        plt.legend()
        plt.tight_layout()

    fig.autofmt_xdate()
    if if_save_imag:
        plt.savefig(if_save_imag)
    plt.show()



def performance_metrics(df, metrics=None, rolling_window=0.1, monthly=False):
    """Compute performance metrics from cross-validation results.

    Computes a suite of performance metrics on the output of cross-validation.
    By default the following metrics are included:
    'mse': mean squared error
    'rmse': root mean squared error
    'mae': mean absolute error
    'mape': mean absolute percent error
    'mdape': median absolute percent error
    'smape': symmetric mean absolute percentage error
    'coverage': coverage of the upper and lower intervals

    A subset of these can be specified by passing a list of names as the
    `metrics` argument.

    Metrics are calculated over a rolling window of cross validation
    predictions, after sorting by horizon. Averaging is first done within each
    value of horizon, and then across horizons as needed to reach the window
    size. The size of that window (number of simulated forecast points) is
    determined by the rolling_window argument, which specifies a proportion of
    simulated forecast points to include in each window. rolling_window=0 will
    compute it separately for each horizon. The default of rolling_window=0.1
    will use 10% of the rows in df in each window. rolling_window=1 will
    compute the metric across all simulated forecast points. The results are
    set to the right edge of the window.

    If rolling_window < 0, then metrics are computed at each datapoint with no
    averaging (i.e., 'mse' will actually be squared error with no mean).

    The output is a dataframe containing column 'horizon' along with columns
    for each of the metrics computed.

    Parameters
    ----------
    df: The dataframe returned by cross_validation.
    metrics: A list of performance metrics to compute. If not provided, will
        use ['mse', 'rmse', 'mae', 'mape', 'mdape', 'smape', 'coverage'].
    rolling_window: Proportion of data to use in each rolling window for
        computing the metrics. Should be in [0, 1] to average.
    monthly: monthly=True will compute horizons as numbers of calendar months 
        from the cutoff date, starting from 0 for the cutoff month.

    Returns
    -------
    Dataframe with a column for each metric, and column 'horizon'
    """
    valid_metrics = ['mse', 'rmse', 'mae', 'mape', 'mdape', 'smape', 'coverage']
    if metrics is None:
        metrics = valid_metrics
    if ('yhat_lower' not in df or 'yhat_upper' not in df) and ('coverage' in metrics):
        metrics.remove('coverage')
    if len(set(metrics)) != len(metrics):
        raise ValueError('Input metrics must be a list of unique values')
    if not set(metrics).issubset(set(valid_metrics)):
        raise ValueError(
            'Valid values for metrics are: {}'.format(valid_metrics)
        )
    df_m = df.copy()
    if monthly:
        df_m['horizon'] = df_m['ds'].dt.to_period('M').astype(int) - df_m['cutoff'].dt.to_period('M').astype(int)
    else:
        df_m['horizon'] = df_m['ds'] - df_m['cutoff']
    df_m.sort_values('horizon', inplace=True)
    if 'mape' in metrics and df_m['y'].abs().min() < 1e-8:
        logger.info('Skipping MAPE because y close to 0')
        metrics.remove('mape')
    if len(metrics) == 0:
        return None
    w = int(rolling_window * df_m.shape[0])
    if w >= 0:
        w = max(w, 1)
        w = min(w, df_m.shape[0])
    # Compute all metrics
    dfs = {}
    for metric in metrics:
        dfs[metric] = eval(metric)(df_m, w)
    res = dfs[metrics[0]]
    for i in range(1, len(metrics)):
        res_m = dfs[metrics[i]]
        assert np.array_equal(res['horizon'].values, res_m['horizon'].values)
        res[metrics[i]] = res_m[metrics[i]]
    return res


def rolling_mean_by_h(x, h, w, name):
    """Compute a rolling mean of x, after first aggregating by h.

    Right-aligned. Computes a single mean for each unique value of h. Each
    mean is over at least w samples.

    Parameters
    ----------
    x: Array.
    h: Array of horizon for each value in x.
    w: Integer window size (number of elements).
    name: Name for metric in result dataframe

    Returns
    -------
    Dataframe with columns horizon and name, the rolling mean of x.
    """
    # Aggregate over h
    df = pd.DataFrame({'x': x, 'h': h})
    df2 = (
        df.groupby('h').agg(['sum', 'count']).reset_index().sort_values('h')
    )
    xs = df2['x']['sum'].values
    ns = df2['x']['count'].values
    hs = df2.h.values

    trailing_i = len(df2) - 1
    x_sum = 0
    n_sum = 0
    # We don't know output size but it is bounded by len(df2)
    res_x = np.empty(len(df2))

    # Start from the right and work backwards
    for i in range(len(df2) - 1, -1, -1):
        x_sum += xs[i]
        n_sum += ns[i]
        while n_sum >= w:
            # Include points from the previous horizon. All of them if still
            # less than w, otherwise weight the mean by the difference
            excess_n = n_sum - w
            excess_x = excess_n * xs[i] / ns[i]
            res_x[trailing_i] = (x_sum - excess_x)/ w
            x_sum -= xs[trailing_i]
            n_sum -= ns[trailing_i]
            trailing_i -= 1

    res_h = hs[(trailing_i + 1):]
    res_x = res_x[(trailing_i + 1):]

    return pd.DataFrame({'horizon': res_h, name: res_x})
    


def rolling_median_by_h(x, h, w, name):
    """Compute a rolling median of x, after first aggregating by h.

    Right-aligned. Computes a single median for each unique value of h. Each
    median is over at least w samples.

    For each h where there are fewer than w samples, we take samples from the previous h,
    moving backwards. (In other words, we ~ assume that the x's are shuffled within each h.)

    Parameters
    ----------
    x: Array.
    h: Array of horizon for each value in x.
    w: Integer window size (number of elements).
    name: Name for metric in result dataframe

    Returns
    -------
    Dataframe with columns horizon and name, the rolling median of x.
    """
    # Aggregate over h
    df = pd.DataFrame({'x': x, 'h': h})
    grouped = df.groupby('h')
    df2 = grouped.size().reset_index().sort_values('h')
    hs = df2['h']

    res_h = []
    res_x = []
    # Start from the right and work backwards
    i = len(hs) - 1
    while i >= 0:
        h_i = hs[i]
        xs = grouped.get_group(h_i).x.tolist()

        # wrap in array so this works if h is pandas Series with custom index or numpy array
        next_idx_to_add = np.array(h == h_i).argmax() - 1
        while (len(xs) < w) and (next_idx_to_add >= 0):
            # Include points from the previous horizon. All of them if still
            # less than w, otherwise just enough to get to w.
            xs.append(x[next_idx_to_add])
            next_idx_to_add -= 1
        if len(xs) < w:
            # Ran out of points before getting enough.
            break
        res_h.append(hs[i])
        res_x.append(np.median(xs))
        i -= 1
    res_h.reverse()
    res_x.reverse()
    return pd.DataFrame({'horizon': res_h, name: res_x})


# The functions below specify performance metrics for cross-validation results.
# Each takes as input the output of cross_validation, and returns the statistic
# as a dataframe, given a window size for rolling aggregation.


def mse(df, w):
    """Mean squared error

    Parameters
    ----------
    df: Cross-validation results dataframe.
    w: Aggregation window size.

    Returns
    -------
    Dataframe with columns horizon and mse.
    """
    se = (df['y'] - df['yhat']) ** 2
    if w < 0:
        return pd.DataFrame({'horizon': df['horizon'], 'mse': se})
    return rolling_mean_by_h(
        x=se.values, h=df['horizon'].values, w=w, name='mse'
    )


def rmse(df, w):
    """Root mean squared error

    Parameters
    ----------
    df: Cross-validation results dataframe.
    w: Aggregation window size.

    Returns
    -------
    Dataframe with columns horizon and rmse.
    """
    res = mse(df, w)
    res['mse'] = np.sqrt(res['mse'])
    res.rename({'mse': 'rmse'}, axis='columns', inplace=True)
    return res


def mae(df, w):
    """Mean absolute error

    Parameters
    ----------
    df: Cross-validation results dataframe.
    w: Aggregation window size.

    Returns
    -------
    Dataframe with columns horizon and mae.
    """
    ae = np.abs(df['y'] - df['yhat'])
    if w < 0:
        return pd.DataFrame({'horizon': df['horizon'], 'mae': ae})
    return rolling_mean_by_h(
        x=ae.values, h=df['horizon'].values, w=w, name='mae'
    )


def mape(df, w):
    """Mean absolute percent error

    Parameters
    ----------
    df: Cross-validation results dataframe.
    w: Aggregation window size.

    Returns
    -------
    Dataframe with columns horizon and mape.
    """
    ape = np.abs((df['y'] - df['yhat']) / df['y'])
    if w < 0:
        return pd.DataFrame({'horizon': df['horizon'], 'mape': ape})
    return rolling_mean_by_h(
        x=ape.values, h=df['horizon'].values, w=w, name='mape'
    )


def mdape(df, w):
    """Median absolute percent error

    Parameters
    ----------
    df: Cross-validation results dataframe.
    w: Aggregation window size.

    Returns
    -------
    Dataframe with columns horizon and mdape.
    """
    ape = np.abs((df['y'] - df['yhat']) / df['y'])
    if w < 0:
        return pd.DataFrame({'horizon': df['horizon'], 'mdape': ape})
    return rolling_median_by_h(
        x=ape.values, h=df['horizon'], w=w, name='mdape'
    )


def smape(df, w):
    """Symmetric mean absolute percentage error
    based on Chen and Yang (2004) formula

    Parameters
    ----------
    df: Cross-validation results dataframe.
    w: Aggregation window size.

    Returns
    -------
    Dataframe with columns horizon and smape.
    """
    sape = np.abs(df['y'] - df['yhat']) / ((np.abs(df['y']) + np.abs(df['yhat'])) / 2)
    if w < 0:
        return pd.DataFrame({'horizon': df['horizon'], 'smape': sape})
    return rolling_mean_by_h(
        x=sape.values, h=df['horizon'].values, w=w, name='smape'
    )


def coverage(df, w):
    """Coverage

    Parameters
    ----------
    df: Cross-validation results dataframe.
    w: Aggregation window size.

    Returns
    -------
    Dataframe with columns horizon and coverage.
    """
    is_covered = (df['y'] >= df['yhat_lower']) & (df['y'] <= df['yhat_upper'])
    if w < 0:
        return pd.DataFrame({'horizon': df['horizon'], 'coverage': is_covered})
    return rolling_mean_by_h(
        x=is_covered.values, h=df['horizon'].values, w=w, name='coverage'
    )