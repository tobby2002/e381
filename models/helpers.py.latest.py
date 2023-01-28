from models.WavePattern import WavePattern
import pandas as pd
import time
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import numba

def timeit(func):
    def wrapper(*arg, **kw):

        t1 = time.perf_counter_ns()
        res = func(*arg, **kw)
        t2 = time.perf_counter_ns()
        print("took:", t2-t1, 'ns')
        return res
    return wrapper


def plot_cycle(df, wave_cycle, title: str = ''):

    data = go.Ohlc(x=df['Date'],
                   open=df['Open'],
                   high=df['High'],
                   low=df['Low'],
                   close=df['Close'])

    monowaves = go.Scatter(x=wave_cycle.dates,
                           y=wave_cycle.values,
                           text=wave_cycle.labels,
                           mode='lines+markers+text',
                           textposition='middle right',
                           textfont=dict(size=15, color='#2c3035'),
                           line=dict(
                               color=('rgb(111, 126, 130)'),
                               width=3),
                           )
    layout = dict(title=title)
    fig = go.Figure(data=[data, monowaves], layout=layout)
    fig.update(layout_xaxis_rangeslider_visible=False)

    fig.show()


def convert_yf_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a yahoo finance OHLC DataFrame to column name(s) used in this project

    old_names = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    new_names = ['Date', 'Open', 'High', 'Low', 'Close']

    :param df:
    :return:
    """
    df_output = pd.DataFrame()

    df_output['Date'] = list(df.index)
    df_output['Date'] = pd.to_datetime(df_output['Date'], format="%Y-%m-%d %H:%M:%S")

    df_output['Open'] = df['Open'].to_list()
    df_output['High'] = df['High'].to_list()
    df_output['Low'] = df['Low'].to_list()
    df_output['Close'] = df['Close'].to_list()


    return df_output

def plot_pattern(df: pd.DataFrame, wave_pattern: WavePattern, title: str = ''):
    data = go.Ohlc(x=df['Date'],
                   open=df['Open'],
                   high=df['High'],
                   low=df['Low'],
                   close=df['Close'])

    monowaves = go.Scatter(x=wave_pattern.dates,
                           y=wave_pattern.values,
                           text=wave_pattern.labels,
                           mode='lines+markers+text',
                           textposition='middle right',
                           textfont=dict(size=15, color='#2c3035'),
                           line=dict(
                               color=('rgb(111, 126, 130)'),
                               width=3),
                           )
    layout = dict(title=title)
    fig = go.Figure(data=[data, monowaves], layout=layout)
    fig.update(layout_xaxis_rangeslider_visible=False)

    fig.show()


def plot_pivots(X, pivots):
    plt.xlim(0, len(X))
    plt.ylim(X.min()*0.99, X.max()*1.01)
    plt.plot(np.arange(len(X)), X, 'k:', alpha=0.5)
    plt.plot(np.arange(len(X))[pivots != 0], X[pivots != 0], 'k-')
    plt.scatter(np.arange(len(X))[pivots == 1], X[pivots == 1], color='g')
    plt.scatter(np.arange(len(X))[pivots == -1], X[pivots == -1], color='r')
    plt.show()
## md
from zigzag import *


# @numba.jit()
def peak_valley_pivots_candlestick(close, high, low, up_thresh, down_thresh):
    """
    Finds the peaks and valleys of a series of HLC (open is not necessary).
    TR: This is modified peak_valley_pivots function in order to find peaks and valleys for OHLC.
    Parameters
    ----------
    close : This is series with closes prices.
    high : This is series with highs  prices.
    low : This is series with lows prices.
    up_thresh : The minimum relative change necessary to define a peak.
    down_thesh : The minimum relative change necessary to define a valley.
    Returns
    -------
    an array with 0 indicating no pivot and -1 and 1 indicating valley and peak
    respectively
    Using Pandas
    ------------
    For the most part, close, high and low may be a pandas series. However, the index must
    either be [0,n) or a DateTimeIndex. Why? This function does X[t] to access
    each element where t is in [0,n).
    The First and Last Elements
    ---------------------------
    The first and last elements are guaranteed to be annotated as peak or
    valley even if the segments formed do not have the necessary relative
    changes. This is a tradeoff between technical correctness and the
    propensity to make mistakes in data analysis. The possible mistake is
    ignoring data outside the fully realized segments, which may bias analysis.
    """
    if down_thresh > 0:
        raise ValueError('The down_thresh must be negative.')

    initial_pivot = identify_initial_pivot(close, up_thresh, down_thresh)

    t_n = len(close)
    pivots = np.zeros(t_n, dtype='i1')
    pivots[0] = initial_pivot

    # Adding one to the relative change thresholds saves operations. Instead
    # of computing relative change at each point as x_j / x_i - 1, it is
    # computed as x_j / x_1. Then, this value is compared to the threshold + 1.
    # This saves (t_n - 1) subtractions.
    up_thresh += 1
    down_thresh += 1

    trend = -initial_pivot
    last_pivot_t = 0
    last_pivot_x = close[0]
    for t in range(1, len(close)):

        if trend == -1:
            x = low[t]
            r = x / last_pivot_x
            if r >= up_thresh:
                pivots[last_pivot_t] = trend
                trend = 1
                last_pivot_x = x
                last_pivot_t = t
            elif x < last_pivot_x:
                last_pivot_x = x
                last_pivot_t = t
        else:
            x = high[t]
            r = x / last_pivot_x
            if r <= down_thresh:
                pivots[last_pivot_t] = trend
                trend = -1
                last_pivot_x = x
                last_pivot_t = t
            elif x > last_pivot_x:
                last_pivot_x = x
                last_pivot_t = t


    if last_pivot_t == t_n-1:
        pivots[last_pivot_t] = trend
    elif pivots[t_n-1] == 0:
        pivots[t_n-1] = trend

    return pivots

def zigzag_df(d, h, l, c, up_thresh=0.01, down_thresh=-0.01):
    pivots = peak_valley_pivots_candlestick(c, h, l, up_thresh, down_thresh)
    Z_ = pd.DataFrame(list(zip(d, h, l, pivots)), columns=['D', 'H', 'L', 'Z'])
    Z = Z_[(Z_['Z'] == 1) | (Z_['Z'] == -1)]
    Z['Z_high_low'] = Z.apply(lambda x: x.H if x.Z == 1 else x.L, axis=1)
    return Z

def plot_pattern_m(df: pd.DataFrame, wave_pattern: WavePattern, wave_options=None, plot_list=None, df_lows_plot=None, df_highs_plot=None,trade_info=None, order_history=None, title: str = ''):
    data = go.Ohlc(x=df['Date'],
                   open=df['Open'],
                   high=df['High'],
                   low=df['Low'],
                   close=df['Close'])

    pattern_l = list()
    pattern_l.append(data)

    # C = np.array(df['Close'].values.tolist())
    # D = np.array(df['Date'].values.tolist())
    # H = np.array(df['High'].values.tolist())
    # L = np.array(df['Low'].values.tolist())
    # pivots = peak_valley_pivots(C, 0.01, -0.01)
    # Z = zigzag_df(D, H, L, C, up_thresh=0.001, down_thresh=-0.001)
    # pattern_l.append(go.Scatter(x=Z['D'].values.tolist(), y=Z['Z_high_low'].values.tolist(), line=dict(color='blue', width=2)))
    # Z_005 = zigzag_df(D, H, L, C, up_thresh=0.005, down_thresh=-0.005)
    # pattern_l.append(go.Scatter(x=Z_005['D'].values.tolist(), y=Z_005['Z_high_low'].values.tolist(), line=dict(color='green', width=3)))
    # Z_5 = zigzag_df(D, H, L, C, up_thresh=0.01, down_thresh=-0.01)
    # pattern_l.append(go.Scatter(x=Z_5['D'].values.tolist(), y=Z_5['Z_high_low'].values.tolist(), line=dict(color='yellow', width=4)))


    # df_zz = df[(df['ZigZag_5_10_1'] != 0)]
    df_zz = df[(df['ZigZag_1_2_1'] != 0)]
    # Z['Z_high_low'] = Z_5_100_1.apply(lambda x: x.H if x.Z_5_100_1 == 0, axis=1)
    # pattern_l.append(go.Scatter(x=df_zz['Date'].values.tolist(), y=df_zz['ZigZag_5_10_1'].values.tolist(), line=dict(color='yellow', width=3)))
    pattern_l.append(go.Scatter(x=df_zz['Date'].values.tolist(), y=df_zz['ZigZag_1_2_1'].values.tolist(), line=dict(color='black', width=2)))

    # plot_pivots(X, pivots)

    # trade_stats = [len(trade_count), winrate, symbol, asset_new, str(round(pnl_percent, 4)) + '%', sum(pnl_history),
    #                sum(fee_history)]
    # order_history = [condi_order_i, position_enter_i, position_pf_i, position_sl_i]
    # h = [trade_stats, order_history, asset_history, trade_count, fee_history, pnl_history, wavepattern]
    if plot_list:
        try:
            for i in plot_list:
                pattern_l.append(go.Scatter(x=df.Date, y=i, line=dict(color='grey', width=1)))
        except:
            pass

    if trade_info:
        order_history = trade_info[1]
        for h in order_history:
            xdata = list()
            ydata = list()
            zlabels = list()

            if h:
                entering = h[0]
                lnst = 'long' if h[2] == True else 'short'
                pfsl = 'PF' if h[3] == '+' else 'SL'
                color_t = 'red' if h[3] == '+' else 'blue'
                if entering:
                    xdata.append(str(entering[0]))
                    ydata.append(entering[-1])
                    zlabels.append('→ e %s %s' % (str(lnst), pfsl))
                outering = h[1]
                if outering:
                    xdata.append(str(outering[0]))
                    ydata.append(outering[1])
                    zlabels.append('← x')

                t = go.Scatter(x=xdata,
                               y=ydata,
                               text=zlabels,
                               mode='lines+markers+text',
                               textposition='middle right',
                               textfont=dict(size=12, color=color_t),
                               line=dict(
                                   color=color_t,
                                   width=2),
                               )
                pattern_l.append(t)

    if wave_pattern:
        for wp in wave_pattern:
            p = go.Scatter(x=wp[3].dates,
                                   y=wp[3].values,
                                   text=wp[3].labels,
                                   mode='lines+markers+text',
                                   textposition='middle right',
                                   textfont=dict(size=11, color='#2c3035'),
                                   line=dict(
                                       color=('rgb(111, 126, 130)'),
                                       width=1),
                                   )
            pattern_l.append(p)

    if wave_options:
        for wo in wave_options:
            o = go.Scatter(
                            x=wo[0],
                            y=wo[1],
                            text='  '+str(wo[2]),
                            mode='lines+markers+text',
                            textposition='middle right',
                            textfont=dict(size=10, color='blue'),
                            line=dict(
                               color=('rgb(0, 149, 59)'),
                               width=2),
                                   )
            pattern_l.append(o)

    if df_lows_plot is not None:
        p = go.Scatter(x=df_lows_plot.Date.tolist(),
                          y=df_lows_plot.Low.tolist(),
                           text=df_lows_plot.index.tolist(),

                       mode='lines+markers+text',
                               textposition='middle right',
                               textfont=dict(size=8, color='red'),
                               line=dict(
                                   color=('rgb(0, 149, 59)'),
                                   width=1),
                               )
        pattern_l.append(p)

    if df_highs_plot is not None:
        p = go.Scatter(x=df_highs_plot.Date.tolist(),
                          y=df_highs_plot.High.tolist(),
                           text=df_highs_plot.index.tolist(),
                       mode='lines+markers+text',
                               textposition='middle right',
                               textfont=dict(size=8, color='red'),
                               line=dict(
                                   color=('rgb(0, 149, 59)'),
                                   width=1),
                               )
        pattern_l.append(p)


    layout = dict(
            title=title,
            font = dict(
                family="Arial",
                size=10,
                color='#000000'
            ))
    fig = go.Figure(data=pattern_l, layout=layout)
    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.show()

    # import dash
    # from dash import dcc
    # from dash import html
    #
    # app = dash.Dash()
    # app.layout = html.Div([
    #     dcc.Graph(figure=fig)
    # ])

def plot_monowave(df, monowave, title: str = ''):
    data = go.Ohlc(x=df['Date'],
                   open=df['Open'],
                   high=df['High'],
                   low=df['Low'],
                   close=df['Close'])

    monowaves = go.Scatter(x=monowave.dates,
                           y=monowave.points,
                           mode='lines+markers+text',
                           textposition='middle right',
                           textfont=dict(size=15, color='#2c3035'),
                           line=dict(
                               color=('rgb(111, 126, 130)'),
                               width=3),
                           )
    layout = dict(title=title)
    fig = go.Figure(data=[data, monowaves], layout=layout)
    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.show()


def plot_monowave_m(df, monowaves, df_plot=None, title: str = ''):
    data = go.Ohlc(x=df['Date'],
                   open=df['Open'],
                   high=df['High'],
                   low=df['Low'],
                   close=df['Close'])
    monowaves_l = list()
    monowaves_l.append(data)

    for monowave in monowaves:
        m = go.Scatter(x=monowave.dates,
                               y=monowave.points,
                               mode='lines+markers+text',
                               textposition='middle right',
                               textfont=dict(size=15, color='#2c3035'),
                               line=dict(
                                   color=('rgb(111, 126, 130)'),
                                   width=1),
                               )
        monowaves_l.append(m)

    if df_plot is not None:
        p = go.Scatter(x=df_plot.Date.tolist(),
                          y=df_plot.Low.tolist(),
                          mode='lines+markers+text',
                               textposition='middle right',
                               textfont=dict(size=8, color='red'),
                               line=dict(
                                   color=('rgb(0, 149, 59)'),
                                   width=1),
                               )
        monowaves_l.append(p)
    layout = dict(title=title)
    fig = go.Figure(data=monowaves_l, layout=layout)
    fig.update(layout_xaxis_rangeslider_visible=False)
    # https://plotly.com/python/ohlc-charts/
    fig.show()

    # import dash
    # from dash import dcc
    # from dash import html
    #
    # app = dash.Dash()
    # app.layout = html.Div([
    #     dcc.Graph(figure=fig)
    # ])
    #
    # app.run_server(debug=True, use_reloader=True)  # Turn off reloader if inside Jupyter