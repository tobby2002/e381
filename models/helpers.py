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

def plot_pattern_m(df: pd.DataFrame, wave_pattern: WavePattern, wave_options=None, plot_list=None, df_lows_plot=None, df_highs_plot=None,trade_info=None, order_history=None, title: str = ''):
    data = go.Ohlc(x=df['Date'],
                   open=df['Open'],
                   high=df['High'],
                   low=df['Low'],
                   close=df['Close'])

    pattern_l = list()
    pattern_l.append(data)

    # trade_stats = [len(trade_count), winrate, symbol, asset_new, str(round(pnl_percent, 4)) + '%', sum(pnl_history), sum(fee_history)]
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
        if len(order_history) > 0 and order_history[-1]:
            h = order_history[-1]
        # for h in order_history:
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


def plot_pattern_n(df: pd.DataFrame, wave_pattern: WavePattern, wave_options=None, plot_list=None, df_lows_plot=None, df_highs_plot=None,trade_info=None, order_history=None, title: str = ''):
    data = go.Ohlc(x=df['Date'],
                   open=df['Open'],
                   high=df['High'],
                   low=df['Low'],
                   close=df['Close'])

    pattern_l = list()
    pattern_l.append(data)

    # trade_stats = [len(trade_count), winrate, symbol, asset_new, str(round(pnl_percent, 4)) + '%', sum(pnl_history), sum(fee_history)]
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
        if len(order_history) > 0 and order_history[-1]:
            h = order_history[-1] # ['LINKUSDT', 7.386, 7.497, 7.36, '2023-04-05 22:54:00', '2023-04-05 23:03:00', False, 'WIN']
            symbol = h[0]
            longshort = h[-2]
            winlose = h[-1]
            et = h[1]
            sl = h[2]
            tp = h[3]
            enter_dt = h[4]
            out_dt = h[5]
        # for h in order_history:
            xdata = list()
            ydata = list()
            zlabels = list()

            if h:
                lnst = 'long' if longshort else 'short'
                pfsl = 'PF' if winlose == 'WIN' else 'SL'
                color_t = 'red' if winlose == 'WIN' else 'blue'
                if enter_dt:
                    xdata.append(enter_dt)
                    ydata.append(et)
                    zlabels.append('→ e %s %s' % (str(lnst), pfsl))
                if out_dt:
                    xdata.append(out_dt)
                    ydata.append(tp if winlose == 'WIN' else sl)
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
