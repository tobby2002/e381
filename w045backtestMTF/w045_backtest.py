from __future__ import annotations
from models.WavePattern import WavePattern
from models.WaveRules import Impulse, DownImpulse
from models.WaveAnalyzer import WaveAnalyzer
from models.WaveOptions import WaveOptionsGenerator5
from models.helpers import plot_pattern_m
import datetime
import pandas as pd
import numpy as np
from binance.client import Client
import datetime as dt
from tapy import Indicators
import shutup; shutup.please()
from datetime import datetime
from typing import Optional
import dateparser
import pytz
import json
import os
import random
import pickle
import math
import logging
from binancefutures.um_futures import UMFutures
from binancefutures.lib.utils import config_logging
from binancefutures.error import ClientError
# config_logging(logging, logging.DEBUG)
from binance.helpers import round_step_size

seq = str(random.randint(10000, 99999))
# 로그 생성
logger = logging.getLogger()
# 로그의 출력 기준 설정
logger.setLevel(logging.INFO)
# log 출력 형식
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(message)s')

# log 출력
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
# log를 파일에 출력
file_handler = logging.FileHandler('logger_%s.log' % seq)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


key = "IkzH8WHKl0lGzOSqiZZ4TnAyKnDpqnC9Xi31kzrRNpwJCp28gP8AuWDxntSqWdrn"
secret = "FwKTmQ2RWSiECMfhZOaY7Hed45JuXqlEPno2xiLGgCzloLq4NMMcmusG6gtMCKa5"

um_futures_client = UMFutures(key=key, secret=secret)

open_order_history = []
open_order_history_seq= 'oohistory_' + seq + '.pkl'


def load_history_pkl():
    try:
        if os.path.isfile(open_order_history_seq):
            with open(open_order_history_seq, 'rb') as f:
                h = pickle.load(f)
                logger.info('load_history_pk:' + str(h))
                return h
    except Exception as e:
        logger.error(e)
        try:
            os.remove(open_order_history_seq)
        except Exception as e:
            logger.error(e)
        return []
    return []


open_order_history = load_history_pkl()


def dump_history_pkl():
    try:
        with open(open_order_history_seq, 'wb') as f:
            pickle.dump(open_order_history, f)
    except Exception as e:
        logger.error(e)


with open('config_backtest.json', 'r') as f:
    config = json.load(f)

version = config['default']['version']
descrition = config['default']['descrition']
exchange = config['default']['exchange']
exchange_symbol = config['default']['exchange_symbol']
futures = config['default']['futures']
type = config['default']['type']
maxleverage = config['default']['maxleverage']
qtyrate = config['default']['qtyrate']
krate_max = config['default']['krate_max']
krate_min = config['default']['krate_min']

high_target = config['default']['high_target']
low_target = config['default']['low_target']
low_target_w2 = config['default']['low_target_w2']

seed = config['default']['seed']
fee = config['default']['fee']
fee_limit = config['default']['fee_limit']
fee_sl = config['default']['fee_sl']
fee_tp = config['default']['fee_tp']
tp_type = config['default']['tp_type']
fee_slippage = config['default']['fee_slippage']

period_days_ago = config['default']['period_days_ago']
period_days_ago_till = config['default']['period_days_ago_till']
period_interval = config['default']['period_interval']
dt_flg = config['default']['dt_flg']
start_datetime = config['default']['start_datetime']
end_datetime = config['default']['end_datetime']

if dt_flg:
    period_days_ago = \
    (dt.date.today() - dt.date(int(start_datetime[0:4]), int(start_datetime[5:7]), int(start_datetime[9:10]))).split(' ')[
        0]  # "start_datetime": "2023-03-08 00:00:00"
    period_days_ago_till = \
    (dt.date.today() - dt.date(int(end_datetime[0:4]), int(end_datetime[5:7]), int(end_datetime[9:10]))).split(' ')[
        0]  # "end_datetime": "2023-03-08 24:00:00"

round_trip_flg = config['default']['round_trip_flg']
round_trip_count = config['default']['round_trip_count']
compounding = config['default']['compounding']
fcnt = config['default']['fcnt']
loop_count = config['default']['loop_count']


timeframe = config['default']['timeframe']
up_to_count = config['default']['up_to_count']
condi_same_date = config['default']['condi_same_date']
condi_compare_before_fractal = config['default']['condi_compare_before_fractal']
condi_compare_before_fractal_strait = config['default']['condi_compare_before_fractal_strait']
if condi_compare_before_fractal_strait:
    condi_compare_before_fractal_shift = 1
condi_compare_before_fractal_shift = config['default']['condi_compare_before_fractal_shift']
condi_compare_before_fractal_mode = config['default']['condi_compare_before_fractal_mode']
if not condi_compare_before_fractal:
    condi_compare_before_fractal_mode = 0



condi_plrate_adaptive = config['default']['condi_plrate_adaptive']
condi_plrate_rate = config['default']['condi_plrate_rate']
condi_plrate_rate_min = config['default']['condi_plrate_rate_min']
condi_kelly_adaptive = config['default']['condi_kelly_adaptive']
condi_kelly_window = config['default']['condi_kelly_window']


c_risk_beyond_flg = config['default']['c_risk_beyond_flg']
c_risk_beyond_max = config['default']['c_risk_beyond_max']
c_risk_beyond_min = config['default']['c_risk_beyond_min']


c_time_beyond_flg = config['default']['c_time_beyond_flg']
c_time_beyond_rate = config['default']['c_time_beyond_rate']
o_fibo = config['default']['o_fibo']
h_fibo = config['default']['h_fibo']
l_fibo = config['default']['l_fibo']
entry_fibo = config['default']['entry_fibo']
target_fibo = config['default']['target_fibo']
sl_fibo = config['default']['sl_fibo']


symbol_random = config['default']['symbol_random']
symbol_last = config['default']['symbol_last']
symbol_length = config['default']['symbol_length']

basic_secret_key = config['basic']['secret_key']
basic_secret_value = config['basic']['secret_value']
futures_secret_key = config['futures']['secret_key']
futures_secret_value = config['futures']['secret_value']


intersect_idx = config['default']['intersect_idx']
compare_account_trade = config['default']['compare_account_trade']
plotview = config['default']['plotview']
printout = config['default']['printout']

# fee_limit_tp = 0
# if tp_type == 'maker':
#     fee_limit_tp = (fee_limit + fee_tp) * qtyrate
# elif tp_type == 'taker':
#     fee_limit_tp = (fee_limit + fee_tp + fee_slippage) * qtyrate
#
# fee_limit_sl = (fee_limit + fee_sl + fee_slippage) * qtyrate


def print_condition():
    logger.info('-------------------------------')
    logger.info('version:%s' % str(version))
    logger.info('descrition:%s' % str(descrition))
    logger.info('exchange:%s' % str(exchange))
    logger.info('exchange_symbol:%s' % str(exchange_symbol))
    logger.info('futures:%s' % str(futures))
    logger.info('type:%s' % str(type))
    logger.info('maxleverage:%s' % str(maxleverage))
    logger.info('qtyrate:%s' % str(qtyrate))
    logger.info('krate_max:%s' % str(krate_max))
    logger.info('krate_min:%s' % str(krate_min))
    logger.info('seed:%s' % str(seed))

    logger.info('timeframe: %s' % timeframe)
    logger.info('period_days_ago: %s' % period_days_ago)
    logger.info('period_days_ago_till: %s' % period_days_ago_till)
    logger.info('period_interval: %s' % period_interval)
    logger.info('dt_flg: %s' % dt_flg)

    logger.info('c_risk_beyond_flg: %s' % c_risk_beyond_flg)
    logger.info('c_risk_beyond_max: %s' % c_risk_beyond_max)
    logger.info('c_risk_beyond_min: %s' % c_risk_beyond_min)

    logger.info('c_time_beyond_flg: %s' % c_time_beyond_flg)
    logger.info('c_time_beyond_rate: %s' % c_time_beyond_rate)

    logger.info('round_trip_count: %s' % round_trip_count)
    logger.info('compounding: %s' % compounding)
    logger.info('fcnt: %s' % fcnt)
    logger.info('loop_count: %s' % loop_count)
    logger.info('symbol_random: %s' % symbol_random)
    logger.info('symbol_last: %s' % symbol_last)
    logger.info('symbol_length: %s' % symbol_length)

    if not dt_flg:
        start_datetime_p = str((pd.to_datetime('today') - pd.Timedelta(str(period_days_ago) + ' days')).date())  # period_days_ago=0 -> today
        end_datetime_p = str((pd.to_datetime('today') - pd.Timedelta(str(period_days_ago_till) + ' days')).date())  # period_days_ago_till=0 -> today
    else:
        start_datetime_p = start_datetime
        end_datetime_p = end_datetime

    logger.info('period: %s ~ %s' % (start_datetime_p, end_datetime_p))
    logger.info('up_to_count: %s' % up_to_count)
    logger.info('condi_same_date: %s' % condi_same_date)
    logger.info('condi_compare_before_fractal: %s' % condi_compare_before_fractal)
    logger.info('condi_compare_before_fractal_mode: %s' % condi_compare_before_fractal_mode)
    logger.info('condi_compare_before_fractal_strait: %s' % condi_compare_before_fractal_strait)
    logger.info('condi_compare_before_fractal_shift: %s' % condi_compare_before_fractal_shift)

    logger.info('condi_plrate_adaptive: %s' % condi_plrate_adaptive)
    logger.info('condi_plrate_rate: %s' % condi_plrate_rate)
    logger.info('condi_plrate_rate_min: %s' % condi_plrate_rate_min)
    logger.info('condi_kelly_adaptive: %s' % condi_kelly_adaptive)
    logger.info('condi_kelly_window: %s' % condi_kelly_window)

    logger.info('o_fibo: %s' % o_fibo)
    logger.info('h_fibo: %s' % h_fibo)
    logger.info('l_fibo: %s' % l_fibo)

    logger.info('entry_fibo: %s' % entry_fibo)
    logger.info('target_fibo: %s' % target_fibo)
    logger.info('sl_fibo: %s' % sl_fibo)

    logger.info('fee:%s%%' % str(fee*100))
    logger.info('fee_limit:%s%%' % str(fee_limit*100))
    logger.info('fee_sl:%s%%' % str(fee_sl*100))
    logger.info('fee_tp:%s%%' % str(fee_tp*100))
    logger.info('tp_type:%s' % str(tp_type))
    # logger.info('fee_slippage:%s%%' % str(round(fee_slippage*100, 4)))
    # logger.info('(fee_limit_sl:%s%%' % round(float(fee_limit_sl)*100, 4))

    logger.info('intersect_idx: %s' % intersect_idx)
    logger.info('compare_account_trade: %s' % compare_account_trade)
    logger.info('plotview: %s' % plotview)
    logger.info('printout: %s' % printout)
    logger.info('-------------------------------')

client = None
if not futures:
    # basic
    client_basic = Client("basic_secret_key", "basic_secret_value")
    client = client_basic
else:
    # futures
    client_futures = Client("futures_secret_key", "futures_secret_value")
    client = client_futures

symbols = list()

## binance symbols
symbols_binance_futures = []
symbols_binance_futures_USDT = []
symbols_binance_futures_BUSD = []
symbols_binance_futures_USDT_BUSD = []

exchange_info = client.futures_exchange_info()

for s in exchange_info['symbols']:
    if s['contractType'] == 'PERPETUAL' and s['symbol'][-4:] == 'USDT':
        symbols_binance_futures_USDT.append(s['symbol'])
        symbols_binance_futures_USDT_BUSD.append(s['symbol'])
    elif s['contractType'] == 'PERPETUAL' and s['symbol'][-4:] == 'BUSD':
        symbols_binance_futures_BUSD.append(s['symbol'])
        symbols_binance_futures_USDT_BUSD.append(s['symbol'])
    symbols_binance_futures.append(s['symbol'])

def get_symbols():
    symbols = []
    if exchange_symbol == 'binance_usdt_perp':
        symbols = symbols_binance_futures_USDT
    elif exchange_symbol == 'binance_busd_perp':
        symbols = symbols_binance_futures_BUSD
    elif exchange_symbol == 'binance_usdt_busd_perp':
        symbols = symbols_binance_futures_USDT_BUSD

    if symbol_last:
        symbols = symbols[symbol_last:]
    if symbol_length:
        symbols = symbols[:symbol_length]
    if symbol_random:
        symbols = random.sample(symbols, len(symbols))
    logger.info(str(len(symbols)) +' ' + str(symbols))
    return symbols

import threading
import functools
import time

def synchronized(wrapped):
    lock = threading.Lock()
    @functools.wraps(wrapped)
    def _wrap(*args, **kwargs):
        with lock:
            # print ("Calling '%s' with Lock %s from thread %s [%s]"
            #        % (wrapped.__name__, id(lock),
            #        threading.current_thread().name, time.time()))
            result = wrapped(*args, **kwargs)
            # print ("Done '%s' with Lock %s from thread %s [%s]"
            #        % (wrapped.__name__, id(lock),
            #        threading.current_thread().name, time.time()))
            return result
    return _wrap


def get_mdd(x):
    """
    MDD(Maximum Draw-Down)
    :return: (peak_upper, peak_lower, mdd rate)
    """
    try:
        arr_v = np.array(x)
        peak_lower = np.argmax(np.maximum.accumulate(arr_v) - arr_v)
        peak_upper = np.argmax(arr_v[:peak_lower])
        return peak_upper, peak_lower, (arr_v[peak_lower] - arr_v[peak_upper]) / arr_v[peak_upper]
    except Exception as e:
        print(e)
    return None, None, None

def get_mdd_1(prices : pd.Series) -> float :
    peak = np.maximum.accumulate(prices) # 현시점까지의 고점
    trough = np.minimum.accumulate(prices) # 현시점까지의 저점
    dd = (trough - peak) / peak # 낙폭
    mdd = min(dd) # 낙폭 중 가장 큰 값, 즉 최대낙폭
    return mdd

def date_to_milliseconds(date_str: str) -> int:
    """Convert UTC date to milliseconds

    If using offset strings add "UTC" to date string e.g. "now UTC", "11 hours ago UTC"

    See dateparse docs for formats http://dateparser.readthedocs.io/en/latest/

    :param date_str: date in readable format, i.e. "January 01, 2018", "11 hours ago UTC", "now UTC"
    """
    # get epoch value in UTC
    epoch: datetime = datetime.utcfromtimestamp(0).replace(tzinfo=pytz.utc)
    # parse our date string
    d: Optional[datetime] = dateparser.parse(date_str, settings={'TIMEZONE': "UTC"})
    # if the date is not timezone aware apply UTC timezone
    if d.tzinfo is None or d.tzinfo.utcoffset(d) is None:
        d = d.replace(tzinfo=pytz.utc)

    # return the difference in time
    return int((d - epoch).total_seconds() * 1000.0)


@synchronized
def get_historical_ohlc_data_start_end(symbol, start_int, end_int, past_days=None, interval=None, futures=False):
    D = None
    start_date_str = None
    end_date_str = None
    try:
        """Returns historcal klines from past for given symbol and interval
        past_days: how many days back one wants to download the data"""
        if not futures:
            # basic
            client_basic = Client("basic_secret_key",
                                  "basic_secret_value")
            client = client_basic
        else:
            # futures
            client_futures = Client("futures_secret_key",
                                    "futures_secret_value")
            client = client_futures

        if not interval:
            interval = '1h'  # default interval 1 hour
        if not past_days:
            past_days = 10  # default past days 10

        # start_date_str = str((pd.to_datetime('today') - pd.Timedelta(str(start_int) + ' days')).date())
        start_date_str = str((pd.to_datetime('today') - pd.Timedelta(str(start_int) + ' days')).timestamp())
        if end_int is not None and end_int > 0:
            end_date_str = str((pd.to_datetime('today') - pd.Timedelta(str(end_int) + ' days')).timestamp())
        else:
            end_date_str = None
        try:
            if exchange_symbol == 'binance_usdt_perp' or exchange_symbol == 'binance_busd_perp' or exchange_symbol == 'binance_usdt_busd_perp':
                if futures:
                    D = pd.DataFrame(
                        client.futures_historical_klines(symbol=symbol, start_str=start_date_str, end_str=end_date_str, interval=interval))
                else:
                    D = pd.DataFrame(client.get_historical_klines(symbol=symbol, start_str=start_date_str, end_str=end_date_str, interval=interval))

        except Exception as e:
            time.sleep(0.1)
            logger.info(e)
            return D, start_date_str, end_date_str

        if D is not None and D.empty:
            return D, start_date_str, end_date_str

        D.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades',
                     'taker_base_vol', 'taker_quote_vol', 'is_best_match']
        D['open_date_time'] = [dt.datetime.fromtimestamp(x / 1000) for x in D.open_time]
        D['symbol'] = symbol
        D = D[['symbol', 'open_date_time', 'open', 'high', 'low', 'close', 'volume', 'num_trades', 'taker_base_vol',
               'taker_quote_vol']]
        D.rename(columns={
                            "open_date_time": "Date",
                            "open": "Open",
                            "high": "High",
                            "low": "Low",
                            "close": "Close",
                            "volume": "Volume",
                          }, inplace=True)
        new_names = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        D['Date'] = D['Date'].astype(str)
        D['Open'] = D['Open'].astype(float)
        D['High'] = D['High'].astype(float)
        D['Low'] = D['Low'].astype(float)
        D['Close'] = D['Close'].astype(float)
        D['Volume'] = D['Volume'].astype(float)
        D = D[new_names]
    except Exception as e:
        print('e in get_historical_ohlc_data_start_end:%s' % e)
        pass
    return D, start_date_str, end_date_str


def fractals_low_loopB(df, loop_count=1):
    for c in range(loop_count):
        i = Indicators(df)
        i.fractals()
        df = i.df
        df = df[~(df[['fractals_low']] == 0).all(axis=1)]
        df = df.dropna()
        df = df.drop(['fractals_high', 'fractals_low'], axis=1)
    return df

def fractals_low_loopA(df, fcnt=51, loop_count=1):
    for c in range(loop_count):
        window = 2 * fcnt + 1
        df['fractals_low'] = df['Low'].rolling(window, center=True).apply(lambda x: x[fcnt] == min(x), raw=True)
        df = df[~(df[['fractals_low']].isin([0, np.nan])).all(axis=1)]
        df = df.dropna()
        df = df.drop(['fractals_low'], axis=1)
    return df

def fractals_high_loopA(df, fcnt=5, loop_count=1):
    for c in range(loop_count):
        window = 2 * fcnt + 1
        df['fractals_high'] = df['High'].rolling(window, center=True).apply(lambda x: x[fcnt] == max(x), raw=True)
        df = df[~(df[['fractals_high']].isin([0, np.nan])).all(axis=1)]
        df = df.dropna()
        df = df.drop(['fractals_high'], axis=1)
    return df

def sma(source, period):
    return pd.Series(source).rolling(period).mean().values

def sma_df(df, p):
    i = Indicators(df)
    i.sma(period=p)
    return i.df

def get_precision(symbol):
    for x in exchange_info['symbols']:
        if x['symbol'] == symbol:
            return x['quantityPrecision']


def set_price(symbol, price, longshort):
    e_info = exchange_info['symbols']  # pull list of symbols
    for x in range(len(e_info)):  # find length of list and run loop
        if e_info[x]['symbol'] == symbol:  # until we find our coin
            a = e_info[x]["filters"][0]['tickSize']  # break into filters pulling tick size
            cost = round_step_size(price, float(a)) # convert tick size from string to float, insert in helper func with cost
            # 아래는 시장가의 비용 및 sleepage 를 보고 나중에 추가 또는 삭제 검토요
            # cost = cost - float(a) if longshort else cost + float(a)
            return cost


def step_size_to_precision(ss):
    return ss.find('1') - 1


def format_value(val, step_size_str):
    precision = step_size_to_precision(step_size_str)
    if precision > 0:
        return "{:0.0{}f}".format(val, precision)
    return math.floor(int(val))


def format_valueDown(val, step_size_str):
    precision = step_size_to_precision(step_size_str)
    if precision > 0:
        return "{:0.0{}f}".format(val, precision)
    return math.trunc(int(val))


def get_quantity_step_size_minqty(symbol):
    symbols = exchange_info['symbols']
    filters = [x['filters'] for x in symbols if x['symbol'] == symbol]
    filter = filters[0]
    stepsize = [x['stepSize'] for x in filter if x['filterType'] == 'LOT_SIZE']
    minqty = [x['minQty'] for x in filter if x['filterType'] == 'LOT_SIZE']
    return stepsize[0], minqty[0]


def backtest_trade45(tf, df, symbol, fcnt, longshort, df_lows_plot, df_highs_plot, wavepattern, trade_info, idx, wavepattern_l, wavepattern_tpsl_l):
    w = wavepattern
    # real condititon by fractal index
    real_condititon1 = True if (fcnt/2 < (w.idx_end - w.idx_start)) and w.idx_start == idx else False
    real_condititon2 = True if df.iloc[idx + int(fcnt/2), 0] < (w.dates[-1]) else False
    if not real_condititon1:
        if printout: logger.info('not real_condititon1 ')
        # print('%s not real_condititon1 ' % symbol)
        return trade_info, False
    if not real_condititon2:
        if printout: logger.info('not real_condititon2 ')
        # print('%s not real_condititon2 ' % symbol)
        return trade_info, False
    t = trade_info
    stats_history = t[0]
    order_history = t[1]
    asset_history = t[2]
    trade_count = t[3]
    fee_history = t[4]
    pnl_history = t[5]
    wavepattern_history = t[6]


    # df_smaN = sma_df(df, fcnt)

    w_start_price = w.values[0]  # wave1
    w_end_price = w.values[-1]  # wave5
    height_price = abs(w_end_price - w_start_price)
    o_fibo_value = height_price * o_fibo / 100 if o_fibo else 0
    # entry_price = w.values[7]  # wave4

    # entry_price = w.values[0] + height_price * entry_fibo if longshort else w.values[0] - height_price * entry_fibo  # 0.381 되돌림가정
    # sl_price = w.values[0] + height_price * sl_fibo if longshort else w.values[0] - height_price * sl_fibo
    # tp_price = entry_price + height_price * target_fibo if longshort else entry_price - height_price * target_fibo  # 0.5 되돌림가정

    w0 = w.values[0]
    w1 = w.values[1]
    w2 = w.values[3]
    w3 = w.values[5]
    w4 = w.values[7]
    w5 = w.values[9]

    # tp_price = w5 + abs(w5-w4)*10/20 if longshort else w5 - abs(w5-w4)*10/20
    tp_price = w5
    entry_price = w4
    # entry_price = w0 + height_price*0.5 if longshort else w0 - height_price*0.5
    sl_price = w0
    out_price = None
    entry_price = set_price(symbol, entry_price, longshort)
    sl_price = set_price(symbol, sl_price, longshort)
    tp_price = set_price(symbol, tp_price, longshort)

    b_symbol = abs(tp_price - entry_price) / abs(sl_price - entry_price)  # one trade profitlose rate
    if condi_plrate_adaptive:
        if b_symbol > condi_plrate_rate or condi_plrate_rate_min > b_symbol:
            return trade_info, False

    qtyrate_k = qtyrate
    if condi_kelly_adaptive:
        if stats_history:
            f_trade_stats = stats_history[-1][9]  # trade_stats index 9, f value kelly_index
            if f_trade_stats <= 0:
                qtyrate_k = krate_min
            elif f_trade_stats <= qtyrate:
                qtyrate_k = qtyrate
            elif f_trade_stats > qtyrate and f_trade_stats< krate_max:
                qtyrate_k = f_trade_stats
            elif f_trade_stats >= krate_max:
                qtyrate_k = krate_max
    # else:
    #     qtyrate_k = qtyrate

    # print('qtyrate_k', qtyrate_k)

    if c_risk_beyond_flg:
        pnl_percent_sl = (abs(entry_price - sl_price) / entry_price) * qtyrate_k
        if pnl_percent_sl >= c_risk_beyond_max:  # decrease max sl rate   0.1 = 10%
            # logger.info(symbol + ' _c_risk_beyond_max : ' + str(pnl_percent_sl))
            return trade_info, False

        pnl_percent_tp = (abs(tp_price - entry_price) / entry_price) * qtyrate_k
        if pnl_percent_tp <= c_risk_beyond_min:  # reduce low tp rate  0.005 = 0.5%
            # logger.info(symbol + ' _c_risk_beyond_min : ' + str(pnl_percent_tp))
            return trade_info, False

    if entry_price == sl_price:
        logger.info(symbol + ' _entry_price == sl_price')
        return trade_info, False

    # here when df_active_next, trade is possible  / if elsecase, out  즉 웨이브가 끝나고 하나의 봉을 더 보고 그 다음부터 거래가 가능
    df_active_next = df[w.idx_end + 1: w.idx_end + 2]
    if not df_active_next.empty:
        df_active_next_high = df_active_next['High'].iat[0]
        df_active_next_low = df_active_next['Low'].iat[0]

        c_next_ohlc_beyond = df_active_next_low < entry_price if longshort else df_active_next_high > entry_price
        if c_next_ohlc_beyond:
            return trade_info, False
    else:
        return trade_info, False


    df_active = df[w.idx_end + 1:]
    # df_active = df[w.idx_end:]  # xxxxx 틀렸음

    dates = df_active.Date.tolist()
    closes = df_active.Close.tolist()
    trends = df_active.High.tolist() if longshort else df_active.Low.tolist()
    detrends = df_active.Low.tolist() if longshort else df_active.High.tolist()

    # try:
    #     active_max_value = max(df_active.High.tolist(), default=w_end_price)
    #     active_min_value = min(df_active.Low.tolist(), default=w_end_price)
    # except Exception as e:
    #     logger.error('active_max_value:' + str(e))
    #     return trade_info, False


    # c_active_min_max = (active_min_value > entry_price and active_max_value < (w_end_price + o_fibo_value)) \
    #                         if longshort else \
    #                     (active_max_value < entry_price and active_min_value > (w_end_price - o_fibo_value))
    #
    # if not c_active_min_max:
    #     print('case c_active_min_max')
    #     return trade_info, False

    c_out_trend_beyond = False
    c_positioning = False
    c_profit = False
    c_stoploss = False

    position_enter_i = []
    position = False
    h_id = dt.datetime.now().timestamp()

    w_idx_width = w.idx_end - w.idx_start
    if closes:
        for i, close in enumerate(closes):
            c_out_idx_width_beyond = True if (i/w_idx_width >= c_time_beyond_rate) else False
            c_out_trend_beyond = trends[i] >= (w_end_price + o_fibo_value) if longshort else trends[i] <= (w_end_price - o_fibo_value)

            c_positioning = (position is False and detrends[i] <= entry_price and detrends[i] > sl_price) if longshort else (position is False and detrends[i] >= entry_price and detrends[i] < sl_price)
            c_profit = (position and trends[i] >= tp_price) if longshort else (position and trends[i] <= tp_price)
            c_stoploss = (position and detrends[i] <= sl_price) if longshort else (position and detrends[i] >= sl_price)
            c_stoploss_direct = (detrends[i] <= sl_price and trends[i] >= entry_price) if longshort else (detrends[i] >= sl_price and trends[i] <= entry_price)

            if c_stoploss_direct:
                position = True
                c_stoploss = True
                position_enter_i = [dates[i], entry_price]
                logger.info('c_stoplost_direct')
                open_order = {
                    'id': h_id,
                    'datetime': dt.datetime.now(),
                    'status': 'NEW',
                    'symbol': symbol,
                    'longshort': longshort,
                    'entry': entry_price,
                    'target': tp_price,
                    'sl_price': sl_price,
                    'limit_orderId': 'order_limit[orderId]',
                    'sl_orderId': 'order_sl[orderId]',
                    'tp_orderId': None,
                    'wavepattern': wavepattern,
                    'data': []
                }
                # create history NEW
                open_order_history.append(open_order)
                # dump_history_pkl()

            if position is False and c_out_idx_width_beyond and c_time_beyond_flg:
                if printout:  logger.info('c_out_idx_width_beyond , ', w_idx_width, i)
                # print('c_out_idx_width_beyond, ', w_idx_width, i)
                return trade_info, False
            elif position is False and c_out_trend_beyond:
                if printout:  logger.info('c_out_trend_beyond ', i, close)
                # print('c_out_trend_beyond xxxxx ', i, close)
                return trade_info, False
            elif position is False and c_positioning:
                position = True
                position_enter_i = [dates[i], entry_price]
                if printout: logger.info('===>c_positioning (c_positioning), ', i, close, position_enter_i)
                # print('c_positioning (c_positioning), ', i, close, position_enter_i)
                open_order = {
                    'id': h_id,
                    'datetime': dt.datetime.now(),
                    'status': 'NEW',
                    'symbol': symbol,
                    'longshort': longshort,
                    'entry': entry_price,
                    'target': tp_price,
                    'sl_price': sl_price,
                    'limit_orderId': 'order_limit[orderId]',
                    'sl_orderId': 'order_sl[orderId]',
                    'tp_orderId': None,
                    'wavepattern': wavepattern,
                    'data': []
                }
                # create history NEW
                # open_order_history.append(open_order)
                # dump_history_pkl()
                if symbol == 'DASHUSDT':
                    print('1')
                if plotview:
                    plot_pattern_m(df=df, wave_pattern=[[i, wavepattern.dates[0], id(wavepattern), wavepattern]],
                                   df_lows_plot=df_lows_plot, df_highs_plot=df_highs_plot, trade_info=trade_info, title=str(
                            symbol + ' %s ' % str(longshort) + str('_entering') + str(position_enter_i)))
            if position is True:

                if c_profit or c_stoploss:


                    fee_limit_tp = 0
                    if tp_type == 'maker':
                        fee_limit_tp = (fee_limit + fee_tp) * qtyrate_k
                    elif tp_type == 'taker':
                        fee_limit_tp = (fee_limit + fee_tp + fee_slippage) * qtyrate_k
                    fee_limit_sl = (fee_limit + fee_sl + fee_slippage) * qtyrate_k

                    fee_percent = 0
                    pnl_percent = 0
                    win_lose_flg = 0
                    trade_inout_i = []
                    if c_stoploss:
                        win_lose_flg = 0
                        position_sl_i = [dates[i], sl_price]
                        pnl_percent = -(abs(entry_price - sl_price) / entry_price) * qtyrate_k
                        fee_percent = fee_limit_sl
                        trade_count.append(0)
                        trade_inout_i = [position_enter_i, position_sl_i, longshort, '-']
                        out_price = sl_price
                        order_history.append(trade_inout_i)
                        if printout: logger.info('- stoploss, ', i, close, position_enter_i, position_sl_i)
                        # update_history_status(open_order_history, symbol, h_id, 'DONE')

                    if c_profit:
                        win_lose_flg = 1
                        position_pf_i = [dates[i], tp_price]
                        pnl_percent = (abs(tp_price - entry_price) / entry_price) * qtyrate_k
                        fee_percent = fee_limit_tp
                        trade_count.append(1)
                        trade_inout_i = [position_enter_i, position_pf_i, longshort, '+']
                        out_price = tp_price
                        order_history.append(trade_inout_i)

                        if printout: logger.info('+ profit, ', i, close, position_enter_i, position_pf_i)
                        # update_history_status(open_order_history, symbol, h_id, 'DONE')


                    asset_history_pre = asset_history[-1] if asset_history else seed
                    asset_new = asset_history_pre * (1 + pnl_percent - fee_percent)
                    pnl_history.append(asset_history_pre*pnl_percent)
                    fee_history.append(asset_history_pre*fee_percent)
                    asset_history.append(asset_new)
                    wavepattern_history.append(wavepattern)

                    winrate = round((sum(trade_count)/len(trade_count))*100, 2)

                    asset_min = seed
                    asset_max = seed

                    if len(stats_history) > 0:
                        asset_last_min = stats_history[-1][-2]
                        asset_min = asset_new if asset_new < asset_last_min else asset_last_min
                        asset_last_max = stats_history[-1][-1]
                        asset_max = asset_new if asset_new > asset_last_max else asset_last_max

                    df_s = pd.DataFrame.from_records(stats_history)

                    p_cum = winrate/100  # win rate
                    b_cum = (df_s[8].sum() + b_symbol)/(len(df_s) + 1) if len(stats_history) > 0 else b_symbol  # mean - profitlose rate, df_s[6] b_cum index
                    q_cum = 1 - p_cum # lose rate
                    tpi_cum = round(p_cum*(1+b_cum), 2)  # trading perfomance index
                    f_cum = round(p_cum - (q_cum/b_cum), 2)  # kelly index

                    f = f_cum
                    tpi = tpi_cum
                    b = b_cum

                    if condi_kelly_adaptive:
                        if len(df_s) >= condi_kelly_window:
                            df_s_window = df_s.iloc[-condi_kelly_window:]
                            p = df_s_window[4].sum()/len(df_s_window)
                            b = (df_s_window[11].sum() + b_symbol)/(len(df_s_window) + 1)  # mean in kelly window - profitlose rate
                            q = 1 - p  # lose rate
                            tpi = round(p * (1 + b), 2)  # trading perfomance index
                            f = round(p - (q / b), 2)  # kelly index

                    trade_stats = [len(trade_count), winrate, asset_new, symbol, win_lose_flg, 'WIN' if win_lose_flg else 'LOSE', f_cum, tpi_cum, b_cum, f, tpi, b, b_symbol, str(qtyrate_k), str(round(pnl_percent, 4)), sum(pnl_history), sum(fee_history), round(asset_min, 2), round(asset_max, 2)]
                    stats_history.append(trade_stats)
                    trade_info = [stats_history, order_history, asset_history, trade_count, fee_history, pnl_history, wavepattern_history]

                    wavepattern_tpsl_l.append([idx, wavepattern.dates[0], id(wavepattern), wavepattern])
                    # if printout: print(str(trade_stats))
                    # print(symbol, fcnt, longshort, trade_inout_i[0][0][2:-3], trade_inout_i[0][1], '~', trade_inout_i[1][0][-8:-3], trade_inout_i[1][1], ' | %s' % str([w.values[0], w.values[-1], w.values[7]]), str(trade_stats))

                    # w2_rate = float(re.findall('\(([^)]+)', wavepattern.labels[3])[0])  # extracts string in bracket()
                    # w3_rate = float(re.findall('\(([^)]+)', wavepattern.labels[5])[0])  # extracts string in bracket()

                    # if float(trade_stats[4]) > 0 :

                    if True :
                    # if w2_rate > 0.9:
                        s_11 = symbol + '           '
                        logger.info('%s %s %s %s %s x%s %s-%s %s %s %s %s %s %s - %s' % (str(qtyrate_k), str(condi_compare_before_fractal_mode)+' :shift='+str(condi_compare_before_fractal_shift), timeframe, s_11[:11], tf, qtyrate_k, period_days_ago, period_days_ago_till,  fcnt, 'L' if longshort else 'S', trade_inout_i[0][0][2:-3], '-',  trade_inout_i[1][0][8:-3], str(trade_stats), str([entry_price, sl_price, tp_price, out_price])))
                              # , w2_rate,
                              #     w3_rate, w3_rate / w2_rate)
                    # print(symbol, trade_inut_i[0][0][2:-3], str(trade_stats))

                    if longshort is not None and len(trade_info[1])>0:
                        if plotview:
                            plot_pattern_m(df=df, wave_pattern=[[i, wavepattern.dates[0], id(wavepattern), wavepattern]], df_lows_plot=df_lows_plot, df_highs_plot=df_highs_plot, trade_info=trade_info, title=str(
                                symbol + ' %s '% str(longshort) + str(trade_stats)))

                    return trade_info, True
    return trade_info, False


def check_has_same_wavepattern(symbol, fcnt, w_l, wavepattern):
    for wl in reversed(w_l):
        if symbol == wl[0] and fcnt == wl[1]:
            a0 = wl[-1].dates[0]
            b0 = wavepattern.dates[0]
            c0 = wl[-1].values[0]
            d0 = wavepattern.values[0]
            c0 = a0 == b0 and c0 == d0
            if condi_same_date:
                if a0 == b0:
                    return True

            a2 = wl[-1].dates[3]
            b2 = wavepattern.dates[3]
            c2 = wl[-1].values[3]
            d2 = wavepattern.values[3]
            c2 = a2 == b2 and c2 == d2

            a3 = wl[-1].dates[5]
            b3 = wavepattern.dates[5]
            c3 = wl[-1].values[5]
            d3 = wavepattern.values[5]
            c3 = a3 == b3 and c3 == d3

            a4 = wl[-1].dates[7]
            b4 = wavepattern.dates[7]
            c4 = wl[-1].values[7]
            d4 = wavepattern.values[7]
            c4 = a4 == b4 and c4 == d4

            a5 = wl[-1].dates[-1]
            b5 = wavepattern.dates[-1]
            c5 = wl[-1].values[-1]
            d5 = wavepattern.values[-1]
            c5 = a5 == b5 and c5 == d5

            if (c0 and c4) or (c4 and c5) or (c0 and c5) or (c0 and c3) or (c0 and c2 and c4) or (c0 and c2 and c3) or (c0 and c3 and c4):
                return True

            eq_dates = np.array_equal(np.array(wavepattern.dates), np.array(wl[-1].dates))
            eq_values = np.array_equal(np.array(wavepattern.values), np.array(wl[-1].values))
            if eq_dates or eq_values:
                return True

    return False

wavepattern_l = list()

def loopsymbol(tf, symbol, i, trade_info):
    ###################
    ## data settting ##
    ###################
    # 1day:1440m

    # try:

    past_days = 1
    timeunit = 'm'
    bin_size = str(tf) + timeunit
    start_int = i
    end_int = start_int - period_interval

    if start_int == 0 or end_int < 0:
        end_int = None

    if end_int is not None and start_int <= end_int:
        return trade_info

    # start = time.perf_counter()
    # logger.info(f'{symbol} start: {time.strftime("%H:%M:%S")}')

    df, start_date, end_date = get_historical_ohlc_data_start_end(symbol,
                                                                      start_int=start_int,
                                                                      end_int=end_int,
                                                                      past_days=past_days,
                                                                      interval=bin_size, futures=futures)
    # print(f'Finished in {round(time.perf_counter() - start, 2)} second(s)')

    if df is not None:
        if df.empty:
            return trade_info

    df_all = df
    longshort = None
    if df is not None and df.empty:
        return trade_info
    try:
        wa = WaveAnalyzer(df=df_all, verbose=True)
    except:
        return trade_info

    wave_options = WaveOptionsGenerator5(up_to=up_to_count)
    idxs = list()
    lows_idxs = list()
    highs_idxs = list()
    df_lows_plot = None
    df_highs_plot = None

    for fc in fcnt:
        if 'long' in type:
            df_lows = fractals_low_loopA(df_all, fcnt=fc, loop_count=loop_count)
            df_lows_plot = df_lows[['Date', 'Low']]

            if condi_compare_before_fractal:
                if not df_lows.empty:
                    for i in range(condi_compare_before_fractal_shift, 0, -1):
                        try:
                            if condi_compare_before_fractal_strait:
                                i = condi_compare_before_fractal_shift
                            df_lows['Low_before'] = df_lows.Low.shift(i).fillna(0)
                            if condi_compare_before_fractal_mode == 1:
                                df_lows['compare_flg'] = df_lows.apply(lambda x: 1 if x['Low'] > x['Low_before'] else 0, axis=1)
                            elif condi_compare_before_fractal_mode == 2:
                                df_lows['compare_flg'] = df_lows.apply(
                                    lambda x: 1 if x['Low'] >= x['Low_before'] else 0, axis=1)
                            elif condi_compare_before_fractal_mode == 3:
                                df_lows['compare_flg'] = df_lows.apply(lambda x: 1 if x['Low'] < x['Low_before'] else 0, axis=1)
                            elif condi_compare_before_fractal_mode == 4:
                                df_lows['compare_flg'] = df_lows.apply(
                                    lambda x: 1 if x['Low'] <= x['Low_before'] else 0, axis=1)
                            elif condi_compare_before_fractal_mode == 5:
                                df_lows['compare_flg'] = df_lows.apply(
                                    lambda x: 1 if x['Low'] == x['Low_before'] else 0, axis=1)
                            df_lows = df_lows.drop(df_lows[df_lows['compare_flg'] == 0].index)

                            if not df_lows.empty:
                                del df_lows['Low_before']
                                del df_lows['compare_flg']
                                pass
                        except:
                            pass


            impulse = Impulse('impulse')
            lows_idxs = df_lows.index.tolist()
            idxs = lows_idxs

        if 'short' in type:
            df_highs = fractals_high_loopA(df_all, fcnt=fc, loop_count=loop_count)
            df_highs_plot = df_highs[['Date', 'High']]

            if condi_compare_before_fractal:
                if not df_highs.empty:
                    for i in range(condi_compare_before_fractal_shift, 0, -1):
                        try:
                            df_highs['High_before'] = df_highs.High.shift(i).fillna(10000000000)
                            if condi_compare_before_fractal_mode == 1:
                                df_highs['compare_flg'] = df_highs.apply(lambda x: 1 if x['High'] < x['High_before'] else 0, axis=1)
                            elif condi_compare_before_fractal_mode == 2:
                                df_highs['compare_flg'] = df_highs.apply(lambda x: 1 if x['High'] <= x['High_before'] else 0, axis=1)
                            elif condi_compare_before_fractal_mode == 3:
                                df_highs['compare_flg'] = df_highs.apply(lambda x: 1 if x['High'] > x['High_before'] else 0, axis=1)
                            elif condi_compare_before_fractal_mode == 4:
                                df_highs['compare_flg'] = df_highs.apply(lambda x: 1 if x['High'] >= x['High_before'] else 0, axis=1)
                            elif condi_compare_before_fractal_mode == 5:
                                df_highs['compare_flg'] = df_highs.apply(lambda x: 1 if x['High'] == x['High_before'] else 0, axis=1)
                            df_highs = df_highs.drop(df_highs[df_highs['compare_flg'] == 0].index)

                            if not df_highs.empty:
                                del df_lows['High_before']
                                del df_lows['compare_flg']
                                pass
                        except:
                            pass

            downimpulse = DownImpulse('downimpulse')
            highs_idxs = df_highs.index.tolist()
            idxs = highs_idxs

        rules_to_check = list()
        wavepatterns = set()
        # wavepattern_l = []
        wavepattern_tpsl_l = []
        wave_option_plot_l = []
        if ('long' in type) and ('short' in type): idxs = sorted(list(set(lows_idxs) | set(highs_idxs)))
        for i in idxs:
            for wave_opt in wave_options.options_sorted:
                if i in lows_idxs:
                    waves = wa.find_impulsive_wave(idx_start=i, wave_config=wave_opt.values)
                    longshort = True
                    rules_to_check = [impulse]
                elif i in highs_idxs:
                    waves = wa.find_downimpulsive_wave(idx_start=i, wave_config=wave_opt.values)
                    longshort = False
                    rules_to_check = [downimpulse]

                if waves:
                    wavepattern = WavePattern(waves, verbose=True)
                    if (wavepattern.idx_end - wavepattern.idx_start + 1) >= fc / 2:
                        for rule in rules_to_check:
                            if wavepattern.check_rule(rule):
                                if wavepattern in wavepatterns:
                                    continue
                                else:
                                    if not check_has_same_wavepattern(symbol, fc, wavepattern_l, wavepattern):
                                        wavepatterns.add(wavepattern)
                                        wavepattern_l.append([symbol, fc, i, wavepattern.dates[0], id(wavepattern), wavepattern])
                                        wave_option_plot_l.append([
                                            [str(wavepattern.dates[-1])],
                                            [wavepattern.values[-1]],
                                            [str(wave_opt.values)]
                                        ])



                                        # same condition with real play
                                        c_check_wave_identical = False
                                        c_check_wave_identical_2_3_4 = False
                                        # if len(open_order_history) > 0:
                                        #     open_order_this_symbol = [x for x in open_order_history if
                                        #                               x['symbol'] == symbol]
                                        #     if len(open_order_this_symbol) > 0:
                                        #         open_order_wavepattern = open_order_this_symbol[0]['wavepattern']
                                        #
                                        #         # check identical check
                                        #         c_check_wave_identical = True if (
                                        #                                                      open_order_wavepattern.dates == wavepattern.dates) and (
                                        #                                                      open_order_wavepattern.values == wavepattern.values) else False
                                        #         # 2,3,4 wave same check
                                        #         c_check_wave_identical_2_3_4 = True if (open_order_wavepattern.dates[
                                        #                                                 2:-2] == wavepattern.dates[
                                        #                                                          2:-2]) and (
                                        #                                                            open_order_wavepattern.values[
                                        #                                                            2:-2] == wavepattern.values[
                                        #                                                                     2:-2]) else False
                                        #
                                        # if c_check_wave_identical or c_check_wave_identical_2_3_4:
                                        #     # print(c_check_wave_identical, c_check_wave_identical_2_3_4)
                                        #     if c_check_wave_identical_2_3_4:
                                        #         # print('234')
                                        #         pass
                                        #     return trade_info



                                        # w2_rate = float(re.findall('\(([^)]+)', wavepattern.labels[3])[
                                        #                     0])  # extracts string in bracket()
                                        # w3_rate = float(re.findall('\(([^)]+)', wavepattern.labels[5])[
                                        #                     0])  # extracts string in bracket()
                                        #
                                        # if w3_rate/w2_rate > 0:
                                        if True:

                                            trade_info, trade_flg = backtest_trade45(tf, df_all, symbol, fc, longshort, df_lows_plot, df_highs_plot, wavepattern, trade_info, i, wavepattern_l, wavepattern_tpsl_l)

                                            if printout:
                                                logger.info(f'{rule.name} found: {wave_opt.values}')
                                                logger.info(f'good... {(wavepattern.idx_end - wavepattern.idx_start + 1) }/{fc}found')

                                            if plotview and trade_flg:
                                                # if len(wavepattern_l) > 0:
                                                # t = bin_size + '_' + str(i) + ':' + rule.name + ' ' + str(wave_opt.values) + ' ' + str(trade_info[0][-1] if trade_info[0] else [])
                                                t = bin_size + '_' + str(i) + ':' + rule.name + ' ' + str(
                                                    wave_opt.values) + ' ' + str(
                                                    trade_info[0][-1] if trade_info[0] else [])
                                                plot_pattern_m(df=df_all, wave_pattern=wavepattern_tpsl_l,
                                                               df_lows_plot=df_lows_plot,
                                                               df_highs_plot=df_highs_plot,
                                                               trade_info=trade_info,
                                                               wave_options=wave_option_plot_l,
                                                               title='tpsl_%s_' % str(fc) + t + ' %s' % str(condi_compare_before_fractal_mode))



                                    else:
                                        if printout:
                                            logger.info(f'{rule.name} found: {wave_opt.values}')
                                            logger.info(f'not good... {(wavepattern.idx_end - wavepattern.idx_start + 1)}/{fc}found')


    # if plotview:
    #     if len(wavepattern_l) > 0:
    #         # t = bin_size + '_' + str(i) + ':' + rule.name + ' ' + str(wave_opt.values) + ' ' + str(trade_info[0][-1] if trade_info[0] else [])
    #         t = bin_size + '_' + str(i) + ':' + 'rule.name' + ' ' + 'str(wave_opt.values)' + ' ' + str(trade_info[0][-1] if trade_info[0] else [])
    #         plot_pattern_m(df=df_all, wave_pattern=wavepattern_tpsl_l,
    #                        df_lows_plot=df_lows_plot,
    #                        df_highs_plot=df_highs_plot,
    #                        trade_info=trade_info,
    #                        wave_options=wave_option_plot_l, title='tpsl_%s_' % str(fc) + t)
    #     else:
    #         print(f'not found {wavepattern_l}')
    #         pass


    return trade_info
    # except Exception as e:
    #     logger.info('loopsymbol Exception : %s ' % e)

def get_i_r(list, key, value):
    r = [[i, x] for i, x in enumerate(list) if x[key] == value]
    if len(r) == 1:
        return r[0][0], r[0][1]
    return None, None


def update_history_status(open_order_history, symbol, h_id, new_status):
    history_idx, history_id = get_i_r(open_order_history, 'id', h_id)
    history_id['status'] = new_status  # update new status
    open_order_history[history_idx] = history_id  # replace history
    # print(symbol + ' _update history to new status:%s' % new_status)
    # logger.info(symbol + ' _update history to new status:%s' % new_status)
    # dump_history_pkl()


def single(symbols, i, trade_info, *args):
    for symbol in symbols:
        for tf in timeframe:
            loopsymbol(tf, symbol, i, trade_info)
    return trade_info


def round_trip(i):
    round_trip = []
    asset_history = [seed]
    trade_count = []
    fee_history = []
    pnl_history = []
    symbols = get_symbols()
    if symbol_random:
        symbols = random.sample(symbols, len(symbols))
    if symbol_last:
        symbols = symbols[symbol_last:]
    if symbol_length:
        symbols = symbols[:symbol_length]
    # single
    single(symbols, i, trade_info)
    round_trip.append(trade_info[2][-1])
    logger.info(i, ' | ', trade_info[2][-1], len(trade_count), ' | ', asset_history[-1])
    return round_trip


def account_trades_df(symbols):
    df = None
    # data = [{'symbol': 'DEFIUSDT', 'id': 32890705, 'orderId': 3932063139, 'side': 'SELL', 'price': '707.2', 'qty': '0.513', 'realizedPnl': '0', 'marginAsset': 'USDT', 'quoteQty': '362.7936', 'commission': '0.00020614', 'commissionAsset': 'BNB', 'time': 1676866475190, 'positionSide': 'SHORT', 'buyer': False, 'maker': True}, {'symbol': 'DEFIUSDT', 'id': 32890706, 'orderId': 3932063139, 'side': 'SELL', 'price': '707.2', 'qty': '3.193', 'realizedPnl': '0', 'marginAsset': 'USDT', 'quoteQty': '2258.0896', 'commission': '0.00128308', 'commissionAsset': 'BNB', 'time': 1676866475192, 'positionSide': 'SHORT', 'buyer': False, 'maker': True}, {'symbol': 'DEFIUSDT', 'id': 32900146, 'orderId': 3932063140, 'side': 'BUY', 'price': '723.5', 'qty': '0.302', 'realizedPnl': '-4.92260000', 'marginAsset': 'USDT', 'quoteQty': '218.4970', 'commission': '0.00024650', 'commissionAsset': 'BNB', 'time': 1676890901682, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32900147, 'orderId': 3932063140, 'side': 'BUY', 'price': '723.5', 'qty': '1.033', 'realizedPnl': '-16.83790000', 'marginAsset': 'USDT', 'quoteQty': '747.3755', 'commission': '0.00084316', 'commissionAsset': 'BNB', 'time': 1676890901683, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32900148, 'orderId': 3932063140, 'side': 'BUY', 'price': '723.7', 'qty': '0.028', 'realizedPnl': '-0.46200000', 'marginAsset': 'USDT', 'quoteQty': '20.2636', 'commission': '0.00002286', 'commissionAsset': 'BNB', 'time': 1676890901683, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32900149, 'orderId': 3932063140, 'side': 'BUY', 'price': '723.7', 'qty': '0.090', 'realizedPnl': '-1.48500000', 'marginAsset': 'USDT', 'quoteQty': '65.1330', 'commission': '0.00007348', 'commissionAsset': 'BNB', 'time': 1676890901683, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32900150, 'orderId': 3932063140, 'side': 'BUY', 'price': '723.7', 'qty': '0.963', 'realizedPnl': '-15.88950000', 'marginAsset': 'USDT', 'quoteQty': '696.9231', 'commission': '0.00078624', 'commissionAsset': 'BNB', 'time': 1676890901683, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32900151, 'orderId': 3932063140, 'side': 'BUY', 'price': '723.7', 'qty': '1.290', 'realizedPnl': '-21.28500000', 'marginAsset': 'USDT', 'quoteQty': '933.5730', 'commission': '0.00105322', 'commissionAsset': 'BNB', 'time': 1676890901683, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32988327, 'orderId': 3936576835, 'side': 'SELL', 'price': '687.6', 'qty': '0.061', 'realizedPnl': '0', 'marginAsset': 'USDT', 'quoteQty': '41.9436', 'commission': '0.00002452', 'commissionAsset': 'BNB', 'time': 1677092791056, 'positionSide': 'SHORT', 'buyer': False, 'maker': True}, {'symbol': 'DEFIUSDT', 'id': 32988328, 'orderId': 3936576835, 'side': 'SELL', 'price': '687.6', 'qty': '0.505', 'realizedPnl': '0', 'marginAsset': 'USDT', 'quoteQty': '347.2380', 'commission': '0.00020296', 'commissionAsset': 'BNB', 'time': 1677092791147, 'positionSide': 'SHORT', 'buyer': False, 'maker': True}, {'symbol': 'DEFIUSDT', 'id': 32988329, 'orderId': 3936576835, 'side': 'SELL', 'price': '687.6', 'qty': '1.922', 'realizedPnl': '0', 'marginAsset': 'USDT', 'quoteQty': '1321.5672', 'commission': '0.00077248', 'commissionAsset': 'BNB', 'time': 1677092791190, 'positionSide': 'SHORT', 'buyer': False, 'maker': True}, {'symbol': 'DEFIUSDT', 'id': 32988330, 'orderId': 3936576835, 'side': 'SELL', 'price': '687.6', 'qty': '1.440', 'realizedPnl': '0', 'marginAsset': 'USDT', 'quoteQty': '990.1440', 'commission': '0.00057876', 'commissionAsset': 'BNB', 'time': 1677092791232, 'positionSide': 'SHORT', 'buyer': False, 'maker': True}, {'symbol': 'DEFIUSDT', 'id': 32988331, 'orderId': 3936576835, 'side': 'SELL', 'price': '687.6', 'qty': '1.215', 'realizedPnl': '0', 'marginAsset': 'USDT', 'quoteQty': '835.4340', 'commission': '0.00048832', 'commissionAsset': 'BNB', 'time': 1677092791337, 'positionSide': 'SHORT', 'buyer': False, 'maker': True}, {'symbol': 'DEFIUSDT', 'id': 32995833, 'orderId': 3936576836, 'side': 'BUY', 'price': '698.6', 'qty': '0.009', 'realizedPnl': '-0.09900000', 'marginAsset': 'USDT', 'quoteQty': '6.2874', 'commission': '0.00000723', 'commissionAsset': 'BNB', 'time': 1677110489921, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32995834, 'orderId': 3936576836, 'side': 'BUY', 'price': '698.6', 'qty': '0.413', 'realizedPnl': '-4.54300000', 'marginAsset': 'USDT', 'quoteQty': '288.5218', 'commission': '0.00033220', 'commissionAsset': 'BNB', 'time': 1677110489921, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32995835, 'orderId': 3936576836, 'side': 'BUY', 'price': '698.6', 'qty': '0.368', 'realizedPnl': '-4.04800000', 'marginAsset': 'USDT', 'quoteQty': '257.0848', 'commission': '0.00029601', 'commissionAsset': 'BNB', 'time': 1677110489921, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32995836, 'orderId': 3936576836, 'side': 'BUY', 'price': '698.6', 'qty': '0.405', 'realizedPnl': '-4.45500000', 'marginAsset': 'USDT', 'quoteQty': '282.9330', 'commission': '0.00032577', 'commissionAsset': 'BNB', 'time': 1677110489921, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32995837, 'orderId': 3936576836, 'side': 'BUY', 'price': '698.6', 'qty': '0.300', 'realizedPnl': '-3.30000000', 'marginAsset': 'USDT', 'quoteQty': '209.5800', 'commission': '0.00024131', 'commissionAsset': 'BNB', 'time': 1677110489921, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32995838, 'orderId': 3936576836, 'side': 'BUY', 'price': '698.7', 'qty': '0.928', 'realizedPnl': '-10.30080000', 'marginAsset': 'USDT', 'quoteQty': '648.3936', 'commission': '0.00074656', 'commissionAsset': 'BNB', 'time': 1677110489921, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32995839, 'orderId': 3936576836, 'side': 'BUY', 'price': '698.7', 'qty': '1.097', 'realizedPnl': '-12.17670000', 'marginAsset': 'USDT', 'quoteQty': '766.4739', 'commission': '0.00088252', 'commissionAsset': 'BNB', 'time': 1677110489921, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32995840, 'orderId': 3936576836, 'side': 'BUY', 'price': '698.7', 'qty': '0.486', 'realizedPnl': '-5.39460000', 'marginAsset': 'USDT', 'quoteQty': '339.5682', 'commission': '0.00039098', 'commissionAsset': 'BNB', 'time': 1677110489921, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32995841, 'orderId': 3936576836, 'side': 'BUY', 'price': '698.8', 'qty': '0.037', 'realizedPnl': '-0.41440000', 'marginAsset': 'USDT', 'quoteQty': '25.8556', 'commission': '0.00002977', 'commissionAsset': 'BNB', 'time': 1677110489921, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32995842, 'orderId': 3936576836, 'side': 'BUY', 'price': '698.8', 'qty': '1.016', 'realizedPnl': '-11.37920000', 'marginAsset': 'USDT', 'quoteQty': '709.9808', 'commission': '0.00081748', 'commissionAsset': 'BNB', 'time': 1677110489921, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32995843, 'orderId': 3936576836, 'side': 'BUY', 'price': '698.9', 'qty': '0.034', 'realizedPnl': '-0.38420000', 'marginAsset': 'USDT', 'quoteQty': '23.7626', 'commission': '0.00002736', 'commissionAsset': 'BNB', 'time': 1677110489921, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32995844, 'orderId': 3936576836, 'side': 'BUY', 'price': '698.9', 'qty': '0.038', 'realizedPnl': '-0.42940000', 'marginAsset': 'USDT', 'quoteQty': '26.5582', 'commission': '0.00003057', 'commissionAsset': 'BNB', 'time': 1677110489921, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32995845, 'orderId': 3936576836, 'side': 'BUY', 'price': '699', 'qty': '0.012', 'realizedPnl': '-0.13680000', 'marginAsset': 'USDT', 'quoteQty': '8.3880', 'commission': '0.00000965', 'commissionAsset': 'BNB', 'time': 1677110489921, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}]
    for s in symbols:
        try:
            time.sleep(0.05)
            trades = um_futures_client.get_account_trades(symbol=s, recvWindow=6000)
            df_trades = pd.DataFrame.from_records(trades)
            if not df_trades.empty:
                df_trades['qty'] = df_trades['qty'].apply(pd.to_numeric)
                df_trades['qty'] = df_trades.groupby(['orderId'])['qty'].transform('sum')
                df_trades['realizedPnl'] = df_trades['realizedPnl'].apply(pd.to_numeric)
                df_trades['realizedPnl'] = df_trades.groupby(['orderId'])['realizedPnl'].transform('sum')
                df_trades['commission'] = df_trades['commission'].apply(pd.to_numeric)
                df_trades['commission'] = df_trades.groupby(['orderId'])['commission'].transform('sum')
                df_trades = df_trades.drop_duplicates(subset=['orderId'])
                df_trades['symbol_side_qty'] = df_trades.apply(lambda x: x['symbol'] + '_' + str(x['positionSide']) + '_' + str(x['qty']), axis=1)

                df_trades['entry_price_trade'] = df_trades.apply(lambda x: x['price'] if x['realizedPnl'] == 0 else 0, axis=1)
                df_trades['out_price_trade'] = df_trades.apply(lambda x: x['price'] if x['realizedPnl'] != 0 else 0, axis=1)
                df_trades['entry_time_trade'] = df_trades.apply(lambda x: x['time'] if x['realizedPnl'] == 0 else 0, axis=1)
                df_trades['out_time_trade'] = df_trades.apply(lambda x: x['time'] if x['realizedPnl'] != 0 else 0, axis=1)

                df_trades['realizedPnl'] = df_trades.groupby(['symbol_side_qty'])['realizedPnl'].transform('sum')
                df_trades['entry_price_trade'] = df_trades['entry_price_trade'].apply(pd.to_numeric)
                df_trades['entry_price_trade'] = df_trades.groupby(['symbol_side_qty'])['entry_price_trade'].transform('sum')
                df_trades['entry_time_trade'] = df_trades.groupby(['symbol_side_qty'])['entry_time_trade'].transform('sum')
                df_trades['out_price_trade'] = df_trades['out_price_trade'].apply(pd.to_numeric)
                df_trades['out_price_trade'] = df_trades.groupby(['symbol_side_qty'])['out_price_trade'].transform('sum')
                df_trades['out_time_trade'] = df_trades.groupby(['symbol_side_qty'])['out_time_trade'].transform('sum')
                df_trades = df_trades.drop_duplicates(subset=['symbol_side_qty'])

                time.sleep(0.05)
                orders = um_futures_client.get_all_orders(symbol=s, recvWindow=6000)
                df_orders = pd.DataFrame.from_records(orders)
                if not df_orders.empty:
                    df_orders = df_orders[df_orders['status'] == 'FILLED']
                    df_orders['entry_price'] = df_orders.apply(lambda x: float(x['price']) if x['status'] == 'FILLED' and (str(x['clientOrderId'])[:6] == 'limit_') else None, axis=1)
                    df_orders['sl_price'] = df_orders.apply(lambda x: float(x['clientOrderId'].split('_')[1]) if x['status'] == 'FILLED' and (str(x['clientOrderId'])[:6] == 'limit_') else None, axis=1)
                    df_orders['tp_price'] = df_orders.apply(lambda x: float(x['clientOrderId'].split('_')[2]) if x['status'] == 'FILLED' and (str(x['clientOrderId'])[:6] == 'limit_') else None, axis=1)
                    df_orders['match_orderId'] = df_orders.apply(lambda x: x['orderId'] if x['status'] == 'FILLED' and (str(x['clientOrderId'])[:6] == 'limit_') else (int(x['clientOrderId'][3:]) if x['status'] == 'FILLED' and (str(x['clientOrderId'])[:3] == 'tp_' or str(x['clientOrderId'])[:3] == 'sl_') else None), axis=1)
                    df_orders['out_price'] = df_orders.apply(lambda x: x['tp_price'] if x['match_orderId'] == x['orderId'] and str(x['clientOrderId'])[:3] == 'tp_' else x['sl_price'], axis=1)

                    df_orders = df_orders.dropna()

                    if not df_orders.empty:
                        del df_orders["symbol"]
                        del df_orders["side"]
                        del df_orders["price"]
                        del df_orders["time"]
                        del df_orders["positionSide"]
                        df_merged = pd.merge(df_trades, df_orders, how='left', left_on=['orderId'], right_on=['orderId'])
                        if not df_merged.empty:
                            if df is None:
                                df = df_merged
                            else:
                                df = df.append(df_merged, ignore_index=True)
                else:
                    # print('symbol:%s (df_orders.empty xxxxxxxxxxx)' % s)
                    pass
            else:
                # print('symbol:%s (df_trades.empty)' % s)
                pass
            pass
        except ClientError as error:
            logging.error(
                "ClientError Found error. status: {}, error code: {}, error message: {}".format(
                    error.status_code, error.error_code, error.error_message
                )
            )
        except KeyError as error:
            logging.error(
                "KeyError Found error. status: {}, error code: {}, error message: {}".format(
                    error.status_code, error.error_code, error.error_message
                )
            )

    # uid
    df['uid'] = df.apply(lambda x: x['symbol'] + '_' + str(x['positionSide']) + '_' + str(x['price']), axis=1)
    # sid
    df['sid'] = df.apply(lambda x: x['symbol'] + '_' + str(x['orderId']), axis=1)
    df['sid_pnl'] = df.apply(lambda x: float(x['realizedPnl']), axis=1)
    df['sid_win_pnl'] = df.apply(lambda x: x['sid_pnl'] if float(x['sid_pnl']) > 0 else 0, axis=1)
    df['sid_loss_pnl'] = df.apply(lambda x: x['sid_pnl'] if float(x['sid_pnl']) <= 0 else 0, axis=1)

    df['sum_sid_pnl'] = df.groupby(['sid'])['sid_pnl'].transform('sum')  # https://sparkbyexamples.com/pandas/pandas-groupby-count-examples/
    df['sum_sid_win_pnl'] = df.groupby(['sid'])['sid_win_pnl'].transform('sum')
    df['sum_sid_loss_pnl'] = df.groupby(['sid'])['sid_loss_pnl'].transform('sum')

    df['sid_trade_1'] = df.apply(lambda x: 1 if float(x['sum_sid_pnl']) != 0 else 0, axis=1)
    df['sid_win_1'] = df.apply(lambda x: 1 if float(x['sum_sid_pnl']) > 0 else 0, axis=1)
    df['sid_loss_1'] = df.apply(lambda x: 1 if float(x['sum_sid_pnl']) < 0 else 0, axis=1)

    df['count_sid_trade_1'] = df['sid_trade_1'].sum()
    df['count_sid_win_1'] = df['sid_win_1'].sum()
    df['count_sid_loss_1'] = df['sid_loss_1'].sum()
    df['winrate_sid'] = df['sid_win_1'].sum() / df['sid_trade_1'].sum()

    df['sid_commission'] = df.apply(lambda x: float(x['commission']), axis=1)
    df['sum_sid_commission'] = df.groupby(['sid'])['sid_commission'].transform('sum')

    # daily
    df['datetime'] = df.apply(
        lambda x: str(dt.datetime.fromtimestamp(float(x['time']) / 1000).strftime('%Y-%m-%d %H:%M:%S')), axis=1)
    df['date'] = df.apply(
        lambda x: str(dt.datetime.fromtimestamp(float(x['time']) / 1000).strftime('%Y-%m-%d')), axis=1)

    df['sum_date_sid_trade_pnl'] = df.groupby(['date'])['sid_pnl'].transform('sum')
    df['sum_date_sid_win_pnl'] = df.groupby(['date'])['sid_win_pnl'].transform('sum')
    df['sum_date_sid_loss_pnl'] = df.groupby(['date'])['sid_loss_pnl'].transform('sum')

    df['count_date_sid_trade_1'] = df.groupby(['date'])['sid_trade_1'].transform('count')
    df['count_date_sid_win_1'] = df.groupby(['date'])['sid_win_1'].transform('count')
    df['count_date_sid_loss_1'] = df.groupby(['date'])['sid_loss_1'].transform('count')

    df['sum_date_sid_commission'] = df.groupby(['date'])['sid_commission'].transform('sum')

    df['date_wins'] = df.apply(lambda x: 1 if float(x['sum_sid_pnl']) > 0 else 0, axis=1)
    df['date_losses'] = df.apply(lambda x: 1 if float(x['sum_sid_pnl']) < 0 else 0, axis=1)
    df['date_winmark'] = df.apply(lambda x: 1 if float(x['sum_date_sid_trade_pnl']) > 0 else 0, axis=1)

    # qty
    df['qty_f'] = df.apply(lambda x: float(x['qty']), axis=1)
    df['quoteQty_f'] = df.apply(lambda x: float(x['quoteQty']), axis=1)
    df['sum_qty'] = df.groupby(['sid'])['qty_f'].transform('sum')
    df['sum_quoteQty'] = df.groupby(['sid'])['quoteQty_f'].transform('sum')
    df['uid'] = df.apply(lambda x: x['symbol'] + '_' + str(x['positionSide']) + '_' + str(x['entry_price']) + '_' + str(x['out_price']), axis=1)

    # print(df)
    return df


def trade_info_df(trade_info):
    df = pd.DataFrame.from_records(trade_info)
    df = df.T

    mapping = {df.columns[0]: 'stats_history', df.columns[1]: 'order_history',
               df.columns[2]: 'asset_history', df.columns[3]: 'trade_count',
               df.columns[4]: 'fee_history', df.columns[5]: 'pnl_history',
               df.columns[6]: 'wavepattern_history'
               }
    df = df.rename(columns=mapping)
    if not df.empty:
        df['symbol'] = df.apply(lambda x: x['stats_history'][2], axis=1)
        df['entry_dt'] = df.apply(lambda x: x['order_history'][0][0], axis=1)
        df['entry_price'] = df.apply(lambda x: x['order_history'][0][1], axis=1)
        df['out_dt'] = df.apply(lambda x: x['order_history'][1][0], axis=1)
        df['out_price'] = df.apply(lambda x: x['order_history'][1][1], axis=1)
        df['longshort'] = df.apply(lambda x: 'LONG' if float(x['out_price']) < float(x['entry_price']) and x['trade_count'] == 0 else 'SHORT', axis=1)
        df['uid'] = df.apply(lambda x: x['symbol'] + '_' + str(x['longshort'])+ '_' + str(x['entry_price']) + '_' + str(x['out_price']), axis=1)
        # print(df)
    return df

if __name__ == '__main__':
    print("""

     _____             _ _              ______       _
    |_   _|           | (_)             | ___ \     | |
      | |_ __ __ _  __| |_ _ __   __ _  | |_/ / ___ | |_
      | | '__/ _` |/ _` | | '_ \ / _` | | ___ \/ _ \| __|
      | | | | (_| | (_| | | | | | (_| | | |_/ / (_) | |_
      \_/_|  \__,_|\__,_|_|_| |_|\__, | \____/ \___/ \__| %s %s
                                  __/ |
                                 |___/

    """ % (version, descrition))
    print_condition()
    logger.info('seq:' + seq)
    rount_trip_total = []
    start = time.perf_counter()

    # symbols = get_symbols()
    # r_df = account_trades_df(symbols)

    if round_trip_flg:
        import multiprocessing
        for i in range(round_trip_count):
            rt = list()
            try:
                start = time.perf_counter()
                # cpucount = multiprocessing.cpu_count() * 1
                # cpucount = multiprocessing.cpu_count()
                cpucount = 1
                logger.info('cpucount:%s' % cpucount)
                pool = multiprocessing.Pool(processes=cpucount)
                rt = pool.map(round_trip, range(period_days_ago, period_days_ago_till, -1 * period_interval))
                pool.close()
                pool.join()
                start_dt = str((pd.to_datetime('today') - pd.Timedelta(str(period_days_ago) + ' days')).date())
                end_dt = str((pd.to_datetime('today') - pd.Timedelta(str(period_days_ago_till) + ' days')).date())
                logger.info(f'Finished in {round(time.perf_counter() - start, 2)} second(s)')
            except Exception as e:
                # print(e)
                pass

            logger.info('============ %s stat.==========' % str(i))
            r = list(map(lambda i: i[0], rt))

            winrate_l = list(map(lambda i: 1 if i[0] > seed else 0, rt))
            meanaverage = None
            winrate = None
            total_gains = None
            if r:
                meanaverage = round((sum(r)/len(r)), 2)
            roundcount = len(rt)
            if winrate_l:
                winrate = str(round((sum(winrate_l))/len(winrate_l)*100, 2))
            if meanaverage:
                total_gains = (meanaverage - seed)*roundcount
            logger.info('round r: %s' % r)
            logger.info('round winrate_l: %s' % str(winrate_l))
            logger.info('round roundcount: %s' % roundcount)
            logger.info('round winrate: %s' % winrate)
            logger.info('round meanaverage: %s' % str(meanaverage))
            logger.info('round total gains: %s' % str(total_gains))
            logger.info('============ %s End All=========='% str(i))

            rount_trip_total.append([meanaverage, roundcount, winrate, total_gains])
        print_condition()
        for i, v in enumerate(rount_trip_total):
            logger.info(i, v)
        logger.info(f'Finished wave_analyzer in {round(time.perf_counter() - start, 2)} second(s)')

    else:

        symbols = get_symbols()
        stats_history = []
        order_history = []
        asset_history = []
        trade_count = []
        fee_history = []
        pnl_history = []
        wavepattern_history = []
        trade_info = [stats_history, order_history, asset_history, trade_count, fee_history, pnl_history, wavepattern_history]
        mdd = None
        r = range(period_days_ago, period_days_ago_till, -1 * period_interval)
        for i in r:
            asset_history_pre = trade_info[2][-1] if trade_info[2] else seed
            single(symbols, i, trade_info)
            asset_history_last = trade_info[2]
            if asset_history_last:
                mdd1 = get_mdd_1(asset_history_last)
                mdd2 = get_mdd(asset_history_last)
                print(str(i)+'/'+str(len(r)), ' now asset: ', asset_history_last[-1], ' | ', len(trade_count), ' | pre seed: ', asset_history_pre, ' | MDD1', mdd1, ' | MDD2', mdd2)
            else:
                print(str(i)+'/'+str(len(r)), ' now asset: ', seed, ' | ', len(trade_count), ' | pre seed: ', seed)

        if compare_account_trade:
            print('######## compare_account_trade start #######')

            # 1. test trades
            t_df = trade_info_df(trade_info)

            # 2. account trades
            r_df = account_trades_df(symbols)
            sid_df = r_df.drop_duplicates(subset=['sid'])
            # sid_df = r_df.drop_duplicates(subset=['sid']).drop('symbol', 1)
            sid_df = sid_df[(sid_df.realizedPnl != '0')]
            sid_df.sort_values(by='time', ascending=False,
                               inplace=True)  # https://sparkbyexamples.com/pandas/sort-pandas-dataframe-by-date/
            print(sid_df)

            new_df = pd.merge(t_df, sid_df, how='left', left_on=['uid'], right_on=['uid'])
            print(new_df)

            new_df2 = pd.merge(sid_df, t_df, how='left', left_on=['uid'], right_on=['uid'])
            print(new_df2)
            print('######## compare_account_trade end #######')


        logger.info('============ %s stat.==========' % str(i))
        winrate_l = list(map(lambda i: 1 if i > 0 else 0, pnl_history))
        meanaverage = None
        if len(asset_history) > 0:
            meanaverage = round((sum(asset_history)/len(asset_history)), 2)
        roundcount = len(trade_count)
        winrate = None
        if len(winrate_l) > 0:
            winrate = str(round((sum(winrate_l))/len(winrate_l)*100, 2))
        logger.info('round r: %s' % roundcount)
        logger.info('round winrate_l: %s' % str(winrate_l))
        logger.info('round roundcount: %s' % roundcount)
        logger.info('round winrate: %s' % winrate)
        logger.info('round meanaverage: %s' % str(meanaverage))
        logger.info('round MDD: %s' % str(mdd))
        logger.info('round total gains: %s' % str(trade_info[-2][-1] if trade_info[2] else 0))
        logger.info('============ %s End All=========='% str(i))
        logger.info(f'Finished wave_analyzer in {round(time.perf_counter() - start, 2)} second(s)')

    print_condition()
    logger.info("good luck done!!")
