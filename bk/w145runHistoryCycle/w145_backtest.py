from __future__ import annotations
from models.WavePattern import WavePattern
from models.WaveCycle import WaveCycle
from models.WaveRules import Impulse, DownImpulse, Correction, DownCorrection, TDWave
from models.WaveAnalyzer import WaveAnalyzer
from models.WaveOptions import WaveOptionsGenerator5, WaveOptionsGenerator3
from models.helpers import plot_pattern_m, plot_cycle
import datetime
import pandas as pd
import numpy as np
from binance.client import Client
import datetime as dt
from tapy import Indicators
import multiprocessing
import random
import shutup; shutup.please()
from datetime import datetime
from typing import Optional
import dateparser
import pytz
import json
import logging

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

# import ccxt
import pandas as pd
import logging
from binancefutures.um_futures import UMFutures
from binancefutures.lib.utils import config_logging
from binancefutures.error import ClientError
# config_logging(logging, logging.DEBUG)
from binance.helpers import round_step_size
import math
key = "IkzH8WHKl0lGzOSqiZZ4TnAyKnDpqnC9Xi31kzrRNpwJCp28gP8AuWDxntSqWdrn"
secret = "FwKTmQ2RWSiECMfhZOaY7Hed45JuXqlEPno2xiLGgCzloLq4NMMcmusG6gtMCKa5"

um_futures_client = UMFutures(key=key, secret=secret)

import pickle
open_order_history = [
    # {'id':'timestamp', 'symbol':'SYMBOLUSDT', 'wavepattern': wavepattern, 'entry':'10000', 'target':'10381', 'status':'NEW or DRAW or TAKE_PROFIT or DONE',  'data': [{limit_result}, {sl_result}, {tp_result}], 'position':[]}
    # {'id':'1234567890.1234', 'symbol': 'BTCUSDT', 'wavepattern': wavepattern, 'entry':'10000', 'target':'10381', 'status':'NEW' 'data': [{'orderId': 1300759837, 'symbol': 'WOOUSDT', 'status': 'NEW', 'clientOrderId': 'waveshortlimit001', 'price': '0.21310', 'avgPrice': '0.00000', 'origQty': '30', 'executedQty': '0', 'cumQty': '0', 'cumQuote': '0', 'timeInForce': 'GTC', 'type': 'LIMIT', 'reduceOnly': False, 'closePosition': False, 'side': 'SELL', 'positionSide': 'SHORT', 'stopPrice': '0', 'workingType': 'CONTRACT_PRICE', 'priceProtect': False, 'origType': 'LIMIT', 'updateTime': 1674269613350}, {'orderId': 1300759838, 'symbol': 'WOOUSDT', 'status': 'NEW', 'clientOrderId': 'waveshortlimit001sl', 'price': '0', 'avgPrice': '0.00000', 'origQty': '30', 'executedQty': '0', 'cumQty': '0', 'cumQuote': '0', 'timeInForce': 'GTC', 'type': 'STOP_MARKET', 'reduceOnly': True, 'closePosition': False, 'side': 'BUY', 'positionSide': 'SHORT', 'stopPrice': '0.22103', 'workingType': 'CONTRACT_PRICE', 'priceProtect': False, 'origType': 'STOP_MARKET', 'updateTime': 1674269613350}]}
]
import os
import random

open_order_history_seq= 'oohistory_' + seq + '.pkl'

def load_history_pkl():
    try:
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

exchange = config['default']['exchange']
exchange_symbol = config['default']['exchange_symbol']
futures = config['default']['futures']
type = config['default']['type']
leverage = config['default']['leverage']

high_target = config['default']['high_target']
low_target = config['default']['low_target']
low_target_w2 = config['default']['low_target_w2']

seed = config['default']['seed']
fee_maker = config['default']['fee_maker']
fee_taker = config['default']['fee_taker']
fee_slippage = config['default']['fee_slippage']

fee_entry_type = config['default']['fee_entry_type']
fee_sl_type = config['default']['fee_sl_type']
fee_tp_type = config['default']['fee_tp_type']

period_days_ago = config['default']['period_days_ago']
period_days_ago_till = config['default']['period_days_ago_till']
period_interval = config['default']['period_interval']

round_trip_flg = config['default']['round_trip_flg']
round_trip_count = config['default']['round_trip_count']
compounding = config['default']['compounding']
fcnt = config['default']['fcnt']
loop_count = config['default']['loop_count']


timeframe = config['default']['timeframe']
up_to_count = config['default']['up_to_count']
condi_same_date = config['default']['condi_same_date']
long = config['default']['long']
o_fibo = config['default']['o_fibo']
h_fibo = config['default']['h_fibo']
l_fibo = config['default']['l_fibo']
entry_fibo = config['default']['entry_fibo']
target_fibo = config['default']['target_fibo']
sl_fibo = config['default']['sl_fibo']


symbol_random = config['default']['symbol_random']
symbol_each = config['default']['symbol_each']
symbol_duplicated = config['default']['symbol_duplicated']
symbol_last = config['default']['symbol_last']
symbol_length = config['default']['symbol_length']

basic_secret_key = config['basic']['secret_key']
basic_secret_value = config['basic']['secret_value']
futures_secret_key = config['futures']['secret_key']
futures_secret_value = config['futures']['secret_value']


intersect_idx = config['default']['intersect_idx']
plotview = config['default']['plotview']
printout = config['default']['printout']

fee_entry_tp = 0
fee_entry_sl = 0

if fee_entry_type == 'taker' and fee_sl_type == 'maker' and fee_tp_type == 'maker':
    fee_entry_tp = (fee_taker + fee_slippage + fee_maker) * leverage
    fee_entry_sl = (fee_taker + fee_slippage + fee_maker) * leverage

elif fee_entry_type == 'maker' and fee_sl_type == 'taker' and fee_tp_type == 'maker':
    fee_entry_tp = (fee_maker + fee_maker) * leverage
    fee_entry_sl = (fee_maker + fee_maker + fee_slippage) * leverage

elif fee_entry_type == 'taker' and fee_sl_type == 'maker' and fee_tp_type == 'taker':
    fee_entry_tp = (fee_taker + fee_slippage + fee_maker) * leverage
    fee_entry_sl = (fee_taker + fee_slippage + fee_taker + fee_slippage) * leverage

elif fee_entry_type == 'taker' and fee_sl_type == 'taker' and fee_tp_type == 'taker':
    fee_entry_tp = (fee_taker + fee_slippage + fee_taker + fee_slippage) * leverage
    fee_entry_sl = (fee_taker + fee_slippage + fee_taker + fee_slippage) * leverage


def print_condition():
    logger.info('-------------------------------')
    logger.info('exchange:%s' % str(exchange))
    logger.info('exchange_symbol:%s' % str(exchange_symbol))
    logger.info('futures:%s' % str(futures))
    logger.info('type:%s' % str(type))
    logger.info('leverage:%s' % str(leverage))
    logger.info('seed:%s' % str(seed))

    logger.info('fee_maker:%s%%' % str(round(fee_maker*100, 4)))
    logger.info('fee_taker:%s%%' % str(round(fee_taker*100, 4)))
    logger.info('fee_slippage:%s%%' % str(round(fee_slippage*100, 4)))

    logger.info('fee_entry_type:%s' % str(fee_entry_type))
    logger.info('fee_sl_type:%s' % str(fee_sl_type))
    logger.info('fee_tp_type:%s' % str(fee_tp_type))

    logger.info('timeframe: %s' % timeframe)
    logger.info('period_days_ago: %s' % period_days_ago)
    logger.info('period_days_ago_till: %s' % period_days_ago_till)
    logger.info('period_interval: %s' % period_interval)
    logger.info('round_trip_count: %s' % round_trip_count)
    logger.info('compounding: %s' % compounding)
    logger.info('fcnt: %s' % fcnt)
    logger.info('loop_count: %s' % loop_count)

    logger.info('symbol_duplicated: %s' % symbol_duplicated)
    logger.info('symbol_random: %s' % symbol_random)
    logger.info('symbol_each: %s' % symbol_each)
    logger.info('symbol_last: %s' % symbol_last)
    logger.info('symbol_length: %s' % symbol_length)

    # logger.info('timeframe: %s' % timeframe)
    start_dt = str((pd.to_datetime('today') - pd.Timedelta(str(period_days_ago) + ' days')).date())
    end_dt = str((pd.to_datetime('today') - pd.Timedelta(str(period_days_ago_till) + ' days')).date())
    logger.info('period: %s ~ %s' % (start_dt, end_dt))
    logger.info('up_to_count: %s' % up_to_count)
    logger.info('condi_same_date: %s' % condi_same_date)
    # logger.info('long: %s' % long)
    logger.info('o_fibo: %s' % o_fibo)
    logger.info('h_fibo: %s' % h_fibo)
    logger.info('l_fibo: %s' % l_fibo)

    logger.info('entry_fibo: %s' % entry_fibo)
    logger.info('target_fibo: %s' % target_fibo)
    logger.info('sl_fibo: %s' % sl_fibo)

    logger.info('intersect_idx: %s' % intersect_idx)
    logger.info('plotview: %s' % plotview)
    logger.info('printout: %s' % printout)
    logger.info('-------------------------------')

client = None
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
    logger.info(str(len(symbols)) +',' + str(symbols))
    return symbols

import threading
import functools
import time

def synchronized(wrapped):
    lock = threading.Lock()
    @functools.wraps(wrapped)
    def _wrap(*args, **kwargs):
        with lock:
            result = wrapped(*args, **kwargs)
            return result
    return _wrap



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
            past_days = 30  # default past days 30.

        start_date_str = str((pd.to_datetime('today') - pd.Timedelta(str(start_int) + ' days')).date())
        if end_int:
            end_date_str = str((pd.to_datetime('today') - pd.Timedelta(str(end_int) + ' days')).date())
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
            time.sleep(0.5)
            # logger.info(e)
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
        # print('e in get_historical_ohlc_data_start_end:%s' % e)
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


def backtest_trade45(df, symbol, fcnt, longshort, df_lows_plot, df_highs_plot, wavepattern, trade_info, idx, wavepattern_l, wavepattern_tpsl_l):
    w = wavepattern
    # real condititon by fractal index
    real_condititon1 = True if (fcnt/2 < (w.idx_end - w.idx_start)) and w.idx_start == idx else False
    real_condititon2 = True if df.iloc[idx + int(fcnt/2), 0] < (w.dates[-1]) else False
    if not real_condititon1:
        if printout: logger.info('not real_condititon1 ')
        return trade_info, False
    if not real_condititon2:
        if printout: logger.info('not real_condititon2 ')
        return trade_info, False

    t = trade_info
    stats_history = t[0]
    order_history = t[1]
    asset_history = t[2]
    trade_count = t[3]
    fee_history = t[4]
    pnl_history = t[5]
    wavepattern_history = t[6]


    df_smaN = sma_df(df, fcnt)

    w_start_price = w.values[0]  # wave1
    w_end_price = w.values[-1]  # wave5
    height_price = abs(w_end_price - w_start_price)
    o_fibo_value = height_price * o_fibo / 100 if o_fibo else 0
    # entry_price = w.values[7]  # wave4

    # entry_price = w.values[0] + height_price * entry_fibo if longshort else w.values[0] - height_price * entry_fibo  # 0.381 되돌림가정
    # sl_price = w.values[0] + height_price * sl_fibo if longshort else w.values[0] - height_price * sl_fibo
    # target_price = entry_price + height_price * target_fibo if longshort else entry_price - height_price * target_fibo  # 0.5 되돌림가정

    w0 = w.values[0]
    w1 = w.values[1]
    w2 = w.values[2]
    w3 = w.values[5]
    w4 = w.values[7]
    w5 = w.values[9]

    # target_price = w5 - abs(w5-w4)*0/10
    # target_price = w5 + height_price * target_fibo if longshort else w5 - height_price * target_fibo  # 0.5 되돌림가정
    # entry_price = w4
    # sl_price = w1


    # target_price = w5
    # entry_price = w4
    # sl_price = w0

    target_price = w5
    entry_price = w4
    sl_price = w0

    entry_price = set_price(symbol, entry_price, longshort)
    sl_price = set_price(symbol, sl_price, longshort)
    target_price = set_price(symbol, target_price, longshort)

    if entry_price == sl_price:
        logger.info(symbol + ' _entry_price == sl_price')
        return trade_info, False

    df_active = df[w.idx_end + 1:]
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
    if closes:
        for i, close in enumerate(closes):
            c_out_trend_beyond = trends[i] >= (w_end_price + o_fibo_value) if longshort else trends[i] <= (w_end_price - o_fibo_value)
            c_positioning = (position is False and detrends[i] <= entry_price and detrends[i] > sl_price) if longshort else (position is False and detrends[i] >= entry_price and detrends[i] < sl_price)
            c_profit = (position and trends[i] >= target_price) if longshort else (position and trends[i] <= target_price)
            c_stoploss = (position and detrends[i] <= sl_price) if longshort else (position and detrends[i] >= sl_price)
            c_stoploss_direct = (detrends[i] <= sl_price and trends[i] >= entry_price) if longshort else (detrends[i] >= sl_price and trends[i] <= entry_price)

            if position is False and c_out_trend_beyond:
                if printout:  logger.info('@@ beyondxxxx @@>c_out_trend (not position and c_out_trend_beyond), ', i, close)
                # print('beyondxxxx @@>c_out_trend (not position and c_out_trend_beyond), ', i, close)
                return trade_info, False

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
                    'target': target_price,
                    'sl_price': sl_price,
                    'limit_orderId': 'order_limit[orderId]',
                    'sl_orderId': 'order_sl[orderId]',
                    'tp_orderId': None,
                    'wavepattern': wavepattern,
                    'data': []
                }
                # create history NEW
                open_order_history.append(open_order)
                dump_history_pkl()


            if position is False and c_positioning:
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
                    'target': target_price,
                    'sl_price': sl_price,
                    'limit_orderId': 'order_limit[orderId]',
                    'sl_orderId': 'order_sl[orderId]',
                    'tp_orderId': None,
                    'wavepattern': wavepattern,
                    'data': []
                }
                # create history NEW
                open_order_history.append(open_order)
                dump_history_pkl()

            if position is True:
                if c_profit or c_stoploss:
                    fee_percent = 0
                    pnl_percent = 0
                    trade_inout_i = []
                    if c_stoploss:
                        position_sl_i = [dates[i], sl_price]
                        pnl_percent = -(abs(entry_price - sl_price) / entry_price) * leverage
                        fee_percent = fee_entry_sl
                        trade_count.append(0)
                        trade_inout_i = [position_enter_i, position_sl_i, longshort, '-']
                        order_history.append(trade_inout_i)
                        if printout: logger.info('- stoploss, ', i, close, position_enter_i, position_sl_i)
                        # update_history_status(open_order_history, symbol, h_id, 'DONE')

                    if c_profit:
                        position_pf_i = [dates[i], target_price]
                        pnl_percent = (abs(target_price - entry_price) / entry_price) * leverage
                        fee_percent = fee_entry_tp
                        trade_count.append(1)
                        trade_inout_i = [position_enter_i, position_pf_i, longshort, '+']
                        order_history.append(trade_inout_i)
                        if printout: logger.info('+ profit, ', i, close, position_enter_i, position_pf_i)
                        # update_history_status(open_order_history, symbol, h_id, 'DONE')


                    asset_history_pre = asset_history[-1] if asset_history else seed
                    asset_new = asset_history_pre * (1 + pnl_percent - fee_percent)
                    pnl_history.append(asset_history_pre*pnl_percent)
                    fee_history.append(asset_history_pre*fee_percent)
                    asset_history.append(asset_new)
                    # wavepattern_history.append(wavepattern)

                    winrate = round((sum(trade_count)/len(trade_count))*100, 2)
                    trade_stats = [len(trade_count), winrate, symbol, asset_new, str(round(pnl_percent, 4)), sum(pnl_history), sum(fee_history)]
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
                        logger.info('%s %s %s %s %s %s %s' % (symbol, fcnt, 'L' if longshort else 'S', trade_inout_i[0][0][2:-3], '-',  trade_inout_i[1][0][11:-3], str(trade_stats)))
                              # , w2_rate,
                              #     w3_rate, w3_rate / w2_rate)
                    # print(symbol, trade_inut_i[0][0][2:-3], str(trade_stats))

                    # if longshort is not None and len(trade_info[1])>0:
                    #     if plotview:
                    #         plot_pattern_m(df=df, wave_pattern=[[i, wavepattern.dates[0], id(wavepattern), wavepattern]], df_lows_plot=df_lows_plot, df_highs_plot=df_highs_plot, trade_info=trade_info, title=str(
                    #             symbol + ' %s '% str(longshort) + str(trade_stats)))

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

def loopsymbol(symbol, i, trade_info):
    ###################
    ## data settting ##
    ###################
    # 1day:1440m

    # try:

    past_days = 1
    timeunit = 'm'
    bin_size = str(timeframe) + timeunit
    start_int = i
    end_int = start_int - period_interval
    if end_int < 0:
        end_int = None
    df, start_date, end_date = get_historical_ohlc_data_start_end(symbol, start_int=start_int,
                                                                      end_int=end_int, past_days=past_days,
                                                                      interval=bin_size, futures=futures)

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
    waveoptions_down = WaveOptionsGenerator3(up_to=up_to_count)

    idxs = list()
    lows_idxs = list()
    highs_idxs = list()
    df_lows_plot = None
    df_highs_plot = None
    trade_flg = False

    for fc in fcnt:
        if 'long' in type:
            df_lows = fractals_low_loopA(df_all, fcnt=fc, loop_count=loop_count)
            df_lows_plot = df_lows[['Date', 'Low']]
            impulse = Impulse('impulse')
            correction = Correction('correction')

            lows_idxs = df_lows.index.tolist()
            idxs = lows_idxs

        if 'short' in type:
            df_highs = fractals_high_loopA(df_all, fcnt=fc, loop_count=loop_count)
            df_highs_plot = df_highs[['Date', 'High']]
            downimpulse = DownImpulse('downimpulse')
            downcorrection = DownCorrection('correction')

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







                                # logger.info('Impulse found! %s' % str(wave_opt.values))
                                # end = waves[4].idx_end
                                #
                                # cycle_complete = False
                                # wave_cycle = None
                                # for new_option_correction in waveoptions_down.options_sorted:
                                #
                                #     if longshort:
                                #         wave_abcs = wa.find_corrective_wave(idx_start=end,
                                #                                           wave_config=new_option_correction.values)
                                #     else:
                                #         wave_abcs = wa.find_upcorrective_wave(idx_start=end,
                                #                                             wave_config=new_option_correction.values)
                                #
                                #     if wave_abcs:
                                #         wavepattern_abc = WavePattern(wave_abcs, verbose=False)
                                #
                                #         check_rule_flg = wavepattern_abc.check_rule(correction) if longshort else wavepattern_abc.check_rule(downcorrection)
                                #         if check_rule_flg:
                                #
                                #             cycle_complete = True
                                #             wave_cycle = WaveCycle(wavepattern, wavepattern_abc)
                                #             print('Corrrection found!', new_option_correction.values)
                                #             # plot_cycle(df=df_all, wave_cycle=wave_cycle, title=symbol)
                                #
                                # if cycle_complete:
                                #     print('abc found!')
                                #
                                #     pass
                                #     # yield wave_cycle
                                # if wave_cycle:
                                #     print(wave_cycle)
















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
                                        if len(open_order_history) > 0:
                                            open_order_this_symbol = [x for x in open_order_history if
                                                                      x['symbol'] == symbol]
                                            if len(open_order_this_symbol) > 0:
                                                open_order_wavepattern = open_order_this_symbol[0]['wavepattern']

                                                # check identical check
                                                c_check_wave_identical = True if (
                                                                                             open_order_wavepattern.dates == wavepattern.dates) and (
                                                                                             open_order_wavepattern.values == wavepattern.values) else False
                                                # 2,3,4 wave same check
                                                c_check_wave_identical_2_3_4 = True if (open_order_wavepattern.dates[
                                                                                        2:-2] == wavepattern.dates[
                                                                                                 2:-2]) and (
                                                                                                   open_order_wavepattern.values[
                                                                                                   2:-2] == wavepattern.values[
                                                                                                            2:-2]) else False

                                        if c_check_wave_identical or c_check_wave_identical_2_3_4:
                                            # print(c_check_wave_identical, c_check_wave_identical_2_3_4)
                                            if c_check_wave_identical_2_3_4:
                                                # print('234')
                                                pass
                                            return trade_info



                                        # w2_rate = float(re.findall('\(([^)]+)', wavepattern.labels[3])[
                                        #                     0])  # extracts string in bracket()
                                        # w3_rate = float(re.findall('\(([^)]+)', wavepattern.labels[5])[
                                        #                     0])  # extracts string in bracket()
                                        #
                                        # if w3_rate/w2_rate > 0:
                                        if True:

                                            trade_info, trade_flg = backtest_trade45(df_all, symbol, fc, longshort, df_lows_plot, df_highs_plot, wavepattern, trade_info, i, wavepattern_l, wavepattern_tpsl_l)

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
                                                               title='tpsl_%s_' % str(fc) + t)



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
    dump_history_pkl()


def single(symbols, i, trade_info, *args):
    for symbol in symbols:
        loopsymbol(symbol, i, trade_info)
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

if __name__ == '__main__':
    logger.info("""

     _____             _ _              ______       _
    |_   _|           | (_)             | ___ \     | |
      | |_ __ __ _  __| |_ _ __   __ _  | |_/ / ___ | |_
      | | '__/ _` |/ _` | | '_ \ / _` | | ___ \/ _ \| __|
      | | | | (_| | (_| | | | | | (_| | | |_/ / (_) | |_
      \_/_|  \__,_|\__,_|_|_| |_|\__, | \____/ \___/ \__| v0.2 BACK TESTOR
                                  __/ |
                                 |___/

    """)
    print_condition()
    logger.info('seq:' + seq)
    rount_trip_total = []
    start = time.perf_counter()
    if round_trip_flg:
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
        r = range(period_days_ago, period_days_ago_till, -1 * period_interval)
        for i in r:
            asset_history_pre = trade_info[2][-1] if trade_info[2] else seed
            single(symbols, i, trade_info)
            if trade_info[2]:
                print(str(i)+'/'+str(len(r)), ' now asset: ', trade_info[2][-1], ' | ', len(trade_count), ' | pre seed: ', asset_history_pre)
            else:
                print(str(i)+'/'+str(len(r)), ' now asset: ', seed, ' | ', len(trade_count), ' | pre seed: ', seed)

        logger.info('============ %s stat.==========' % str(i))
        winrate_l = list(map(lambda i: 1 if i > 0 else 0, pnl_history))
        meanaverage = round((sum(asset_history)/len(asset_history)), 2)
        roundcount = len(trade_count)
        winrate = str(round((sum(winrate_l))/len(winrate_l)*100, 2))
        logger.info('round r: %s' % roundcount)
        logger.info('round winrate_l: %s' % str(winrate_l))
        logger.info('round roundcount: %s' % roundcount)
        logger.info('round winrate: %s' % winrate)
        logger.info('round meanaverage: %s' % str(meanaverage))
        logger.info('round total gains: %s' % str(trade_info[-2][-1] if trade_info[2] else 0))
        logger.info('============ %s End All=========='% str(i))
        logger.info(f'Finished wave_analyzer in {round(time.perf_counter() - start, 2)} second(s)')

    print_condition()
    logger.info("good luck done!!")