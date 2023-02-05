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
import multiprocessing
import random
import shutup; shutup.please()
from datetime import datetime
from typing import Optional
import dateparser
import pytz
import json
import logging

# 로그 생성
logger = logging.getLogger()
# 로그의 출력 기준 설정
logger.setLevel(logging.INFO)
# log 출력 형식
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# log 출력
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
# log를 파일에 출력
file_handler = logging.FileHandler('logger.log')
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
def load_history_pkl():
    try:
        with open('open_order_history.pkl', 'rb') as f:
            h = pickle.load(f)
            logger.info('load_history_pk:' + str(h))
            return h
    except Exception as e:
        logger.error(e)
        try:
            os.remove("open_order_history.pkl")
        except Exception as e:
            logger.error(e)
        return []
    return []

open_order_history = load_history_pkl()

def dump_history_pkl():
    try:
        with open('open_order_history.pkl', 'wb') as f:
            # print('dump_history_pkl')
            pickle.dump(open_order_history, f)
    except Exception as e:
        logger.error(e)


with open('w145config.json', 'r') as f:
    config = json.load(f)

exchange = config['default']['exchange']
exchange_symbol = config['default']['exchange_symbol']
futures = config['default']['futures']
type = config['default']['type']
leverage = config['default']['leverage']
qtyrate = config['default']['qtyrate']


high_target = config['default']['high_target']
low_target = config['default']['low_target']
low_target_w2 = config['default']['low_target_w2']

seed = config['default']['seed']
fee = config['default']['fee']
fee_maker = config['default']['fee_maker']
fee_taker = config['default']['fee_taker']
fee_slippage = config['default']['fee_slippage']

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


def print_condition():
    logger.info('-------------------------------')
    logger.info('exchange:%s' % str(exchange))
    logger.info('exchange_symbol:%s' % str(exchange_symbol))
    logger.info('futures:%s' % str(futures))
    logger.info('type:%s' % str(type))
    logger.info('leverage:%s' % str(leverage))
    logger.info('qtyrate:%s' % str(qtyrate))
    logger.info('seed:%s' % str(seed))
    logger.info('fee:%s%%' % str(fee*100))
    logger.info('fee_maker:%s%%' % str(fee_maker*100))
    logger.info('fee_taker:%s%%' % str(fee_taker*100))
    logger.info('fee_slippage:%s%%' % str(round(fee_slippage*100, 4)))
    if futures:
        fee_maker_maker = (fee_maker + fee_maker) * leverage  # fee_maker buy:0.02%(0.0002) sell:0.02%(0.0002), sleepage:0.01%(0.0001)
        logger.info('(fee_maker_maker:%s%%' % round(float(fee_maker_maker)*100, 4))

        fee_maker_taker_slippage = (fee_maker + fee_taker + fee_slippage) * leverage  # fee_maker buy:0.02%(0.0002) sell:0.02%(0.0002), sleepage:0.01%(0.0001)
        logger.info('(fee_maker_taker_slippage:%s%%' % round(float(fee_maker_taker_slippage)*100, 4))

    else:
        fee_maker_maker = (fee_maker + fee_maker) * leverage  # fee_maker buy:0.02%(0.0002) sell:0.02%(0.0002), sleepage:0.01%(0.0001)
        logger.info('(fee_maker_maker:%s%%' % round(float(fee_maker_maker)*100, 4))

        fee_maker_taker_slippage = (fee_maker + fee_taker + fee_slippage) * leverage  # fee_maker buy:0.02%(0.0002) sell:0.02%(0.0002), sleepage:0.01%(0.0001)
        logger.info('(fee_maker_taker_slippage:%s%%' % round(float(fee_maker_taker_slippage)*100, 4))

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

symbols_binace_info = client.futures_exchange_info()

allsymbols = symbols_binace_info['symbols']
for s in allsymbols:
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
    logger.info(str(len(symbols)) + ':' + str(symbols))
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
        # if not futures:
        #     # basic
        #     client_basic = Client("basic_secret_key",
        #                           "basic_secret_value")
        #     client = client_basic
        # else:
        #     # futures
        #     client_futures = Client("futures_secret_key",
        #                             "futures_secret_value")
        #     client = client_futures

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
            logger.error(str(e))
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
        pass
    return D, start_date_str, end_date_str

def get_fetch_dohlcv(symbol,
                     # start_int,
                     # end_int,
                     interval=None,
                     limit=500):
    # startTime = start_int
    # endTime = end_int
    datalist = um_futures_client.klines(symbol, interval, limit=limit)
    D = pd.DataFrame(datalist)
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
    return D

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

def sma_df(df, period=7):
    i = Indicators(df)
    i.sma(period=7)
    return i.df

def new_batch_order(params):
    try:
        response = um_futures_client.new_batch_order(params)
        logger.info(response)
    except ClientError as error:
        logger.error(
            "Found error. status: {}, error code: {}, error message: {}".format(
                error.status_code, error.error_code, error.error_message
            )
        )
        return None
    except Exception as e:
        logger.error(
            "Found error. status: {}, error code: {}, error message: {}".format(
                e.status_code, e.error_code, e.error_message
            )
        )
        # try one more
        try:
            time.sleep(0.1)  # 0.1초
            response = um_futures_client.new_batch_order(params)
            logger.info(response)
        except Exception as e:
            logger.error(e)

        return None

    return response


info = client.futures_exchange_info()


def get_precision(symbol):
    for x in info['symbols']:
        if x['symbol'] == symbol:
            return x['quantityPrecision']

def set_price(symbol, price, longshort):
    data = client.futures_exchange_info()  # request data
    info = data['symbols']  # pull list of symbols
    for x in range(len(info)):  # find length of list and run loop
        if info[x]['symbol'] == symbol:  # until we find our coin
            a = info[x]["filters"][0]['tickSize']  # break into filters pulling tick size
            cost = round_step_size(price, float(a)) # convert tick size from string to float, insert in helper func with cost
            # 아래는 시장가의 비용 및 sleepage 를 보고 나중에 추가 또는 삭제 검토요
            # cost = cost - float(a) if longshort else cost + float(a)
            return cost

def my_available_balance(exchange_symbol):
    response = um_futures_client.balance(recvWindow=6000)
    if exchange_symbol == 'binance_usdt_perp':
        my_marginavailable_l = [x['marginAvailable'] for x in response if x['asset'] == 'USDT']
        my_marginbalance_l = [x['availableBalance'] for x in response if x['asset'] == 'USDT']
        my_walletbalance_l = [x['balance'] for x in response if x['asset'] == 'USDT']
        if len(my_marginbalance_l) == 1:
            return my_marginavailable_l[0], float(my_marginbalance_l[0]), float(my_walletbalance_l[0])
    elif exchange_symbol == 'binance_busd_perp':
        my_marginavailable_l = [x['marginAvailable'] for x in response if x['asset'] == 'BUSD']
        my_marginbalance_l = [x['availableBalance'] for x in response if x['asset'] == 'BUSD']
        my_walletbalance_l = [x['balance'] for x in response if x['asset'] == 'BUSD']
        if len(my_marginbalance_l) == 1:
            return my_marginavailable_l[0], float(my_marginbalance_l[0]), float(my_walletbalance_l[0])
    return False, 0


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
    response = um_futures_client.exchange_info()
    symbols = response['symbols']
    filters = [x['filters'] for x in symbols if x['symbol'] == symbol]
    filter = filters[0]
    stepsize = [x['stepSize'] for x in filter if x['filterType'] == 'LOT_SIZE']
    minqty = [x['minQty'] for x in filter if x['filterType'] == 'LOT_SIZE']
    return stepsize[0], minqty[0]


def blesstrade_new_limit_order(df, symbol, fcnt, longshort, df_lows_plot, df_highs_plot, wavepattern, idx):
    w = wavepattern
    # real condititon by fractal index
    real_condititon1 = True if (fcnt/2 < (w.idx_end - w.idx_start)) and w.idx_start == idx else False
    real_condititon2 = True if df.iloc[idx + int(fcnt/2), 0] < (w.dates[-1]) else False
    if not real_condititon1:
        if printout: print('not real_condititon1 ')
        return
    if not real_condititon2:
        if printout: print('not real_condititon2 ')
        return


    w_start_price = w.values[0]  # wave1
    w_end_price = w.values[-1]  # wave5
    height_price = abs(w_end_price - w_start_price)
    o_fibo_value = height_price * o_fibo / 100 if o_fibo else 0
    # entry_price = w.values[7]  # wave4

    # entry_price = w.values[0] + height_price * entry_fibo if longshort else w.values[0] - height_price * entry_fibo  # 0.05
    # sl_price = w.values[0] + height_price * sl_fibo if longshort else w.values[0] - height_price * sl_fibo
    # target_price = entry_price + height_price * target_fibo if longshort else entry_price - height_price * target_fibo  # 0.351

    w1 = w.values[0]
    w2 = w.values[2]
    w3 = w.values[5]
    w4 = w.values[7]
    w5 = w.values[9]

    target_price = w5
    entry_price = w4
    sl_price = w1


    df_active = df[w.idx_end + 1:]
    dates = df_active.Date.tolist()
    closes = df_active.Close.tolist()
    trends = df_active.High.tolist() if longshort else df_active.Low.tolist()
    detrends = df_active.Low.tolist() if longshort else df_active.High.tolist()
    try:
        active_max_value = max(df_active.High.tolist(), default=w_end_price)
        active_min_value = min(df_active.Low.tolist(), default=w_end_price)
    except Exception as e:
        logger.error('active_max_value:' + str(e))
        return

    # check order or position
    current_price = float(um_futures_client.ticker_price(symbol)['price'])
    position = False
    c_active_min_max = (active_min_value > entry_price and active_max_value < (w_end_price + o_fibo_value)) \
                            if longshort else \
                        (active_max_value < entry_price and active_min_value > (w_end_price - o_fibo_value))

    # case 1.
    # c_current_price = (current_price > entry_price and current_price < (w_end_price + o_fibo_value)) \
    #                         if longshort else \
    #                         (current_price < entry_price and current_price > (w_end_price - o_fibo_value))

    half_entry_target = entry_price + abs(target_price - entry_price)/2 if longshort else entry_price - abs(target_price - entry_price)/2
    # case 2. 좀 더 많은 기회를 갖기 위한 것일까
    # c_current_price = (current_price > entry_price and current_price < target_price) \
    #                         if longshort else \
    #                         (current_price < entry_price and current_price > target_price)

    c_current_price = (current_price > entry_price and current_price < half_entry_target) \
                            if longshort else \
                            (current_price < entry_price and current_price > half_entry_target)

    if c_active_min_max and c_current_price:

        margin_available, available_balance, wallet_balance = my_available_balance(exchange_symbol)

        if not margin_available:
            logger.info('margin_available : False')
            logger.info('symbol:%s, available_balance:%s, wallet_balance:%s' % (symbol, str(available_balance), str(wallet_balance)))
            return

        # max_quantity = float(available_balance) * int(leverage) / current_price
        # quantity = max_quantity * qtyrate

        # max_quantity = wallet_balance * qtyrate * int(leverage) / current_price
        max_quantity = wallet_balance * qtyrate / current_price
        quantity = max_quantity

        step_size, minqty = get_quantity_step_size_minqty(symbol)
        quantity = format_value(quantity, step_size)

        if available_balance <= wallet_balance * qtyrate:
            # logger.info('available_balance <= wallet_balance * qtyrate')
            # logger.info('symbol:%s, available_balance:%s, wallet_balance:%s' % (symbol, str(available_balance), str(wallet_balance)))
            return

        if float(quantity) < float(minqty):
            logger.info('float(quantity) <= float(minqty)')
            logger.info('symbol:%s, quantity:%s, minqty:%s' % (symbol, str(quantity), str(minqty)))
            logger.info('symbol:%s, available_balance:%s, wallet_balance:%s' % (symbol, str(available_balance), str(wallet_balance)))
            return

        if not quantity:
            logger.info('quantity:' + str(quantity))
            logger.info('available_balance:%s, wallet_balance:%s' % (str(available_balance), str(wallet_balance)))
            return

        entry_price = set_price(symbol, entry_price, longshort)
        sl_price = set_price(symbol, sl_price, longshort)
        target_price = set_price(symbol, target_price, longshort)

        if entry_price == sl_price:
            logger.info(symbol + ' _entry_price == sl_price')
            return

        #####  이중 new limit order 방지 로직 start #####
        c_no_double_order = True
        history_new = [x for x in open_order_history if
                       (x['symbol'] == symbol and x['status'] == 'NEW')]


        if len(history_new) > 0:
            for history in history_new:
                h_id = history['id']
                h_status = history['status']
                longshort = history['longshort']
                h_limit_orderId = history['limit_orderId']
                h_sl_orderId = history['sl_orderId']
                h_tp_orderId = history['tp_orderId']

                if h_limit_orderId:
                    r_query_limit = um_futures_client.query_order(symbol=symbol,
                                                                  orderId=h_limit_orderId,
                                                                  recvWindow=6000)

                    if r_query_limit['status'] == 'FILLED':
                        # 대상외
                        return

                    if float(r_query_limit['price']) == float(entry_price) or float(r_query_limit['clientOrderId'].split('_')[2]) == float(target_price):
                        # and r_query_limit['newClientOrderId'] == target_price:  # when limit order, set newClientOrderId": str(target_price), therefore ..
                        return

        #####  이중 new limit order 방지 로직 start #####

        if c_no_double_order:
            # 1. new_order LIMIT
            params = [
                {
                    "symbol": symbol,
                    "side": "BUY" if longshort else "SELL",
                    "type": "LIMIT",
                    "positionSide": "LONG" if longshort else "SHORT",
                    "quantity": str(quantity),
                    "timeInForce": "GTC",
                    "price": str(entry_price),
                    "newClientOrderId": 'limit_' + str(sl_price) + '_' + str(target_price),
                }
            ]
            result_limit = new_batch_order(params)
            for order_limit in result_limit:
                try:
                    if order_limit['orderId']:
                        # 2. new_order STOP_MARKET
                        params = [
                            {
                                "symbol": symbol,
                                "side": "SELL" if longshort else "BUY",
                                "type": "STOP_MARKET",
                                "positionSide": "LONG" if longshort else "SHORT",
                                "quantity": str(quantity),
                                "timeInForce": "GTC",
                                "stopPrice": str(sl_price),
                                "priceProtect": "TRUE",
                                "closePosition": "TRUE",
                                "workingType": "CONTRACT_PRICE",
                                "newClientOrderId": "sl_" + str(order_limit['orderId']),
                            }
                        ]
                        result_sl = new_batch_order(params)

                        for order_sl in result_sl:
                            try:
                                ## success perfect and save open_order_history
                                if order_sl['orderId']:
                                    # {'symbol':'SYMBOLUSDT', 'wavepattern': wavepattern, 'entry':'10000', 'target':'10381', 'status':'NEW or DRAW or DONE',  'data': [{limit_result}, {sl_result}, {tp_result}], 'position':[]}
                                    open_order = {
                                        'id': dt.datetime.now().timestamp(),
                                        'datetime': dt.datetime.now(),
                                        'status': 'NEW',
                                        'symbol': symbol,
                                        'longshort': longshort,
                                        'entry': entry_price,
                                        'target': target_price,
                                        'sl_price': sl_price,
                                        'limit_orderId': order_limit['orderId'],
                                        'sl_orderId': order_sl['orderId'],
                                        'tp_orderId': None,
                                        'wavepattern': wavepattern,
                                        'data': [order_limit, order_sl]
                                    }

                                    # create history NEW
                                    open_order_history.append(open_order)
                                    logger.info(symbol + ' _NEW order:' + str(open_order))
                                    dump_history_pkl()

                                    if plotview:
                                        plot_pattern_m(df=df,
                                                       wave_pattern=[[1, wavepattern.dates[0], id(wavepattern), wavepattern]],
                                                       df_lows_plot=df_lows_plot, df_highs_plot=df_highs_plot,
                                                       trade_info=None, title=str(
                                                symbol + ' %s ' % str(longshort) + str(current_price)))

                                    return

                            except Exception as e:
                                logger.error(symbol + ' _FAIL SL NEW order :' + str(e))
                                logger.error(symbol + 'result_sl:' + str(result_sl))
                                # 2-1. when new_order STOP_MARKET's fail cancel LIMIT order
                                response = um_futures_client.cancel_batch_order(
                                    symbol=symbol, orderIdList=[order_limit['orderId']], origClientOrderIdList=[],
                                    recvWindow=6000
                                )

                                for order_tp in response:
                                    try:
                                        if order_tp['orderId']:
                                            logger.info(symbol + ' _CANCEL BATCH [LIMIT NEW order by SL FAIL] success')
                                            return
                                    except Exception as e:
                                        logger.error(symbol + ' _CANCEL BATCH [LIMIT NEW order by SL FAIL] error:' + str(e))
                                return

                except Exception as e:
                    logger.error(symbol + ':' + str(e))
                    logger.error(symbol + ' _LIMIT_new order error, result_limit:' + str(result_limit))

                    response = um_futures_client.cancel_batch_order(
                        symbol=symbol, orderIdList=[order_limit['orderId']], origClientOrderIdList=[],
                        recvWindow=6000
                    )

                    for order_tp in response:
                        try:
                            if order_tp['orderId']:
                                logger.info(symbol + ' _CANCEL BATCH [LIMIT NEW order by Exception] success')
                        except Exception as e:
                            logger.error(symbol + ' _CANCEL BATCH [LIMIT NEW order by Exception] error:' + str(e))
    return


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


def rename_symbol(s):
    if s[-4:] == 'USDT':
        return s.replace('USDT', '/USDT')
    if s[-4:] == 'BUSD':
        return s.replace('BUSD', '/BUSD')



def loopsymbol(symbol, i):
    ###################
    ## data settting ##
    ###################
    timeunit = 'm'
    bin_size = str(timeframe) + timeunit
    delta = (10 * fcnt[0] + 1)
    df = get_fetch_dohlcv(symbol,
                        interval=bin_size,
                        limit=delta)
    if df is not None:
        if df.empty:
            return

    df_all = df
    longshort = None
    if df is not None and df.empty:
        return
    try:
        wa = WaveAnalyzer(df=df_all, verbose=True)
    except:
        return

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
            impulse = Impulse('impulse')
            lows_idxs = df_lows.index.tolist()
            idxs = lows_idxs

        if 'short' in type:
            df_highs = fractals_high_loopA(df_all, fcnt=fc, loop_count=loop_count)
            df_highs_plot = df_highs[['Date', 'High']]
            downimpulse = DownImpulse('downimpulse')
            highs_idxs = df_highs.index.tolist()
            idxs = highs_idxs

        rules_to_check = list()

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

                                c_check_wave_identical = False
                                c_check_wave_identical_2_3_4 = False
                                if len(open_order_history) > 0:
                                    open_order_this_symbol = [x for x in open_order_history if x['symbol'] == symbol]
                                    if len(open_order_this_symbol) > 0:
                                        open_order_wavepattern = open_order_this_symbol[0]['wavepattern']

                                        # check identical check
                                        c_check_wave_identical = True if (open_order_wavepattern.dates == wavepattern.dates) and (open_order_wavepattern.values == wavepattern.values) else False
                                        # 2,3,4 wave same check
                                        c_check_wave_identical_2_3_4 = True if (open_order_wavepattern.dates[2:-2] == wavepattern.dates[2:-2]) and (open_order_wavepattern.values[2:-2] == wavepattern.values[2:-2]) else False

                                c_query_new = False
                                history_new = [x for x in open_order_history if
                                               (x['symbol'] == symbol and x['status'] == 'NEW')]

                                if len(history_new) > 0:
                                    for history in history_new:
                                        h_id = history['id']
                                        h_status = history['status']
                                        longshort = history['longshort']
                                        h_limit_orderId = history['limit_orderId']
                                        h_sl_orderId = history['sl_orderId']
                                        h_tp_orderId = history['tp_orderId']

                                        r_query_longshort = None
                                        r_query_status = None

                                        if h_limit_orderId:
                                            r_query_limit = um_futures_client.query_order(symbol=symbol,
                                                                                          orderId=h_limit_orderId,
                                                                                          recvWindow=6000)

                                            r_q_longshort = True if r_query_limit['positionSide'] == 'LONG' else False
                                            r_query_status = r_query_limit['status']

                                            if r_query_limit['status'] == 'NEW' or r_query_limit['status'] == 'FILLED':
                                                c_query_new = True

                                # if not (c_check_wave_identical and c_query_new):
                                if not ((c_check_wave_identical or c_check_wave_identical_2_3_4) and c_query_new):
                                    blesstrade_new_limit_order(df_all, symbol, fc, longshort, df_lows_plot, df_highs_plot, wavepattern, i)

    return

def get_i_r(list, key, value):
    r = [[i, x] for i, x in enumerate(list) if x[key] == value]
    if len(r) == 1:
        return r[0][0], r[0][1]
    return None, None


def new_tp_order(symbol, longshort, target, quantity, limit_orderId):
    params = [
        {
            "symbol": symbol,
            "side": "SELL" if longshort else "BUY",
            "type": "LIMIT",
            "positionSide": "LONG" if longshort else "SHORT",
            "price": str(target),
            "quantity": str(quantity),
            "priceProtect": "TRUE",
            "timeInForce": "GTC",
            "postOnly": "TRUE",
            "workingType": "CONTRACT_PRICE",
            "newClientOrderId": 'tp_' + str(limit_orderId)
        }
    ]
    result_tp = new_batch_order(params)
    for order_tp in result_tp:
        try:
            if order_tp['orderId']:
                logger.info(symbol + ' _TAKE_PROFIT new order success')
                # find history limit order and update it
                update_history_by_limit_id(open_order_history, symbol, 'TAKE_PROFIT', limit_orderId)
                return True
        except Exception as e:
            try:
                if order_tp['code'] == -2021:  # [{'code': -2021, 'msg': 'Order would immediately trigger.'}]
                    current_price = float(um_futures_client.ticker_price(symbol)['price'])
                    compare_price_flg = current_price > target if longshort else current_price < target
                    if compare_price_flg:
                        order_market = um_futures_client.new_order(
                            symbol=symbol,
                            side="SELL" if longshort else "BUY",
                            positionSide="LONG" if longshort else "SHORT",
                            type="MARKET",
                            quantity=quantity,
                            newClientOrderId="tp_" + str(limit_orderId)
                        )
                        if order_market['orderId']:
                            logger.info(symbol + ' _TP MARKET over TP -> success')
                            # find history limit order and update it
                            update_history_by_limit_id(open_order_history, symbol, 'DONE', limit_orderId)
                            return True
                pass
            except Exception as e:
                logger.error(symbol + ' _TP SELL MARKET error:' + str(e))

            logger.error(symbol + ' _TAKE_PROFIT Exception error:' + str(e))
            return False
    return False


def cancel_batch_order(symbol, orderIdList):
    try:
        response = um_futures_client.cancel_batch_order(
            symbol=symbol, orderIdList=orderIdList, origClientOrderIdList=[],
            recvWindow=6000
        )
        if len(response) > 0:
            try:
                if response[0]['orderId']:
                    return True
            except Exception as e:
                if response[0]['code']:
                    logger.error(symbol + ' _cancel_batch_order error:' + str(response[0]['msg']))
                    return False
    except Exception as e:
        logger.error(symbol + ' _cancel_batch_order Exception error:' + str(e))
        return False
    return False


def update_history_status(open_order_history, symbol, h_id, new_status):
    history_idx, history_id = get_i_r(open_order_history, 'id', h_id)
    history_id['status'] = new_status  # update new status
    open_order_history[history_idx] = history_id  # replace history
    print(symbol + ' _update history to new status:%s' % new_status)
    logger.info(symbol + ' _update history to new status:%s' % new_status)
    dump_history_pkl()


def update_history_by_limit_id(open_order_history, symbol, description, limit_orderId):
    history_matched = [x for x in open_order_history if
                       (x['symbol'] == symbol and x['limit_orderId'] == float(limit_orderId))]
    if len(history_matched) > 0:
        h_id = history_matched[0]['id']
        update_history_status(open_order_history, symbol, h_id, description)

def delete_history_status(open_order_history, symbol, h_id, event):
    history_idx, history_id = get_i_r(open_order_history, 'id', h_id)
    open_order_history.pop(history_idx)
    print(symbol + ' _delete history by %s' % event)
    dump_history_pkl()


def monitoring_orders_positions(symbol):
    try:
        # 1. check orders
        all_orders = um_futures_client.get_all_orders(symbol=symbol, recvWindow=6000)
        new_sles = [x for x in all_orders if (x['status'] == 'NEW' and x['type'] == 'STOP_MARKET')]
        if len(new_sles) > 0:
            for new_sl in new_sles:
                sl_orderId = new_sl['orderId']
                longshort = True if new_sl['positionSide'] == 'LONG' else False
                limit_orderId = new_sl['clientOrderId'][3:]  # trace limit's orderId by 'sl_limitOrderId'

                # 1. check limit status
                r_query_limit = um_futures_client.query_order(symbol=symbol,
                                                              orderId=limit_orderId,
                                                              recvWindow=6000)
                if r_query_limit['status'] == 'FILLED':
                    target_price = r_query_limit['clientOrderId'].split('_')[2]  # target_price
                    quantity = r_query_limit['executedQty']  # target_price

                    # 2. check if there is TP or TP's status with this origClientOrderId
                    key = 'clientOrderId'
                    value = 'tp_' + limit_orderId
                    tp_index, tp_order = get_i_r(all_orders, key, value)
                    if tp_index:
                        if tp_order['status'] == 'FILLED':
                            # force cancel sl
                            success_cancel = cancel_batch_order(symbol, [sl_orderId])
                            if success_cancel:
                                update_history_by_limit_id(open_order_history, symbol, 'TP_FILLED', limit_orderId)
                    else:
                        try:
                            success = new_tp_order(symbol, longshort, target_price, quantity, limit_orderId)
                        except Exception as e:
                            logging.error(
                                "Found monitoring_orders_positions/r_query_tp error. symbol: {}, status: {}, error code: {}, error message: {}".format(
                                    symbol, e.status_code, e.error_code, e.error_message
                                )
                            )

                            result_position = um_futures_client.get_position_risk(symbol=symbol, recvWindow=6000)
                            result_position_filtered = [x for x in result_position if x['entryPrice'] != '0.0']
                            if len(result_position_filtered) > 0:  # 2. if in position
                                for p in result_position_filtered:
                                    all_orders = um_futures_client.get_all_orders(symbol=symbol, recvWindow=6000)
                                    filled_limits = [x for x in all_orders if (
                                                x['status'] == 'FILLED' and x['type'] == 'LIMIT' and x['price'] == p[
                                            'entryPrice'])]
                                    if len(filled_limits) > 0:
                                        limit_orderId = filled_limits[0]['orderId']
                                        longshort = True if filled_limits[0]['positionSide'] == 'LONG' else False
                                        quantity = p['positionAmt']
                                        order_market = um_futures_client.new_order(
                                            symbol=symbol,
                                            side="SELL" if longshort else "BUY",
                                            positionSide="LONG" if longshort else "SHORT",
                                            type="MARKET",
                                            quantity=quantity,
                                            newClientOrderId="tp_" + str(limit_orderId)
                                        )
                                        if order_market['orderId']:
                                            logger.info(symbol + ' FORCE MARKET BY ONLY POSI WITH NO SL_TP' + str(p))
                                            # find history limit order and update it
                                            update_history_by_limit_id(open_order_history, symbol, 'DONE',
                                                                       limit_orderId)
    except Exception as e:
        logging.error(
            "Found monitoring_orders_positions error. symbol: {}, status: {}, error code: {}, error message: {}".format(
                symbol, e.status_code, e.error_code, e.error_message
            )
        )


def set_status_manager_when_new_or_tp(symbol):
    history_new_tp = [x for x in open_order_history if
                   (x['symbol'] == symbol and (x['status'] == 'NEW' or x['status'] == 'TAKE_PROFIT'))]

    if len(history_new_tp) > 0:
        for history in history_new_tp:
            h_id = history['id']
            h_status = history['status']
            longshort = history['longshort']
            h_limit_orderId = history['limit_orderId']
            h_sl_orderId = history['sl_orderId']
            h_tp_orderId = history['tp_orderId']


            r_query_limit = um_futures_client.query_order(symbol=symbol,
                                                          orderId=h_limit_orderId,
                                                          recvWindow=6000)
            r_query_sl = um_futures_client.query_order(symbol=symbol,
                                                          orderId=h_sl_orderId,
                                                          recvWindow=6000)
            # CASE TP
            if h_tp_orderId:
                r_query_tp = um_futures_client.query_order(symbol=symbol,
                                                              orderId=h_tp_orderId,
                                                              recvWindow=6000)
                if r_query_limit['status'] == 'FILLED' and r_query_sl['status'] == 'NEW' and r_query_tp['status'] == 'NEW':
                    return
                if r_query_limit['status'] == 'FILLED' and r_query_sl['status'] == 'FILLED' and r_query_tp['status'] == 'EXPIRED':
                    # update_history_status(open_order_history, symbol, h_id, 'DONE')
                    delete_history_status(open_order_history, symbol, h_id, 'F-F-E')
                    return
                if r_query_limit['status'] == 'FILLED' and r_query_sl['status'] == 'NEW' and r_query_tp['status'] == 'FILLED':
                    success_cancel = cancel_batch_order(symbol, [h_sl_orderId])
                    if success_cancel:
                        # update_history_status(open_order_history, symbol, h_id, 'DONE')
                        delete_history_status(open_order_history, symbol, h_id, 'F-N-F')
                    return
            # CASE NO TP
            else:
                if r_query_limit['status'] == 'NEW' and r_query_sl['status'] == 'NEW':
                    ###################
                    # case1. outzone
                    ###################
                    entry_price = float(r_query_limit['price'])
                    target_price = float(r_query_limit['clientOrderId'].split('_')[2])

                    half_entry_target = entry_price + abs(
                        target_price - entry_price) / 2 if longshort else entry_price - abs(
                        target_price - entry_price) / 2

                    current_price = float(um_futures_client.ticker_price(symbol)['price'])
                    c_current_price = current_price > half_entry_target if longshort else current_price < half_entry_target

                    if c_current_price:
                        # OUTZONE
                        success_cancel = cancel_batch_order(symbol, [r_query_limit['orderId'], h_sl_orderId])
                        if success_cancel:
                            # update_history_status(open_order_history, symbol, h_id, 'OUTZONE')
                            delete_history_status(open_order_history, symbol, h_id, 'N-N-OUTZONE')

                    ###################
                    # case2. beyond
                    ###################
                    w = history['wavepattern']
                    delta = (10 * fcnt[0] + 1)
                    df = get_fetch_dohlcv(symbol,
                                          interval=(str(timeframe) + 'm'),
                                          limit=delta)

                    w_start_price = w.values[0]  # wave1
                    w_end_price = w.values[-1]  # wave5
                    height_price = abs(w_end_price - w_start_price)
                    o_fibo_value = height_price * o_fibo / 100 if o_fibo else 0
                    # entry_price = w.values[7]  # wave4

                    entry_price = w.values[0] + height_price * entry_fibo if longshort else w.values[
                                                                                                0] - height_price * entry_fibo  # 0.05
                    wave5_datetime = w.dates[-1]
                    # df_active = df[w.idx_end + 1:]
                    df_active = df.loc[df['Date'] > wave5_datetime]

                    try:
                        active_max_value = max(df_active.High.tolist(), default=w_end_price)
                        active_min_value = min(df_active.Low.tolist(), default=w_end_price)
                    except Exception as e:
                        logger.error('active_max_value:' + str(e))
                        return

                    c_active_min_max = (
                                active_min_value > entry_price and active_max_value < (w_end_price + o_fibo_value)) \
                        if longshort else \
                        (active_max_value < entry_price and active_min_value > (w_end_price - o_fibo_value))
                    if not c_active_min_max:
                        success_cancel = cancel_batch_order(symbol, [str(h_limit_orderId),
                                                        str(h_sl_orderId)])
                        if success_cancel:
                            delete_history_status(open_order_history, symbol, h_id, 'N-N-BEYOND')
                            return
                    return

                if (r_query_limit['status'] == 'FILLED' or r_query_limit['status'] == 'CANCELED') and r_query_sl['status'] == 'NEW':

                    result_position = um_futures_client.get_position_risk(symbol=symbol, recvWindow=6000)
                    result_position_filtered = [x for x in result_position if x['entryPrice'] != '0.0']
                    if len(result_position_filtered) == 0:  # no position
                        success_cancel = cancel_batch_order(symbol, [str(h_sl_orderId)])
                        if success_cancel:
                            delete_history_status(open_order_history, symbol, h_id, 'F-N-DONE')
                            return
                    return
                if r_query_limit['status'] == 'FILLED' and r_query_sl['status'] == 'FILLED':
                    delete_history_status(open_order_history, symbol, h_id, 'F-F-DONE')
                    return


def single(symbols, i, *args):
    for symbol in symbols:
        try:
            monitoring_orders_positions(symbol)
            if len(open_order_history) > 0:
                set_status_manager_when_new_or_tp(symbol)
            loopsymbol(symbol, i)
        except Exception as e:
            logger.error("ERROR in single:" + str(e))
            print("ERROR in single:" + str(e))
    # print(f' {i} end: {time.strftime("%H:%M:%S")}')
    return


def set_leverage_allsymbol(symbols, leverage):
    logger.info('set  x %s leverage start' % str(leverage))
    for symbol in symbols:
        tryleverage = leverage
        while True:
            try:
                response = um_futures_client.change_leverage(
                    symbol=symbol, leverage=tryleverage, recvWindow=6000
                )
                time.sleep(0.1)
                logger.info(response)
                break
            except ClientError as error:
                tryleverage = tryleverage - 4
                logger.error(
                    "Found error. status: {}, error code: {}, error message: {}".format(
                        error.status_code, error.error_code, error.error_message
                    )
                )

    logger.info('set  x %s leverage done' % str(leverage))

if __name__ == '__main__':
    print("""

     _____             _ _              ______       _
    |_   _|           | (_)             | ___ \     | |
      | |_ __ __ _  __| |_ _ __   __ _  | |_/ / ___ | |_
      | | '__/ _` |/ _` | | '_ \ / _` | | ___ \/ _ \| __|
      | | | | (_| | (_| | | | | | (_| | | |_/ / (_) | |_
      \_/_|  \__,_|\__,_|_|_| |_|\__, | \____/ \___/ \__| v0.2
                                  __/ |
                                 |___/

    """)
    print_condition()
    set_leverage_allsymbol(symbols_binance_futures, leverage)

    start = time.perf_counter()

    symbols = get_symbols()
    i = 1
    logger.info('PID:' + str(os.getpid()))
    while True:
        if i % 10 == 1:
            logger.info(f'{i} start: {time.strftime("%H:%M:%S")}')
        single(symbols, i)
        i += 1

    print(f'Finished in {round(time.perf_counter() - start, 2)} second(s)')
    print_condition()
    print("good luck done!!")
