from __future__ import annotations
from models.WavePattern import WavePattern
from models.WaveRules import Impulse, DownImpulse
from models.WaveAnalyzer import WaveAnalyzer
from models.WaveOptions import WaveOptionsGenerator5
from models.helpers import plot_pattern_m, plot_pattern_n, plot_pattern_k
import datetime as dt
import pandas as pd
import numpy as np
from ratelimit import limits, sleep_and_retry, RateLimitException
from backoff import on_exception, expo
from binancefutures.um_futures import UMFutures
from binancefutures.error import ClientError
from binance.client import Client
from binance.helpers import round_step_size
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from tapy import Indicators

import shutup;

shutup.please()
import json
import os
import random
import pickle
import math
import logging
import threading
import functools
import time
from random import uniform, randrange

with open('w045configmode.json', 'r') as f:
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
walletrate = config['default']['walletrate']

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

round_trip_flg = config['default']['round_trip_flg']
round_trip_count = config['default']['round_trip_count']
compounding = config['default']['compounding']
fcnt = config['default']['fcnt']
loop_count = config['default']['loop_count']

timeframe = config['default']['timeframe']

up_to_count = config['default']['up_to_count']
c_same_date = config['default']['c_same_date']
c_sma_n = config['default']['c_sma_n']

c_compare_before_fractal = config['default']['c_compare_before_fractal']
c_compare_before_fractal_strait = config['default']['c_compare_before_fractal_strait']
if c_compare_before_fractal_strait:
    c_compare_before_fractal_shift = 1
c_compare_before_fractal_shift = config['default']['c_compare_before_fractal_shift']
c_compare_before_fractal_mode = config['default']['c_compare_before_fractal_mode']
if not c_compare_before_fractal:
    c_compare_before_fractal_shift = 0
    c_compare_before_fractal_strait = False
    c_compare_before_fractal_mode = 0

c_plrate_adaptive = config['default']['c_plrate_adaptive']
c_plrate_rate = config['default']['c_plrate_rate']
c_plrate_rate_min = config['default']['c_plrate_rate_min']
c_kelly_adaptive = config['default']['c_kelly_adaptive']
c_kelly_window = config['default']['c_kelly_window']

et_zone_rate = config['default']['et_zone_rate']

c_time_beyond_flg = config['default']['c_time_beyond_flg']
c_time_beyond_rate = config['default']['c_time_beyond_rate']

c_risk_beyond_flg = config['default']['c_risk_beyond_flg']
c_risk_beyond_max = config['default']['c_risk_beyond_max']
c_risk_beyond_min = config['default']['c_risk_beyond_min']

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
plotview = config['default']['plotview']
printout = config['default']['printout']
init_running_trade = config['default']['init_running_trade']
reset_leverage = config['default']['reset_leverage']
trade_mode = config['default']['trade_mode']
paper_flg = config['default']['paper_flg']

seq = dt.datetime.now().strftime("%Y%m%d_%H%M%S") + str([timeframe, fcnt, period_days_ago, period_days_ago_till])
# seq = '{date:%Y-%m-%d_%H:%M:%S}'.format(date=dt.datetime.now())

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s')  # ('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
file_handler = logging.FileHandler('logger_%s.log' % seq)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

botfather_token = config['message']['botfather_token']


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
    logger.info('walletrate:%s' % str(walletrate))
    logger.info('seed:%s' % str(seed))
    logger.info('fee:%s%%' % str(fee * 100))
    logger.info('fee_slippage:%s%%' % str(round(fee_slippage * 100, 4)))
    logger.info('timeframe: %s' % timeframe)
    logger.info('fcnt: %s' % fcnt)

    logger.info('period_days_ago: %s' % period_days_ago)
    logger.info('period_days_ago_till: %s' % period_days_ago_till)
    logger.info('period_interval: %s' % period_interval)

    logger.info('c_time_beyond_flg: %s' % c_time_beyond_flg)
    logger.info('c_time_beyond_rate: %s' % c_time_beyond_rate)

    logger.info('c_risk_beyond_flg: %s' % c_risk_beyond_flg)
    logger.info('c_risk_beyond_max: %s' % c_risk_beyond_max)
    logger.info('c_risk_beyond_min: %s' % c_risk_beyond_min)

    logger.info('round_trip_count: %s' % round_trip_count)
    logger.info('compounding: %s' % compounding)
    logger.info('loop_count: %s' % loop_count)
    logger.info('symbol_random: %s' % symbol_random)
    logger.info('symbol_last: %s' % symbol_last)
    logger.info('symbol_length: %s' % symbol_length)

    start_dt = str((pd.to_datetime('today') - pd.Timedelta(str(period_days_ago) + ' days')).date())
    end_dt = str((pd.to_datetime('today') - pd.Timedelta(str(period_days_ago_till) + ' days')).date())
    logger.info('period: %s ~ %s' % (start_dt, end_dt))
    logger.info('up_to_count: %s' % up_to_count)
    logger.info('c_same_date: %s' % c_same_date)
    logger.info('c_sma_n: %s' % c_sma_n)
    logger.info('c_compare_before_fractal: %s' % c_compare_before_fractal)
    logger.info('c_compare_before_fractal_mode: %s' % c_compare_before_fractal_mode)
    logger.info('c_compare_before_fractal_strait: %s' % c_compare_before_fractal_strait)
    logger.info('c_compare_before_fractal_shift: %s' % c_compare_before_fractal_shift)

    logger.info('c_plrate_adaptive: %s' % c_plrate_adaptive)
    logger.info('c_plrate_rate: %s' % c_plrate_rate)
    logger.info('c_plrate_rate_min: %s' % c_plrate_rate_min)
    logger.info('c_kelly_adaptive: %s' % c_kelly_adaptive)
    logger.info('c_kelly_window: %s' % c_kelly_window)

    logger.info('et_zone_rate: %s' % et_zone_rate)
    logger.info('o_fibo: %s' % o_fibo)
    logger.info('h_fibo: %s' % h_fibo)
    logger.info('l_fibo: %s' % l_fibo)

    logger.info('entry_fibo: %s' % entry_fibo)
    logger.info('target_fibo: %s' % target_fibo)
    logger.info('sl_fibo: %s' % sl_fibo)

    logger.info('fee:%s%%' % str(fee * 100))
    logger.info('fee_limit:%s%%' % str(fee_limit * 100))
    logger.info('fee_sl:%s%%' % str(fee_sl * 100))
    logger.info('fee_tp:%s%%' % str(fee_tp * 100))
    logger.info('tp_type:%s' % str(tp_type))

    logger.info('intersect_idx: %s' % intersect_idx)
    logger.info('plotview: %s' % plotview)
    logger.info('printout: %s' % printout)
    logger.info('init_running_trade: %s' % init_running_trade)
    logger.info('reset_leverage: %s' % reset_leverage)
    logger.info('trade_mode: %s' % trade_mode)
    logger.info('paper_flg: %s' % paper_flg)
    logger.info('-------------------------------')


um_futures_client = UMFutures(key=futures_secret_key, secret=futures_secret_value)
client = Client(futures_secret_key, futures_secret_value)


# @on_exception(expo, RateLimitException, max_tries=3)
# @sleep_and_retry
# @limits(calls=2400, period=60)  # 2400call/60sec
def api_call(method, arglist, **kwargs):
    response = None
    try:
        if method == 'klines':
            symbol = arglist[0]
            interval = arglist[1]
            limit = arglist[2]
            response = um_futures_client.klines(symbol, interval, limit=limit)
        elif method == 'ticker_price':
            symbol = arglist[0]
            response = um_futures_client.ticker_price(symbol)
        elif method == 'new_order':
            symbol = arglist[0]
            side = arglist[1]
            positionSide = arglist[2]
            type = arglist[3]
            quantity = arglist[4]
            newClientOrderId = arglist[5]
            response = um_futures_client.new_order(
                symbol=symbol,
                side=side,
                positionSide=positionSide,
                type=type,
                quantity=quantity,
                newClientOrderId=newClientOrderId
            )
        elif method == 'new_batch_order':
            params = arglist[0]
            try:
                response = um_futures_client.new_batch_order(params)
            except:
                return None
        elif method == 'query_order':
            symbol = arglist[0]
            orderId = arglist[1]
            try:
                response = um_futures_client.query_order(symbol=symbol, orderId=orderId, recvWindow=6000)
            except:
                return None
        elif method == 'get_open_orders':
            symbol = arglist[0]
            orderId = arglist[1]
            try:
                response = um_futures_client.get_open_orders(symbol=symbol, orderId=orderId, recvWindow=6000)
            except:
                return None
        elif method == 'get_all_orders':
            symbol = arglist[0]
            response = um_futures_client.get_all_orders(symbol=symbol, recvWindow=6000)
        elif method == 'get_position_risk':
            symbol = arglist[0]
            response = um_futures_client.get_position_risk(symbol=symbol, recvWindow=6000)
        elif method == 'get_account_trades':
            symbol = arglist[0]
            response = um_futures_client.get_account_trades(symbol=symbol, recvWindow=6000)
        elif method == 'cancel_batch_order':
            symbol = arglist[0]
            orderIdList = arglist[1]
            origClientOrderIdList = arglist[2]
            response = um_futures_client.cancel_batch_order(symbol=symbol, orderIdList=orderIdList,
                                                            origClientOrderIdList=origClientOrderIdList,
                                                            recvWindow=6000)
        elif method == 'cancel_open_orders':
            symbol = arglist[0]
            response = um_futures_client.cancel_open_orders(symbol=symbol, recvWindow=6000)
        elif method == 'leverage_brackets':
            symbol = arglist[0]
            response = um_futures_client.leverage_brackets(symbol=symbol, recvWindow=6000)
        elif method == 'change_leverage':
            symbol = arglist[0]
            leverage = arglist[1]
            response = um_futures_client.change_leverage(symbol=symbol, leverage=leverage, recvWindow=6000)
        elif method == 'change_margin_type':
            symbol = arglist[0]
            margin_type = arglist[1]
            response = um_futures_client.change_margin_type(symbol=symbol, marginType=margin_type, recvWindow=6000)
        elif method == 'balance':
            response = um_futures_client.balance(recvWindow=6000)
        elif method == 'account':
            response = um_futures_client.account()
        elif method == 'exchange_info':
            response = um_futures_client.exchange_info()

    except ClientError as error:
        logging.error('API_CALL_ClientError: ' + method + ' : ' + str(arglist) +
                      " Found error. status: {}, error code: {}, error message: {}".format(
                          error.status_code, error.error_code, error.error_message))
        try:
            if error.status_code == 429 and error.error_code == -1003:
                time.sleep(5)
            if error.status_code == 418 and error.error_code == -1003:
                err_msg = error.error_message
                logging.error('err_msg', err_msg)
                untiltime = err_msg.split('banned until ')[1]
                untiltime = untiltime.split('. Please')[0]
                logging.error('untiltime', untiltime)
                if untiltime:
                    untiltime = dt.datetime.fromtimestamp(float(str(untiltime)) / 1000)
                    now = dt.datetime.now()
                    time.sleep((untiltime - now).total_seconds())
        except Exception as e:
            logging.error('API_CALL_ClientError xxxxxx error.status_code == 418' + str(e))
            return None

    except Exception as e:
        logging.error('API_CALL_Exception: ' + method + ' : ' + str(arglist) + " Found Exception. e: {}".format(str(e)))
        return None
    return response


open_order_history = []
# open_order_history_seq= 'history_' + seq + '.pkl'
# def load_history_pkl():
#     try:
#         if os.path.isfile(open_order_history_seq):
#             with open(open_order_history_seq, 'rb') as f:
#                 h = pickle.load(f)
#                 logger.info('load_history_pk:' + str(h))
#                 return h
#     except Exception as e:
#         logger.error(e)
#         try:
#             os.remove(open_order_history_seq)
#         except Exception as e:
#             logger.error(e)
#         return []
#     return []


# open_order_history = load_history_pkl()


# def dump_history_pkl():
#     try:
#         with open(open_order_history_seq, 'wb') as f:
#             pickle.dump(open_order_history, f)
#     except Exception as e:
#         logger.error(e)

symbols_binance_futures = []
symbols_binance_futures_USDT = []
symbols_binance_futures_BUSD = []
symbols_binance_futures_USDT_BUSD = []

exchange_info = api_call('exchange_info', [])
if exchange_info:
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

    if symbol_random:
        symbols = random.sample(symbols, len(symbols))
    if symbol_last:
        symbols = symbols[symbol_last:]
    if symbol_length:
        symbols = symbols[:symbol_length]

    logger.info(str(len(symbols)) + ':' + str(symbols))
    return symbols


def get_fetch_dohlcv(symbol,
                     interval=None,
                     limit=500):
    datalist = api_call('klines', [symbol, interval, limit])
    if datalist:
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


def sma(source, period):
    return pd.Series(source).rolling(period).mean().values


def sma_df(df, p):
    i = Indicators(df)
    i.sma(period=p)
    return i.df


def ichi_df(df):
    i = Indicators(df)
    i.ichimoku_kinko_hyo(period_tenkan_sen=9, period_kijun_sen=26, period_senkou_span_b=52, column_name_chikou_span='chikou_span', column_name_tenkan_sen='tenkan_sen', column_name_kijun_sen='kijun_sen', column_name_senkou_span_a='senkou_span_a', column_name_senkou_span_b='senkou_span_b')
    return i.df


def fractals_low_loopA(df, fcnt=None, loop_count=1):
    for c in range(loop_count):
        window = 2 * fcnt + 1
        df['fractals_low'] = df['Low'].rolling(window, center=True).apply(lambda x: x[fcnt] == min(x), raw=True)
        df = df[~(df[['fractals_low']].isin([0, np.nan])).all(axis=1)]
        df = df.dropna()
        df = df.drop(['fractals_low'], axis=1)
    return df



def fractals_high_loopA(df, fcnt=None, loop_count=1):
    for c in range(loop_count):
        window = 2 * fcnt + 1
        df['fractals_high'] = df['High'].rolling(window, center=True).apply(lambda x: x[fcnt] == max(x), raw=True)
        df = df[~(df[['fractals_high']].isin([0, np.nan])).all(axis=1)]
        df = df.dropna()
        df = df.drop(['fractals_high'], axis=1)
    return df


def get_precision(symbol):
    for x in exchange_info['symbols']:
        if x['symbol'] == symbol:
            return x['quantityPrecision']


def set_price(symbol, price, longshort):
    e_info = exchange_info['symbols']  # pull list of symbols
    for x in range(len(e_info)):  # find length of list and run loop
        if e_info[x]['symbol'] == symbol:  # until we find our coin
            a = e_info[x]["filters"][0]['tickSize']  # break into filters pulling tick size
            cost = round_step_size(price,
                                   float(a))  # convert tick size from string to float, insert in helper func with cost
            # 아래는 시장가의 비용 및 sleepage 를 보고 나중에 추가 또는 삭제 검토요
            # cost = cost - float(a) if longshort else cost + float(a)
            return cost


def set_price_for_tp(o_his, symbol, price, longshort, t_mode):
    try:
        if o_his:
            symbol_order_history = [x for x in o_his if x['symbol'] == symbol
                                    and x['trade_mode'] == t_mode
                                    and x['status'] == 'ETSL']
            symbol_order_history_last_10 = symbol_order_history
            if len(symbol_order_history) >= 10:
                symbol_order_history_last_10 = symbol_order_history[-10:]

            tp_prices = [x['tp_price'] for x in symbol_order_history_last_10]

            if price in tp_prices:
                tp_prices_sorted = sorted(tp_prices, key=lambda x: float(x), reverse=(True if longshort else False))
                # print(price, 'in', tp_prices_sorted)
                e_info = exchange_info['symbols']  # pull list of symbols
                for x in range(len(e_info)):  # find length of list and run loop
                    if e_info[x]['symbol'] == symbol:  # until we find our coin
                        tickSize = e_info[x]["filters"][0]['tickSize']  # break into filters pulling tick size

                        # price = round_step_size(price - float(tickSize), float(tickSize))  # make it 'one tickSize' different for more taker tp done

                        for other_tp in tp_prices_sorted:
                            if price == other_tp:
                                price = round_step_size(price - float(tickSize), float(tickSize)) \
                                    if longshort \
                                    else round_step_size(price + float(tickSize), float(tickSize))

                        # print(price, 'changed tp price')
                        return price
    except Exception as e:
        print(e)
    return price


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


def get_i_r(li, key, value):
    r = [[i, x] for i, x in enumerate(li) if x[key] == value]
    if len(r) == 1:
        return r[0][0], r[0][1]
    return None, None


def get_wave_prices(symbol, longshort, w):
    w0 = w.values[0]
    # w1 = w.values[1]
    # w2 = w.values[3]
    # w3 = w.values[5]
    w4 = w.values[7]
    w5 = w.values[9]

    et_price = set_price(symbol, w4, longshort)
    sl_price = set_price(symbol, w0, longshort)
    tp_price_w5 = set_price(symbol, w5, longshort)
    return et_price, sl_price, tp_price_w5


def get_trade_prices(o_his, symbol, longshort, w, t_mode):
    et_price, sl_price, tp_price_w5 = get_wave_prices(symbol, longshort, w)
    tp_price = set_price_for_tp(o_his, symbol, tp_price_w5, longshort, t_mode)
    return et_price, sl_price, tp_price, tp_price_w5


def new_et_order_real(symbol, tf, fc, longshort, qtyrate_k, et_price, sl_price, tp_price, tp_price_w5, quantity, wavepattern, o_his, t_mode):
    try:
        params_et = [
            {
                "symbol": symbol,
                "side": "BUY" if longshort else "SELL",
                "type": "LIMIT",
                "positionSide": "LONG" if longshort else "SHORT",
                "quantity": str(float(quantity)),
                "timeInForce": "GTC",
                "price": str(et_price),
                "newClientOrderId": str(tf) + '_' + str(fc) + '_' + str(sl_price) + '_' + str(tp_price_w5),
            }
        ]
        r1 = api_call('new_batch_order', [params_et])

        if r1 is not None:
            try:
                order_et = r1[0]
                if order_et['orderId']:
                    params_sl = [
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
                            "newClientOrderId": "sl_" + str(tf) + '_' + str(fc) + '_' + str(order_et['orderId']),
                        }
                    ]
                    r2 = api_call('new_batch_order', [params_sl])
                    if r2:
                        try:
                            order_sl = r2[0]
                            if order_sl['orderId']:

                                o_his = add_etsl_history(o_his, symbol, tf, fc, longshort, qtyrate_k, wavepattern, et_price,
                                                             sl_price, tp_price, tp_price_w5, quantity, order_et['orderId'],
                                                             order_sl['orderId'],
                                                             order_et,
                                                             order_sl,
                                                             t_mode
                                                         )
                                o_his = get_order_history_etsl_and_new_tp_order(o_his, symbol, order_et['orderId'], t_mode)
                                return True, o_his
                        except Exception as e:
                            r3 = api_call('cancel_batch_order', [symbol, [order_et['orderId']], []])
                            if r3:
                                try:
                                    order_cancel_et = r3[0]
                                    if order_cancel_et['code']:
                                        logger.error('_NEWET ET CANCEL FAIL ' + order_cancel_et['code'])
                                        return False, o_his
                                except:
                                    logger.error('_NEWET ET CANCEL FAIL2 ' + order_cancel_et['code'])
                                    return False, o_his
            except Exception as e:
                r3 = api_call('cancel_batch_order', [symbol, [order_et['orderId']], []])
                if r3:
                    try:
                        order_cancel_et = r3[0]
                        if order_cancel_et['code']:
                            logger.error('_NEWET ET CANCEL FAIL ' + order_cancel_et['code'])
                            return False, o_his
                    except:
                        logger.error('_NEWET ET CANCEL FAIL1 ' + order_cancel_et['code'])
                        return False, o_his
        else:
            logger.info(symbol + ' _FAIL ET ITSELF FAIL0' + str(
                (symbol, tf, fc, longshort, et_price, sl_price, tp_price, quantity, wavepattern)))
    except Exception as e:
        logger.error( symbol + ' new_et_order_real order_et error : e ' + str(e))
    return False, o_his


def new_et_order_test(symbol, tf, fc, longshort, qtyrate_k, et_price, sl_price, tp_price, tp_price_w5, quantity, wavepattern, et_orderid_test, o_his, t_mode):
    order_et_bt = {
        "symbol": symbol,
        "side": "BUY" if longshort else "SELL",
        "type": "LIMIT",
        "positionSide": "LONG" if longshort else "SHORT",
        "quantity": str(float(quantity)),
        "timeInForce": "GTC",
        "price": str(et_price),
        "newClientOrderId": str(tf) + '_' + str(fc) + '_' + str(sl_price) + '_' + str(tp_price),
    }
    order_sl_bt = {
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
        "newClientOrderId": "sl_" + str(tf) + '_' + str(fc) + '_' + str(et_orderid_test),
    }
    o_his = add_etsl_history(o_his, symbol, tf, fc, longshort, qtyrate_k, wavepattern, et_price,
                             sl_price, tp_price, tp_price_w5, quantity, et_orderid_test,
                             randrange(10000000000, 99999999999, 1),
                             order_et_bt,
                             order_sl_bt, t_mode)

    o_his = get_order_history_etsl_and_new_tp_order(o_his, symbol, et_orderid_test, t_mode)
    return True, o_his


def new_tp_order(symbol, tf, fc, longshort, tp_price, quantity, et_orderId_p, t_mode, o_his):
    params = [
        {
            "symbol": symbol,
            "side": "SELL" if longshort else "BUY",
            "type": "LIMIT",
            "positionSide": "LONG" if longshort else "SHORT",
            "price": str(tp_price),
            "quantity": str(quantity),
            "priceProtect": "TRUE",
            "timeInForce": "GTC",
            "postOnly": "TRUE",
            "workingType": "CONTRACT_PRICE",
            "newClientOrderId": 'tp_' + str(tf) + '_' + str(fc) + '_' + str(et_orderId_p)
        }
    ]
    if t_mode == 'REAL':
        result_tp = api_call('new_batch_order', [params])
        if result_tp:
            order_tp = result_tp[0]
            o_his = add_tp_history(o_his, symbol, et_orderId_p, order_tp['orderId'], order_tp)
            return True, o_his
    elif t_mode in ['BACKTEST', 'PAPER']:
        order_tp_bt = {
            "symbol": symbol,
            "side": "SELL" if longshort else "BUY",
            "type": "LIMIT",
            "positionSide": "LONG" if longshort else "SHORT",
            "price": str(tp_price),
            "quantity": str(quantity),
            "priceProtect": "TRUE",
            "timeInForce": "GTC",
            "postOnly": "TRUE",
            "workingType": "CONTRACT_PRICE",
            "newClientOrderId": 'tp_' + str(tf) + '_' + str(fc) + '_' + str(et_orderId_p)
        }
        o_his = add_tp_history(o_his, symbol, et_orderId_p, randrange(10000000000, 99999999999, 1), order_tp_bt)
        return True, o_his

    return False, o_his


def cancel_batch_order(symbol, order_id_l, desc):
    orderIdList = order_id_l
    origClientOrderIdList = []
    response = api_call('cancel_batch_order', [symbol, orderIdList, origClientOrderIdList])
    cnt_success = 0
    if response:
        for rs in response:
            try:
                if rs['orderId']:
                    # logger.info(symbol + (' _CANCELBATCH success, %s : ' % desc) + str(rs))
                    cnt_success += 1
            except:
                if rs['code']:
                    logger.info(symbol + (' _CANCEBATCH error, %s : ' % desc) + str(rs))
        if cnt_success == len(response):
            return True
    return False


def c_in_plrate_adaptive(o_his, symbol, longshort, w, t_mode):
    try:
        et_price, sl_price, tp_price, tp_price_w5 = get_trade_prices(o_his, symbol, longshort, w, t_mode)
        b_symbol = abs(tp_price - et_price) / abs(sl_price - et_price)  # one trade profit/lose rate
        if c_plrate_adaptive:
            if b_symbol > c_plrate_rate or c_plrate_rate_min > b_symbol:
                return False
    except Exception as e:
        logger.error(symbol + ' ' + str(e))
    return True


def c_in_no_double_ordering(o_his, symbol, longshort, tf, fc, w, t_mode):
    #####  이중 new limit order 방지 로직 start #####
    history_new = [x for x in o_his if
                   (x['symbol'] == symbol
                    and x['trade_mode'] == t_mode
                    and x['status'] == 'ETSL'
                    and x['timeframe'] == tf and x['fcnt'] == fc)]

    et_price, sl_price, tp_price, tp_price_w5 = get_trade_prices(o_his, symbol, longshort, w, t_mode)

    if len(history_new) > 0:
        for history in history_new:
            h_longshort = history['longshort']
            h_tf = history['timeframe']
            h_fc = history['fcnt']
            h_et_orderId = history['et_orderId']
            if h_et_orderId:
                r_query_limit = api_call('query_order', [symbol, h_et_orderId])
                if r_query_limit['status'] == 'FILLED':
                    # 대상외
                    return False

                if float(r_query_limit['price']) == float(et_price) \
                        and float(r_query_limit['clientOrderId'].split('_')[2]) == float(sl_price) \
                        and float(r_query_limit['clientOrderId'].split('_')[3]) == float(tp_price):
                    # print('c_in_no_double_ordering:' + str([symbol, longshort, tf, fc, et_price, sl_price, tp_price, tp_price_w5]))
                    # logger.info('c_in_no_double_ordering:' + str([symbol, longshort, tf, fc, et_price, sl_price, tp_price, tp_price_w5]))
                    return False
    #####  이중 new limit order 방지 로직 start #####
    return True


def c_real_condition_by_fractal_index(df, fcnt, w, idx):  # real condititon by fractal index
    notreal = True
    try:
        real_condititon1 = True if ((2 * fcnt + 1) / 2 > (w.idx_end - w.idx_start)) and w.idx_start == idx else False
        real_condititon2 = True if df.iloc[idx + int((2 * fcnt + 1) / 2), 0] > (w.dates[-1]) else False
        if not (real_condititon1 and real_condititon2):
            return notreal
    except Exception as e:
        logger.error('c_real_condition_by_fractal_index:%s' % str(e))
        return notreal
    return not notreal


def c_active_no_empty(df, w):  # 거의 영향을 안줌
    df_active = df.loc[df['Date'] > w.dates[-1]]  # 2023.3.13 after liqu  # df[w.idx_end + 1:]
    if df_active.empty:
        return False
    return True


def c_active_next_bean_ok(df, o_his, symbol, longshort, w):  ##### 거래건수 차이가 많이 남 없으면, 많아짐
    et_price, sl_price, tp_price_w5 = get_wave_prices(symbol, longshort, w)
    df_active_next = df[w.idx_end + 1: w.idx_end + 2]  # 웨이브가 끝나고 하나의 봉을 더 보고 그 다음부터 거래가 가능
    if not df_active_next.empty:
        df_active_next_high = df_active_next['High'].iat[0]
        df_active_next_low = df_active_next['Low'].iat[0]

        c_next_ohlc_beyond = df_active_next_low < et_price if longshort else df_active_next_high > et_price
        if c_next_ohlc_beyond:
            return False
    return True


def c_active_in_time(df, w):
    df_active = df.loc[df['Date'] > w.dates[-1]]  # 2023.3.13 after liqu  # df[w.idx_end + 1:]
    w_idx_width = w.idx_end - w.idx_start
    s = df_active.size
    c_beyond_idx_width = True if (s / w_idx_width >= c_time_beyond_rate) else False
    if c_beyond_idx_width and c_time_beyond_flg:
        return False
    return True


def c_current_price_in_zone(symbol, longshort, w):
    w0 = w.values[0]
    w1 = w.values[1]
    w2 = w.values[3]
    w3 = w.values[5]
    w4 = w.values[7]
    w5 = w.values[9]

    tp_price = w5
    et_price = w4
    sl_price = w0
    between_entry_target = et_price + abs(tp_price - et_price) * (et_zone_rate) if longshort else et_price - abs(
        tp_price - et_price) * (et_zone_rate)
    c_price = float(api_call('ticker_price', [symbol])['price'])
    c_price_in_zone = (c_price > sl_price and c_price < between_entry_target) \
        if longshort else \
        (c_price < sl_price and c_price > between_entry_target)

    if not c_price_in_zone:
        return False
    return True


def c_current_price_in_zone_by_prices(symbol, longshort, et_price, tp_price):
    between_entry_target = et_price + abs(tp_price - et_price) * (et_zone_rate) if longshort else et_price - abs(tp_price - et_price) * (et_zone_rate)
    c_price = float(api_call('ticker_price', [symbol])['price'])
    c_price_in_zone = (c_price > et_price and c_price < between_entry_target) \
        if longshort else \
        (c_price < et_price and c_price > between_entry_target)

    if not c_price_in_zone:
        return False
    return True


def c_current_price_in_zone_by_prices_k_close(symbol, longshort, et_price, tp_price, c_price):
    between_entry_target = et_price + abs(tp_price - et_price) * (et_zone_rate) if longshort else et_price - abs(tp_price - et_price) * (et_zone_rate)
    c_price_in_zone = (c_price > et_price and c_price < between_entry_target) \
        if longshort else \
        (c_price < et_price and c_price > between_entry_target)
    if not c_price_in_zone:
        return False
    return True


def c_active_in_zone(df, longshort, w, et_price, sl_price, tp_price, tp_price_w5, price_base):
    w_start_price = w.values[0]  # wave1
    w2_price = w.values[3]
    w_end_price = w.values[-1]  # wave5
    height_price = abs(w_end_price - w_start_price)
    o_fibo_value = height_price * o_fibo / 100 if o_fibo else 0
    df_active = df.loc[df['Date'] > w.dates[-1]]  # 2023.3.13 after liqu  # df[w.idx_end + 1:]
    if not df_active.empty:
        try:
            active_max_value = max(df_active.High.tolist(), default=tp_price_w5)
            active_min_value = min(df_active.Low.tolist(), default=tp_price_w5)
        except Exception as e:
            logger.error('active_max_value:' + str(e))

        if price_base == 'ET_PRICE':
            c_active_min_max_in_zone = (active_min_value > et_price and active_max_value < (w_end_price + o_fibo_value)) \
                                    if longshort else \
                                (active_max_value < et_price and active_min_value > (w_end_price - o_fibo_value))
        elif price_base == 'W2_PRICE':
            c_active_min_max_in_zone = (active_min_value > w2_price and active_max_value < (w_end_price + o_fibo_value)) \
                                    if longshort else \
                                (active_max_value < w2_price and active_min_value > (w_end_price - o_fibo_value))
        elif price_base == 'SL_PRICE':
            c_active_min_max_in_zone = (active_min_value > sl_price and active_max_value < (tp_price_w5 + o_fibo_value)) \
                if longshort else \
                (active_max_value < sl_price and active_min_value > (tp_price_w5 - o_fibo_value))
        if not c_active_min_max_in_zone:
            return False
    return True


def c_in_no_risk(o_his, symbol, longshort, w, t_info, qtyrate, t_mode):
    qtyrate_k = get_qtyrate_k(t_info, qtyrate)
    et_price, sl_price, tp_price, tp_price_w5 = get_trade_prices(o_his, symbol, longshort, w, t_mode)

    if c_risk_beyond_flg:
        pnl_percent_sl = (abs(et_price - sl_price) / et_price) * qtyrate_k
        if pnl_percent_sl >= c_risk_beyond_max:  # decrease max sl rate   0.1 = 10%
            # logger.info(symbol + ' _c_risk_beyond_max : ' + str(pnl_percent_sl))
            return False

        pnl_percent_tp = (abs(tp_price - et_price) / et_price) * qtyrate_k
        if pnl_percent_tp <= c_risk_beyond_min:  # reduce low tp rate  0.005 = 0.5%
            # logger.info(symbol + ' _c_risk_beyond_min : ' + str(pnl_percent_tp))
            return False

    if et_price == sl_price:
        logger.info(symbol + ' _et_price == sl_price')
        return False
    return True


def check_cons_for_new_etsl_order(o_his, df, symbol, tf, fc, longshort, w, et_price, sl_price, tp_price, tp_price_w5, ix, t_mode, t_info, qtyrate):
    try:
        if not c_in_plrate_adaptive(o_his, symbol, longshort, w, t_mode):
            return False
        # if not c_real_condition_by_fractal_index(df, fc, w, ix):
        #     return False
        if not c_active_no_empty(df, w):
            return False
        # if not c_active_next_bean_ok(df, o_his, symbol, longshort, w): # 미적용하는게 휠씬 이익이 극대화 된다.
        #     return False
        if not c_in_no_risk(o_his, symbol, longshort, w, t_info, qtyrate, t_mode):
            return False
        if t_mode in ['REAL']:
            # if not c_active_in_time(df, w):
            #     return False
            if not c_in_no_double_ordering(o_his, symbol, longshort, tf, fc, w, t_mode):
                return False
            if not c_active_in_zone(df, longshort, w, et_price, sl_price, tp_price, tp_price_w5, 'W2_PRICE'):
                return False
            if not c_current_price_in_zone(symbol, longshort, w):
                return False
    except Exception as e:
        print('check_cons_for_new_etsl_order e: %s ' % str(e))
    return True


def my_available_balance(s, exchange_symbol):
    response = api_call('balance', [])
    if response:
        if exchange_symbol == 'binance_usdt_busd_perp':
            if s[-4:] == 'USDT':
                my_marginavailable_l = [x['marginAvailable'] for x in response if x['asset'] == 'USDT']
                my_marginbalance_l = [x['availableBalance'] for x in response if x['asset'] == 'USDT']
                my_walletbalance_l = [x['balance'] for x in response if x['asset'] == 'USDT']
                if len(my_marginbalance_l) == 1:
                    return my_marginavailable_l[0], float(my_marginbalance_l[0]), float(my_walletbalance_l[0])
            if s[-4:] == 'BUSD':
                my_marginavailable_l = [x['marginAvailable'] for x in response if x['asset'] == 'BUSD']
                my_marginbalance_l = [x['availableBalance'] for x in response if x['asset'] == 'BUSD']
                my_walletbalance_l = [x['balance'] for x in response if x['asset'] == 'BUSD']
                if len(my_marginbalance_l) == 1:
                    return my_marginavailable_l[0], float(my_marginbalance_l[0]), float(my_walletbalance_l[0])
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
    return False, 0, 0


def c_balance_and_calc_quanty(symbol):
    margin_available, available_balance, wallet_balance = my_available_balance(symbol, exchange_symbol)
    if not margin_available:
        logger.info('margin_available : False')
        logger.info('symbol:%s, available_balance:%s, wallet_balance:%s' % (
            symbol, str(available_balance), str(wallet_balance)))
        return False, None

    # max_quantity = float(available_balance) * int(leveragexxxxx) / c_price
    # quantity = max_quantity * qtyrate
    c_price = float(api_call('ticker_price', [symbol])['price'])
    quantity = wallet_balance * qtyrate / c_price
    step_size, minqty = get_quantity_step_size_minqty(symbol)
    quantity = format_value(quantity, step_size)

    if available_balance <= wallet_balance * walletrate:
        logger.info('available_balance <= wallet_balance * %s' % str(walletrate))
        logger.info('symbol:%s, available_balance:%s, wallet_balance:%s' % (
            symbol, str(available_balance), str(wallet_balance)))
        return False, None

    if float(quantity) < float(minqty):
        logger.info('float(quantity) < float(minqty)')
        logger.info('symbol:%s, quantity:%s, minqty:%s' % (symbol, str(quantity), str(minqty)))
        logger.info('symbol:%s, available_balance:%s, wallet_balance:%s' % (
            symbol, str(available_balance), str(wallet_balance)))
        return False, None

    if not quantity:
        logger.info('quantity:' + str(quantity))
        logger.info('available_balance:%s, wallet_balance:%s' % (str(available_balance), str(wallet_balance)))
        return False, None
    return True, quantity


def c_check_valid_wave_in_history(o_his, symbol, tf, fc, wavepattern, et_price, sl_price, tp_price, tp_price_w5, t_mode):
    try:
        if o_his:
            order_filter = [x for x in o_his
                            if
                            x['trade_mode'] == t_mode
                            and x['symbol'] == symbol
                            # and x['timeframe'] == tf
                            # and x['fcnt'] == fc
                            and x['status'] in ['ETSL', 'TP', 'WIN', 'LOSE', 'FORCE']
                            # and x['et_price'] == et_price
                            and x['sl_price'] == sl_price
                            # and x['tp_price'] == tp_price
                            # and x['tp_price_w5'] == tp_price_w5
                            # and x['wavepattern'].dates == wavepattern.dates
                            # and x['wavepattern'].values == wavepattern.values
                            ]
            if order_filter:
                return False
    except Exception as e:
        logger.error(symbol + ' c_check_valid_wave_in_history' + str(e))
    return True


# def c_check_valid_etslwave_in_history(o_his, symbol, tf, wavepattern):
#     if o_his:
#         open_order_this_symbol = [x for x in o_his
#                                     if x['symbol'] == symbol and x['timeframe'] == tf
#                                     and x['status'] in ['ETSL']
#                                     and x['wavepattern'].dates == wavepattern.dates
#                                     and x['wavepattern'].values == wavepattern.values
#                                   ]
#         if open_order_this_symbol:
#             for history in open_order_this_symbol:
#                 h_et_orderId = history['et_orderId']
#                 r_query_et = api_call('query_order', [symbol, h_et_orderId])
#                 if r_query_et:
#                     r_query_limit = r_query_et[0]
#                     if r_query_limit['status'] == 'NEW' or r_query_limit['status'] == 'FILLED':
#                         return False
#     return True


def c_compare_before_fractal(df_lows, c_compare_before_fractal, c_compare_before_fractal_shift):
    if c_compare_before_fractal:
        if not df_lows.empty:
            for i in range(c_compare_before_fractal_shift, 0, -1):
                try:
                    if c_compare_before_fractal_strait:
                        i = c_compare_before_fractal_shift
                    df_lows['Low_before'] = df_lows.Low.shift(i).fillna(0)
                    if c_compare_before_fractal_mode == 1:
                        df_lows['compare_flg'] = df_lows.apply(lambda x: 1 if x['Low'] > x['Low_before'] else 0, axis=1)
                    elif c_compare_before_fractal_mode == 2:
                        df_lows['compare_flg'] = df_lows.apply(
                            lambda x: 1 if x['Low'] >= x['Low_before'] else 0, axis=1)
                    elif c_compare_before_fractal_mode == 3:
                        df_lows['compare_flg'] = df_lows.apply(lambda x: 1 if x['Low'] < x['Low_before'] else 0, axis=1)
                    elif c_compare_before_fractal_mode == 4:
                        df_lows['compare_flg'] = df_lows.apply(
                            lambda x: 1 if x['Low'] <= x['Low_before'] else 0, axis=1)
                    elif c_compare_before_fractal_mode == 5:
                        df_lows['compare_flg'] = df_lows.apply(
                            lambda x: 1 if x['Low'] == x['Low_before'] else 0, axis=1)
                    df_lows = df_lows.drop(df_lows[df_lows['compare_flg'] == 0].index)

                    if not df_lows.empty:
                        del df_lows['Low_before']
                        del df_lows['compare_flg']
                        pass
                except:
                    pass
    return df_lows


def c_allowed_intersect_df(df_active, line_price, cross_cnt):
    try:
        cross_count = 0
        if intersect_idx:
            # df_active = df[w.idx_end:]  # w5 stick 포함 or --> maybe here, right -> No포함 (df[w.idx_end + 1:])
            if not df_active.empty and df_active.size != 0:
                df_active['cross'] = df_active.apply(lambda x: 1 if x['High'] >= line_price and line_price >= x['Low'] else 0,
                                                     axis=1)
                cross_count = df_active['cross'].sum(axis=0)

                if cross_count > cross_cnt:
                    return False, cross_count
    except Exception as e:
        logger.error('c_allowed_intersect_df e:' + str(e))
    return True, cross_count


def get_cross_count_intersect_df(df_active, line_price):
    try:
        cross_count = 0
        if intersect_idx:
            # df_active = df[w.idx_end:]  # w5 stick 포함 or --> maybe here, right -> No포함 (df[w.idx_end + 1:])
            if not df_active.empty and df_active.size != 0:
                df_active['cross'] = df_active.apply(lambda x: 1 if x['High'] >= line_price and line_price >= x['Low'] else 0,
                                                     axis=1)
                cross_count = df_active['cross'].sum(axis=0)

    except Exception as e:
        logger.error('c_allowed_intersect_df e:' + str(e))
    return cross_count

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


@synchronized
def get_historical_ohlc_data_start_end(symbol, start_int, end_int, past_days=None, interval=None, futures=False, sma_n=0):
    D = None
    start_date_str = None
    end_date_str = None
    try:
        # start_date_str = str((pd.to_datetime('today') - pd.Timedelta(str(start_int) + ' days')).date())
        if sma_n:
            min_trans = int(interval.replace('m', '')) * sma_n
            # start_date_str = str((pd.to_datetime('today') - pd.Timedelta(str(start_int) + ' days')).timestamp())
            start_date_str = str((pd.to_datetime('today') - pd.Timedelta(str(start_int) + ' days') - pd.Timedelta(str(min_trans) + ' minutes')).timestamp())
        else:
            start_date_str = str((pd.to_datetime('today') - pd.Timedelta(str(start_int) + ' days')).timestamp())

        if end_int is not None and end_int > 0:
            end_date_str = str((pd.to_datetime('today') - pd.Timedelta(str(end_int) + ' days')).timestamp())
        else:
            end_date_str = None
        try:
            if exchange_symbol == 'binance_usdt_perp' or exchange_symbol == 'binance_busd_perp' or exchange_symbol == 'binance_usdt_busd_perp':
                if futures:
                    D = pd.DataFrame(
                        client.futures_historical_klines(symbol=symbol, start_str=start_date_str, end_str=end_date_str,
                                                         interval=interval))
                else:
                    D = pd.DataFrame(
                        client.get_historical_klines(symbol=symbol, start_str=start_date_str, end_str=end_date_str,
                                                     interval=interval))

        except Exception as e:
            time.sleep(0.1)
            logger.info(symbol + ' in futures_historical_klines e :' + str(e))
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
        time.sleep(0.1)
        logger.info(symbol + ' in get_historical_ohlc_data_start_end e :' + str(e))
    return D, start_date_str, end_date_str


def get_sma_n_df(df, n):
    sma_n = sma_df(df, n)
    sma_n['sma_-1'] = sma_n.sma.shift(-1).fillna(0)
    sma_n['sma_gradient'] = sma_n.apply(lambda x: 1 if x['sma'] > x['sma_-1'] else 0, axis=1)
    df = sma_n[n - 1:]
    df.reset_index(drop=True, inplace=True)
    return df


def c_ichi(df, longshort, wavepattern):
    w2 = wavepattern.waves['wave2']
    w2_price = w2.high
    w2_high_idx = w2.high_idx
    senkou_span_a = df['senkou_span_a'].iat[w2_high_idx]
    senkou_span_b = df['senkou_span_b'].iat[w2_high_idx]

    c_ichi = w2_price > senkou_span_a and w2_price > senkou_span_b if longshort else w2_price < senkou_span_a and w2_price < senkou_span_b
    if c_ichi:
        return True
    return False


def get_waves(symbol, tf, t_info, t_mode, i=None):
    n = c_sma_n
    if t_mode in ['REAL', 'PAPER']:
        if n:
            delta = (4 * fcnt[-1] + 1) + n
        else:
            delta = (4 * fcnt[-1] + 1)
        df = get_fetch_dohlcv(symbol,
                              interval=tf,
                              limit=delta)
    elif t_mode in ['BACKTEST']:
        start_int = i
        end_int = start_int - period_interval

        if start_int == 0 or end_int < 0:
            end_int = None

        if end_int is not None and start_int <= end_int or start_int == 0:
            return t_info

        df, start_date, end_date = get_historical_ohlc_data_start_end(symbol,
                                                                      start_int=start_int,
                                                                      end_int=end_int,
                                                                      past_days=None,
                                                                      interval=tf,
                                                                      futures=futures,
                                                                      sma_n=n
                                                                      )
    if df is not None and df.empty:
        return
    if df is None:
        return

    if n:
        df = get_sma_n_df(df, n)

    df = ichi_df(df)

    try:
        wa = WaveAnalyzer(df=df, verbose=True)
    except:
        return

    wave_options = WaveOptionsGenerator5(up_to=up_to_count)
    idxs = list()
    lows_idxs = list()
    highs_idxs = list()
    df_lows_plot = None
    df_highs_plot = None
    wave_list = list()
    for fc in fcnt:
        if 'long' in type:
            df_lows = fractals_low_loopA(df, fcnt=fc, loop_count=loop_count)
            df_lows_plot = df_lows[['Date', 'Low']]
            df_lows = c_compare_before_fractal(df_lows, c_compare_before_fractal,
                                               c_compare_before_fractal_shift)


            if n:
                df_lows = df_lows[df_lows['sma_gradient'] == 1]

            impulse = Impulse('impulse')
            lows_idxs = df_lows.index.tolist()
            idxs = lows_idxs

        if 'short' in type:
            df_highs = fractals_high_loopA(df, fcnt=fc, loop_count=loop_count)
            df_highs_plot = df_highs[['Date', 'High']]
            df_highs = c_compare_before_fractal(df_highs, c_compare_before_fractal,
                                               c_compare_before_fractal_shift)

            if n:
                df_highs = df_highs[df_highs['sma_gradient'] == 0]

            downimpulse = DownImpulse('downimpulse')
            highs_idxs = df_highs.index.tolist()
            idxs = highs_idxs

        rules_to_check = list()
        wavepatterns = set()

        if ('long' in type) and ('short' in type): idxs = sorted(list(set(lows_idxs) | set(highs_idxs)))

        if idxs:
            for ix in idxs:
                longshort = None
                for wave_opt in wave_options.options_sorted:
                    if ix in lows_idxs:
                        wave = wa.find_impulsive_wave(idx_start=ix, wave_config=wave_opt.values)
                        longshort = True
                        rules_to_check = [impulse]
                    elif ix in highs_idxs:
                        wave = wa.find_downimpulsive_wave(idx_start=ix, wave_config=wave_opt.values)
                        longshort = False
                        rules_to_check = [downimpulse]

                    if wave:
                        wavepattern = WavePattern(wave, verbose=True)
                        if (wavepattern.idx_end - wavepattern.idx_start + 1) >= fc / 2:
                            for rule in rules_to_check:
                                if wavepattern.check_rule(rule):
                                    if wavepattern in wavepatterns:
                                        continue
                                    else:
                                        if c_ichi(df, longshort, wavepattern):
                                            wave_list.append(
                                                [symbol, df, tf, fc, longshort, wavepattern, ix, wave_opt, df_lows_plot,
                                                 df_highs_plot])
                                        wavepatterns.add(wavepattern)

    return wave_list


def real_trade(symbol, tf, fc, longshort, et_price, sl_price, tp_price, tp_price_w5, w, t_mode, o_his, df_lows_plot, df_highs_plot, df, t_info):
    available, quantity = c_balance_and_calc_quanty(symbol)
    if available:
        df_active_with_w5_stick = df[w.idx_end:]
        c_interset, cross_cnt = c_allowed_intersect_df(df_active_with_w5_stick, et_price, 1)  # w5 stick 포함하여 계산, 1번 초과 제외
        if not c_interset:
            return t_info, o_his
        if cross_cnt <= 1:
            # check current price
            c_price = float(api_call('ticker_price', [symbol])['price'])
            w2_price = w.values[3]
            c_c_price_position = c_price < et_price and c_price > w2_price if longshort else c_price > et_price and c_price < w2_price
            if not c_c_price_position:
                return t_info, o_his
        try:
            qtyrate_k = get_qtyrate_k(t_info, qtyrate)
            r_order, o_his = new_et_order_real(symbol, tf, fc, longshort, qtyrate_k, et_price, sl_price, tp_price, tp_price_w5, quantity, w, o_his, 'REAL')
            if r_order:
                if plotview:
                    try:
                        plot_pattern_n(df=df,
                                       wave_pattern=[[1, w.dates[0], id(w), w]],
                                       df_lows_plot=df_lows_plot, df_highs_plot=df_highs_plot,
                                       trade_info=None,
                                       title=str(t_mode + ' ' + symbol + ' %s ' % str('LONG' if longshort else 'SHORT') + '%sm %s' % (
                                           tf, fc) + ', ET: ' + str(et_price)))
                    except Exception as e:
                        logger.error(symbol + ' ' + str(e))
            pass
        except Exception as e:
            logger.error('real_trade: %s' % str(e))
    return t_info, o_his


def real_trade_market_T(symbol, tf, fc, longshort, et_price, sl_price, tp_price, tp_price_w5, w, t_mode, o_his, df_lows_plot, df_highs_plot, wavepatterns, wavepattern_tpsl_l, wave_option_plot_l, wave_opt, ix, df, t_info):
    wavepatterns.add(w)
    # wavepattern_l.append([symbol, fc, ix, w.dates[0], id(w), w])
    wave_option_plot_l.append([
        [str(w.dates[-1])],
        [w.values[-1]],
        [str(wave_opt.values)]
    ])

    w = w
    t = t_info
    stats_history = t[0]
    order_history = t[1]
    asset_history = t[2]
    trade_count = t[3]
    fee_history = t[4]
    pnl_history = t[5]
    wavepattern_history = t[6]

    w_start_price = w.values[0]  # wave1
    w_end_price = w.values[-1]  # wave5

    out_price = None

    qtyrate_k = get_qtyrate_k(t_info, qtyrate)

    df_active = df[w.idx_end:]  # w5 stick 포함해서 전체 active
    closes = df_active.Close.tolist()

    if closes:
        position = False
        c_stoploss = False
        c_profit = False
        c_price_et_price = None
        for k, close in enumerate(closes):
            try:
                df_k = df[:w.idx_end + (k + 1)]
                df_active_k = df_active[:(k + 1)]  # k = 0 번째까지 active

                dates = df_active_k.Date.tolist()
                closes = df_active_k.Close.tolist()
                highs = df_active_k.High.tolist()
                lows = df_active_k.Low.tolist()

            except Exception as e:
                print('here e: %s' % str(e))

            if not position:
                df_active_with_w5_stick = df_active_k[w.idx_end:]
                cross_cnt = get_cross_count_intersect_df(df_active_with_w5_stick, et_price)  # w5 stick 포함하여 계산, 1번 초과 제외
                if cross_cnt >= 2:
                    return t_info, o_his
                if cross_cnt == 1:
                    # check current price
                    c_price = float(api_call('ticker_price', [symbol])['price'])
                    w2_price = w.values[3]
                    c_c_price_position = c_price < et_price and c_price > w2_price if longshort else c_price > et_price and c_price < w2_price
                    if not c_c_price_position:
                        continue
                        # return t_info, o_his
                    else:
                        position = True
                        available, quantity = c_balance_and_calc_quanty(symbol)
                        if available:
                            try:
                                qtyrate_k = get_qtyrate_k(t_info, qtyrate)
                                r_order, o_his = new_et_order_real(symbol, tf, fc, longshort, qtyrate_k, et_price,
                                                                   sl_price, tp_price, tp_price_w5, quantity, w, o_his,
                                                                   'REAL')
                                if r_order:
                                    if plotview:
                                        try:
                                            plot_pattern_n(df=df,
                                                           wave_pattern=[[1, w.dates[0], id(w), w]],
                                                           df_lows_plot=df_lows_plot, df_highs_plot=df_highs_plot,
                                                           trade_info=None,
                                                           title=str(t_mode + ' ' + symbol + ' %s ' % str(
                                                               'LONG' if longshort else 'SHORT') + '%sm %s' % (
                                                                         tf, fc) + ', ET: ' + str(et_price)))
                                        except Exception as e:
                                            logger.error(symbol + ' ' + str(e))
                                pass
                            except Exception as e:
                                logger.error('real_trade: %s' % str(e))
                        break
    return t_info, o_his


def f_trade_status(pre_status, sl_c, et_c, tp_c, out_c):
    if pre_status == 0 and sl_c == 0 and et_c == 0 and tp_c == 0 and out_c == 0:  # STANDBY
        return 0
    elif pre_status == 0 and sl_c == 0 and et_c == 0 and out_c == 1:  # OUT
        return -1
    elif pre_status in [-1, 1, 2, 3]:  # DONE
        return -1
    elif pre_status == 0 and sl_c == 0 and et_c == 1 and tp_c == 0 and out_c == 0:  # ETSL
        return 1
    elif pre_status == 1 and sl_c == 1:  # LOSE et - sl
        return 2
    elif pre_status == 0 and sl_c == 1 and et_c == 1:  # LOSE direct et - sl
        return 2
    elif pre_status == 1 and sl_c == 0 and tp_c == 1:  # WIN et - tp
        return 3
    else:
        return np.nan


# https://www.backtrader.com/docu/order/
# Order.Created, order.Submitted, Order.Partial, order.Accepted,
# order.Completed, order.Cancelled, order.Margin, Order.Rejected, Order.Expired
def f_order_status(sl_c, et_c, tp_c, out_c, sub_c, can_c):
    status_l = list()
    if sub_c == 1:
        status_l.append('Submitted')
    if can_c == 1:
        status_l.append('Cancelled')
    if et_c == 1:
        status_l.append('ET')
    if tp_c == 1:
        status_l.append('WIN')
    if sl_c == 1:
        status_l.append('LOSE')
    if out_c == 1:
        status_l.append('Rejected')
    return status_l


def real_test_trade_df_apply(symbol, tf, fc, longshort, et_price, sl_price, tp_price, tp_price_w5, w, t_mode, o_his, df_lows_plot, df_highs_plot, wavepatterns, wavepattern_tpsl_l, wave_option_plot_l, wave_opt, ix, df, t_info):
    wavepatterns.add(w)
    # wavepattern_l.append([symbol, fc, ix, w.dates[0], id(w), w])
    wave_option_plot_l.append([
        [str(w.dates[-1])],
        [w.values[-1]],
        [str(wave_opt.values)]
    ])

    w = w
    t = t_info
    stats_history = t[0]
    order_history = t[1]
    asset_history = t[2]
    trade_count = t[3]
    fee_history = t[4]
    pnl_history = t[5]
    wavepattern_history = t[6]

    w_start_price = w.values[0]  # wave1
    w_end_price = w.values[-1]  # wave5

    out_price = None

    qtyrate_k = get_qtyrate_k(t_info, qtyrate)

    # df_raw = df
    # df = df_raw.dropna()

    # if w.idx_start < df.index[0]:
    #     return t_info, o_his

    df_active_raw = df[w.idx_end:]  # w5 stick 포함해서 전체 active
    df_active = df_active_raw.dropna()
    df_active = df_active_raw

    position_enter_i = []
    et_orderid_test = randrange(10000000000, 99999999999, 1)


    position = False
    c_profit = False
    c_stoploss = False

    available, quantity = c_balance_and_calc_quanty(symbol)
    if available:
        df_active_with_w5_stick = df[w.idx_end:]
        c_interset, cross_cnt = c_allowed_intersect_df(df_active_with_w5_stick, et_price, 1)  # w5 stick 포함하여 계산, 1번 초과 제외
        if not c_interset:
            return t_info, o_his
        if cross_cnt <= 1:
            # check current price
            c_price = float(api_call('ticker_price', [symbol])['price'])
            w2_price = w.values[3]
            c_c_price_position = c_price < et_price and c_price > w2_price if longshort else c_price > et_price and c_price < w2_price
            if not c_c_price_position:
                return t_info, o_his

    if not df_active.empty and df_active.size != 0:
        w_start_price = w.values[0]  # wave1
        w2_price = w.values[3]
        w_end_price = w.values[-1]  # wave5
        height_price = abs(w_end_price - w_start_price)
        o_fibo_value = height_price * o_fibo / 100 if o_fibo else 0


        out_price = w_end_price + o_fibo_value if longshort else w_end_price - o_fibo_value
        submit_price = et_price + abs(tp_price - et_price)/2 if longshort else et_price - abs(tp_price - et_price)/2
        cancel_price = tp_price

        df_active['sl_price'] = sl_price
        df_active['et_price'] = et_price
        df_active['tp_price'] = tp_price
        df_active['out_price'] = out_price
        df_active['sub_price'] = submit_price
        df_active['can_price'] = cancel_price

        df_active['sl_cross'] = df_active.apply(lambda x: 1 if x['High'] >= sl_price and sl_price >= x['Low'] else 0, axis=1)
        df_active['et_cross'] = df_active.apply(lambda x: 1 if x['High'] >= et_price and et_price >= x['Low'] else 0, axis=1)
        df_active['tp_cross'] = df_active.apply(lambda x: 1 if x['High'] >= tp_price and tp_price >= x['Low'] else 0, axis=1)
        df_active['out_cross'] = df_active.apply(lambda x: 1 if x['High'] >= out_price and out_price >= x['Low'] else 0, axis=1)
        df_active['sub_cross'] = df_active.apply(lambda x: 1 if x['High'] >= submit_price and submit_price >= x['Low'] else 0, axis=1)
        df_active['can_cross'] = df_active.apply(lambda x: 1 if x['High'] >= cancel_price and cancel_price >= x['Low'] else 0, axis=1)

        df_active['status_raw'] = df_active.apply(lambda x: f_order_status(x['sl_cross'], x['et_cross'], x['tp_cross'], x['out_cross'], x['sub_cross'], x['can_cross']), axis=1)
        status_raw_list = df_active['status_raw'].tolist()
        close_list = df_active['Close'].tolist()
        et_price_list = df_active['et_price'].tolist()
        can_price_list = df_active['can_price'].tolist()

        current_action = list()
        pre_action_list = list()
        for i, status in enumerate(status_raw_list):
            # pre_action_list = list(dict.fromkeys(pre_action_list)) #  순서유지
            if status:
                if len(pre_action_list) == 0 and len(status) == 3 \
                        and ('Submitted' in status) and ('Cancelled' in status) and ('WIN' in status):
                    c_submitted = close_list[i] < can_price_list[i] if longshort else close_list[i] > can_price_list[i]
                    if c_submitted:
                        pre_action_list = ['Submitted']
                        current_action.append('Submitted')
                    else:
                        current_action.append(np.nan)
                elif len(pre_action_list) == 0 and ('Rejected' in status):
                    current_action.append('Rejected')
                    for k in status_raw_list[i+1:]:
                        current_action.append(np.nan)
                    break
                elif len(pre_action_list) == 0 and len(status) == 1 and ('Cancelled' in status):
                    pre_action_list = []
                    current_action.append(np.nan)
                elif len(pre_action_list) == 0 and len(status) == 2 and ('Cancelled' in status):
                    pre_action_list = []
                    current_action.append(np.nan)
                elif len(pre_action_list) == 0 and len(status) == 2 and ('Submitted' in status) and ('ET' in status):
                    pre_action_list = ['Submitted', 'ET']
                    current_action.append('ET')
                elif len(pre_action_list) == 0 and len(status) == 1 and ('Submitted' in status):
                    pre_action_list = ['Submitted']
                    current_action.append('Submitted')
                elif len(pre_action_list) == 1 and len(status) == 2 and ('Submitted' in pre_action_list) and ('Cancelled' in status) and ('WIN' in status):
                    pre_action_list = []
                    current_action.append(np.nan)
                elif len(pre_action_list) == 1 and ('ET' in status):
                    pre_action_list = ['Submitted', 'ET']
                    current_action.append('ET')
                elif len(pre_action_list) == 2 and ('WIN' in status):
                    pre_action_list = ['Submitted', 'ET', 'WIN']
                    current_action.append('WIN')
                    for k in status_raw_list[i+1:]:
                        current_action.append(np.nan)
                    break
                elif len(pre_action_list) == 2 and ('LOSE' in status):
                    pre_action_list = ['Submitted', 'ET', 'LOSE']
                    current_action.append('LOSE')
                    for k in status_raw_list[i+1:]:
                        current_action.append(np.nan)
                    break
                else:
                    current_action.append(np.nan)
            else:
                current_action.append(np.nan)

        df_active['action'] = current_action
        current_action_undupl = list(dict.fromkeys(current_action)) #  순서유지


        # status_raw_list_release = list()
        # for s in status_raw_list:
        #     for j in s:
        #         status_raw_list_release.append(j)
        # status_raw_list_undupl = list(dict.fromkeys(status_raw_list_release)) #  순서유지


        # 1
        # LOSE -->['Submitted', 'Cancelled', 'WIN', 'ET', 'LOSE']
        # WIN -->['Submitted', 'Cancelled', 'WIN', 'ET', 'LOSE']

        # 2
        # LOSE -->['Submitted', 'Cancelled', 'ET', 'WIN', 'LOSE']
        # WIN -->['Submitted', 'Cancelled', 'ET', 'WIN', 'LOSE']

        # if 'LOSE' in status_raw_list_undupl:
        #     print('LOSE-->' + str(status_raw_list_undupl))
        # if 'Rejected' in status_raw_list_undupl:
        #     print('Rejected-->' +str(status_raw_list_undupl))
        # if 'WIN' in status_raw_list_undupl:
        #     print('WIN-->' +str(status_raw_list_undupl))

        # if 'Submitted' in current_action_undupl:
        #     # print(str(current_action_undupl))
        #     position = True
        #     c_profit = True
        #     c_stoploss = False
        #     if plotview:
        #         plot_pattern_n(
        #             symbol + '_' + str(tf) + '_' + str(fc),
        #             df=df,
        #             w=w,
        #             # k=k,
        #             wave_pattern=wavepattern_tpsl_l,
        #             df_lows_plot=df_lows_plot,
        #             df_highs_plot=df_highs_plot,
        #             trade_info=t_info,
        #             wave_options=wave_option_plot_l,
        #             title='APPLY Submitted ' + symbol + ' ' +
        #                   str([et_price, sl_price, tp_price, tp_price_w5]))
        #     return trade_info, o_his

        if 'ET' in current_action_undupl:
            # print(str(current_action_undupl))
            position = True
            c_profit = True
            c_stoploss = False
            position_enter_i = [symbol, et_price, sl_price, tp_price, 'dates[k]']

            if plotview:
                plot_pattern_n(df=df,
                               wave_pattern=[[1, w.dates[0], id(w), w]],
                               df_lows_plot=df_lows_plot, df_highs_plot=df_highs_plot,
                               trade_info=None,
                               title=str(
                                   t_mode + ' ' + symbol + ' %s ' % str('LONG' if longshort else 'SHORT') + '%sm %s' % (
                                       tf, fc) + ', ET(APPLY): ' + str(et_price)))
            return trade_info, o_his
        if 'WIN' in current_action_undupl:
            position = True
            c_profit = True
            c_stoploss = False
            position_enter_i = [symbol, et_price, sl_price, tp_price, 'dates[k]']

            # if plotview:
            #     plot_pattern_n(
            #         symbol + '_' + str(tf) + '_' + str(fc),
            #         df=df,
            #         w=w,
            #         # k=k,
            #         wave_pattern=wavepattern_tpsl_l,
            #         df_lows_plot=df_lows_plot,
            #         df_highs_plot=df_highs_plot,
            #         trade_info=t_info,
            #         wave_options=wave_option_plot_l,
            #         title='BACKTEST TP/SL ' + str('trade_stats') + ' ' +
            #               str([et_price, sl_price, tp_price, tp_price_w5]))
        if 'LOSE' in current_action_undupl:
            position = True
            c_profit = False
            c_stoploss = True
            position_enter_i = [symbol, et_price, sl_price, tp_price, 'dates[k]']

            # if plotview:
            #     plot_pattern_n(
            #         symbol + '_' + str(tf) + '_' + str(fc),
            #         df=df,
            #         w=w,
            #         # k=k,
            #         wave_pattern=wavepattern_tpsl_l,
            #         df_lows_plot=df_lows_plot,
            #         df_highs_plot=df_highs_plot,
            #         trade_info=t_info,
            #         wave_options=wave_option_plot_l,
            #         title='BACKTEST TP/SL ' + str('trade_stats') + ' ' +
            #               str([et_price, sl_price, tp_price, tp_price_w5]))

    # if position is True:
    if False:
        position_enter_i = [symbol, et_price, sl_price, tp_price, 'dates[k]']

        if c_profit or c_stoploss:
            win_lose_flg = None
            if t_mode in ['BACKTEST']:
                r_order, o_his = new_et_order_test(symbol, tf, fc, longshort, qtyrate_k, et_price, sl_price,
                                                   tp_price, tp_price_w5, 1, w, et_orderid_test, o_his, t_mode)

            fee_limit_tp = 0
            fee_limit_sl = 0
            if tp_type == 'maker':
                fee_limit_tp = fee_limit + fee_tp
            elif tp_type == 'taker':
                fee_limit_tp = fee_limit + fee_tp + fee_slippage
            fee_limit_sl = fee_limit + fee_sl + fee_slippage

            fee_percent = 0
            pnl_percent = 0
            win_lose_flg = 0
            seed_pre = asset_history[-1] if asset_history else seed

            if c_stoploss and c_profit:
                print('XXXXXXXX XXXXXXXX XXXXXXX')

            if c_stoploss and not c_profit:
                win_lose_flg = 0

                # https://www.binance.com/en/support/faq/how-to-use-binance-futures-calculator-360036498511
                pnl_percent = -(abs(sl_price - et_price) / sl_price)         #  PnL  =  (1 - entry price / exit price)
                fee_percent = fee_limit_sl

                pnl = pnl_percent * seed_pre * qtyrate_k
                fee = fee_percent * seed_pre * qtyrate_k

                trade_count.append(0)
                trade_inout_i = [position_enter_i[0],
                                 position_enter_i[1],
                                 position_enter_i[2],
                                 position_enter_i[3],
                                 position_enter_i[4],' dates[k]',
                                 longshort, 'LOSE']
                order_history.append(trade_inout_i)
                o_his = update_history_status(o_his, symbol, et_orderid_test, 'LOSE')

            if c_profit and not c_stoploss:
                win_lose_flg = 1
                pnl_percent = (abs(tp_price - et_price) / tp_price)
                fee_percent = fee_limit_tp

                pnl = pnl_percent * seed_pre * qtyrate_k
                fee = fee_percent * seed_pre * qtyrate_k

                trade_count.append(1)
                trade_inout_i = [position_enter_i[0],
                                 position_enter_i[1],
                                 position_enter_i[2],
                                 position_enter_i[3],
                                 position_enter_i[4], 'dates[k]',
                                 longshort, 'WIN']
                order_history.append(trade_inout_i)
                o_his = update_history_status(o_his, symbol, et_orderid_test, 'WIN')


            if win_lose_flg is not None:
                asset_new = seed_pre + pnl - fee
                pnl_history.append(pnl)
                fee_history.append(fee)
                asset_history.append(asset_new)
                wavepattern_history.append(w)

                winrate = (sum(trade_count) / len(trade_count)) * 100

                asset_min = seed
                asset_max = seed

                if len(stats_history) > 0:
                    asset_last_min = stats_history[-1][-2]
                    asset_min = asset_new if asset_new < asset_last_min else asset_last_min
                    asset_last_max = stats_history[-1][-1]
                    asset_max = asset_new if asset_new > asset_last_max else asset_last_max

                df_s = pd.DataFrame.from_records(stats_history)
                b_symbol = abs(tp_price - et_price) / abs(
                    sl_price - et_price)  # one trade profitlose rate

                b_cum = (df_s[8].sum() + b_symbol) / (
                        len(df_s) + 1) if len(
                    stats_history) > 0 else b_symbol  # mean - profitlose rate, df_s[6] b_cum index
                p_cum = winrate / 100  # win rate
                q_cum = 1 - p_cum  # lose rate
                tpi_cum = round(p_cum * (1 + b_cum), 2)  # trading perfomance index
                f_cum = round(p_cum - (q_cum / b_cum),
                              2)  # kelly index https://igotit.tistory.com/1526

                f = f_cum
                tpi = tpi_cum
                b = b_cum

                if c_kelly_adaptive:
                    if len(df_s) >= c_kelly_window:
                        df_s_window = df_s.iloc[-c_kelly_window:]
                        p = df_s_window[4].sum() / len(df_s_window)
                        b = (df_s_window[11].sum() + b_symbol) / (len(
                            df_s_window) + 1)  # mean in kelly window - profitlose rate
                        q = 1 - p  # lose rate
                        tpi = round(p * (1 + b),
                                    2)  # trading perfomance index
                        f = round(p - (q / b), 2)  # kelly index

                trade_stats = [len(trade_count), round(winrate, 2),
                               asset_new, symbol, win_lose_flg,
                               'WIN' if win_lose_flg else 'LOSE', f_cum,
                               tpi_cum, b_cum, f, tpi, b, b_symbol,
                               str(qtyrate_k),
                               str(round(pnl_percent, 4)),
                               sum(pnl_history), sum(fee_history),
                               round(asset_min, 2), round(asset_max, 2)]
                stats_history.append(trade_stats)
                t_info = [stats_history, order_history,
                          asset_history, trade_count, fee_history,
                          pnl_history, wavepattern_history]

                wavepattern_tpsl_l.append(
                    [ix, w.dates[0], id(w), w])

                s_11 = symbol + '           '
                trade_in = trade_inout_i[4][2:-3]
                trade_out = trade_inout_i[5][6:-3]

                trade_stats_print = [len(trade_count), round(winrate, 2),
                               asset_new, symbol, win_lose_flg,
                               'WIN' if win_lose_flg else 'LOSE', f_cum,
                               # tpi_cum, b_cum, f, tpi, b, b_symbol,
                               str(qtyrate_k),
                               str(round(pnl_percent, 4)),
                               sum(pnl_history), sum(fee_history),
                               round(asset_min, 2), round(asset_max, 2)]
                logger.info(
                    '%s %s %s %s x%s %s-%s %s %s %s %s %s %s - %s' % (t_mode[:1] + ' :',
                        timeframe, s_11[:11], tf, qtyrate_k,
                        period_days_ago, period_days_ago_till, ' x ' + str(fc),
                        'L' if longshort else 'S', trade_in, '-',
                        trade_out, str(trade_stats_print),
                        str([et_price, sl_price, tp_price, tp_price_w5])))

                if plotview:
                    plot_pattern_k(
                        symbol + '_' + str(tf) + '_' + str(fc),
                        df=df,
                        w=w,
                        k=k,
                        wave_pattern=wavepattern_tpsl_l,
                        df_lows_plot=df_lows_plot,
                        df_highs_plot=df_highs_plot,
                        trade_info=t_info,
                        wave_options=wave_option_plot_l,
                        title='BACKTEST TP/SL ' + str(trade_stats) + ' ' +
                        str([et_price, sl_price, tp_price, tp_price_w5]))

    return t_info, o_his


def test_trade_market_df_apply(symbol, tf, fc, longshort, et_price, sl_price, tp_price, tp_price_w5, w, t_mode, o_his, df_lows_plot, df_highs_plot, wavepatterns, wavepattern_tpsl_l, wave_option_plot_l, wave_opt, ix, df, t_info):
    wavepatterns.add(w)
    # wavepattern_l.append([symbol, fc, ix, w.dates[0], id(w), w])
    wave_option_plot_l.append([
        [str(w.dates[-1])],
        [w.values[-1]],
        [str(wave_opt.values)]
    ])

    w = w
    t = t_info
    stats_history = t[0]
    order_history = t[1]
    asset_history = t[2]
    trade_count = t[3]
    fee_history = t[4]
    pnl_history = t[5]
    wavepattern_history = t[6]

    w_start_price = w.values[0]  # wave1
    w_end_price = w.values[-1]  # wave5

    out_price = None

    qtyrate_k = get_qtyrate_k(t_info, qtyrate)

    # df_raw = df
    # df = df_raw.dropna()

    # if w.idx_start < df.index[0]:
    #     return t_info, o_his

    df_active_raw = df[w.idx_end:]  # w5 stick 포함해서 전체 active
    # df_active = df_active_raw.dropna()
    df_active = df_active_raw

    position_enter_i = []
    et_orderid_test = randrange(10000000000, 99999999999, 1)


    position = False
    c_profit = False
    c_stoploss = False

    if not df_active.empty and df_active.size != 0:
        w_start_price = w.values[0]  # wave1
        w2_price = w.values[3]
        w_end_price = w.values[-1]  # wave5
        height_price = abs(w_end_price - w_start_price)
        o_fibo_value = height_price * o_fibo / 100 if o_fibo else 0


        out_price = w_end_price + o_fibo_value if longshort else w_end_price - o_fibo_value
        submit_price = et_price + abs(tp_price - et_price)/2 if longshort else et_price - abs(tp_price - et_price)/2
        cancel_price = tp_price

        df_active['sl_price'] = sl_price
        df_active['et_price'] = et_price
        df_active['tp_price'] = tp_price
        df_active['out_price'] = out_price
        df_active['sub_price'] = submit_price
        df_active['can_price'] = cancel_price

        df_active['sl_cross'] = df_active.apply(lambda x: 1 if x['High'] >= sl_price and sl_price >= x['Low'] else 0, axis=1)
        df_active['et_cross'] = df_active.apply(lambda x: 1 if x['High'] >= et_price and et_price >= x['Low'] else 0, axis=1)
        df_active['tp_cross'] = df_active.apply(lambda x: 1 if x['High'] >= tp_price and tp_price >= x['Low'] else 0, axis=1)
        df_active['out_cross'] = df_active.apply(lambda x: 1 if x['High'] >= out_price and out_price >= x['Low'] else 0, axis=1)
        df_active['sub_cross'] = df_active.apply(lambda x: 1 if x['High'] >= submit_price and submit_price >= x['Low'] else 0, axis=1)
        df_active['can_cross'] = df_active.apply(lambda x: 1 if x['High'] >= cancel_price and cancel_price >= x['Low'] else 0, axis=1)

        df_active['status_raw'] = df_active.apply(lambda x: f_order_status(x['sl_cross'], x['et_cross'], x['tp_cross'], x['out_cross'], x['sub_cross'], x['can_cross']), axis=1)
        status_raw_list = df_active['status_raw'].tolist()

        current_action = list()
        pre_action_list = list()
        for i, status in enumerate(status_raw_list):
            # pre_action_list = list(dict.fromkeys(pre_action_list)) #  순서유지
            if status:
                if len(pre_action_list) == 0 and ('Rejected' in status):
                    current_action.append('Rejected')
                    for k in status_raw_list[i+1:]:
                        current_action.append(np.nan)
                    break
                elif len(pre_action_list) == 0 and len(status) == 1 and ('Cancelled' in status):
                    pre_action_list = []
                    current_action.append(np.nan)
                elif len(pre_action_list) == 0 and len(status) == 2 and ('Cancelled' in status):
                    pre_action_list = []
                    current_action.append(np.nan)
                elif len(pre_action_list) == 0 and len(status) == 2 and ('Submitted' in status) and ('ET' in status):
                    pre_action_list = ['Submitted', 'ET']
                    current_action.append('ET')
                elif len(pre_action_list) == 0 and len(status) == 1 and ('Submitted' in status):
                    pre_action_list = ['Submitted']
                    current_action.append('Submitted')
                elif len(pre_action_list) == 1 and ('ET' in status):
                    pre_action_list = ['Submitted', 'ET']
                    current_action.append('ET')
                elif len(pre_action_list) == 2 and ('WIN' in status):
                    pre_action_list = ['Submitted', 'ET', 'WIN']
                    current_action.append('WIN')
                    for k in status_raw_list[i+1:]:
                        current_action.append(np.nan)
                    break
                elif len(pre_action_list) == 2 and ('LOSE' in status):
                    pre_action_list = ['Submitted', 'ET', 'LOSE']
                    current_action.append('LOSE')
                    for k in status_raw_list[i+1:]:
                        current_action.append(np.nan)
                    break
                else:
                    current_action.append(np.nan)
            else:
                current_action.append(np.nan)

        df_active['action'] = current_action
        current_action_undupl = list(dict.fromkeys(current_action)) #  순서유지


        # status_raw_list_release = list()
        # for s in status_raw_list:
        #     for j in s:
        #         status_raw_list_release.append(j)
        # status_raw_list_undupl = list(dict.fromkeys(status_raw_list_release)) #  순서유지


        # 1
        # LOSE -->['Submitted', 'Cancelled', 'WIN', 'ET', 'LOSE']
        # WIN -->['Submitted', 'Cancelled', 'WIN', 'ET', 'LOSE']

        # 2
        # LOSE -->['Submitted', 'Cancelled', 'ET', 'WIN', 'LOSE']
        # WIN -->['Submitted', 'Cancelled', 'ET', 'WIN', 'LOSE']

        # if 'LOSE' in status_raw_list_undupl:
        #     print('LOSE-->' + str(status_raw_list_undupl))
        # if 'Rejected' in status_raw_list_undupl:
        #     print('Rejected-->' +str(status_raw_list_undupl))
        # if 'WIN' in status_raw_list_undupl:
        #     print('WIN-->' +str(status_raw_list_undupl))

        # if 'Submitted' in current_action_undupl:
        #     # print(str(current_action_undupl))
        #     position = True
        #     c_profit = True
        #     c_stoploss = False
        #     if plotview:
        #         plot_pattern_n(
        #             symbol + '_' + str(tf) + '_' + str(fc),
        #             df=df,
        #             w=w,
        #             # k=k,
        #             wave_pattern=wavepattern_tpsl_l,
        #             df_lows_plot=df_lows_plot,
        #             df_highs_plot=df_highs_plot,
        #             trade_info=t_info,
        #             wave_options=wave_option_plot_l,
        #             title='APPLY Submitted ' + symbol + ' ' +
        #                   str([et_price, sl_price, tp_price, tp_price_w5]))
        #     return trade_info, o_his

        if 'ET' in current_action_undupl:
            # print(str(current_action_undupl))
            position = True
            c_profit = True
            c_stoploss = False
            # if plotview:
            #     plot_pattern_n(
            #         symbol + '_' + str(tf) + '_' + str(fc),
            #         df=df,
            #         w=w,
            #         # k=k,
            #         wave_pattern=wavepattern_tpsl_l,
            #         df_lows_plot=df_lows_plot,
            #         df_highs_plot=df_highs_plot,
            #         trade_info=t_info,
            #         wave_options=wave_option_plot_l,
            #         title='APPLY ET ' + symbol + ' ' +
            #               str([et_price, sl_price, tp_price, tp_price_w5]))
            # return trade_info, o_his
        if 'WIN' in current_action_undupl:
            position = True
            c_profit = True
            c_stoploss = False
            # if plotview:
            #     plot_pattern_n(
            #         symbol + '_' + str(tf) + '_' + str(fc),
            #         df=df,
            #         w=w,
            #         # k=k,
            #         wave_pattern=wavepattern_tpsl_l,
            #         df_lows_plot=df_lows_plot,
            #         df_highs_plot=df_highs_plot,
            #         trade_info=t_info,
            #         wave_options=wave_option_plot_l,
            #         title='BACKTEST TP/SL ' + str('trade_stats') + ' ' +
            #               str([et_price, sl_price, tp_price, tp_price_w5]))
        if 'LOSE' in current_action_undupl:
            position = True
            c_profit = False
            c_stoploss = True
            # if plotview:
            #     plot_pattern_n(
            #         symbol + '_' + str(tf) + '_' + str(fc),
            #         df=df,
            #         w=w,
            #         # k=k,
            #         wave_pattern=wavepattern_tpsl_l,
            #         df_lows_plot=df_lows_plot,
            #         df_highs_plot=df_highs_plot,
            #         trade_info=t_info,
            #         wave_options=wave_option_plot_l,
            #         title='BACKTEST TP/SL ' + str('trade_stats') + ' ' +
            #               str([et_price, sl_price, tp_price, tp_price_w5]))

    if position is True:
    # if False:
        position_enter_i = [symbol, et_price, sl_price, tp_price, 'dates[k]']

        if c_profit or c_stoploss:
            win_lose_flg = None
            if t_mode in ['BACKTEST']:
                r_order, o_his = new_et_order_test(symbol, tf, fc, longshort, qtyrate_k, et_price, sl_price,
                                                   tp_price, tp_price_w5, 1, w, et_orderid_test, o_his, t_mode)

            fee_limit_tp = 0
            fee_limit_sl = 0
            if tp_type == 'maker':
                fee_limit_tp = fee_limit + fee_tp
            elif tp_type == 'taker':
                fee_limit_tp = fee_limit + fee_tp + fee_slippage
            fee_limit_sl = fee_limit + fee_sl + fee_slippage

            fee_percent = 0
            pnl_percent = 0
            win_lose_flg = 0
            seed_pre = asset_history[-1] if asset_history else seed

            if c_stoploss and c_profit:
                print('XXXXXXXX XXXXXXXX XXXXXXX')

            if c_stoploss and not c_profit:
                win_lose_flg = 0

                # https://www.binance.com/en/support/faq/how-to-use-binance-futures-calculator-360036498511
                pnl_percent = -(abs(sl_price - et_price) / sl_price)         #  PnL  =  (1 - entry price / exit price)
                fee_percent = fee_limit_sl

                pnl = pnl_percent * seed_pre * qtyrate_k
                fee = fee_percent * seed_pre * qtyrate_k

                trade_count.append(0)
                trade_inout_i = [position_enter_i[0],
                                 position_enter_i[1],
                                 position_enter_i[2],
                                 position_enter_i[3],
                                 position_enter_i[4],' dates[k]',
                                 longshort, 'LOSE']
                order_history.append(trade_inout_i)
                o_his = update_history_status(o_his, symbol, et_orderid_test, 'LOSE')

            if c_profit and not c_stoploss:
                win_lose_flg = 1
                pnl_percent = (abs(tp_price - et_price) / tp_price)
                fee_percent = fee_limit_tp

                pnl = pnl_percent * seed_pre * qtyrate_k
                fee = fee_percent * seed_pre * qtyrate_k

                trade_count.append(1)
                trade_inout_i = [position_enter_i[0],
                                 position_enter_i[1],
                                 position_enter_i[2],
                                 position_enter_i[3],
                                 position_enter_i[4], 'dates[k]',
                                 longshort, 'WIN']
                order_history.append(trade_inout_i)
                o_his = update_history_status(o_his, symbol, et_orderid_test, 'WIN')


            if win_lose_flg is not None:
                asset_new = seed_pre + pnl - fee
                pnl_history.append(pnl)
                fee_history.append(fee)
                asset_history.append(asset_new)
                wavepattern_history.append(w)

                winrate = (sum(trade_count) / len(trade_count)) * 100

                asset_min = seed
                asset_max = seed

                if len(stats_history) > 0:
                    asset_last_min = stats_history[-1][-2]
                    asset_min = asset_new if asset_new < asset_last_min else asset_last_min
                    asset_last_max = stats_history[-1][-1]
                    asset_max = asset_new if asset_new > asset_last_max else asset_last_max

                df_s = pd.DataFrame.from_records(stats_history)
                b_symbol = abs(tp_price - et_price) / abs(
                    sl_price - et_price)  # one trade profitlose rate

                b_cum = (df_s[8].sum() + b_symbol) / (
                        len(df_s) + 1) if len(
                    stats_history) > 0 else b_symbol  # mean - profitlose rate, df_s[6] b_cum index
                p_cum = winrate / 100  # win rate
                q_cum = 1 - p_cum  # lose rate
                tpi_cum = round(p_cum * (1 + b_cum), 2)  # trading perfomance index
                f_cum = round(p_cum - (q_cum / b_cum),
                              2)  # kelly index https://igotit.tistory.com/1526

                f = f_cum
                tpi = tpi_cum
                b = b_cum

                if c_kelly_adaptive:
                    if len(df_s) >= c_kelly_window:
                        df_s_window = df_s.iloc[-c_kelly_window:]
                        p = df_s_window[4].sum() / len(df_s_window)
                        b = (df_s_window[11].sum() + b_symbol) / (len(
                            df_s_window) + 1)  # mean in kelly window - profitlose rate
                        q = 1 - p  # lose rate
                        tpi = round(p * (1 + b),
                                    2)  # trading perfomance index
                        f = round(p - (q / b), 2)  # kelly index

                trade_stats = [len(trade_count), round(winrate, 2),
                               asset_new, symbol, win_lose_flg,
                               'WIN' if win_lose_flg else 'LOSE', f_cum,
                               tpi_cum, b_cum, f, tpi, b, b_symbol,
                               str(qtyrate_k),
                               str(round(pnl_percent, 4)),
                               sum(pnl_history), sum(fee_history),
                               round(asset_min, 2), round(asset_max, 2)]
                stats_history.append(trade_stats)
                t_info = [stats_history, order_history,
                          asset_history, trade_count, fee_history,
                          pnl_history, wavepattern_history]

                wavepattern_tpsl_l.append(
                    [ix, w.dates[0], id(w), w])

                s_11 = symbol + '           '
                trade_in = trade_inout_i[4][2:-3]
                trade_out = trade_inout_i[5][6:-3]

                trade_stats_print = [len(trade_count), round(winrate, 2),
                               asset_new, symbol, win_lose_flg,
                               'WIN' if win_lose_flg else 'LOSE', f_cum,
                               # tpi_cum, b_cum, f, tpi, b, b_symbol,
                               str(qtyrate_k),
                               str(round(pnl_percent, 4)),
                               sum(pnl_history), sum(fee_history),
                               round(asset_min, 2), round(asset_max, 2)]
                logger.info(
                    '%s %s %s %s x%s %s-%s %s %s %s %s %s %s - %s' % (t_mode[:1] + ' :',
                        timeframe, s_11[:11], tf, qtyrate_k,
                        period_days_ago, period_days_ago_till, ' x ' + str(fc),
                        'L' if longshort else 'S', trade_in, '-',
                        trade_out, str(trade_stats_print),
                        str([et_price, sl_price, tp_price, tp_price_w5])))

                if plotview:
                    plot_pattern_k(
                        symbol + '_' + str(tf) + '_' + str(fc),
                        df=df,
                        w=w,
                        k=k,
                        wave_pattern=wavepattern_tpsl_l,
                        df_lows_plot=df_lows_plot,
                        df_highs_plot=df_highs_plot,
                        trade_info=t_info,
                        wave_options=wave_option_plot_l,
                        title='BACKTEST TP/SL ' + str(trade_stats) + ' ' +
                        str([et_price, sl_price, tp_price, tp_price_w5]))

    return t_info, o_his


def test_trade_market_real_xxxxxxxx(symbol, tf, fc, longshort, et_price, sl_price, tp_price, tp_price_w5, w, t_mode, o_his, df_lows_plot, df_highs_plot, wavepatterns, wavepattern_tpsl_l, wave_option_plot_l, wave_opt, ix, df, t_info):
    wavepatterns.add(w)
    # wavepattern_l.append([symbol, fc, ix, w.dates[0], id(w), w])
    wave_option_plot_l.append([
        [str(w.dates[-1])],
        [w.values[-1]],
        [str(wave_opt.values)]
    ])

    w = w
    t = t_info
    stats_history = t[0]
    order_history = t[1]
    asset_history = t[2]
    trade_count = t[3]
    fee_history = t[4]
    pnl_history = t[5]
    wavepattern_history = t[6]

    w_start_price = w.values[0]  # wave1
    w_end_price = w.values[-1]  # wave5

    out_price = None

    qtyrate_k = get_qtyrate_k(t_info, qtyrate)

    df_active = df[w.idx_end:]  # w5 stick 포함해서 전체 active
    closes = df_active.Close.tolist()

    position_enter_i = []
    et_orderid_test = randrange(10000000000, 99999999999, 1)

    if closes:
        position = False
        c_stoploss = False
        c_profit = False
        c_price_et_price = None
        for k, close in enumerate(closes):
            try:
                df_k = df[:w.idx_end + (k + 1)]
                df_active_k = df_active[:(k + 1)]  # k = 0 번째까지 active

                dates = df_active_k.Date.tolist()
                closes = df_active_k.Close.tolist()
                highs = df_active_k.High.tolist()
                lows = df_active_k.Low.tolist()

            except Exception as e:
                print('here e: %s' % str(e))

            if not position:
                df_active_k_with_w5_stick = df_active_k[w.idx_end:]
                cross_cnt = get_cross_count_intersect_df(df_active_k_with_w5_stick, et_price)  # w5 stick 포함하여 계산, 1번 초과 제외
                if cross_cnt >= 2:
                    return t_info, o_his

                if cross_cnt == 0:
                    continue

                if cross_cnt == 1:
                    # check current price
                    # c_price = float(api_call('ticker_price', [symbol])['price'])
                    # a = close
                    # a_k = closes[k]
                    # c_price = lows[k] if longshort else highs[k]
                    c_price = et_price
                    w2_price = w.values[3]
                    position = True
                    position_enter_i = [symbol, c_price, sl_price, tp_price, dates[k]]

                    c_price_et_price = et_price
                    c_stoploss_direct = lows[k] <= sl_price if longshort else highs[k] >= sl_price
                    if c_stoploss_direct:
                        c_stoploss = True
                        c_profit = False
                        logger.info('c_stoplost_direct')

                        if True:
                            plot_pattern_k(
                                            symbol + '_' + str(tf) + '_' + str(fc),
                                            df=df,
                                            w=w,
                                            k=k,
                                            wave_pattern=[[ix, w.dates[0], id(w), w]],
                                            df_lows_plot=df_lows_plot,
                                            df_highs_plot=df_highs_plot,
                                            # trade_info=t_info,
                                            wave_options=wave_option_plot_l,
                                            title='BACKTEST c_stoplost_direct %s' % str(position_enter_i))


            c_stoploss = (position and lows[k] <= sl_price) if longshort else (position and highs[k] >= sl_price)
            c_profit = (position and highs[k] >= tp_price) if longshort else (position and lows[k] <= tp_price)

            if position is True:
                if c_profit or c_stoploss:
                    win_lose_flg = None
                    if t_mode in ['BACKTEST']:
                        et_price = c_price_et_price
                        r_order, o_his = new_et_order_test(symbol, tf, fc, longshort, qtyrate_k, et_price, sl_price,
                                                           tp_price, tp_price_w5, 1, w, et_orderid_test, o_his, t_mode)

                    fee_limit_tp = 0
                    fee_limit_sl = 0
                    if tp_type == 'maker':
                        fee_limit_tp = fee_limit + fee_tp
                    elif tp_type == 'taker':
                        fee_limit_tp = fee_limit + fee_tp + fee_slippage
                    fee_limit_sl = fee_limit + fee_sl + fee_slippage

                    fee_percent = 0
                    pnl_percent = 0
                    win_lose_flg = 0
                    seed_pre = asset_history[-1] if asset_history else seed

                    if c_stoploss and c_profit:
                        print('XXXXXXXX XXXXXXXX XXXXXXX')

                    if c_stoploss and not c_profit:
                        win_lose_flg = 0

                        # https://www.binance.com/en/support/faq/how-to-use-binance-futures-calculator-360036498511
                        pnl_percent = -(abs(sl_price - et_price) / sl_price)         #  PnL  =  (1 - entry price / exit price)
                        fee_percent = fee_limit_sl

                        pnl = pnl_percent * seed_pre * qtyrate_k
                        fee = fee_percent * seed_pre * qtyrate_k

                        trade_count.append(0)
                        trade_inout_i = [position_enter_i[0],
                                         position_enter_i[1],
                                         position_enter_i[2],
                                         position_enter_i[3],
                                         position_enter_i[4], dates[k],
                                         longshort, 'LOSE']
                        order_history.append(trade_inout_i)
                        o_his = update_history_status(o_his, symbol, et_orderid_test, 'LOSE')

                    if c_profit and not c_stoploss:
                        win_lose_flg = 1
                        pnl_percent = (abs(tp_price - et_price) / tp_price)
                        fee_percent = fee_limit_tp

                        pnl = pnl_percent * seed_pre * qtyrate_k
                        fee = fee_percent * seed_pre * qtyrate_k

                        trade_count.append(1)
                        trade_inout_i = [position_enter_i[0],
                                         position_enter_i[1],
                                         position_enter_i[2],
                                         position_enter_i[3],
                                         position_enter_i[4], dates[k],
                                         longshort, 'WIN']
                        order_history.append(trade_inout_i)
                        o_his = update_history_status(o_his, symbol, et_orderid_test, 'WIN')


                    if win_lose_flg is not None:
                        asset_new = seed_pre + pnl - fee
                        pnl_history.append(pnl)
                        fee_history.append(fee)
                        asset_history.append(asset_new)
                        wavepattern_history.append(w)

                        winrate = (sum(trade_count) / len(trade_count)) * 100

                        asset_min = seed
                        asset_max = seed

                        if len(stats_history) > 0:
                            asset_last_min = stats_history[-1][-2]
                            asset_min = asset_new if asset_new < asset_last_min else asset_last_min
                            asset_last_max = stats_history[-1][-1]
                            asset_max = asset_new if asset_new > asset_last_max else asset_last_max

                        df_s = pd.DataFrame.from_records(stats_history)
                        b_symbol = abs(tp_price - et_price) / abs(
                            sl_price - et_price)  # one trade profitlose rate

                        b_cum = (df_s[8].sum() + b_symbol) / (
                                len(df_s) + 1) if len(
                            stats_history) > 0 else b_symbol  # mean - profitlose rate, df_s[6] b_cum index
                        p_cum = winrate / 100  # win rate
                        q_cum = 1 - p_cum  # lose rate
                        tpi_cum = round(p_cum * (1 + b_cum), 2)  # trading perfomance index
                        f_cum = round(p_cum - (q_cum / b_cum),
                                      2)  # kelly index https://igotit.tistory.com/1526

                        f = f_cum
                        tpi = tpi_cum
                        b = b_cum

                        if c_kelly_adaptive:
                            if len(df_s) >= c_kelly_window:
                                df_s_window = df_s.iloc[-c_kelly_window:]
                                p = df_s_window[4].sum() / len(df_s_window)
                                b = (df_s_window[11].sum() + b_symbol) / (len(
                                    df_s_window) + 1)  # mean in kelly window - profitlose rate
                                q = 1 - p  # lose rate
                                tpi = round(p * (1 + b),
                                            2)  # trading perfomance index
                                f = round(p - (q / b), 2)  # kelly index

                        trade_stats = [len(trade_count), round(winrate, 2),
                                       asset_new, symbol, win_lose_flg,
                                       'WIN' if win_lose_flg else 'LOSE', f_cum,
                                       tpi_cum, b_cum, f, tpi, b, b_symbol,
                                       str(qtyrate_k),
                                       str(round(pnl_percent, 4)),
                                       sum(pnl_history), sum(fee_history),
                                       round(asset_min, 2), round(asset_max, 2)]
                        stats_history.append(trade_stats)
                        t_info = [stats_history, order_history,
                                  asset_history, trade_count, fee_history,
                                  pnl_history, wavepattern_history]

                        wavepattern_tpsl_l.append(
                            [ix, w.dates[0], id(w), w])

                        s_11 = symbol + '           '
                        trade_in = trade_inout_i[4][2:-3]
                        trade_out = trade_inout_i[5][6:-3]

                        trade_stats_print = [len(trade_count), round(winrate, 2),
                                       asset_new, symbol, win_lose_flg,
                                       'WIN' if win_lose_flg else 'LOSE', f_cum,
                                       # tpi_cum, b_cum, f, tpi, b, b_symbol,
                                       str(qtyrate_k),
                                       str(round(pnl_percent, 4)),
                                       sum(pnl_history), sum(fee_history),
                                       round(asset_min, 2), round(asset_max, 2)]
                        logger.info(
                            '%s %s %s %s x%s %s-%s %s %s %s %s %s %s - %s' % (t_mode[:1] + ' :',
                                timeframe, s_11[:11], tf, qtyrate_k,
                                period_days_ago, period_days_ago_till, ' x ' + str(fc),
                                'L' if longshort else 'S', trade_in, '-',
                                trade_out, str(trade_stats_print),
                                str([et_price, sl_price, tp_price, tp_price_w5])))

                        if plotview:
                            plot_pattern_k(
                                symbol + '_' + str(tf) + '_' + str(fc),
                                df=df,
                                w=w,
                                k=k,
                                wave_pattern=wavepattern_tpsl_l,
                                df_lows_plot=df_lows_plot,
                                df_highs_plot=df_highs_plot,
                                trade_info=t_info,
                                wave_options=wave_option_plot_l,
                                title='BACKTEST TP/SL ' + str(trade_stats) + ' ' +
                                str([et_price, sl_price, tp_price, tp_price_w5]))
                        break
    return t_info, o_his


def test_trade(symbol, tf, fc, longshort, et_price, sl_price, tp_price, tp_price_w5, w, t_mode, o_his, df_lows_plot, df_highs_plot, wavepatterns, wavepattern_tpsl_l, wave_option_plot_l, wave_opt, ix, df, t_info):
    wavepatterns.add(w)
    # wavepattern_l.append([symbol, fc, ix, w.dates[0], id(w), w])
    wave_option_plot_l.append([
        [str(w.dates[-1])],
        [w.values[-1]],
        [str(wave_opt.values)]
    ])

    w = w
    t = t_info
    stats_history = t[0]
    order_history = t[1]
    asset_history = t[2]
    trade_count = t[3]
    fee_history = t[4]
    pnl_history = t[5]
    wavepattern_history = t[6]

    w_start_price = w.values[0]  # wave1
    w_end_price = w.values[-1]  # wave5
    height_price = abs(w_end_price - w_start_price)

    out_price = None

    qtyrate_k = get_qtyrate_k(t_info, qtyrate)

    df_active = df[w.idx_end + 1:]
    df_active_with_w5_stick = df[w.idx_end:]
    # df_active = df[w.idx_end:]  # xxxxx 틀렸음

    dates = df_active.Date.tolist()
    closes = df_active.Close.tolist()
    highs = df_active.High.tolist()
    lows = df_active.Low.tolist()

    position_enter_i = []
    position = False
    et_exec_flg = False
    et_orderid_test = randrange(10000000000, 99999999999, 1)

    if dates:
        for k, close in enumerate(closes):
            try:
                df_k = df[:(w.idx_end + 1) + (k + 1)]
                df_active_k = df_active[:(k + 1)]
            except Exception as e:
                print('here e: %s' % str(e))

            # if not c_active_in_time(df_i, w):
            #     return False

            # if not c_active_in_zone(df_active_k, longshort, w, et_price, sl_price, tp_price, tp_price_w5, 'ET_PRICE'):
            #     return False
            #
            # c_interset, cross_cnt = c_allowed_intersect_df(df_active_k, et_price, 1)  # 1번 초과 제외
            # if not c_interset:
            #     return t_info, o_his
            if t_mode in ['PAPER']:
                if paper_flg:
                    if not et_exec_flg:

                        c_interset, cross_cnt = c_allowed_intersect_df(df_active_with_w5_stick, et_price, 0)  # 0번 초과 제외
                        if not c_interset:
                            return t_info, o_his

                        if not c_active_in_zone(df_active_with_w5_stick, longshort, w, et_price, sl_price, tp_price, tp_price_w5,
                                                'ET_PRICE'):
                            return t_info, o_his

                        if c_current_price_in_zone_by_prices_k_close(symbol, longshort, et_price, tp_price, close):
                            plot_pattern_k(
                                           symbol+'_'+str(tf)+'_'+str(fc),
                                           df=df,
                                           w=w,
                                           k=k,
                                           wave_pattern=[[ix, w.dates[0], id(w), w]],
                                           df_lows_plot=df_lows_plot,
                                           df_highs_plot=df_highs_plot,
                                           # trade_info=t_info,
                                           wave_options=wave_option_plot_l,
                                           title='B ET/test_trade %s' % str([symbol, et_price, sl_price, tp_price, dates[k]]))
                            r_order, o_his = new_et_order_test(symbol, tf, fc, longshort, qtyrate_k, et_price, sl_price, tp_price, tp_price_w5, 1, w, et_orderid_test, o_his, t_mode)

                            et_exec_flg = True
                    else:
                        if not c_current_price_in_zone_by_prices_k_close(symbol, longshort, et_price, tp_price, close):
                            o_his = delete_history_status(o_his, symbol, et_orderid_test, 'CANCEL')
                            et_exec_flg = False

            c_positioning = (
                    position is False
                    and et_price >= lows[k] > sl_price) if longshort else (
                    position is False and et_price <= highs[k] < sl_price
            )

            c_stoploss = (position and lows[k] <= sl_price) if longshort else (position and highs[k] >= sl_price)
            c_profit = (position and highs[k] >= tp_price) if longshort else (position and lows[k] <= tp_price)

            c_stoploss_direct = (lows[k] <= sl_price and highs[k] >= et_price) if longshort else (
                        highs[k] >= sl_price and lows[k] <= et_price)

            if c_stoploss_direct:
                position = True
                c_stoploss = True
                position_enter_i = [symbol, et_price, sl_price, tp_price, dates[k]]

                logger.info('c_stoplost_direct')

                # if plotview:
                #     plot_pattern_k(
                #                     symbol + '_' + str(tf) + '_' + str(fc),
                #                     df=df,
                #                     w=w,
                #                     k=k,
                #                     wave_pattern=[[ix, w.dates[0], id(w), w]],
                #                     df_lows_plot=df_lows_plot,
                #                     df_highs_plot=df_highs_plot,
                #                     # trade_info=t_info,
                #                     wave_options=wave_option_plot_l,
                #                     title='BACKTEST ET POSITION (DIRECT) %s' % str(position_enter_i))

            elif position is False and c_positioning:
                position = True
                position_enter_i = [symbol, et_price, sl_price, tp_price, dates[k]]

                # if plotview:
                #     plot_pattern_k(
                #         symbol + '_' + str(tf) + '_' + str(fc),
                #         df=df,
                #         w=w,
                #         k=k,
                #         wave_pattern=[[ix, w.dates[0], id(w), w]],
                #         df_lows_plot=df_lows_plot,
                #         df_highs_plot=df_highs_plot,
                #         # trade_info=t_info,
                #         wave_options=wave_option_plot_l,
                #         title='BACKTEST  ET POSITION  %s' % str(position_enter_i))


            if position is True:
                if c_profit or c_stoploss:
                    if t_mode in ['BACKTEST']:
                        r_order, o_his = new_et_order_test(symbol, tf, fc, longshort, qtyrate_k, et_price, sl_price,
                                                           tp_price, tp_price_w5, 1, w, et_orderid_test, o_his, t_mode)

                    fee_limit_tp = 0
                    fee_limit_sl = 0
                    if tp_type == 'maker':
                        fee_limit_tp = fee_limit + fee_tp
                    elif tp_type == 'taker':
                        fee_limit_tp = fee_limit + fee_tp + fee_slippage
                    fee_limit_sl = fee_limit + fee_sl + fee_slippage

                    fee_percent = 0
                    pnl_percent = 0
                    win_lose_flg = 0
                    seed_pre = asset_history[-1] if asset_history else seed

                    if c_stoploss and not c_profit:
                        win_lose_flg = 0

                        # https://www.binance.com/en/support/faq/how-to-use-binance-futures-calculator-360036498511
                        pnl_percent = -(abs(sl_price - et_price) / sl_price)  # PnL  =  (1 - entry price / exit price)
                        fee_percent = fee_limit_sl

                        pnl = pnl_percent * seed_pre * qtyrate_k
                        fee = fee_percent * seed_pre * qtyrate_k

                        trade_count.append(0)
                        trade_inout_i = [position_enter_i[0],
                                         position_enter_i[1],
                                         position_enter_i[2],
                                         position_enter_i[3],
                                         position_enter_i[4], dates[k],
                                         longshort, 'LOSE']
                        order_history.append(trade_inout_i)
                        o_his = update_history_status(o_his, symbol, et_orderid_test, 'LOSE')

                    if c_profit and not c_stoploss:
                        win_lose_flg = 1
                        pnl_percent = (abs(tp_price - et_price) / tp_price)
                        fee_percent = fee_limit_tp

                        pnl = pnl_percent * seed_pre * qtyrate_k
                        fee = fee_percent * seed_pre * qtyrate_k

                        trade_count.append(1)
                        trade_inout_i = [position_enter_i[0],
                                         position_enter_i[1],
                                         position_enter_i[2],
                                         position_enter_i[3],
                                         position_enter_i[4], dates[k],
                                         longshort, 'WIN']
                        order_history.append(trade_inout_i)
                        o_his = update_history_status(o_his, symbol, et_orderid_test, 'WIN')

                    asset_new = seed_pre + pnl - fee
                    pnl_history.append(pnl)
                    fee_history.append(fee)
                    asset_history.append(asset_new)
                    wavepattern_history.append(w)

                    winrate = (sum(trade_count) / len(trade_count)) * 100

                    asset_min = seed
                    asset_max = seed

                    if len(stats_history) > 0:
                        asset_last_min = stats_history[-1][-2]
                        asset_min = asset_new if asset_new < asset_last_min else asset_last_min
                        asset_last_max = stats_history[-1][-1]
                        asset_max = asset_new if asset_new > asset_last_max else asset_last_max

                    df_s = pd.DataFrame.from_records(stats_history)
                    b_symbol = abs(tp_price - et_price) / abs(
                        sl_price - et_price)  # one trade profitlose rate

                    b_cum = (df_s[8].sum() + b_symbol) / (
                            len(df_s) + 1) if len(
                        stats_history) > 0 else b_symbol  # mean - profitlose rate, df_s[6] b_cum index
                    p_cum = winrate / 100  # win rate
                    q_cum = 1 - p_cum  # lose rate
                    tpi_cum = round(p_cum * (1 + b_cum), 2)  # trading perfomance index
                    f_cum = round(p_cum - (q_cum / b_cum),
                                  2)  # kelly index https://igotit.tistory.com/1526

                    f = f_cum
                    tpi = tpi_cum
                    b = b_cum

                    if c_kelly_adaptive:
                        if len(df_s) >= c_kelly_window:
                            df_s_window = df_s.iloc[-c_kelly_window:]
                            p = df_s_window[4].sum() / len(df_s_window)
                            b = (df_s_window[11].sum() + b_symbol) / (len(
                                df_s_window) + 1)  # mean in kelly window - profitlose rate
                            q = 1 - p  # lose rate
                            tpi = round(p * (1 + b),
                                        2)  # trading perfomance index
                            f = round(p - (q / b), 2)  # kelly index

                    trade_stats = [len(trade_count), round(winrate, 2),
                                   asset_new, symbol, win_lose_flg,
                                   'WIN' if win_lose_flg else 'LOSE', f_cum,
                                   tpi_cum, b_cum, f, tpi, b, b_symbol,
                                   str(qtyrate_k),
                                   str(round(pnl_percent, 4)),
                                   sum(pnl_history), sum(fee_history),
                                   round(asset_min, 2), round(asset_max, 2)]
                    stats_history.append(trade_stats)
                    t_info = [stats_history, order_history,
                              asset_history, trade_count, fee_history,
                              pnl_history, wavepattern_history]

                    wavepattern_tpsl_l.append(
                        [ix, w.dates[0], id(w), w])

                    s_11 = symbol + '           '
                    trade_in = trade_inout_i[4][2:-3]
                    trade_out = trade_inout_i[5][6:-3]

                    trade_stats_print = [len(trade_count), round(winrate, 2),
                                   asset_new, symbol, win_lose_flg,
                                   'WIN' if win_lose_flg else 'LOSE', f_cum,
                                   # tpi_cum, b_cum, f, tpi, b, b_symbol,
                                   str(qtyrate_k),
                                   str(round(pnl_percent, 4)),
                                   sum(pnl_history), sum(fee_history),
                                   round(asset_min, 2), round(asset_max, 2)]
                    logger.info(
                        '%s %s %s %s x%s %s-%s %s %s %s %s %s %s - %s' % (t_mode[:1] + ' :',
                            timeframe, s_11[:11], tf, qtyrate_k,
                            period_days_ago, period_days_ago_till, ' x ' + str(fc),
                            'L' if longshort else 'S', trade_in, '-',
                            trade_out, str(trade_stats_print),
                            str([et_price, sl_price, tp_price, tp_price_w5])))

                    if plotview:
                        plot_pattern_k(
                            symbol + '_' + str(tf) + '_' + str(fc),
                            df=df,
                            w=w,
                            k=k,
                            wave_pattern=wavepattern_tpsl_l,
                            df_lows_plot=df_lows_plot,
                            df_highs_plot=df_highs_plot,
                            trade_info=t_info,
                            wave_options=wave_option_plot_l,
                            title='BACKTEST TP/SL ' + str(trade_stats) + ' ' +
                            str([et_price, sl_price, tp_price, tp_price_w5]))
                    break
    return t_info, o_his


def monitor_wave_and_action(symbol_p, tf_p, t_mode, t_info, o_his, i=None):
    wave_list = get_waves(symbol_p, tf_p, t_info, t_mode, i=i)
    if wave_list:
        wavepatterns = set()
        wavepattern_tpsl_l = []
        wave_option_plot_l = []
        for wave in wave_list:
            symbol = wave[0]
            df = wave[1]
            tf = wave[2]
            fc = wave[3]
            longshort = wave[4]
            w = wave[5]
            ix = wave[6]
            wave_opt = wave[7]
            df_lows_plot = wave[8]
            df_highs_plot = wave[9]
            et_price, sl_price, tp_price, tp_price_w5 = get_trade_prices(o_his, symbol, longshort, w, t_mode)
            if c_check_valid_wave_in_history(o_his, symbol, tf, fc, w, et_price, sl_price, tp_price, tp_price_w5, t_mode):
                if check_cons_for_new_etsl_order(o_his, df, symbol, tf, fc, longshort, w, et_price, sl_price, tp_price, tp_price_w5, ix, t_mode, t_info, qtyrate):
                    if t_mode in ['REAL']:
                        t_info, o_his = real_trade(symbol, tf, fc, longshort, et_price, sl_price, tp_price, tp_price_w5, w, t_mode, o_his, df_lows_plot, df_highs_plot, df, t_info)
                        # a, b = real_test_trade_df_apply(symbol, tf, fc, longshort, et_price, sl_price, tp_price, tp_price_w5, w, t_mode, o_his, df_lows_plot, df_highs_plot, wavepatterns, wavepattern_tpsl_l, wave_option_plot_l, wave_opt, ix, df, t_info)


                    elif t_mode in ['PAPER']:
                        t_info, o_his = test_trade(symbol, tf, fc, longshort, et_price, sl_price, tp_price, tp_price_w5, w,
                                                   t_mode, o_his, df_lows_plot, df_highs_plot, wavepatterns,
                                                   wavepattern_tpsl_l, wave_option_plot_l, wave_opt, ix, df, t_info)
                        t_info, o_his = real_trade(symbol, tf, fc, longshort, et_price, sl_price, tp_price, tp_price_w5, w,
                                                   t_mode, o_his, df_lows_plot, df_highs_plot, df, t_info)
                    elif t_mode in ['BACKTEST']:
                        t_info, o_his = test_trade_market_df_apply(symbol, tf, fc, longshort, et_price, sl_price, tp_price, tp_price_w5, w, t_mode, o_his, df_lows_plot, df_highs_plot, wavepatterns, wavepattern_tpsl_l, wave_option_plot_l, wave_opt, ix, df, t_info)
                        # t_info, o_his = test_trade(symbol, tf, fc, longshort, et_price, sl_price, tp_price, tp_price_w5, w, t_mode, o_his, df_lows_plot, df_highs_plot, wavepatterns, wavepattern_tpsl_l, wave_option_plot_l, wave_opt, ix, df, t_info)

    return t_info, o_his


def add_etsl_history(o_his, symbol, tf, fc, longshort, qtyrate_k, w, et_price, sl_price, tp_price,
                     tp_price_w5, quantity, et_orderId_p, sl_orderId, order_et, order_sl, t_mode):
    now = dt.datetime.now()
    history = {
        'trade_mode': t_mode,
        'symbol': symbol,
        'id': et_orderId_p,
        'create_datetime': now,
        'status': 'ETSL',
        'timeframe': tf,
        'fcnt': fc,
        'qtyrate_k': qtyrate_k,
        'longshort': longshort,
        'side': 'LONG' if longshort else 'SHORT',
        'wavepattern': w,
        'et_price': et_price,
        'sl_price': sl_price,
        'tp_price': tp_price,
        'tp_price_w5': tp_price_w5,
        'quantity': quantity,
        'et_orderId': et_orderId_p,
        'sl_orderId': sl_orderId,
        'tp_orderId': None,
        'etsl_datetime': now,
        'tp_datetime': None,
        'update_datetime': None,
        'et_data': order_et,
        'sl_data': order_sl,
        'tp_data': None
    }
    o_his.append(history)
    if t_mode in ['REAL', 'PAPER']:
        logger.info(t_mode + ' : ' + symbol + ' _HS add_etsl_history %s:' % 'ETSL' + str(history))
    elif t_mode in ['BACKTEST']:
        # logger.info(t_mode + ' : ' + symbol + ' _HS add_etsl_history %s:' % 'ETSL' + str(history))
        pass
    # dump_history_pkl()
    return o_his


def add_tp_history(o_his, symbol, et_orderId, tp_orderId, tp_data):
    if o_his:
        history_idx, history_id = get_i_r(o_his, 'id', et_orderId)
        history_id['status'] = 'TP'
        history_id['tp_orderId'] = tp_orderId
        history_id['tp_datetime'] = dt.datetime.now()
        history_id['update_datetime'] = dt.datetime.now()
        history_id['tp_data'] = tp_data
        o_his[history_idx] = history_id  # replace history
        # logger.info(symbol + ' _HS add_tp_history %s:' % str(tp_data))
        # dump_history_pkl()
    return o_his


def update_history_status(o_his, symbol, h_id, new_status):
    if o_his:
        history_idx, history_id = get_i_r(o_his, 'id', h_id)
        history_id['status'] = new_status  # update new status
        history_id['update_datetime'] = dt.datetime.now()
        o_his[history_idx] = history_id  # replace history
        # logger.info(symbol + ' _HS update_history_status id: %s status: %s:' % (str(h_id), new_status))
        # dump_history_pkl()
    return o_his


def delete_history_status(o_his, symbol, h_id, event):
    if o_his:
        history_idx, history_id = get_i_r(o_his, 'id', h_id)
        o_his.pop(history_idx)
        # logger.info(symbol + ' _HS delete_history_status %s' % event)
        # dump_history_pkl()
    return o_his


def close_position_by_symbol(symbol, quantity, longshort, et_orderId):
    positions = api_call('account', [])['positions']
    side = "LONG" if longshort else "SHORT"
    position_filtered = [x for x in positions if
                         x['symbol'] == symbol and x['entryPrice'] != '0.0' and x['positionSide'] == side]
    for p in position_filtered:
        # quantity = str(abs(float(p['positionAmt'])))
        order_market = api_call('new_order',
                                [symbol, "SELL" if longshort else "BUY",
                                 "LONG" if longshort else "SHORT", "MARKET",
                                 quantity, "tp_" + str(et_orderId)])
        if order_market:
            logger.info(symbol + ' _CLOSE POSITION close_position_by_symbol success' + str(order_market))
        else:
            logger.info(symbol + ' _CLOSE POSITION close_position_by_symbol fail' + str(order_market))


def get_qtyrate_k(t_info, qtyrate):
    t = t_info
    stats_history = t[0]
    qtyrate_k = qtyrate
    if c_kelly_adaptive:
        if stats_history:
            f_trade_stats = stats_history[-1][9]  # trade_stats index 9, f value kelly_index
            if f_trade_stats <= 0:
                qtyrate_k = krate_min
            elif f_trade_stats <= qtyrate:
                qtyrate_k = qtyrate
            elif f_trade_stats > qtyrate and f_trade_stats < krate_max:
                qtyrate_k = f_trade_stats
            elif f_trade_stats >= krate_max:
                qtyrate_k = krate_max
    return qtyrate_k


def update_trade_info(t_info, c_profit, c_stoploss, o_his, symbol, h_id):
    try:
        h = None
        if o_his:
            history_idx, history_id = get_i_r(o_his, 'id', h_id)
            o_his[history_idx] = history_id  # replace history
            h = o_his[history_idx]

        t = t_info
        stats_history = t[0]
        order_history = t[1]
        asset_history = t[2]
        trade_count = t[3]
        fee_history = t[4]
        pnl_history = t[5]
        wavepattern_history = t[6]
        position = True

        dates = str(h['update_datetime'])[:19]  # '2023-01-15 21:24:00' #TODO how to find real entry date
        tf = h['timeframe']
        timeframe = h['timeframe']
        longshort = h['longshort']
        qtyrate_k = h['qtyrate_k']
        et_price = h['et_price']
        sl_price = h['sl_price']
        tp_price = h['tp_price']
        wavepattern = h['wavepattern']

        b_symbol = abs(tp_price - et_price) / abs(sl_price - et_price)  # one trade profitlose rate

        position_enter_i = []
        if position is True:
            if c_profit or c_stoploss:

                fee_limit_tp = 0
                fee_limit_sl = 0
                if tp_type == 'maker':
                    fee_limit_tp = fee_limit + fee_tp
                elif tp_type == 'taker':
                    fee_limit_tp = fee_limit + fee_tp + fee_slippage
                fee_limit_sl = fee_limit + fee_sl + fee_slippage

                fee_percent = 0
                pnl_percent = 0
                win_lose_flg = 0
                seed_pre = asset_history[-1] if asset_history else seed

                if c_stoploss:
                    win_lose_flg = 0

                    # https://www.binance.com/en/support/faq/how-to-use-binance-futures-calculator-360036498511
                    pnl_percent = -(abs(sl_price - et_price) / sl_price)  # PnL  =  (1 - entry price / exit price)
                    fee_percent = fee_limit_sl

                    pnl = pnl_percent * seed_pre * qtyrate_k
                    fee = fee_percent * seed_pre * qtyrate_k

                    trade_count.append(0)
                    trade_inout_i = [position_enter_i[0],
                                     position_enter_i[1],
                                     position_enter_i[2],
                                     position_enter_i[3],
                                     position_enter_i[4], 'dates[k]',
                                     longshort, 'LOSE']
                    order_history.append(trade_inout_i)

                elif c_profit:
                    win_lose_flg = 1
                    pnl_percent = (abs(tp_price - et_price) / tp_price)
                    fee_percent = fee_limit_tp

                    pnl = pnl_percent * seed_pre * qtyrate_k
                    fee = fee_percent * seed_pre * qtyrate_k

                    trade_count.append(1)
                    trade_inout_i = [position_enter_i[0],
                                     position_enter_i[1],
                                     position_enter_i[2],
                                     position_enter_i[3],
                                     position_enter_i[4], 'dates[k]',
                                     longshort, 'WIN']
                    order_history.append(trade_inout_i)

                asset_new = seed_pre + pnl - fee
                pnl_history.append(pnl)
                fee_history.append(fee)
                asset_history.append(asset_new)
                wavepattern_history.append(w)

                winrate = (sum(trade_count) / len(trade_count)) * 100

                asset_min = seed
                asset_max = seed

                if len(stats_history) > 0:
                    asset_last_min = stats_history[-1][-2]
                    asset_min = asset_new if asset_new < asset_last_min else asset_last_min
                    asset_last_max = stats_history[-1][-1]
                    asset_max = asset_new if asset_new > asset_last_max else asset_last_max

                df_s = pd.DataFrame.from_records(stats_history)
                b_cum = (df_s[8].sum() + b_symbol) / (len(df_s) + 1) if len(
                    stats_history) > 0 else b_symbol  # mean - profitlose rate, df_s[6] b_cum index
                p_cum = winrate / 100  # win rate
                q_cum = 1 - p_cum  # lose rate
                tpi_cum = round(p_cum * (1 + b_cum), 2)  # trading perfomance index
                f_cum = round(p_cum - (q_cum / b_cum), 2)  # kelly index https://igotit.tistory.com/1526

                f = f_cum
                tpi = tpi_cum
                b = b_cum

                if c_kelly_adaptive:
                    if len(df_s) >= c_kelly_window:
                        df_s_window = df_s.iloc[-c_kelly_window:]
                        p = df_s_window[4].sum() / len(df_s_window)
                        b = (df_s_window[11].sum() + b_symbol) / (
                                len(df_s_window) + 1)  # mean in kelly window - profitlose rate
                        q = 1 - p  # lose rate
                        tpi = round(p * (1 + b), 2)  # trading perfomance index
                        f = round(p - (q / b), 2)  # kelly index

                # c_price = float(api_call('ticker_price', ['BNBUSDT'])['price'])

                trade_stats = [len(trade_count), round(winrate, 2), asset_new, symbol, win_lose_flg,
                               'WIN' if win_lose_flg else 'LOSE', f_cum, tpi_cum, b_cum, f, tpi, b, b_symbol,
                               str(qtyrate_k), str(round(pnl_percent, 4)), sum(pnl_history), sum(fee_history),
                               round(asset_min, 2), round(asset_max, 2)]
                stats_history.append(trade_stats)
                t_info = [stats_history, order_history, asset_history, trade_count, fee_history, pnl_history,
                          wavepattern_history]

                if True:
                    s_11 = symbol + '           '
                    trade_in = 'trade_in'  # trade_inout_i[0][0][2:-3]
                    trade_out = 'trade_out'  # trade_inout_i[1][0][8:-3]
                    ll = 'OOOOOOO %s %s %s %s %s x%s %s-%s %s %s %s %s %s %s - %s' % (str(qtyrate_k), str(
                        c_compare_before_fractal_mode) + ' :shift=' + str(c_compare_before_fractal_shift),
                                                                                          timeframe, s_11[:11], tf,
                                                                                          qtyrate_k,
                                                                                          period_days_ago,
                                                                                          period_days_ago_till,
                                                                                          fcnt,
                                                                                          'L' if longshort else 'S',
                                                                                          trade_in, '-',
                                                                                          trade_out,
                                                                                          str(trade_stats), str(
                        [et_price, sl_price, tp_price, 'out_price']))
                    # print(ll)
                    logger.info(ll)

                # if longshort is not None and len(t_info[1]) > 0:
                #     if plotview:
                #         plot_pattern_m(df=df, wave_pattern=[[i, wavepattern.dates[0], id(wavepattern), wavepattern]],
                #                        df_lows_plot=df_lows_plot, df_highs_plot=df_highs_plot, trade_info=t_info,
                #                        title=str(
                #                            symbol + ' %s ' % str(longshort) + str(trade_stats)))
    except Exception as e:
        print(symbol + ' update_trade_info ' + str(h_id) + ' e:' + str(e))
        logger.error(symbol + ' update_trade_info ' + str(h_id) + ' e:' + str(e))
    return t_info


def get_account_trades(symbol, et_orderId, sl_orderId, tp_orderId):
    pnl = 0
    commission = 0
    try:
        ids_list = list()
        ids_list.append(et_orderId)
        if sl_orderId:
            ids_list.append(sl_orderId)
        if tp_orderId:
            ids_list.append(tp_orderId)
        # rt = um_futures_client.get_account_trades(symbol=symbol, recvWindow=6000)
        rt = api_call('get_account_trades', [symbol])

        r = [x for i, x in enumerate(rt) if x['orderId'] in ids_list]
        df = pd.DataFrame.from_records(r)
        df['qty'] = df['qty'].apply(pd.to_numeric)
        df['quoteQty'] = df['quoteQty'].apply(pd.to_numeric)
        df['realizedPnl'] = df['realizedPnl'].apply(pd.to_numeric)
        df['commission'] = df['commission'].apply(pd.to_numeric)

        df['qty'] = df.groupby(['orderId'])['qty'].transform('sum')
        df['quoteQty'] = df.groupby(['orderId'])['quoteQty'].transform('sum')
        df['realizedPnl'] = df.groupby(['orderId'])['realizedPnl'].transform('sum')
        df['commission'] = df.groupby(['orderId'])['commission'].transform('sum')
        df = df.drop_duplicates(subset=['orderId'])
        df['time_dt'] = df.apply(
            lambda x: str(dt.datetime.fromtimestamp(float(x['time']) / 1000).strftime('%Y-%m-%d %H:%M:%S')), axis=1)
        df['date'] = df.apply(lambda x: str(dt.datetime.fromtimestamp(float(x['time']) / 1000).strftime('%Y-%m-%d')),
                              axis=1)
        df.sort_values(by='time_dt', ascending=False,
                       inplace=True)  # https://sparkbyexamples.com/pandas/sort-pandas-dataframe-by-date/

        df['commission_tot'] = df.groupby(['symbol'])['commission'].transform('sum')
        df['realizedPnl_tot'] = df.groupby(['symbol'])['realizedPnl'].transform('sum')

        # print(df)
        pnl = df['realizedPnl_tot'].iat[0]
        commission = df['commission_tot'].iat[0]
        marginAsset = df['marginAsset'].iat[0]
        commissionAsset = df['commissionAsset'].iat[0]
        trans_price = 1
        if marginAsset == 'BUSD' or commissionAsset == 'BNB':
            # trans_price = float(um_futures_client.ticker_price('BNBUSDT')['price'])
            trans_price = float(api_call('ticker_price', ['BNBUSDT'])['price'])
        if marginAsset == 'BUSD':
            pnl = float(pnl) * trans_price
        if commissionAsset == 'BNB':
            commission = float(commission) * trans_price
    except Exception as e:
        print('get_account_trades:%s' % str(e))
        logger.error('get_account_trades:%s' % str(e))
    return df, pnl, commission


def get_order_history_etsl_and_new_tp_order(o_his, symbol, et_orderId_p, t_mode):
    if o_his:
        history_new = [x for x in o_his if (x['et_orderId'] == et_orderId_p
                                            and x['symbol'] == symbol
                                            and x['trade_mode'] == t_mode
                                            and x['status'] == 'ETSL')]
        for new in history_new:
            symbol = new['symbol']
            et_orderId = new['et_orderId']
            sl_orderId = new['sl_orderId']
            tp_price = new['tp_price']
            tf = new['timeframe']
            longshort = new['longshort']
            quantity = new['quantity']
            fc = new['fcnt']

            if t_mode  == 'REAL':
                r_get_open_orders_et = api_call('get_open_orders', [symbol, et_orderId])
                r_get_open_orders_et_flg = True if r_get_open_orders_et else False
                r_query_et = api_call('query_order', [symbol, et_orderId])

                r_get_open_orders_sl = api_call('get_open_orders', [symbol, sl_orderId])
                r_get_open_orders_sl_flg = True if r_get_open_orders_sl else False
                r_query_sl = api_call('query_order', [symbol, sl_orderId])

                if r_get_open_orders_et_flg is False and r_get_open_orders_sl_flg is True and r_query_et['status'] == 'FILLED' and r_query_sl['status'] == 'NEW':
                    # NEW_TP
                    _, o_his = new_tp_order(symbol, tf, fc, longshort, tp_price, quantity, et_orderId, t_mode, o_his)
            elif t_mode == 'BACKTEST':
                try:
                    _, o_his = new_tp_order(symbol, tf, fc, longshort, tp_price, quantity, et_orderId_p, t_mode, o_his)
                except Exception as e:
                    print('get_order_history_etsl_and_new_tp_order e: %s' % str(e))
    return o_his


def monitor_history_and_action(symbol, t_mode, t_info, o_his):
    if o_his:
        history_new = [x for x in o_his if (x['symbol'] == symbol
                                            and x['trade_mode'] == t_mode
                                            and x['status'] == 'ETSL')]
        if history_new:
            for new in history_new:
                et_orderId = new['et_orderId']
                sl_orderId = new['sl_orderId']
                et_price = new['et_price']
                tp_price = new['tp_price']
                tf = new['timeframe']
                longshort = new['longshort']
                quantity = new['quantity']
                fc = new['fcnt']

                r_get_open_orders_et = api_call('get_open_orders', [symbol, et_orderId])
                r_get_open_orders_et_flg = True if r_get_open_orders_et else False
                r_query_et = api_call('query_order', [symbol, et_orderId])

                r_get_open_orders_sl = api_call('get_open_orders', [symbol, sl_orderId])
                r_get_open_orders_sl_flg = True if r_get_open_orders_sl else False
                r_query_sl = api_call('query_order', [symbol, sl_orderId])

                if r_get_open_orders_et_flg is True and r_get_open_orders_sl_flg is True and r_query_et[
                    'status'] == 'NEW' and r_query_sl['status'] == 'NEW':
                    if not c_current_price_in_zone_by_prices(symbol, longshort, float(et_price), float(tp_price)):
                        # CANCEL_ETSL
                        response_cancel = cancel_batch_order(symbol, [str(et_orderId), str(sl_orderId)], 'CANCEL ETSL')
                        if response_cancel:
                            o_his = delete_history_status(o_his, symbol, et_orderId, 'CANCEL')
                elif r_get_open_orders_et_flg is False and r_get_open_orders_sl_flg is True and r_query_et[
                    'status'] == 'FILLED' and r_query_sl['status'] == 'NEW':
                    # NEW_TP
                    _, o_his = new_tp_order(symbol, tf, fc, longshort, tp_price, quantity, et_orderId, t_mode, o_his)
                elif r_get_open_orders_et_flg is False and r_get_open_orders_sl_flg is False and r_query_et[
                    'status'] == 'FILLED' and r_query_sl['status'] == 'FILLED':
                    o_his = update_history_status(o_his, symbol, et_orderId, 'LOSE')
                    logger.info(symbol + ' IN ETSLETSL Ooooooooooo  ESSL DIRECT LOSE  OOOOOoooOOOOOoooOOOO')

                elif r_get_open_orders_et_flg is True and r_get_open_orders_sl_flg is False and r_query_et[
                    'status'] == 'NEW' and r_query_sl['status'] == 'CANCELED':
                    cancel_batch_order(symbol, [et_orderId], 'FORCE CLICK SL IN ETSL, REMAIN ET CLEAR')
                    o_his = delete_history_status(o_his, symbol, et_orderId, 'CANCEL')
                elif r_get_open_orders_et_flg is False and r_get_open_orders_sl_flg is True and r_query_et[
                    'status'] == 'CANCELED' and r_query_sl['status'] == 'NEW':
                    cancel_batch_order(symbol, [sl_orderId], 'FORCE CLICK ET IN ETSL, REMAIN SL CLEAR')
                    o_his = delete_history_status(o_his, symbol, et_orderId, 'CANCEL')


                elif r_get_open_orders_et_flg is False and r_get_open_orders_sl_flg is False and r_query_et[
                    'status'] == 'FILLED' and r_query_sl['status'] == 'CANCELED':
                    close_position_by_symbol(symbol, quantity, longshort, et_orderId)  # AFTER FORCE SL CLICK, REMAIN TP FILLED AND CLOSE POSI ET
                    if rt is not None:
                        logger.info(symbol + ' IN ETSLETSL Ooooooooooo  AFTER FORCE SL CLICK, REMAIN TP FILLED AND CLOSE POSI ET OOOOOoooOOOOOoooOOOO' + str(
                                rt))
                        o_his = update_history_status(o_his, symbol, et_orderId, 'TP')  # TODO check win or lose
                    else:
                        logger.info(symbol + ' IN ETSLETSL Ooooooooooo  AFTER FORCE SL CLICK, REMAIN TP FILLED AND CLOSE POSI ET FAIL ERROR OOOOOoooOOOOOoooOOOO' + str(
                                rt))

                elif r_get_open_orders_et_flg is False and r_get_open_orders_sl_flg is True and r_query_et[
                    'status'] == 'NEW' and r_query_sl['status'] == 'CANCELED':
                    cancel_batch_order(symbol, [sl_orderId], 'FORCE MARKET CLICK IN ETSL, REMAIN SL CLEAR')
                    o_his = delete_history_status(o_his, symbol, et_orderId, 'CANCEL')

                elif r_get_open_orders_et_flg is False and r_get_open_orders_sl_flg is False and r_query_et[
                    'status'] == 'CANCELED' and r_query_sl['status'] == 'CANCELED':
                    o_his = delete_history_status(o_his, symbol, et_orderId, 'CANCEL')
                    logger.info(symbol + ' IN ETSLETSL OooooooooooOOO    TWO ET AND SL ARE CANCELED OOoooOOOOOoooOOOO')

                elif r_get_open_orders_et_flg is True and r_get_open_orders_sl_flg is True and r_query_et[
                    'status'] == 'PARTIALLY_FILLED' and r_query_sl['status'] == 'NEW':
                    rt, o_his = new_tp_order(symbol, tf, fc, longshort, tp_price, r_query_et['executedQty'], et_orderId, o_his)
                    logger.info(symbol + ' IN ETSLETSL PARTIALLY_FILLED monitoring_orders_positions ' + str(rt))
                    # # # force cancel limit(o), sl(x)
                    cancel_batch_order(symbol, [int(et_orderId)], 'CANCEL ETSL PARTIALLY ')

                elif r_get_open_orders_et_flg is False and r_get_open_orders_sl_flg is False and r_query_et[
                    'status'] == 'FILLED' and r_query_sl['status'] == 'EXPIRED':
                    o_his = update_history_status(o_his, symbol, et_orderId, 'FORCE')  # TWO ET AND SL ARE CANCELED
                    logger.info(symbol + ' IN ETSLETSL ET:FILLED and SL:EXPIRED monitoring_orders_positions ' + str(rt))

                else:
                    logger.info(symbol + ' IN ETSLETSL OooooooooooOOOOOoooOOOOOooESSLESSLoOOOO')
                    logger.info('IN ETSL: %s %s %s %s %s ' % (
                        symbol, str(r_get_open_orders_et_flg), str(r_get_open_orders_sl_flg), str(r_query_et['status']),
                        str(r_query_sl['status'])))

        history_tp = [x for x in o_his if (x['symbol'] == symbol
                                           and x['trade_mode'] == t_mode
                                           and x['status'] == 'TP')]
        if history_tp:
            for tp in history_tp:
                et_orderId = tp['et_orderId']
                sl_orderId = tp['sl_orderId']
                tp_orderId = tp['tp_orderId']
                longshort = tp['longshort']
                quantity = tp['quantity']

                r_get_open_orders_tp = api_call('get_open_orders', [symbol, tp_orderId])
                r_get_open_orders_tp_flg = True if r_get_open_orders_tp else False
                r_query_tp = api_call('query_order', [symbol, tp_orderId])

                r_get_open_orders_sl = api_call('get_open_orders', [symbol, sl_orderId])
                r_get_open_orders_sl_flg = True if r_get_open_orders_sl else False
                r_query_sl = api_call('query_order', [symbol, sl_orderId])

                if r_get_open_orders_tp_flg is True and r_get_open_orders_sl_flg is True and r_query_tp[
                    'status'] == 'NEW' and r_query_sl['status'] == 'NEW':
                    # case general TP
                    pass

                elif r_get_open_orders_tp_flg is True and r_get_open_orders_sl_flg is True and r_query_tp[
                    'status'] == 'PARTIALLY_FILLED' and r_query_sl['status'] == 'NEW':
                    # case wait to TP full filled or SL filled
                    logger.info(symbol + ' IN TPTP PARTIALLY_FILLED and NEW OooxxxTTTTTPPPPPPxxxO')
                    pass

                # AUTO
                elif r_get_open_orders_tp_flg is False and r_get_open_orders_sl_flg is True and r_query_tp[
                    'status'] == 'FILLED' and r_query_sl['status'] == 'NEW':
                    cancel_batch_order(symbol, [sl_orderId], 'AUTO TP FILLED, REMAIN SL CLEAR')
                    t_info = update_trade_info(t_info, True, False, o_his, symbol, int(et_orderId))
                    o_his = update_history_status(o_his, symbol, et_orderId, 'WIN')  # AUTO TP FILLED

                    df_t, realizedPnl_tot, commission_tot = get_account_trades(symbol, et_orderId, None, tp_orderId)
                    logger.info(str([symbol, int(et_orderId), df_t, realizedPnl_tot, commission_tot]))

                elif r_query_tp['status'] == 'FILLED' and r_query_sl['status'] == 'EXPIRED':
                    t_info = update_trade_info(t_info, True, False, o_his, symbol, int(et_orderId))
                    o_his = update_history_status(o_his, symbol, et_orderId, 'WIN')  # AUTO TP FILLED
                    df_t, realizedPnl_tot, commission_tot = get_account_trades(symbol, et_orderId, None, tp_orderId)
                    logger.info(str([symbol, int(et_orderId), df_t, realizedPnl_tot, commission_tot]))

                elif r_get_open_orders_tp_flg is False and r_get_open_orders_sl_flg is False and r_query_tp[
                    'status'] == 'EXPIRED' and r_query_sl['status'] == 'FILLED':
                    # cancel_batch_order(symbol, [tp_orderId], 'AUTO SL FILLED, REMAIN TP CLEAR')
                    t_info = update_trade_info(t_info, False, True, o_his, symbol, int(et_orderId))
                    o_his = update_history_status(o_his, symbol, et_orderId, 'LOSE')  # AUTO SL FILLED
                    df_t, realizedPnl_tot, commission_tot = get_account_trades(symbol, et_orderId, sl_orderId, None)
                    logger.info(str([symbol, int(et_orderId), df_t, realizedPnl_tot, commission_tot]))

                elif r_get_open_orders_tp_flg is False and r_get_open_orders_sl_flg is False and r_query_tp[
                    'status'] == 'EXPIRED' and r_query_sl['status'] == 'EXPIRED':
                    logger.info('IN TPTP EXPIRED EXPIRED: %s %s %s %s %s ' % (
                        symbol, str(r_get_open_orders_tp_flg), str(r_get_open_orders_sl_flg), str(r_query_tp['status']),
                        str(r_query_sl['status'])))

                    t_info = update_trade_info(t_info, False, True, o_his, symbol, int(et_orderId))
                    o_his = update_history_status(o_his, symbol, et_orderId, 'LOSE')  # AUTO SL FILLED
                    df_t, realizedPnl_tot, commission_tot = get_account_trades(symbol, et_orderId, sl_orderId, None)
                    logger.info(str([symbol, int(et_orderId), df_t, realizedPnl_tot, commission_tot]))


                # FORCE
                elif r_get_open_orders_tp_flg is False and r_get_open_orders_sl_flg is False and r_query_tp[
                    'status'] == 'FILLED' and r_query_sl['status'] == 'FILLED':
                    logger.info('IN TPTP FILLED FILLED: %s %s %s %s %s ' % (
                        symbol, str(r_get_open_orders_tp_flg), str(r_get_open_orders_sl_flg), str(r_query_tp['status']),
                        str(r_query_sl['status'])))
                    o_his = update_history_status(o_his, symbol, et_orderId, 'FORCE')  # AUTO TP FILLED  # TODO check win or lose

                elif r_get_open_orders_tp_flg is False and r_get_open_orders_sl_flg is True and r_query_tp[
                    'status'] == 'NEW' and r_query_sl['status'] == 'NEW':
                    logger.info('IN TP: %s %s %s %s %s ' % (
                        symbol, str(r_get_open_orders_tp_flg), str(r_get_open_orders_sl_flg), str(r_query_tp['status']),
                        str(r_query_sl['status'])))
                    cancel_batch_order(symbol, [tp_orderId], 'FORCE MARKET CLICK, REMAIN TP CLEAR')
                    cancel_batch_order(symbol, [sl_orderId], 'FORCE MARKET CLICK, REMAIN SL CLEAR')
                    o_his = update_history_status(o_his, symbol, et_orderId, 'FORCE')  # TODO check win or lose
                elif r_get_open_orders_tp_flg is False and r_get_open_orders_sl_flg is True and r_query_tp[
                    'status'] == 'EXPIRED' and r_query_sl['status'] == 'NEW':
                    cancel_batch_order(symbol, [sl_orderId], 'FORCE MARKET CLICK AND TIME PASSED, REMAIN SL CLEAR')
                    o_his = update_history_status(o_his, symbol, et_orderId, 'FORCE')  # TODO check win or lose
                elif r_get_open_orders_tp_flg is False and r_get_open_orders_sl_flg is True and r_query_tp[
                    'status'] == 'CANCELED' and r_query_sl['status'] == 'NEW':
                    close_position_by_symbol(symbol, quantity, longshort, et_orderId)  # FORCE TP CLICK, REMAIN SL CLEAR
                    cancel_batch_order(symbol, [sl_orderId], 'FORCE TP CLICK, REMAIN SL CLEAR')
                    o_his = update_history_status(o_his, symbol, et_orderId, 'FORCE')  # TODO check win or lose

                elif r_get_open_orders_tp_flg is False and r_get_open_orders_sl_flg is False and r_query_tp[
                    'status'] == 'CANCELED' and r_query_sl['status'] == 'CANCELED':
                    close_position_by_symbol(symbol, quantity, longshort, et_orderId)  # FORCE TP and SL CLICK , CACEL POSITION BY FORCE
                    logger.info('IN TPTP CANCELED CANCELED : %s %s %s %s %s ' % (
                        symbol, str(r_get_open_orders_tp_flg), str(r_get_open_orders_sl_flg), str(r_query_tp['status']),
                        str(r_query_sl['status'])))
                    o_his = update_history_status(o_his, symbol, et_orderId, 'FORCE')  # TODO check win or lose

                elif r_get_open_orders_tp_flg is True and r_get_open_orders_sl_flg is True and r_query_tp[
                    'status'] == 'NEW' and r_query_sl['status'] == 'CANCELED':
                    close_position_by_symbol(symbol, quantity, longshort, et_orderId)  # FORCE SL, BEFORE CHECK TP CHEKER
                    cancel_batch_order(symbol, [tp_orderId], 'FORCE SL CLICK, REMAIN SL CLEAR')
                    o_his = update_history_status(o_his, symbol, et_orderId, 'FORCE')  # TODO check win or lose
                elif r_get_open_orders_tp_flg is True and r_get_open_orders_sl_flg is False and r_query_tp[
                    'status'] == 'NEW' and r_query_sl['status'] == 'CANCELED':
                    close_position_by_symbol(symbol, quantity, longshort, et_orderId)  # FORCE SL AFTER CHECK TP CHEKER
                    cancel_batch_order(symbol, [tp_orderId], 'FORCE SL CLICK, REMAIN SL CLEAR')
                    o_his = update_history_status(o_his, symbol, et_orderId, 'FORCE')  # TODO check win or lose

                else:
                    logger.info(symbol + ' IN TPTP OooooooooooOOOOOoooOOOOOTTTTTPPPPPPoooOOOO')
                    logger.info('IN TPTP: %s %s %s %s %s ' % (
                        symbol, str(r_get_open_orders_tp_flg), str(r_get_open_orders_sl_flg), str(r_query_tp['status']),
                        str(r_query_sl['status'])))
    return t_info, o_his


def get_symbols_in_order_position(o_his, t_mode):
    symbols_filter = list()
    if o_his:
        symbols_filter = [x['symbol'] for x in o_his if x['status'] in ['ETSL', 'TP']
                          and x['trade_mode'] == t_mode
                          ]
        symbols_filter = list(set(symbols_filter))
    return symbols_filter


def single(symbol_list, t_mode, t_info, o_his, i):
    roof_cnt = i
    logger.info(f'{roof_cnt} in monitor_wave_and_action: {time.strftime("%H:%M:%S")}')
    for symbol in symbol_list:
        for tf in timeframe:
            try:
                t_info, o_his = monitor_wave_and_action(symbol, tf, t_mode, t_info, o_his, i=i)
            except Exception as e:
                print('monitor_wave_and_action: %s' % str(e))
                logger.error('monitor_wave_and_action: %s' % str(e))

        if t_mode in ['REAL']:
            if roof_cnt % 20 == 0:
                logger.info(f'{roof_cnt} in monitor_history_and_action: {time.strftime("%H:%M:%S")}')
                try:
                    symbol_l = get_symbols_in_order_position(o_his, trade_mode)
                    for sym in symbol_l:
                        t_info, o_his = monitor_history_and_action(sym, t_mode, t_info, o_his)
                except Exception as e:
                    print('monitor_history_and_action: %s' % str(e))
                    logger.error('monitor_history_and_action: %s' % str(e))
        roof_cnt += 1
    return t_info, o_his


def set_max_leverage_margin_type_all_symbol(symbol_list):
    logger.info('set_max_leverage_margin_type_all_symbol start')
    for symbol in symbol_list:
        try:
            r = api_call('leverage_brackets', [symbol])
            time.sleep(0.2)
            logger.info(str(r))

            if r:
                max_leverage = r[0]['brackets'][0]['initialLeverage']
                if max_leverage:
                    rt_c = api_call('change_leverage', [symbol, max_leverage])

                margin_type = 'CROSSED'
                rt_m = api_call('change_margin_type', [symbol, margin_type])
                logger.info(str(rt_c) + str(rt_m))
        except ClientError as error:
            logger.error(
                "Found set_maxleverage_allsymbol error. status: {}, error code: {}, error message: {}".format(
                    error.status_code, error.error_code, error.error_message
                )
            )
    time.sleep(2)
    logger.info('set_max_leverage_margin_type_all_symbol done')


def cancel_all_positions():
    positions = api_call('account', [])['positions']

    result_position_filtered = [x for x in positions if x['entryPrice'] != '0.0']
    print(result_position_filtered)
    for p in result_position_filtered:
        time.sleep(0.5)
        s = p['symbol']
        longshort = True if p['positionSide'] == 'LONG' else False
        quantity = str(abs(float(p['positionAmt'])))
        order_market = api_call('new_order',
                                [s, "SELL" if longshort else "BUY",
                                 "LONG" if longshort else "SHORT", "MARKET",
                                 quantity, "tp_" + str(1010101010)])
        logging.info('cancel_all_positions %s: %s ' % (s, str(order_market)))


def cancel_all_open_orders(symbol_list):
    for symbol in symbol_list:
        try:
            time.sleep(0.5)
            all_orders = api_call('get_all_orders', [symbol])
            newes = [x for x in all_orders if (x['status'] == 'NEW')]
            if len(newes) > 0:
                time.sleep(0.5)
                response = api_call('cancel_open_orders', [symbol])
                logging.info('cancel_all_open_orders %s: %s ' % (symbol, str(response)))
        except ClientError as error:
            logging.error(
                "Found error. status: {}, error code: {}, error message: {}".format(
                    error.status_code, error.error_code, error.error_message
                )
            )


def cancel_all_closes():
    try:
        if init_running_trade:
            cancel_all_positions()
            cancel_all_open_orders(get_symbols())
    except ClientError as error:
        logging.error(
            "Found error. status: {}, error code: {}, error message: {}".format(
                error.status_code, error.error_code, error.error_message
            )
        )
    except Exception as e:
        logging.error('cancel_all_closes Exception e:' + str(e))


def sendMessage(update, context, msg=None, photo=None, corrector=True):
    try:
        if photo:
            context.bot.send_photo(
                chat_id=update.effective_chat.id,
                caption=msg,
                photo=open(photo, 'rb'))
        else:
            if msg and corrector:
                # Omit reserved characters
                exceptions = ['`']
                reserved = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+',
                            '-', '=', '|', '{', '}', '.', '!']
                msg = ''.join(['\\' + s if s in reserved and s not in exceptions
                               else s for s in msg])

            context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=msg,
                parse_mode='MarkdownV2')

    except Exception as e:
        print('Error on sendMessage:', e)


def messageListener(update, context):
    if context.user_data['access']:
        try:
            msg = update.message.text.lstrip().lower()

            if msg:
                sendMessage(update, context, msg)
            else:
                sendMessage(update, context, 'Sorry, I did not understand that')

        except Exception as e:
            print('Error on messageListener:', e)


def message_bot_main(botfather_token):
    try:
        updater = Updater(token=botfather_token, use_context=True)
        dispatcher = updater.dispatcher

        # Set the commands that our bot will handle.
        dispatcher.add_handler(CommandHandler('start', start))
        dispatcher.add_handler(MessageHandler(Filters.text, messageListener))

        updater.start_polling()
        updater.idle()
    except Exception as e:
        logger.error('message_bot_main e: %s' % str(e))


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


def get_mdd_1(prices: pd.Series) -> float:
    peak = np.maximum.accumulate(prices)  # 현시점까지의 고점
    trough = np.minimum.accumulate(prices)  # 현시점까지의 저점
    dd = (trough - peak) / peak  # 낙폭
    mdd = min(dd)  # 낙폭 중 가장 큰 값, 즉 최대낙폭
    return mdd


if __name__ == '__main__':
    import time
    from apscheduler.schedulers.blocking import BlockingScheduler

    sched = BlockingScheduler()  # https://hello-bryan.tistory.com/216
    print("""

     _____             _ _              ______       _
    |_   _|           | (_)             | ___ \     | |
      | |_ __ __ _  __| |_ _ __   __ _  | |_/ / ___ | |_
      | | '__/ _` |/ _` | | '_ \ / _` | | ___ \/ _ \| __|
      | | | | (_| | (_| | | | | | (_| | | |_/ / (_) | |_
      \_/_|  \__,_|\__,_|_|_| |_|\__, | \____/ \___/ \__| %s %s %s
                                  __/ |
                                 |___/
    %s  
    """ % (version, descrition, trade_mode, trade_mode * 16))

    print_condition()
    logger.info(trade_mode * 16)
    message_bot_main(botfather_token)

    if reset_leverage:
        set_max_leverage_margin_type_all_symbol(symbols_binance_futures)
    start = time.perf_counter()
    cancel_all_closes()
    symbols = get_symbols()
    logger.info('PID:' + str(os.getpid()))
    logger.info('seq:' + seq)

    stats_history = []
    order_history = []
    asset_history = []
    trade_count = []
    fee_history = []
    pnl_history = []
    wavepattern_history = []
    trade_info = [stats_history, order_history, asset_history, trade_count, fee_history, pnl_history,
                  wavepattern_history]

    if trade_mode in ['REAL', 'PAPER']:
        i = 1
        while True:
            # if i % 10 == 1:
            logger.info(f'{i} start: {time.strftime("%H:%M:%S")}')
            trade_info, open_order_history = single(symbols, trade_mode, trade_info, open_order_history, i)
            i += 1
    elif trade_mode in ['BACKTEST']:
        mdd = None
        r = range(period_days_ago, period_days_ago_till, -1 * period_interval)
        for i in r:
            asset_history_pre = trade_info[2][-1] if trade_info[2] else seed
            trade_info, open_order_history = single(symbols, trade_mode, trade_info, open_order_history, i)
            asset_history_last = trade_info[2]
            if asset_history_last:
                mdd1 = get_mdd_1(asset_history_last)
                mdd2 = get_mdd(asset_history_last)
                print(str(i) + '/' + str(len(r)) + ' now asset: ' + str(asset_history_last[-1]) + ' | ' + str(len(trade_count)) + ' | pre seed: ' + str(asset_history_pre) + ' | MDD1' + str(mdd1) + ' | MDD2' + str(mdd2))
                logger.info(str(i) + '/' + str(len(r)) + ' now asset: ' + str(asset_history_last[-1]) + ' | ' + str(len(trade_count)) + ' | pre seed: ' + str(asset_history_pre) + ' | MDD1' + str(mdd1) + ' | MDD2' + str(mdd2))
            else:
                print(str(i) + '/' + str(len(r)) + ' now asset: ' + str(seed) + ' | ' + len(trade_count) + ' | pre seed: ' + str(seed))
                logger.info(str(i) + '/' + str(len(r)) + ' now asset: ' + str(seed) + ' | ' + len(trade_count) + ' | pre seed: ' + str(seed))

        logger.info('============ %s stat.==========' % str(i))
        winrate_l = list(map(lambda i: 1 if i > 0 else 0, pnl_history))
        meanaverage = None
        if len(asset_history) > 0:
            meanaverage = round((sum(asset_history) / len(asset_history)), 2)
        roundcount = len(trade_count)
        winrate = None
        if len(winrate_l) > 0:
            winrate = str(round((sum(winrate_l)) / len(winrate_l) * 100, 2))
        logger.info('round r: %s' % roundcount)
        logger.info('round winrate_l: %s' % str(winrate_l))
        logger.info('round roundcount: %s' % roundcount)
        logger.info('round winrate: %s' % winrate)
        logger.info('round meanaverage: %s' % str(meanaverage))
        logger.info('round MDD: %s' % str(mdd))
        logger.info('round total gains: %s' % str(trade_info[-2][-1] if trade_info[2] else 0))
        logger.info('============ %s End All==========' % str(i))
        logger.info(f'Finished wave_analyzer in {round(time.perf_counter() - start, 2)} second(s)')

    print_condition()
    logger.info("good luck done!!")
