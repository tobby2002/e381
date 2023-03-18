from __future__ import annotations
from models.WavePattern import WavePattern
from models.WaveRules import Impulse, DownImpulse
from models.WaveAnalyzer import WaveAnalyzer
from models.WaveOptions import WaveOptionsGenerator5
from models.helpers import plot_pattern_m
import datetime as dt
import pandas as pd
import numpy as np
from ratelimit import limits, sleep_and_retry, RateLimitException
from backoff import on_exception, expo
from binancefutures.um_futures import UMFutures
from binancefutures.error import ClientError
from binance.helpers import round_step_size

import shutup; shutup.please()
import json
import os
import random
import pickle
import math
import logging
import time


with open('w045config.json', 'r') as f:
    config = json.load(f)

version = config['default']['version']
descrition = config['default']['descrition']
exchange = config['default']['exchange']
exchange_symbol = config['default']['exchange_symbol']
futures = config['default']['futures']
type = config['default']['type']
maxleverage = config['default']['maxleverage']
qtyrate = config['default']['qtyrate']
walletrate = config['default']['walletrate']


high_target = config['default']['high_target']
low_target = config['default']['low_target']
low_target_w2 = config['default']['low_target_w2']

seed = config['default']['seed']
fee = config['default']['fee']
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

seq = dt.datetime.now().strftime("%Y%m%d_%H%M%S") + str([timeframe, fcnt, period_days_ago, period_days_ago_till])

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')  # ('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
file_handler = logging.FileHandler('logger_%s.log' % seq)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


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
    logger.info('walletrate:%s' % str(walletrate))
    logger.info('seed:%s' % str(seed))
    logger.info('fee:%s%%' % str(fee*100))
    logger.info('fee_slippage:%s%%' % str(round(fee_slippage*100, 4)))
    logger.info('timeframe: %s' % timeframe)
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
    logger.info('fcnt: %s' % fcnt)
    logger.info('loop_count: %s' % loop_count)
    logger.info('symbol_random: %s' % symbol_random)
    logger.info('symbol_last: %s' % symbol_last)
    logger.info('symbol_length: %s' % symbol_length)

    start_dt = str((pd.to_datetime('today') - pd.Timedelta(str(period_days_ago) + ' days')).date())
    end_dt = str((pd.to_datetime('today') - pd.Timedelta(str(period_days_ago_till) + ' days')).date())
    logger.info('period: %s ~ %s' % (start_dt, end_dt))
    logger.info('up_to_count: %s' % up_to_count)
    logger.info('condi_same_date: %s' % condi_same_date)
    logger.info('et_zone_rate: %s' % et_zone_rate)
    logger.info('o_fibo: %s' % o_fibo)
    logger.info('h_fibo: %s' % h_fibo)
    logger.info('l_fibo: %s' % l_fibo)

    logger.info('entry_fibo: %s' % entry_fibo)
    logger.info('target_fibo: %s' % target_fibo)
    logger.info('sl_fibo: %s' % sl_fibo)

    logger.info('intersect_idx: %s' % intersect_idx)
    logger.info('plotview: %s' % plotview)
    logger.info('printout: %s' % printout)
    logger.info('init_running_trade: %s' % init_running_trade)
    logger.info('reset_leverage: %s' % reset_leverage)
    logger.info('-------------------------------')


um_futures_client = UMFutures(key=futures_secret_key, secret=futures_secret_value)

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
            response = um_futures_client.new_batch_order(params)
        elif method == 'query_order':
            symbol = arglist[0]
            orderId = arglist[1]
            response = um_futures_client.query_order(symbol=symbol, orderId=orderId, recvWindow=6000)
        elif method == 'get_open_orders':
            symbol = arglist[0]
            orderId = arglist[1]
            try:
                response = um_futures_client.get_open_orders(symbol=symbol, orderId=orderId, recvWindow=6000)
            except ClientError as error:
                if error.status_code == 400 and error.error_code == -2013:  # error message: Order does not exist
                    return None
        elif method == 'get_all_orders':
            symbol = arglist[0]
            response = um_futures_client.get_all_orders(symbol=symbol, recvWindow=6000)
        elif method == 'get_position_risk':
            symbol = arglist[0]
            response = um_futures_client.get_position_risk(symbol=symbol, recvWindow=6000)
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
open_order_history_seq= 'history_' + seq + '.pkl'
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

    if symbol_last:
        symbols = symbols[symbol_last:]
    if symbol_length:
        symbols = symbols[:symbol_length]
    if symbol_random:
        symbols = random.sample(symbols, len(symbols))
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
            cost = round_step_size(price, float(a)) # convert tick size from string to float, insert in helper func with cost
            # 아래는 시장가의 비용 및 sleepage 를 보고 나중에 추가 또는 삭제 검토요
            # cost = cost - float(a) if longshort else cost + float(a)
            return cost


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


def get_i_r(list, key, value):
    r = [[i, x] for i, x in enumerate(list) if x[key] == value]
    if len(r) == 1:
        return r[0][0], r[0][1]
    return None, None


def get_trade_prices(symbol, w):
    w0 = w.values[0]
    w1 = w.values[1]
    w2 = w.values[3]
    w3 = w.values[5]
    w4 = w.values[7]
    w5 = w.values[9]

    tp_price = w5
    et_price = w4
    sl_price = w0

    et_price = set_price(symbol, et_price, None)
    sl_price = set_price(symbol, sl_price, None)
    tp_price = set_price(symbol, tp_price, None)
    return et_price, sl_price, tp_price


def new_et_order(symbol, tf, fc, longshort, et_price, sl_price, tp_price, quantity, wavepattern):
    params = [
        {
            "symbol": symbol,
            "side": "BUY" if longshort else "SELL",
            "type": "LIMIT",
            "positionSide": "LONG" if longshort else "SHORT",
            "quantity": str(float(quantity)),
            "timeInForce": "GTC",
            "price": str(et_price),
            "newClientOrderId": 'et_' + str(tf) + '_' + str(fc) + '_' + str(sl_price) + '_' + str(tp_price),
        }
    ]
    r = api_call('new_batch_order', [params])
    if r:
        order_et = r[0]
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
                "newClientOrderId": "sl_" + str(tf) + '_' + str(fc) + '_' + str(order_et['orderId']),
            }
        ]
        r = api_call('new_batch_order', [params])
        if r:
            order_sl = r[0]
            add_etsl_history(open_order_history, symbol, tf, fc, longshort, wavepattern, et_price,
                            sl_price, tp_price, quantity, order_et['orderId'], order_sl['orderId'], order_et,
                            order_sl)
            return True
        else:
            r = api_call('cancel_batch_order', [symbol, [order_et['orderId']], []])
            if r:
                order_tp = r[0]
                logger.info(symbol + ' _CANCEL ET: ' + str(order_tp))
    else:
        logger.info(symbol + ' _FAIL ET' + str((symbol, tf, fc, longshort, et_price, sl_price, tp_price, quantity, wavepattern)))
    return False


def new_tp_order(symbol, tf, fc, longshort, tp_price, quantity, et_orderId):
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
            "newClientOrderId": 'tp_' + str(tf) + '_' + str(fc) + '_' + str(et_orderId)
        }
    ]
    result_tp = api_call('new_batch_order', [params])
    if result_tp:
        order_tp = result_tp[0]
        add_tp_history(open_order_history, symbol, et_orderId, order_tp['orderId'], order_tp)
        return True
    return False


def cancel_batch_order(symbol, order_id_l, desc):
    orderIdList = order_id_l
    origClientOrderIdList = []
    response = api_call('cancel_batch_order', [symbol, orderIdList, origClientOrderIdList])
    cnt_success = 0
    if response:
        for rs in response:
            try:
                if rs['orderId']:
                    logger.info(symbol + (' _CANCELBATCH success, %s : ' % desc) + str(rs))
                    cnt_success += 1
            except:
                if rs['code']:
                    logger.info(symbol + (' _CANCEBATCH error, %s : ' % desc) + str(rs))
        if cnt_success == len(response):
            return True
    return False


def c_in_no_double_ordering(symbol, tf, fc, w):
    #####  이중 new limit order 방지 로직 start #####
    history_new = [x for x in open_order_history if
                   (x['symbol'] == symbol and x['status'] == 'ETSL'
                    and x['timeframe'] == tf and x['fcnt'] == fc)]

    et_price, sl_price, tp_price = get_trade_prices(symbol, w)

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

                if float(r_query_limit['price']) == float(et_price) or float(r_query_limit['clientOrderId'].split('_')[4]) == float(tp_price):
                    # and r_query_limit['newClientOrderId'] == tp_price:  # when limit order, set newClientOrderId": str(tp_price), therefore ..
                    return False
    #####  이중 new limit order 방지 로직 start #####
    return True


def c_real_condition_by_fractal_index(df, fcnt, w, idx):  # real condititon by fractal index
    real_condititon1 = True if (fcnt/2 < (w.idx_end - w.idx_start)) and w.idx_start == idx else False
    real_condititon2 = True if df.iloc[idx + int(fcnt/2), 0] < (w.dates[-1]) else False
    if not (real_condititon1 and real_condititon2):
        if printout: print('real_condititon ')
        return False
    return True


def c_active_no_empty(df, w):
    df_active = df.loc[df['Date'] > w.dates[-1]]  # 2023.3.13 after liqu  # df[w.idx_end + 1:]
    if df_active.empty:
        return False
    return True


def c_active_next_bean_ok(df, symbol, longshort, w):
    et_price, sl_price, tp_price = get_trade_prices(symbol, w)
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
    i = df_active.size
    c_beyond_idx_width = True if (i / w_idx_width >= c_time_beyond_rate) else False
    if c_beyond_idx_width and c_time_beyond_flg:
        return False
    return True


def c_currentprice_in_zone(symbol, longshort, w):
    w0 = w.values[0]
    w1 = w.values[1]
    w2 = w.values[3]
    w3 = w.values[5]
    w4 = w.values[7]
    w5 = w.values[9]

    tp_price = w5
    et_price = w4
    sl_price = w0
    between_entry_target = et_price + abs(tp_price - et_price)*(et_zone_rate) if longshort else et_price - abs(tp_price - et_price)*(et_zone_rate)
    current_price = float(api_call('ticker_price', [symbol])['price'])
    c_current_price_in_zone = (current_price > et_price and current_price < between_entry_target) \
                            if longshort else \
                            (current_price < et_price and current_price > between_entry_target)

    if not c_current_price_in_zone:
        return False
    return True

def c_currentprice_in_zone_by_prices(symbol, longshort, et_price, tp_price):
    between_entry_target = et_price + abs(tp_price - et_price)*(et_zone_rate) if longshort else et_price - abs(tp_price - et_price)*(et_zone_rate)
    current_price = float(api_call('ticker_price', [symbol])['price'])
    c_current_price_in_zone = (current_price > et_price and current_price < between_entry_target) \
                            if longshort else \
                            (current_price < et_price and current_price > between_entry_target)

    if not c_current_price_in_zone:
        return False
    return True


def c_active_in_zone(df, symbol, longshort, w):
    w_start_price = w.values[0]  # wave1
    w_end_price = w.values[-1]  # wave5
    height_price = abs(w_end_price - w_start_price)
    o_fibo_value = height_price * o_fibo / 100 if o_fibo else 0

    et_price, sl_price, tp_price = get_trade_prices(symbol, w)

    df_active = df.loc[df['Date'] > w.dates[-1]]  # 2023.3.13 after liqu  # df[w.idx_end + 1:]

    if not df_active.empty:
        try:
            active_max_value = max(df_active.High.tolist(), default=tp_price)
            active_min_value = min(df_active.Low.tolist(), default=tp_price)
        except Exception as e:
            logger.error('active_max_value:' + str(e))

        c_active_min_max_in_zone = (active_min_value > et_price and active_max_value < (w_end_price + o_fibo_value)) \
                                if longshort else \
                            (active_max_value < et_price and active_min_value > (w_end_price - o_fibo_value))

        if not c_active_min_max_in_zone:
            return False
    return True


def c_in_no_risk(symbol, w):
    et_price, sl_price, tp_price = get_trade_prices(symbol, w)
    if c_risk_beyond_flg:
        pnl_percent_sl = (abs(et_price - sl_price) / et_price) * qtyrate
        if pnl_percent_sl >= c_risk_beyond_max:  # decrease max sl rate   0.1 = 10%
            # logger.info(symbol + ' _c_risk_beyond_max : ' + str(pnl_percent_sl))
            return False

        pnl_percent_tp = (abs(tp_price - et_price) / et_price) * qtyrate
        if pnl_percent_tp <= c_risk_beyond_min:  # reduce low tp rate  0.005 = 0.5%
            # logger.info(symbol + ' _c_risk_beyond_min : ' + str(pnl_percent_tp))
            return False

    if et_price == sl_price:
        logger.info(symbol + ' _et_price == sl_price')
        return False
    return True


def check_cons_for_new_etsl_order(df, symbol, tf, fc, longshort, w, idx):
    if not c_real_condition_by_fractal_index(df, fc, w, idx):
        return False
    if not c_active_no_empty(df, w):
        return False
    if not c_active_next_bean_ok(df, symbol, longshort, w):
        return False
    if not c_active_in_zone(df, symbol, longshort, w):
        return False
    if not c_active_in_time(df, w):
        return False
    if not c_currentprice_in_zone(symbol, longshort, w):
        return False
    if not c_in_no_risk(symbol, w):
        return False
    if not c_in_no_double_ordering(symbol, tf, fc, w):
        return False
    return True


def c_balance_and_calc_quanty(symbol):
    margin_available, available_balance, wallet_balance = my_available_balance(symbol, exchange_symbol)
    if not margin_available:
        logger.info('margin_available : False')
        logger.info('symbol:%s, available_balance:%s, wallet_balance:%s' % (symbol, str(available_balance), str(wallet_balance)))
        return False, None

    # max_quantity = float(available_balance) * int(leveragexxxxx) / current_price
    # quantity = max_quantity * qtyrate
    current_price = float(api_call('ticker_price', [symbol])['price'])
    quantity = wallet_balance * qtyrate / current_price
    step_size, minqty = get_quantity_step_size_minqty(symbol)
    quantity = format_value(quantity, step_size)

    if available_balance <= wallet_balance * walletrate:
        logger.info('available_balance <= wallet_balance * %s' % str(walletrate))
        logger.info('symbol:%s, available_balance:%s, wallet_balance:%s' % (symbol, str(available_balance), str(wallet_balance)))
        return False, None

    if float(quantity) < float(minqty):
        logger.info('float(quantity) < float(minqty)')
        logger.info('symbol:%s, quantity:%s, minqty:%s' % (symbol, str(quantity), str(minqty)))
        logger.info('symbol:%s, available_balance:%s, wallet_balance:%s' % (symbol, str(available_balance), str(wallet_balance)))
        return False, None

    if not quantity:
        logger.info('quantity:' + str(quantity))
        logger.info('available_balance:%s, wallet_balance:%s' % (str(available_balance), str(wallet_balance)))
        return False, None
    return True, quantity


def rename_symbol(s):
    if s[-4:] == 'USDT':
        return s.replace('USDT', '/USDT')
    if s[-4:] == 'BUSD':
        return s.replace('BUSD', '/BUSD')


def c_check_valid_wave_in_history(open_order_history, symbol, tf, fc, wavepattern):
    if open_order_history:
        open_order_this_symbol = [x for x in open_order_history
                                    if x['symbol'] == symbol
                                    and x['timeframe'] == tf
                                    and x['fcnt'] == fc
                                    and x['status'] in ['ETSL', 'TP', 'WIN', 'LOSE']
                                    and x['wavepattern'].dates == wavepattern.dates
                                    and x['wavepattern'].values == wavepattern.values
                                  ]
        if open_order_this_symbol:
            return False
    return True


# def c_check_valid_etslwave_in_history(open_order_history, symbol, tf, wavepattern):
#     if open_order_history:
#         open_order_this_symbol = [x for x in open_order_history
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


def moniwave_and_action(symbol, tf):
    timeunit = 'm'
    bin_size = str(tf) + timeunit
    delta = (4 * fcnt[-1] + 1)
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
                                if c_check_valid_wave_in_history(open_order_history, symbol, tf, fc, wavepattern):
                                    if check_cons_for_new_etsl_order(df_all, symbol, tf, fc, longshort, wavepattern, i):
                                        available, quantity = c_balance_and_calc_quanty(symbol)
                                        if available:
                                            et_price, sl_price, tp_price = get_trade_prices(symbol, wavepattern)
                                            r = new_et_order(symbol, tf, fc, longshort, et_price, sl_price, tp_price, quantity, wavepattern)
                                            if r:
                                                if plotview:
                                                    plot_pattern_m(df=df,
                                                                   wave_pattern=[[1, wavepattern.dates[0], id(wavepattern), wavepattern]],
                                                                   df_lows_plot=df_lows_plot, df_highs_plot=df_highs_plot,
                                                                   trade_info=None, title=str(symbol + ' %s ' % str('LONG' if longshort else 'SHORT') + '%sm %s' % (tf, fc) +', ET: ' + str(et_price)))

    return


def add_etsl_history(open_order_history, symbol, tf, fc, longshort, w, et_price, sl_price, tp_price, quantity, et_orderId, sl_orderId, order_et, order_sl):
    now = dt.datetime.now()
    history = {
        'id': et_orderId,
        'create_datetime': now,
        'status': 'ETSL',
        'symbol': symbol,
        'timeframe': tf,
        'fcnt': fc,
        'longshort': longshort,
        'side': 'LONG' if longshort else 'SHORT',
        'wavepattern': w,
        'et_price': et_price,
        'sl_price': sl_price,
        'tp_price': tp_price,
        'quantity': quantity,
        'et_orderId': et_orderId,
        'sl_orderId': sl_orderId,
        'tp_orderId': None,
        'etsl_datetime': now,
        'tp_datetime': None,
        'update_datetime': None,
        'et_data': order_et,
        'sl_data': order_sl,
        'tp_data': None
    }
    open_order_history.append(history)
    logger.info(symbol + ' _HS add_etsl_history %s:' % 'ETSL' + str(history))
    dump_history_pkl()


def add_tp_history(open_order_history, symbol, et_orderId, tp_orderId, tp_data):
    if open_order_history:
        history_idx, history_id = get_i_r(open_order_history, 'id', et_orderId)
        history_id['status'] = 'TP'
        history_id['tp_orderId'] = tp_orderId
        history_id['tp_datetime'] = dt.datetime.now()
        history_id['update_datetime'] = dt.datetime.now()
        history_id['tp_data'] = tp_data
        open_order_history[history_idx] = history_id  # replace history
        logger.info(symbol + ' _HS add_tp_history %s:' % str(tp_data))
        dump_history_pkl()


def update_history_status(open_order_history, symbol, h_id, new_status):
    if open_order_history:
        history_idx, history_id = get_i_r(open_order_history, 'id', h_id)
        history_id['status'] = new_status  # update new status
        history_id['update_datetime'] = dt.datetime.now()
        open_order_history[history_idx] = history_id  # replace history
        logger.info(symbol + ' _HS update_history_status %s:' % new_status)
        dump_history_pkl()


def delete_history_status(open_order_history, symbol, h_id, event):
    if open_order_history:
        history_idx, history_id = get_i_r(open_order_history, 'id', h_id)
        open_order_history.pop(history_idx)
        logger.info(symbol + ' _HS delete_history_status %s' % event)
        dump_history_pkl()


# def symbol_orders_df(symbol):
#     response = api_call('get_all_orders', [symbol])
#     if response:
#         df = pd.DataFrame.from_records(response)
#         df['sid'] = df.apply(lambda x: x['symbol'] + '_' + str(x['orderId']), axis=1)
#         df['time_dt'] = df.apply(
#             lambda x: str(dt.datetime.fromtimestamp(float(x['time']) / 1000).strftime('%Y-%m-%d %H:%M:%S')), axis=1)
#         df['updateTime_dt'] = df.apply(
#             lambda x: str(dt.datetime.fromtimestamp(float(x['updateTime']) / 1000).strftime('%Y-%m-%d %H:%M:%S')), axis=1)
#         df['date'] = df.apply(lambda x: str(dt.datetime.fromtimestamp(float(x['time']) / 1000).strftime('%Y-%m-%d')), axis=1)
#         df.sort_values(by='time_dt', ascending=False, inplace=True)  # https://sparkbyexamples.com/pandas/sort-pandas-dataframe-by-date/
#         try:
#             def f_clientOrderId(x):
#                 if x.split('_')[0] == 'tp':
#                     if len(x.split('_')) == 4:
#                         return x.split('_')[3]
#                     else:
#                         return x.split('_')[1]
#                 elif x.split('_')[0] == 'sl':
#                     if len(x.split('_')) == 4:
#                         return x.split('_')[3]
#                     else:
#                         return x.split('_')[1]
#                 elif x.split('_')[0] == 'limit':
#                     return x
#             df['gid'] = df['clientOrderId'].map(f_clientOrderId)
#             df['gid'] = df.apply(lambda x: str(x['orderId']) if x['clientOrderId'] == x['gid'] else x['gid'], axis=1)
#             df['gid'] = df['gid'].apply(str)
#             df.drop(df[(df['gid'] == '1010101010')].index, inplace=True)
#             df.drop(df[(df['gid'] == '10101010101')].index, inplace=True)
#             df.drop(df[(df['gid'] == 'None')].index, inplace=True)
#             df['gid'] = df['gid'].apply(int)
#
#             def f_action(x):
#                 if x.split('_')[0] == 'tp':
#                     return 'TP'
#                 elif x.split('_')[0] == 'sl':
#                     return 'SL'
#                 elif x.split('_')[0] == 'limit':
#                     return 'ET'
#             df['action'] = df['clientOrderId'].map(f_action)
#         except Exception as e:
#             print('xxxx' + str(e))
#         return df
#     return None
#
#
# def action_event(df):
#     glist = df['gid'].to_list()
#     glist = list(dict.fromkeys(glist))
#     for gid in glist:
#         df_et = df.query("gid == %s and action == 'ET'" % gid)
#         df_sl = df.query("gid == %s and action == 'SL'" % gid)
#         df_tp = df.query("gid == %s and action == 'TP'" % gid)
#         status_et = df_et['status'].iat[0] if not df_et.empty else ''
#         status_sl = df_sl['status'].iat[0] if not df_sl.empty else ''
#         status_tp = df_tp['status'].iat[0] if not df_tp.empty else ''
#
#         try:
#             if not df_et.empty:
#                 et_orderId = df_et['orderId'].iat[0]
#                 symbol = df_et['symbol'].iat[0]
#
#                 # case no event
#                 if status_et == 'CANCELED' and status_sl == 'CANCELED' and status_tp == '':
#                     pass
#                 elif status_et == 'CANCELED' and status_sl == 'EXPIRED' and status_tp == 'EXPIRED':
#                     pass
#                 elif status_et == 'CANCELED' and status_sl == 'EXPIRED' and status_tp == 'FILLED':
#                     pass
#                 elif status_et == 'CANCELED' and status_sl == '' and status_tp == '':
#                     # ??? need CHECK
#                     pass
#                 elif status_et == 'FILLED' and status_sl == 'FILLED' and status_tp == '':  # order (limit, sl) -> no tp order -> filled sl
#                     # ??? need HT LOG
#                     pass
#                 elif status_et == 'FILLED' and status_sl == 'FILLED' and status_tp == 'EXPIRED':
#                     pass
#                 elif status_et == 'FILLED' and status_sl == 'CANCELED' and status_tp == 'FILLED':
#                     # check history TP and update history xxxxxxx
#                     pass
#                 elif status_et == 'FILLED' and status_sl == 'CANCELED' and status_tp == 'EXPIRED':
#                     pass
#                 elif status_et == 'FILLED' and status_sl == 'CANCELED' and status_tp == 'CANCELED':
#                     pass
#                 elif status_et == 'FILLED' and status_sl == 'CANCELED' and status_tp == '':
#                     # ??? CHECK ???
#                     pass
#                 elif status_et == 'FILLED' and status_sl == 'EXPIRED':
#                     pass
#                 elif status_et == 'FILLED' and status_sl == 'EXPIRED' and status_tp == 'EXPIRED':
#                     pass
#
#
#                 # case of action event
#                 elif status_et == 'FILLED' and status_sl == 'NEW' and status_tp == '':
#                     if len(df_et['clientOrderId'].iat[0].split('_')) >= 5:
#                         tf = df_et['clientOrderId'].iat[0].split('_')[1]
#                         fc = df_et['clientOrderId'].iat[0].split('_')[2]
#                         tp_price = df_et['clientOrderId'].iat[0].split('_')[4]
#                         longshort = True if df_et['positionSide'].iat[0] == 'LONG' else False
#                         quantity = df_et['executedQty'].iat[0]
#                         new_tp_order(symbol, tf, fc, longshort, tp_price, quantity, et_orderId)
#
#                 elif status_et == 'PARTIALLY_FILLED' and status_sl == 'NEW':
#                     if len(df_et['clientOrderId'].iat[0].split('_')) >= 5:
#                         tf = df_et['clientOrderId'].iat[0].split('_')[1]
#                         fc = df_et['clientOrderId'].iat[0].split('_')[2]
#                         tp_price = df_et['clientOrderId'].iat[0].split('_')[4]
#                         longshort = True if df_et['positionSide'].iat[0] == 'LONG' else False
#                         quantity = df_et['executedQty'].iat[0]
#
#                         success_tp_order = new_tp_order(symbol, tf, fc, longshort, tp_price, quantity, et_orderId)
#                         if success_tp_order:
#                             cancel_batch_order(symbol, [int(et_orderId)], 'CANCEL PARTIAL')
#
#                 elif status_et == 'FILLED' and status_sl == 'NEW' and status_tp == 'NEW':
#                     # CHECK_ZONE WAIT OTHER xxxx nocase?
#                     pass
#                 elif status_et == 'NEW' and status_sl == 'NEW' and status_tp == '':
#                     r_get_open_orders_et = api_call('get_open_orders', [symbol, et_orderId])
#                     if r_get_open_orders_et:
#                         # CHECK_ZONE
#                         if len(df_et['clientOrderId'].iat[0].split('_')) >= 5:
#                             et_price = df_et['price'].iat[0]
#                             tp_price = df_et['clientOrderId'].iat[0].split('_')[4]
#                             longshort = True if df_et['positionSide'].iat[0] == 'LONG' else False
#                         if not c_currentprice_in_zone_by_prices(symbol, longshort, float(et_price), float(tp_price)):
#                             # CANCEL ETSL BY OUTZONE
#                             sl_orderId = df_sl['orderId'].iat[0]
#                             response_cancel = cancel_batch_order(symbol, [str(et_orderId), str(sl_orderId)], 'CANCEL NEW')
#                             if response_cancel:
#                                 update_history_status(open_order_history, symbol, et_orderId, 'CANCEL')
#                     else:
#                         r_get_open_orders_sl = api_call('get_open_orders', [symbol, sl_orderId])
#
#
#                 # case of CLOSE
#                 elif status_et == 'FILLED' and status_sl == 'FILLED' and status_tp == 'NEW':
#                     # CLOSE_TP
#                     # action_event.append([gid, df_et, df_sl, df_tp, ['CLOSE_TP']])
#                     pass
#
#                 elif status_et == 'FILLED' and status_sl == 'NEW' and status_tp == 'FILLED':
#                     # CLOSE_SL
#                     # action_event.append([gid, df_et, df_sl, df_tp, ['CLOSE_SL']])
#                     pass
#                 elif status_et == 'FILLED' and status_sl == 'CANCELED' and status_tp == 'NEW':
#                     # FORCE_CLOSE_POSITION
#                     # action_event.append([gid, df_et, df_sl, df_tp, ['FORCE_CLOSE_POSITION']])
#                     pass
#         except Exception as e:
#             print('action_event Exception : %s ' % str(e))
#
#
# def moniorders_and_action(symbol):
#     df = symbol_orders_df(symbol)
#     if df is not None and not df.empty:
#         action_event(df)


def monihistory_and_action(open_order_history, symbol):  # ETSL -> CANCEL        or       TP -> WIN or LOSE
    if open_order_history:
        history_new = [x for x in open_order_history if (x['symbol'] == symbol and x['status'] == 'ETSL')]
        if history_new:
            for new in history_new:
                et_orderId = new['et_orderId']
                sl_orderId = new['sl_orderId']
                et_price = new['et_price']
                tp_price = new['tp_price']
                tf = new['timeframe']
                longshort = new['longshort']
                side = new['side']
                quantity = new['quantity']
                fc = new['fcnt']
                try:
                    r_get_open_orders_et = api_call('get_open_orders', [symbol, et_orderId])
                except:
                    r_get_open_orders_et = None

                if r_get_open_orders_et:
                    if not c_currentprice_in_zone_by_prices(symbol, longshort, float(et_price), float(tp_price)):
                        # CANCEL_ETSL
                        response_cancel = cancel_batch_order(symbol, [str(et_orderId), str(sl_orderId)], 'CANCEL ETSL')
                        if response_cancel:
                            update_history_status(open_order_history, symbol, et_orderId, 'CANCEL')
                else:
                    try:
                        r_query_et = api_call('query_order', [symbol, et_orderId])
                    except:
                        r_query_et = None

                    try:
                        r_query_sl = api_call('query_order', [symbol, sl_orderId])
                    except:
                        r_query_sl = None

                    if r_query_et['status'] == 'FILLED' and r_query_sl['status'] == 'NEW':
                        try:
                            r_get_open_orders_sl = api_call('get_open_orders', [symbol, sl_orderId])
                        except:
                            r_get_open_orders_sl = None

                        if r_get_open_orders_sl:
                            # NEW_TP
                            new_tp_order(symbol, tf, fc, longshort, tp_price, quantity, et_orderId)

                        # try:
                        #     result_position = api_call('get_position_risk', [symbol])
                        # except:
                        #     result_position = None
                        # if result_position:
                        #     result_position_filtered = [x for x in result_position if x['entryPrice'] != '0.0']
                        #     if result_position_filtered:  # no tp and positioning
                        #         for p in result_position_filtered:
                        #             if side == p['positionSide'] and float(quantity) <= float(p['positionAmt']):
                        #                 try:
                        #                     r_get_open_orders_sl = api_call('get_open_orders', [symbol, sl_orderId])
                        #                 except:
                        #                     r_get_open_orders_sl = None
                        #
                        #                 if r_get_open_orders_sl:
                        #                     # NEW_TP
                        #                     new_tp_order(symbol, tf, fc, longshort, tp_price, quantity, et_orderId)

                    elif r_query_et['status'] == 'FILLED' and r_query_sl['status'] == 'FILLED':
                        update_history_status(open_order_history, symbol, et_orderId, 'LOSE')

        history_tp = [x for x in open_order_history if (x['symbol'] == symbol and x['status'] == 'TP')]
        if history_tp:
            for tp in history_tp:
                # POSI_CHECK
                et_orderId = tp['et_orderId']
                sl_orderId = tp['sl_orderId']
                tp_orderId = tp['tp_orderId']
                try:
                    r_get_open_orders_tp = api_call('get_open_orders', [symbol, tp_orderId])
                except:
                    r_get_open_orders_tp = None

                if not r_get_open_orders_tp:
                    # CASE WIN
                    response_cancel = cancel_batch_order(symbol, [sl_orderId], 'WIN')
                    if response_cancel:
                        update_history_status(open_order_history, symbol, et_orderId, 'WIN')
                else:
                    try:
                        r_get_open_orders_sl = api_call('get_open_orders', [symbol, sl_orderId])
                    except:
                        r_get_open_orders_sl = None

                    if not r_get_open_orders_sl:
                        # CASE LOSE
                        response_cancel = cancel_batch_order(symbol, [tp_orderId], 'LOSE')
                        if response_cancel:
                            update_history_status(open_order_history, symbol, et_orderId, 'LOSE')


def single(symbols, i, *args):
    for symbol in symbols:
        try:
            monihistory_and_action(open_order_history, symbol)
        except Exception as e:
            print('monihistory_and_action: %s' % str(e))

        for tf in timeframe:
            try:
                moniwave_and_action(symbol, tf)
            except Exception as e:
                print('moniwave_and_action: %s' % str(e))

        # try:
        #     moniorders_and_action(symbol)
        # except Exception as e:
        #     print('moniwave_and_action: %s' % str(e))


def set_maxleverage_allsymbol(symbols):
    logger.info('set  set_maxleverage_allsymbol start')
    for symbol in symbols:
        try:
            r = api_call('leverage_brackets', [symbol])
            max_leverage = r[0]['brackets'][0]['initialLeverage']
            time.sleep(0.1)
            rt = api_call('change_leverage', [symbol, max_leverage])

            logger.info(rt)
        except ClientError as error:
            logger.error(
                "Found set_maxleverage_allsymbol error. status: {}, error code: {}, error message: {}".format(
                    error.status_code, error.error_code, error.error_message
                )
            )
    time.sleep(2)
    logger.info('set_maxleverage_allsymbol done')


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


def cancel_all_open_orders(symbols):
    for s in symbols:
        try:
            time.sleep(0.5)
            all_orders = api_call('get_all_orders', [s])
            newes = [x for x in all_orders if (x['status'] == 'NEW')]
            if len(newes) > 0:
                time.sleep(0.5)
                response = api_call('cancel_open_orders', [s])
                logging.info('cancel_all_open_orders %s: %s ' % (s, str(response)))
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
            symbols = get_symbols()
            cancel_all_open_orders(symbols)

    except ClientError as error:
        logging.error(
            "Found error. status: {}, error code: {}, error message: {}".format(
                error.status_code, error.error_code, error.error_message
            )
        )
    except Exception as e:
        logging.error('cancel_all_closes Exception e:' + str(e))


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
    if reset_leverage:
        set_maxleverage_allsymbol(symbols_binance_futures)
    start = time.perf_counter()
    cancel_all_closes()
    symbols = get_symbols()
    i = 1
    logger.info('PID:' + str(os.getpid()))
    logger.info('seq:' + seq)
    while True:
        # if i % 10 == 1:
        logger.info(f'{i} start: {time.strftime("%H:%M:%S")}')
        single(symbols, i)
        i += 1
    print(f'Finished in {round(time.perf_counter() - start, 2)} second(s)')
    print_condition()
    print("good luck done!!")
