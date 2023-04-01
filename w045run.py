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
condi_same_date = config['default']['condi_same_date']
condi_compare_before_fractal = config['default']['condi_compare_before_fractal']
condi_compare_before_fractal_strait = config['default']['condi_compare_before_fractal_strait']
if condi_compare_before_fractal_strait:
    condi_compare_before_fractal_shift = 1
condi_compare_before_fractal_shift = config['default']['condi_compare_before_fractal_shift']
condi_compare_before_fractal_mode = config['default']['condi_compare_before_fractal_mode']
if not condi_compare_before_fractal:
    condi_compare_before_fractal_shift = 0
    condi_compare_before_fractal_mode = 0

condi_plrate_adaptive = config['default']['condi_plrate_adaptive']
condi_plrate_rate = config['default']['condi_plrate_rate']
condi_plrate_rate_min = config['default']['condi_plrate_rate_min']
condi_kelly_adaptive = config['default']['condi_kelly_adaptive']
condi_kelly_window = config['default']['condi_kelly_window']

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
    logger.info('krate_max:%s' % str(krate_max))
    logger.info('krate_min:%s' % str(krate_min))
    logger.info('walletrate:%s' % str(walletrate))
    logger.info('seed:%s' % str(seed))
    logger.info('fee:%s%%' % str(fee*100))
    logger.info('fee_slippage:%s%%' % str(round(fee_slippage*100, 4)))
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

    logger.info('et_zone_rate: %s' % et_zone_rate)
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


def set_price_for_tp(order_history, symbol, price, longshort):  # if has same price, make it different
    if order_history:
        symbol_order_history = [x for x in order_history if x['symbol'] == symbol and x['status'] == 'ETSL']
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
                                else round_step_size(price + float(tickSize),float(tickSize))

                    # print(price, 'changed tp price')
                    return price
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


def get_i_r(list, key, value):
    r = [[i, x] for i, x in enumerate(list) if x[key] == value]
    if len(r) == 1:
        return r[0][0], r[0][1]
    return None, None


def get_trade_prices(order_history, symbol, longshort, w):
    w0 = w.values[0]
    w1 = w.values[1]
    w2 = w.values[3]
    w3 = w.values[5]
    w4 = w.values[7]
    w5 = w.values[9]

    et_price = set_price(symbol, w4, longshort)
    sl_price = set_price(symbol, w0, longshort)
    tp_price_w5 = set_price(symbol, w5, longshort)
    tp_price = set_price_for_tp(order_history, symbol, tp_price_w5, longshort)
    return et_price, sl_price, tp_price, tp_price_w5


def new_et_order(symbol, tf, fc, longshort, qtyrate_k, et_price, sl_price, tp_price, tp_price_w5, quantity, wavepattern):
    params = [
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
    r1 = api_call('new_batch_order', [params])
    if r1 is not None:
        try:
            order_et = r1[0]
            if order_et['code']:
                return False
        except:
            pass

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
        r2 = api_call('new_batch_order', [params])
        if r2:
            try:
                order_sl = r2[0]
                if order_sl['code']:
                    print(symbol, tf, fc, longshort, et_price, sl_price, tp_price, quantity, wavepattern)
                    logger.error('_NEWET ET FAIL 2 ' + str(order_sl))
                    api_call('cancel_batch_order', [symbol, [order_et['orderId']], []])
                    return False
            except:
                pass
            add_etsl_history(open_order_history, symbol, tf, fc, longshort, qtyrate_k, wavepattern, et_price,
                            sl_price, tp_price, tp_price_w5, quantity, order_et['orderId'], order_sl['orderId'], order_et,
                            order_sl)
            return True
        else:
            r3 = api_call('cancel_batch_order', [symbol, [order_et['orderId']], []])
            if r3:
                try:
                    order_cancel_et = r3[0]
                    if order_cancel_et['code']:
                        logger.error('_NEWET ET CANCEL FAIL ' + order_cancel_et['code'])
                        return False
                except:
                    return False
    else:
        logger.info(symbol + ' _FAIL ET ITSELF FAIL' + str((symbol, tf, fc, longshort, et_price, sl_price, tp_price, quantity, wavepattern)))
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
                    # logger.info(symbol + (' _CANCELBATCH success, %s : ' % desc) + str(rs))
                    cnt_success += 1
            except:
                if rs['code']:
                    logger.info(symbol + (' _CANCEBATCH error, %s : ' % desc) + str(rs))
        if cnt_success == len(response):
            return True
    return False


def c_plrate_adaptive(open_order_history, symbol, longshort, w):
    et_price, sl_price, tp_price, tp_price_w5 = get_trade_prices(open_order_history, symbol, longshort, w)
    b_symbol = abs(tp_price - et_price) / abs(sl_price - et_price)  # one trade profitlose rate
    if condi_plrate_adaptive:
        if b_symbol > condi_plrate_rate or condi_plrate_rate_min > b_symbol:
            return False
    return True

def c_in_no_double_ordering(open_order_history, symbol, longshort, tf, fc, w):
    #####  이중 new limit order 방지 로직 start #####
    history_new = [x for x in open_order_history if
                   (x['symbol'] == symbol and x['status'] == 'ETSL'
                    and x['timeframe'] == tf and x['fcnt'] == fc)]

    et_price, sl_price, tp_price, tp_price_w5 = get_trade_prices(open_order_history, symbol, longshort, w)

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
                        and float(r_query_limit['clientOrderId'].split('_')[3]) == float(tp_price_w5):
                    print(str([symbol, longshort, tf, fc, et_price, sl_price, tp_price, tp_price_w5]))
                    logger.info(str([symbol, longshort, tf, fc, et_price, sl_price, tp_price, tp_price_w5]))
                    return False
    #####  이중 new limit order 방지 로직 start #####
    return True


def c_real_condition_by_fractal_index(df, fcnt, w, idx):  # real condititon by fractal index
    try:
        real_condititon1 = True if (fcnt/2 < (w.idx_end - w.idx_start)) and w.idx_start == idx else False
        real_condititon2 = True if df.iloc[idx + int(fcnt/2), 0] < (w.dates[-1]) else False
    except Exception as e:
        print(e)
    if not (real_condititon1 and real_condititon2):
        if printout: print('real_condititon ')
        return False
    return True


def c_active_no_empty(df, w):
    df_active = df.loc[df['Date'] > w.dates[-1]]  # 2023.3.13 after liqu  # df[w.idx_end + 1:]
    if df_active.empty:
        return False
    return True


def c_active_next_bean_ok(df, open_order_history, symbol, longshort, w):
    et_price, sl_price, tp_price, tp_price_w5 = get_trade_prices(open_order_history, symbol, longshort, w)
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


def c_active_in_zone(df, open_order_history, symbol, longshort, w):
    w_start_price = w.values[0]  # wave1
    w_end_price = w.values[-1]  # wave5
    height_price = abs(w_end_price - w_start_price)
    o_fibo_value = height_price * o_fibo / 100 if o_fibo else 0

    et_price, sl_price, tp_price, tp_price_w5 = get_trade_prices(open_order_history, symbol, longshort, w)

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


def c_in_no_risk(symbol, longshort, w, trade_info, qtyrate):

    qtyrate_k = get_qtyrate_k(trade_info, qtyrate)
    et_price, sl_price, tp_price, tp_price_w5 = get_trade_prices(open_order_history, symbol, longshort, w)

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


def check_cons_for_new_etsl_order(open_order_history, df, symbol, tf, fc, longshort, w, idx, trade_info, qtyrate):
    if not c_plrate_adaptive(open_order_history, symbol, longshort, w):
        return False
    if not c_real_condition_by_fractal_index(df, fc, w, idx):
        return False
    if not c_active_no_empty(df, w):
        return False
    if not c_active_next_bean_ok(df, open_order_history, symbol, longshort, w):
        return False
    if not c_active_in_zone(df, open_order_history, symbol, longshort, w):
        return False
    if not c_active_in_time(df, w):
        return False
    if not c_currentprice_in_zone(symbol, longshort, w):
        return False
    if not c_in_no_risk(symbol, longshort, w, trade_info, qtyrate):
        return False
    if not c_in_no_double_ordering(open_order_history, symbol, longshort, tf, fc, w):
        return False
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


def moniwave_and_action(symbol, tf, trade_info):
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

            if condi_compare_before_fractal:
                if not df_lows.empty:
                    for i in range(condi_compare_before_fractal_shift, 0, -1):
                        try:
                            if condi_compare_before_fractal_strait:
                                i = condi_compare_before_fractal_shift
                            df_lows['Low_before'] = df_lows.Low.shift(i).fillna(0)
                            if condi_compare_before_fractal_mode == 1:
                                df_lows['compare_flg'] = df_lows.apply(lambda x: 1 if x['Low'] > x['Low_before'] else 0,
                                                                       axis=1)
                            elif condi_compare_before_fractal_mode == 2:
                                df_lows['compare_flg'] = df_lows.apply(
                                    lambda x: 1 if x['Low'] >= x['Low_before'] else 0, axis=1)
                            elif condi_compare_before_fractal_mode == 3:
                                df_lows['compare_flg'] = df_lows.apply(lambda x: 1 if x['Low'] < x['Low_before'] else 0,
                                                                       axis=1)
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
                                df_highs['compare_flg'] = df_highs.apply(
                                    lambda x: 1 if x['High'] < x['High_before'] else 0, axis=1)
                            elif condi_compare_before_fractal_mode == 2:
                                df_highs['compare_flg'] = df_highs.apply(
                                    lambda x: 1 if x['High'] <= x['High_before'] else 0, axis=1)
                            elif condi_compare_before_fractal_mode == 3:
                                df_highs['compare_flg'] = df_highs.apply(
                                    lambda x: 1 if x['High'] > x['High_before'] else 0, axis=1)
                            elif condi_compare_before_fractal_mode == 4:
                                df_highs['compare_flg'] = df_highs.apply(
                                    lambda x: 1 if x['High'] >= x['High_before'] else 0, axis=1)
                            elif condi_compare_before_fractal_mode == 5:
                                df_highs['compare_flg'] = df_highs.apply(
                                    lambda x: 1 if x['High'] == x['High_before'] else 0, axis=1)
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
                                    if check_cons_for_new_etsl_order(open_order_history, df_all, symbol, tf, fc, longshort, wavepattern, i, trade_info, qtyrate):
                                        available, quantity = c_balance_and_calc_quanty(symbol)
                                        if available:
                                            et_price, sl_price, tp_price, tp_price_w5 = get_trade_prices(open_order_history, symbol, longshort, wavepattern)
                                            try:
                                                # plot_pattern_m(df=df,
                                                #                wave_pattern=[[1, wavepattern.dates[0], id(wavepattern),
                                                #                               wavepattern]],
                                                #                df_lows_plot=df_lows_plot, df_highs_plot=df_highs_plot,
                                                #                trade_info=None, title=str(symbol + ' %s ' % str(
                                                #         'LONG' if longshort else 'SHORT') + '%sm %s' % (
                                                #                                           tf, fc) + ', ET: ' + str(
                                                #         et_price)))
                                                qtyrate_k = get_qtyrate_k(trade_info, qtyrate)
                                                r = new_et_order(symbol, tf, fc, longshort, qtyrate_k, et_price, sl_price, tp_price, tp_price_w5, quantity, wavepattern)
                                                if r:
                                                    if plotview:
                                                        plot_pattern_m(df=df,
                                                                       wave_pattern=[[1, wavepattern.dates[0], id(wavepattern), wavepattern]],
                                                                       df_lows_plot=df_lows_plot, df_highs_plot=df_highs_plot,
                                                                       trade_info=None, title=str(symbol + ' %s ' % str('LONG' if longshort else 'SHORT') + '%sm %s' % (tf, fc) +', ET: ' + str(et_price)))

                                            except Exception as e:
                                                logger.error('new_et_order: %s' % str(e))

    return


def add_etsl_history(open_order_history, symbol, tf, fc, longshort, qtyrate_k, w, et_price, sl_price, tp_price, tp_price_w5, quantity, et_orderId, sl_orderId, order_et, order_sl):
    now = dt.datetime.now()
    history = {
        'id': et_orderId,
        'create_datetime': now,
        'status': 'ETSL',
        'symbol': symbol,
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
        logger.info(symbol + ' _HS update_history_status id: %s status: %s:' % (str(h_id), new_status))
        dump_history_pkl()


def delete_history_status(open_order_history, symbol, h_id, event):
    if open_order_history:
        history_idx, history_id = get_i_r(open_order_history, 'id', h_id)
        open_order_history.pop(history_idx)
        logger.info(symbol + ' _HS delete_history_status %s' % event)
        dump_history_pkl()


def close_position_by_symbol(symbol, quantity, longshort, et_orderId):
    positions = api_call('account', [])['positions']
    side = "LONG" if longshort else "SHORT"
    result_position_filtered = [x for x in positions if x['symbol'] == symbol and x['entryPrice'] != '0.0' and x['positionSide'] == side]
    for p in result_position_filtered:
        # quantity = str(abs(float(p['positionAmt'])))
        order_market = api_call('new_order',
                                [symbol, "SELL" if longshort else "BUY",
                                 "LONG" if longshort else "SHORT", "MARKET",
                                 quantity, "tp_" + str(et_orderId)])
        if order_market:
            logger.info(symbol + ' _CLOSE POSITION close_position_by_symbol success' + str(order_market))
        else:
            logger.info(symbol + ' _CLOSE POSITION close_position_by_symbol fail' + str(order_market))


def get_qtyrate_k(trade_info, qtyrate):
    t = trade_info
    stats_history = t[0]
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
    return qtyrate_k


def update_trade_info(trade_info, c_profit, c_stoploss, open_order_history, symbol, h_id):
    try:
        h = None
        if open_order_history:
            history_idx, history_id = get_i_r(open_order_history, 'id', h_id)
            open_order_history[history_idx] = history_id  # replace history
            h = open_order_history[history_idx]

        t = trade_info
        stats_history = t[0]
        order_history = t[1]
        asset_history = t[2]
        trade_count = t[3]
        fee_history = t[4]
        pnl_history = t[5]
        wavepattern_history = t[6]
        position = True


        dates = str(h['update_datetime'])[:19]  #'2023-01-15 21:24:00' #TODO how to find real entry date
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
                    position_sl_i = [dates, sl_price]
                    pnl_percent = -(abs(et_price - sl_price) / et_price) * qtyrate_k
                    fee_percent = fee_limit_sl
                    trade_count.append(0)
                    trade_inout_i = [position_enter_i, position_sl_i, longshort, '-']
                    out_price = sl_price
                    order_history.append(trade_inout_i)

                elif c_profit:
                    win_lose_flg = 1
                    position_pf_i = [dates, tp_price]
                    pnl_percent = (abs(tp_price - et_price) / et_price) * qtyrate_k
                    fee_percent = fee_limit_tp
                    trade_count.append(1)
                    trade_inout_i = [position_enter_i, position_pf_i, longshort, '+']
                    out_price = tp_price
                    order_history.append(trade_inout_i)

                asset_history_pre = asset_history[-1] if asset_history else seed
                asset_new = asset_history_pre * (1 + pnl_percent - fee_percent)
                pnl_history.append(asset_history_pre * pnl_percent)
                fee_history.append(asset_history_pre * fee_percent)
                asset_history.append(asset_new)
                wavepattern_history.append(wavepattern)

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

                if condi_kelly_adaptive:
                    if len(df_s) >= condi_kelly_window:
                        df_s_window = df_s.iloc[-condi_kelly_window:]
                        p = df_s_window[4].sum() / len(df_s_window)
                        b = (df_s_window[11].sum() + b_symbol) / (
                                    len(df_s_window) + 1)  # mean in kelly window - profitlose rate
                        q = 1 - p  # lose rate
                        tpi = round(p * (1 + b), 2)  # trading perfomance index
                        f = round(p - (q / b), 2)  # kelly index

                # current_price = float(api_call('ticker_price', ['BNBUSDT'])['price'])

                trade_stats = [len(trade_count), round(winrate, 2), asset_new, symbol, win_lose_flg,
                               'WIN' if win_lose_flg else 'LOSE', f_cum, tpi_cum, b_cum, f, tpi, b, b_symbol,
                               str(qtyrate_k), str(round(pnl_percent, 4)), sum(pnl_history), sum(fee_history),
                               round(asset_min, 2), round(asset_max, 2)]
                stats_history.append(trade_stats)
                trade_info = [stats_history, order_history, asset_history, trade_count, fee_history, pnl_history,
                              wavepattern_history]

                # wavepattern_tpsl_l.append([idx, wavepattern.dates[0], id(wavepattern), wavepattern])

                if True:
                    s_11 = symbol + '           '
                    trade_in = 'trade_in'  # trade_inout_i[0][0][2:-3]
                    trade_out = 'trade_out'  # trade_inout_i[1][0][8:-3]
                    ll = '%s %s %s %s %s x%s %s-%s %s %s %s %s %s %s - %s' % (str(qtyrate_k), str(
                        condi_compare_before_fractal_mode) + ' :shift=' + str(condi_compare_before_fractal_shift),
                                                                                     timeframe, s_11[:11], tf, qtyrate_k,
                                                                                     period_days_ago, period_days_ago_till,
                                                                                     fcnt, 'L' if longshort else 'S',
                                                                                     trade_in, '-',
                                                                                     trade_out,
                                                                                     str(trade_stats), str(
                        [et_price, sl_price, tp_price, out_price]))
                    print(ll)
                    logger.info(ll)

                # if longshort is not None and len(trade_info[1]) > 0:
                #     if plotview:
                #         plot_pattern_m(df=df, wave_pattern=[[i, wavepattern.dates[0], id(wavepattern), wavepattern]],
                #                        df_lows_plot=df_lows_plot, df_highs_plot=df_highs_plot, trade_info=trade_info,
                #                        title=str(
                #                            symbol + ' %s ' % str(longshort) + str(trade_stats)))
    except Exception as e:
        print(symbol + ' update_trade_info ' + str(h_id) + ' e:' + str(e))
        logger.error(symbol + ' update_trade_info ' + str(h_id) + ' e:' + str(e))
    return trade_info

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
        df['date'] = df.apply(lambda x: str(dt.datetime.fromtimestamp(float(x['time']) / 1000).strftime('%Y-%m-%d')), axis=1)
        df.sort_values(by='time_dt', ascending=False, inplace=True)  # https://sparkbyexamples.com/pandas/sort-pandas-dataframe-by-date/

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


def monihistory_and_action(open_order_history, symbol, trade_info):  # ETSL -> CANCEL        or       TP -> WIN or LOSE
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
                quantity = new['quantity']
                fc = new['fcnt']

                r_get_open_orders_et = api_call('get_open_orders', [symbol, et_orderId])
                r_get_open_orders_et_flg = True if r_get_open_orders_et else False
                r_query_et = api_call('query_order', [symbol, et_orderId])

                r_get_open_orders_sl = api_call('get_open_orders', [symbol, sl_orderId])
                r_get_open_orders_sl_flg = True if r_get_open_orders_sl else False
                r_query_sl = api_call('query_order', [symbol, sl_orderId])


                if r_get_open_orders_et_flg is True and r_get_open_orders_sl_flg is True and r_query_et['status'] =='NEW' and r_query_sl['status'] == 'NEW':
                    if not c_currentprice_in_zone_by_prices(symbol, longshort, float(et_price), float(tp_price)):
                        # CANCEL_ETSL
                        response_cancel = cancel_batch_order(symbol, [str(et_orderId), str(sl_orderId)], 'CANCEL ETSL')
                        if response_cancel:
                            delete_history_status(open_order_history, symbol, et_orderId, 'CANCEL')
                            # update_history_status(open_order_history, symbol, et_orderId, 'CANCEL')
                elif r_get_open_orders_et_flg is False and r_get_open_orders_sl_flg is True and r_query_et['status'] == 'FILLED' and r_query_sl['status'] == 'NEW':
                        # NEW_TP
                        new_tp_order(symbol, tf, fc, longshort, tp_price, quantity, et_orderId)
                elif r_get_open_orders_et_flg is False and r_get_open_orders_sl_flg is False and r_query_et['status'] == 'FILLED' and r_query_sl['status'] == 'FILLED':
                        update_history_status(open_order_history, symbol, et_orderId, 'LOSE')
                        print(symbol, ' IN ETSLETSL OooooooooooOOOOOo  ESSL DIRECT LOSE ooOOOOOoooOOOO')
                        logger.info(symbol + ' IN ETSLETSL Ooooooooooo  ESSL DIRECT LOSE  OOOOOoooOOOOOoooOOOO')

                elif r_get_open_orders_et_flg is True and r_get_open_orders_sl_flg is False and r_query_et['status'] =='NEW' and r_query_sl['status'] == 'CANCELED':
                    cancel_batch_order(symbol, [et_orderId], 'FORCE CLICK SL IN ETSL, REMAIN ET CLEAR')
                    delete_history_status(open_order_history, symbol, et_orderId, 'CANCEL')
                    # update_history_status(open_order_history, symbol, et_orderId, 'CANCEL')
                elif r_get_open_orders_et_flg is False and r_get_open_orders_sl_flg is True and r_query_et['status'] =='CANCELED' and r_query_sl['status'] == 'NEW':
                    cancel_batch_order(symbol, [sl_orderId], 'FORCE CLICK ET IN ETSL, REMAIN SL CLEAR')
                    delete_history_status(open_order_history, symbol, et_orderId, 'CANCEL')
                    # update_history_status(open_order_history, symbol, et_orderId, 'CANCEL')


                elif r_get_open_orders_et_flg is False and r_get_open_orders_sl_flg is False and r_query_et['status'] == 'FILLED' and r_query_sl['status'] == 'CANCELED':
                    rt = close_position_by_symbol(symbol, quantity, longshort, et_orderId)  # AFTER FORCE SL CLICK, REMAIN TP FILLED AND CLOSE POSI ET
                    if rt is not None:
                        logger.info(
                            symbol + ' IN ETSLETSL Ooooooooooo  AFTER FORCE SL CLICK, REMAIN TP FILLED AND CLOSE POSI ET OOOOOoooOOOOOoooOOOO'+ str(rt))
                        update_history_status(open_order_history, symbol, et_orderId, 'TP')  # TODO check win or lose
                    else:
                        print(symbol,
                              ' IN ETSLETSL OooooooooooOOOOOo  AFTER FORCE SL CLICK, REMAIN TP FILLED AND CLOSE POSI ET  FAIL ERROR  ooOOOOOoooOOOO' + str(rt))
                        logger.info(
                            symbol + ' IN ETSLETSL Ooooooooooo  AFTER FORCE SL CLICK, REMAIN TP FILLED AND CLOSE POSI ET FAIL ERROR OOOOOoooOOOOOoooOOOO'+ str(rt))

                elif r_get_open_orders_et_flg is False and r_get_open_orders_sl_flg is True and r_query_et['status'] == 'NEW' and r_query_sl['status'] == 'CANCELED':
                    cancel_batch_order(symbol, [sl_orderId], 'FORCE MARKET CLICK IN ETSL, REMAIN SL CLEAR')
                    delete_history_status(open_order_history, symbol, et_orderId, 'CANCEL')
                    # update_history_status(open_order_history, symbol, et_orderId, 'CANCEL')

                elif r_get_open_orders_et_flg is False and r_get_open_orders_sl_flg is False and r_query_et['status'] == 'CANCELED' and r_query_sl['status'] == 'CANCELED':
                    delete_history_status(open_order_history, symbol, et_orderId, 'CANCEL')
                    # update_history_status(open_order_history, symbol, et_orderId, 'CANCEL')  # TWO ET AND SL ARE CANCELED
                    print(symbol, ' IN ETSLETSL OooooooooooOOOOO    TWO ET AND SL ARE CANCELED oooOOOOOoooOOOO')
                    logger.info(symbol + ' IN ETSLETSL OooooooooooOOO    TWO ET AND SL ARE CANCELED OOoooOOOOOoooOOOO')



                # elif r_get_open_orders_et_flg is False and r_get_open_orders_sl_flg is False and r_query_et['status'] == 'PARTIALLY_FILLED' and r_query_sl['status'] == 'NEW':
                #     rt = close_position_by_symbol(symbol, quantity, longshort, et_orderId)  # AFTER PARTIALLY_FILLED, REMAIN TCLOSE POSI ET
                #     if rt is not None:
                #         update_history_status(open_order_history, symbol, et_orderId, 'FORCE')  # TODO check win or lose
                #     else:
                #         print(symbol,
                #               ' IN ETSLETSL OooooooooooOOOOOo  AFTER PARTIALLY_FILLED, REMAIN TCLOSE POSI ET  FAIL ERROR  ooOOOOOoooOOOO' + str(rt))
                #         logger.info(
                #             symbol + ' IN ETSLETSL Ooooooooooo  AFTER PARTIALLY_FILLED, REMAIN TCLOSE POSI ET ERROR OOOOOoooOOOOOoooOOOO'+ str(rt))

                elif r_get_open_orders_et_flg is True and r_get_open_orders_sl_flg is True and r_query_et['status'] == 'PARTIALLY_FILLED' and r_query_sl['status'] == 'NEW':
                    rt = new_tp_order(symbol, tf, fc, longshort, tp_price, r_query_et['executedQty'], et_orderId)
                    logger.info(symbol + ' IN ETSLETSL PARTIALLY_FILLED monitoring_orders_positions ' + str(rt))
                    # # # force cancel limit(o), sl(x)
                    cancel_batch_order(symbol, [int(et_orderId)], 'CANCEL ETSL PARTIALLY ')

                elif r_get_open_orders_et_flg is False and r_get_open_orders_sl_flg is False and r_query_et['status'] == 'FILLED' and r_query_sl['status'] == 'EXPIRED':
                    update_history_status(open_order_history, symbol, et_orderId, 'FORCE')  # TWO ET AND SL ARE CANCELED
                    logger.info(symbol + ' IN ETSLETSL ET:FILLED and SL:EXPIRED monitoring_orders_positions ' + str(rt))

                else:
                    print(symbol, ' IN ETSLETSL OooooooooooOOOOOoooOOOOOooESSLESSLoOOOO')
                    logger.info(symbol + ' IN ETSLETSL OooooooooooOOOOOoooOOOOOooESSLESSLoOOOO')
                    print('IN ETSL: ', symbol, str(r_get_open_orders_et_flg), str(r_get_open_orders_sl_flg), r_query_et['status'], r_query_sl['status'])
                    logger.info('IN ETSL: %s %s %s %s %s ' % (symbol, str(r_get_open_orders_et_flg), str(r_get_open_orders_sl_flg), str(r_query_et['status']), str(r_query_sl['status'])))


        history_tp = [x for x in open_order_history if (x['symbol'] == symbol and x['status'] == 'TP')]
        if history_tp:
            for tp in history_tp:
                # POSI_CHECK
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


                if r_get_open_orders_tp_flg is True and r_get_open_orders_sl_flg is True and r_query_tp['status'] =='NEW' and r_query_sl['status'] == 'NEW':
                    # case general TP
                    pass

                elif r_get_open_orders_tp_flg is True and r_get_open_orders_sl_flg is True and r_query_tp['status'] =='PARTIALLY_FILLED' and r_query_sl['status'] == 'NEW':
                    # case wait to TP full filled or SL filled
                    print(symbol, ' IN TPTP PARTIALLY_FILLED and NEW OooxxxTTTTTPPPPPPxxxO')
                    logger.info(symbol + ' IN TPTP PARTIALLY_FILLED and NEW OooxxxTTTTTPPPPPPxxxO')
                    pass

                # AUTO
                elif r_get_open_orders_tp_flg is False and r_get_open_orders_sl_flg is True and r_query_tp['status'] == 'FILLED' and r_query_sl['status'] == 'NEW':
                    cancel_batch_order(symbol, [sl_orderId], 'AUTO TP FILLED, REMAIN SL CLEAR')
                    trade_info = update_trade_info(trade_info, True, False, open_order_history, symbol, int(et_orderId))
                    update_history_status(open_order_history, symbol, et_orderId, 'WIN')  # AUTO TP FILLED

                    df_t, realizedPnl_tot, commission_tot = get_account_trades(symbol, et_orderId, None, tp_orderId)
                    print(df_t, realizedPnl_tot, commission_tot)

                elif r_query_tp['status'] == 'FILLED' and r_query_sl['status'] == 'EXPIRED':
                    trade_info = update_trade_info(trade_info, True, False, open_order_history, symbol, int(et_orderId))
                    update_history_status(open_order_history, symbol, et_orderId, 'WIN')  # AUTO TP FILLED
                    df_t, realizedPnl_tot, commission_tot = get_account_trades(symbol, et_orderId, None, tp_orderId)
                    print(df_t, realizedPnl_tot, commission_tot)

                elif r_get_open_orders_tp_flg is False and r_get_open_orders_sl_flg is False and r_query_tp['status'] == 'EXPIRED' and r_query_sl['status'] == 'FILLED':
                    # cancel_batch_order(symbol, [tp_orderId], 'AUTO SL FILLED, REMAIN TP CLEAR')
                    trade_info = update_trade_info(trade_info, False, True, open_order_history, symbol, int(et_orderId))
                    update_history_status(open_order_history, symbol, et_orderId, 'LOSE')  # AUTO SL FILLED
                    df_t, realizedPnl_tot, commission_tot = get_account_trades(symbol, et_orderId, sl_orderId, None)
                    print(df_t, realizedPnl_tot, commission_tot)
                elif r_get_open_orders_tp_flg is False and r_get_open_orders_sl_flg is False and r_query_tp['status'] == 'EXPIRED' and r_query_sl['status'] == 'EXPIRED':
                    logger.info('IN TPTP EXPIRED EXPIRED: %s %s %s %s %s ' % (symbol, str(r_get_open_orders_tp_flg), str(r_get_open_orders_sl_flg), str(r_query_tp['status']), str(r_query_sl['status'])))
                    update_history_status(open_order_history, symbol, et_orderId, 'LOSE')  # TODO check win or lose  # when duplicate order's case, before orderID take this orderId's realizedPnl, so decide this to LOSE it happen just LOSE

                # FORCE
                elif r_get_open_orders_tp_flg is False and r_get_open_orders_sl_flg is False and r_query_tp['status'] == 'FILLED' and r_query_sl['status'] == 'FILLED':
                    logger.info('IN TPTP FILLED FILLED: %s %s %s %s %s ' % (symbol, str(r_get_open_orders_tp_flg), str(r_get_open_orders_sl_flg), str(r_query_tp['status']), str(r_query_sl['status'])))
                    update_history_status(open_order_history, symbol, et_orderId, 'FORCE')  # AUTO TP FILLED  # TODO check win or lose

                elif r_get_open_orders_tp_flg is False and r_get_open_orders_sl_flg is True and r_query_tp['status']=='NEW' and r_query_sl['status'] == 'NEW':
                    print('IN TP: r_get_open_orders_tp_flg is False and r_get_open_orders_sl_flg is True and r_query_tp[status]==NEW and r_query_sl[status] == NEW', symbol, True if r_get_open_orders_tp else False,
                          True if r_get_open_orders_sl else False, r_query_tp['status'], r_query_sl['status'])
                    logger.info('IN TP: %s %s %s %s %s ' % (symbol, str(r_get_open_orders_tp_flg), str(r_get_open_orders_sl_flg), str(r_query_tp['status']), str(r_query_sl['status'])))
                    cancel_batch_order(symbol, [tp_orderId], 'FORCE MARKET CLICK, REMAIN TP CLEAR')
                    cancel_batch_order(symbol, [sl_orderId], 'FORCE MARKET CLICK, REMAIN SL CLEAR')
                    update_history_status(open_order_history, symbol, et_orderId, 'FORCE')  # TODO check win or lose
                elif r_get_open_orders_tp_flg is False and r_get_open_orders_sl_flg is True and r_query_tp['status'] == 'EXPIRED' and r_query_sl['status'] == 'NEW':
                    cancel_batch_order(symbol, [sl_orderId], 'FORCE MARKET CLICK AND TIME PASSED, REMAIN SL CLEAR')
                    update_history_status(open_order_history, symbol, et_orderId, 'FORCE')  # TODO check win or lose
                elif r_get_open_orders_tp_flg is False and r_get_open_orders_sl_flg is True and r_query_tp['status'] == 'CANCELED' and r_query_sl['status'] == 'NEW':
                    close_position_by_symbol(symbol, quantity, longshort, et_orderId)  # FORCE TP CLICK, REMAIN SL CLEAR
                    cancel_batch_order(symbol, [sl_orderId], 'FORCE TP CLICK, REMAIN SL CLEAR')
                    update_history_status(open_order_history, symbol, et_orderId, 'FORCE')  # TODO check win or lose

                elif r_get_open_orders_tp_flg is False and r_get_open_orders_sl_flg is False and r_query_tp['status'] == 'CANCELED' and r_query_sl['status'] == 'CANCELED':
                    close_position_by_symbol(symbol, quantity, longshort, et_orderId)  # FORCE TP and SL CLICK
                    logger.info('IN TPTP CANCELED CANCELED : %s %s %s %s %s ' % (symbol, str(r_get_open_orders_tp_flg), str(r_get_open_orders_sl_flg), str(r_query_tp['status']), str(r_query_sl['status'])))
                    update_history_status(open_order_history, symbol, et_orderId, 'FORCE')  # TODO check win or lose

                elif r_get_open_orders_tp_flg is True and r_get_open_orders_sl_flg is True and r_query_tp['status'] =='NEW' and r_query_sl['status'] == 'CANCELED':
                    close_position_by_symbol(symbol, quantity, longshort, et_orderId)  # FORCE SL, BEFORE CHECK TP CHEKER
                    cancel_batch_order(symbol, [tp_orderId], 'FORCE SL CLICK, REMAIN SL CLEAR')
                    update_history_status(open_order_history, symbol, et_orderId, 'FORCE')  # TODO check win or lose
                elif r_get_open_orders_tp_flg is True and r_get_open_orders_sl_flg is False and r_query_tp['status'] =='NEW' and r_query_sl['status'] == 'CANCELED':
                    close_position_by_symbol(symbol, quantity, longshort, et_orderId)  # FORCE SL AFTER CHECK TP CHEKER
                    cancel_batch_order(symbol, [tp_orderId], 'FORCE SL CLICK, REMAIN SL CLEAR')
                    update_history_status(open_order_history, symbol, et_orderId, 'FORCE')  # TODO check win or lose

                else:
                    print(symbol, ' IN TPTP OooooooooooOOOOOoooOOOOOTTTTTPPPPPPoooOOOO')
                    logger.info(symbol + ' IN TPTP OooooooooooOOOOOoooOOOOOTTTTTPPPPPPoooOOOO')
                    print('IN TP: ', symbol, True if r_get_open_orders_tp else False,
                          True if r_get_open_orders_sl else False, r_query_tp['status'], r_query_sl['status'])
                    logger.info('IN TPTP: %s %s %s %s %s ' % (symbol, str(r_get_open_orders_tp_flg), str(r_get_open_orders_sl_flg), str(r_query_tp['status']), str(r_query_sl['status'])))



def single(symbols, i, trade_info, *args):
    for symbol in symbols:
        try:
            monihistory_and_action(open_order_history, symbol, trade_info)
        except Exception as e:
            print('monihistory_and_action: %s' % str(e))

        for tf in timeframe:
            try:
                moniwave_and_action(symbol, tf, trade_info)
            except Exception as e:
                print('moniwave_and_action: %s' % str(e))


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
    import time
    from apscheduler.schedulers.blocking import BlockingScheduler

    sched = BlockingScheduler()  # https://hello-bryan.tistory.com/216
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

    stats_history = []
    order_history = []
    asset_history = []
    trade_count = []
    fee_history = []
    pnl_history = []
    wavepattern_history = []
    trade_info = [stats_history, order_history, asset_history, trade_count, fee_history, pnl_history,
                  wavepattern_history]
    while True:
        # if i % 10 == 1:
        logger.info(f'{i} start: {time.strftime("%H:%M:%S")}')
        single(symbols, i, trade_info)
        i += 1
    print(f'Finished in {round(time.perf_counter() - start, 2)} second(s)')
    print_condition()
    print("good luck done!!")
