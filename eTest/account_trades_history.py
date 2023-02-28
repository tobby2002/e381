#!/usr/bin/env python
import logging
from binancefutures.um_futures import UMFutures
from binancefutures.lib.utils import config_logging
from binancefutures.error import ClientError
import time
import datetime as dt

config_logging(logging, logging.DEBUG)

key = "IkzH8WHKl0lGzOSqiZZ4TnAyKnDpqnC9Xi31kzrRNpwJCp28gP8AuWDxntSqWdrn"
secret = "FwKTmQ2RWSiECMfhZOaY7Hed45JuXqlEPno2xiLGgCzloLq4NMMcmusG6gtMCKa5"

um_futures_client = UMFutures(key=key, secret=secret)

try:
    data = um_futures_client.get_account_trades(symbol="LITUSDT", recvWindow=6000)
    logging.info(data)
except ClientError as error:
    logging.error(
        "Found error. status: {}, error code: {}, error message: {}".format(
            error.status_code, error.error_code, error.error_message
        )
    )


# data = [{'symbol': 'DEFIUSDT', 'id': 32890705, 'orderId': 3932063139, 'side': 'SELL', 'price': '707.2', 'qty': '0.513', 'realizedPnl': '0', 'marginAsset': 'USDT', 'quoteQty': '362.7936', 'commission': '0.00020614', 'commissionAsset': 'BNB', 'time': 1676866475190, 'positionSide': 'SHORT', 'buyer': False, 'maker': True}, {'symbol': 'DEFIUSDT', 'id': 32890706, 'orderId': 3932063139, 'side': 'SELL', 'price': '707.2', 'qty': '3.193', 'realizedPnl': '0', 'marginAsset': 'USDT', 'quoteQty': '2258.0896', 'commission': '0.00128308', 'commissionAsset': 'BNB', 'time': 1676866475192, 'positionSide': 'SHORT', 'buyer': False, 'maker': True}, {'symbol': 'DEFIUSDT', 'id': 32900146, 'orderId': 3932063140, 'side': 'BUY', 'price': '723.5', 'qty': '0.302', 'realizedPnl': '-4.92260000', 'marginAsset': 'USDT', 'quoteQty': '218.4970', 'commission': '0.00024650', 'commissionAsset': 'BNB', 'time': 1676890901682, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32900147, 'orderId': 3932063140, 'side': 'BUY', 'price': '723.5', 'qty': '1.033', 'realizedPnl': '-16.83790000', 'marginAsset': 'USDT', 'quoteQty': '747.3755', 'commission': '0.00084316', 'commissionAsset': 'BNB', 'time': 1676890901683, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32900148, 'orderId': 3932063140, 'side': 'BUY', 'price': '723.7', 'qty': '0.028', 'realizedPnl': '-0.46200000', 'marginAsset': 'USDT', 'quoteQty': '20.2636', 'commission': '0.00002286', 'commissionAsset': 'BNB', 'time': 1676890901683, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32900149, 'orderId': 3932063140, 'side': 'BUY', 'price': '723.7', 'qty': '0.090', 'realizedPnl': '-1.48500000', 'marginAsset': 'USDT', 'quoteQty': '65.1330', 'commission': '0.00007348', 'commissionAsset': 'BNB', 'time': 1676890901683, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32900150, 'orderId': 3932063140, 'side': 'BUY', 'price': '723.7', 'qty': '0.963', 'realizedPnl': '-15.88950000', 'marginAsset': 'USDT', 'quoteQty': '696.9231', 'commission': '0.00078624', 'commissionAsset': 'BNB', 'time': 1676890901683, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32900151, 'orderId': 3932063140, 'side': 'BUY', 'price': '723.7', 'qty': '1.290', 'realizedPnl': '-21.28500000', 'marginAsset': 'USDT', 'quoteQty': '933.5730', 'commission': '0.00105322', 'commissionAsset': 'BNB', 'time': 1676890901683, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32988327, 'orderId': 3936576835, 'side': 'SELL', 'price': '687.6', 'qty': '0.061', 'realizedPnl': '0', 'marginAsset': 'USDT', 'quoteQty': '41.9436', 'commission': '0.00002452', 'commissionAsset': 'BNB', 'time': 1677092791056, 'positionSide': 'SHORT', 'buyer': False, 'maker': True}, {'symbol': 'DEFIUSDT', 'id': 32988328, 'orderId': 3936576835, 'side': 'SELL', 'price': '687.6', 'qty': '0.505', 'realizedPnl': '0', 'marginAsset': 'USDT', 'quoteQty': '347.2380', 'commission': '0.00020296', 'commissionAsset': 'BNB', 'time': 1677092791147, 'positionSide': 'SHORT', 'buyer': False, 'maker': True}, {'symbol': 'DEFIUSDT', 'id': 32988329, 'orderId': 3936576835, 'side': 'SELL', 'price': '687.6', 'qty': '1.922', 'realizedPnl': '0', 'marginAsset': 'USDT', 'quoteQty': '1321.5672', 'commission': '0.00077248', 'commissionAsset': 'BNB', 'time': 1677092791190, 'positionSide': 'SHORT', 'buyer': False, 'maker': True}, {'symbol': 'DEFIUSDT', 'id': 32988330, 'orderId': 3936576835, 'side': 'SELL', 'price': '687.6', 'qty': '1.440', 'realizedPnl': '0', 'marginAsset': 'USDT', 'quoteQty': '990.1440', 'commission': '0.00057876', 'commissionAsset': 'BNB', 'time': 1677092791232, 'positionSide': 'SHORT', 'buyer': False, 'maker': True}, {'symbol': 'DEFIUSDT', 'id': 32988331, 'orderId': 3936576835, 'side': 'SELL', 'price': '687.6', 'qty': '1.215', 'realizedPnl': '0', 'marginAsset': 'USDT', 'quoteQty': '835.4340', 'commission': '0.00048832', 'commissionAsset': 'BNB', 'time': 1677092791337, 'positionSide': 'SHORT', 'buyer': False, 'maker': True}, {'symbol': 'DEFIUSDT', 'id': 32995833, 'orderId': 3936576836, 'side': 'BUY', 'price': '698.6', 'qty': '0.009', 'realizedPnl': '-0.09900000', 'marginAsset': 'USDT', 'quoteQty': '6.2874', 'commission': '0.00000723', 'commissionAsset': 'BNB', 'time': 1677110489921, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32995834, 'orderId': 3936576836, 'side': 'BUY', 'price': '698.6', 'qty': '0.413', 'realizedPnl': '-4.54300000', 'marginAsset': 'USDT', 'quoteQty': '288.5218', 'commission': '0.00033220', 'commissionAsset': 'BNB', 'time': 1677110489921, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32995835, 'orderId': 3936576836, 'side': 'BUY', 'price': '698.6', 'qty': '0.368', 'realizedPnl': '-4.04800000', 'marginAsset': 'USDT', 'quoteQty': '257.0848', 'commission': '0.00029601', 'commissionAsset': 'BNB', 'time': 1677110489921, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32995836, 'orderId': 3936576836, 'side': 'BUY', 'price': '698.6', 'qty': '0.405', 'realizedPnl': '-4.45500000', 'marginAsset': 'USDT', 'quoteQty': '282.9330', 'commission': '0.00032577', 'commissionAsset': 'BNB', 'time': 1677110489921, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32995837, 'orderId': 3936576836, 'side': 'BUY', 'price': '698.6', 'qty': '0.300', 'realizedPnl': '-3.30000000', 'marginAsset': 'USDT', 'quoteQty': '209.5800', 'commission': '0.00024131', 'commissionAsset': 'BNB', 'time': 1677110489921, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32995838, 'orderId': 3936576836, 'side': 'BUY', 'price': '698.7', 'qty': '0.928', 'realizedPnl': '-10.30080000', 'marginAsset': 'USDT', 'quoteQty': '648.3936', 'commission': '0.00074656', 'commissionAsset': 'BNB', 'time': 1677110489921, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32995839, 'orderId': 3936576836, 'side': 'BUY', 'price': '698.7', 'qty': '1.097', 'realizedPnl': '-12.17670000', 'marginAsset': 'USDT', 'quoteQty': '766.4739', 'commission': '0.00088252', 'commissionAsset': 'BNB', 'time': 1677110489921, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32995840, 'orderId': 3936576836, 'side': 'BUY', 'price': '698.7', 'qty': '0.486', 'realizedPnl': '-5.39460000', 'marginAsset': 'USDT', 'quoteQty': '339.5682', 'commission': '0.00039098', 'commissionAsset': 'BNB', 'time': 1677110489921, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32995841, 'orderId': 3936576836, 'side': 'BUY', 'price': '698.8', 'qty': '0.037', 'realizedPnl': '-0.41440000', 'marginAsset': 'USDT', 'quoteQty': '25.8556', 'commission': '0.00002977', 'commissionAsset': 'BNB', 'time': 1677110489921, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32995842, 'orderId': 3936576836, 'side': 'BUY', 'price': '698.8', 'qty': '1.016', 'realizedPnl': '-11.37920000', 'marginAsset': 'USDT', 'quoteQty': '709.9808', 'commission': '0.00081748', 'commissionAsset': 'BNB', 'time': 1677110489921, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32995843, 'orderId': 3936576836, 'side': 'BUY', 'price': '698.9', 'qty': '0.034', 'realizedPnl': '-0.38420000', 'marginAsset': 'USDT', 'quoteQty': '23.7626', 'commission': '0.00002736', 'commissionAsset': 'BNB', 'time': 1677110489921, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32995844, 'orderId': 3936576836, 'side': 'BUY', 'price': '698.9', 'qty': '0.038', 'realizedPnl': '-0.42940000', 'marginAsset': 'USDT', 'quoteQty': '26.5582', 'commission': '0.00003057', 'commissionAsset': 'BNB', 'time': 1677110489921, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32995845, 'orderId': 3936576836, 'side': 'BUY', 'price': '699', 'qty': '0.012', 'realizedPnl': '-0.13680000', 'marginAsset': 'USDT', 'quoteQty': '8.3880', 'commission': '0.00000965', 'commissionAsset': 'BNB', 'time': 1677110489921, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}]

# def timestamp2datetime(x):
#     return str(datetime.datetime.fromtimestamp(float(x) / 1000).strftime('%Y-%m-%d %H:%M:%S'))

import pandas as pd


symbols = ['AGIXUSDT', 'BTCUSDT', 'LITUSDT', 'TUSDT']


def account_trades_df(symbols):
    data_all = list()
    for s in symbols:
        try:
            time.sleep(0.2)
            data = um_futures_client.get_account_trades(symbol=s, recvWindow=6000)
            if len(data) > 0:
                # logging.info(data)
                data_all.extend(data)
        except ClientError as error:
            logging.error(
                "Found error. status: {}, error code: {}, error message: {}".format(
                    error.status_code, error.error_code, error.error_message
                )
            )


    # data = [{'symbol': 'DEFIUSDT', 'id': 32890705, 'orderId': 3932063139, 'side': 'SELL', 'price': '707.2', 'qty': '0.513', 'realizedPnl': '0', 'marginAsset': 'USDT', 'quoteQty': '362.7936', 'commission': '0.00020614', 'commissionAsset': 'BNB', 'time': 1676866475190, 'positionSide': 'SHORT', 'buyer': False, 'maker': True}, {'symbol': 'DEFIUSDT', 'id': 32890706, 'orderId': 3932063139, 'side': 'SELL', 'price': '707.2', 'qty': '3.193', 'realizedPnl': '0', 'marginAsset': 'USDT', 'quoteQty': '2258.0896', 'commission': '0.00128308', 'commissionAsset': 'BNB', 'time': 1676866475192, 'positionSide': 'SHORT', 'buyer': False, 'maker': True}, {'symbol': 'DEFIUSDT', 'id': 32900146, 'orderId': 3932063140, 'side': 'BUY', 'price': '723.5', 'qty': '0.302', 'realizedPnl': '-4.92260000', 'marginAsset': 'USDT', 'quoteQty': '218.4970', 'commission': '0.00024650', 'commissionAsset': 'BNB', 'time': 1676890901682, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32900147, 'orderId': 3932063140, 'side': 'BUY', 'price': '723.5', 'qty': '1.033', 'realizedPnl': '-16.83790000', 'marginAsset': 'USDT', 'quoteQty': '747.3755', 'commission': '0.00084316', 'commissionAsset': 'BNB', 'time': 1676890901683, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32900148, 'orderId': 3932063140, 'side': 'BUY', 'price': '723.7', 'qty': '0.028', 'realizedPnl': '-0.46200000', 'marginAsset': 'USDT', 'quoteQty': '20.2636', 'commission': '0.00002286', 'commissionAsset': 'BNB', 'time': 1676890901683, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32900149, 'orderId': 3932063140, 'side': 'BUY', 'price': '723.7', 'qty': '0.090', 'realizedPnl': '-1.48500000', 'marginAsset': 'USDT', 'quoteQty': '65.1330', 'commission': '0.00007348', 'commissionAsset': 'BNB', 'time': 1676890901683, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32900150, 'orderId': 3932063140, 'side': 'BUY', 'price': '723.7', 'qty': '0.963', 'realizedPnl': '-15.88950000', 'marginAsset': 'USDT', 'quoteQty': '696.9231', 'commission': '0.00078624', 'commissionAsset': 'BNB', 'time': 1676890901683, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32900151, 'orderId': 3932063140, 'side': 'BUY', 'price': '723.7', 'qty': '1.290', 'realizedPnl': '-21.28500000', 'marginAsset': 'USDT', 'quoteQty': '933.5730', 'commission': '0.00105322', 'commissionAsset': 'BNB', 'time': 1676890901683, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32988327, 'orderId': 3936576835, 'side': 'SELL', 'price': '687.6', 'qty': '0.061', 'realizedPnl': '0', 'marginAsset': 'USDT', 'quoteQty': '41.9436', 'commission': '0.00002452', 'commissionAsset': 'BNB', 'time': 1677092791056, 'positionSide': 'SHORT', 'buyer': False, 'maker': True}, {'symbol': 'DEFIUSDT', 'id': 32988328, 'orderId': 3936576835, 'side': 'SELL', 'price': '687.6', 'qty': '0.505', 'realizedPnl': '0', 'marginAsset': 'USDT', 'quoteQty': '347.2380', 'commission': '0.00020296', 'commissionAsset': 'BNB', 'time': 1677092791147, 'positionSide': 'SHORT', 'buyer': False, 'maker': True}, {'symbol': 'DEFIUSDT', 'id': 32988329, 'orderId': 3936576835, 'side': 'SELL', 'price': '687.6', 'qty': '1.922', 'realizedPnl': '0', 'marginAsset': 'USDT', 'quoteQty': '1321.5672', 'commission': '0.00077248', 'commissionAsset': 'BNB', 'time': 1677092791190, 'positionSide': 'SHORT', 'buyer': False, 'maker': True}, {'symbol': 'DEFIUSDT', 'id': 32988330, 'orderId': 3936576835, 'side': 'SELL', 'price': '687.6', 'qty': '1.440', 'realizedPnl': '0', 'marginAsset': 'USDT', 'quoteQty': '990.1440', 'commission': '0.00057876', 'commissionAsset': 'BNB', 'time': 1677092791232, 'positionSide': 'SHORT', 'buyer': False, 'maker': True}, {'symbol': 'DEFIUSDT', 'id': 32988331, 'orderId': 3936576835, 'side': 'SELL', 'price': '687.6', 'qty': '1.215', 'realizedPnl': '0', 'marginAsset': 'USDT', 'quoteQty': '835.4340', 'commission': '0.00048832', 'commissionAsset': 'BNB', 'time': 1677092791337, 'positionSide': 'SHORT', 'buyer': False, 'maker': True}, {'symbol': 'DEFIUSDT', 'id': 32995833, 'orderId': 3936576836, 'side': 'BUY', 'price': '698.6', 'qty': '0.009', 'realizedPnl': '-0.09900000', 'marginAsset': 'USDT', 'quoteQty': '6.2874', 'commission': '0.00000723', 'commissionAsset': 'BNB', 'time': 1677110489921, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32995834, 'orderId': 3936576836, 'side': 'BUY', 'price': '698.6', 'qty': '0.413', 'realizedPnl': '-4.54300000', 'marginAsset': 'USDT', 'quoteQty': '288.5218', 'commission': '0.00033220', 'commissionAsset': 'BNB', 'time': 1677110489921, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32995835, 'orderId': 3936576836, 'side': 'BUY', 'price': '698.6', 'qty': '0.368', 'realizedPnl': '-4.04800000', 'marginAsset': 'USDT', 'quoteQty': '257.0848', 'commission': '0.00029601', 'commissionAsset': 'BNB', 'time': 1677110489921, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32995836, 'orderId': 3936576836, 'side': 'BUY', 'price': '698.6', 'qty': '0.405', 'realizedPnl': '-4.45500000', 'marginAsset': 'USDT', 'quoteQty': '282.9330', 'commission': '0.00032577', 'commissionAsset': 'BNB', 'time': 1677110489921, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32995837, 'orderId': 3936576836, 'side': 'BUY', 'price': '698.6', 'qty': '0.300', 'realizedPnl': '-3.30000000', 'marginAsset': 'USDT', 'quoteQty': '209.5800', 'commission': '0.00024131', 'commissionAsset': 'BNB', 'time': 1677110489921, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32995838, 'orderId': 3936576836, 'side': 'BUY', 'price': '698.7', 'qty': '0.928', 'realizedPnl': '-10.30080000', 'marginAsset': 'USDT', 'quoteQty': '648.3936', 'commission': '0.00074656', 'commissionAsset': 'BNB', 'time': 1677110489921, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32995839, 'orderId': 3936576836, 'side': 'BUY', 'price': '698.7', 'qty': '1.097', 'realizedPnl': '-12.17670000', 'marginAsset': 'USDT', 'quoteQty': '766.4739', 'commission': '0.00088252', 'commissionAsset': 'BNB', 'time': 1677110489921, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32995840, 'orderId': 3936576836, 'side': 'BUY', 'price': '698.7', 'qty': '0.486', 'realizedPnl': '-5.39460000', 'marginAsset': 'USDT', 'quoteQty': '339.5682', 'commission': '0.00039098', 'commissionAsset': 'BNB', 'time': 1677110489921, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32995841, 'orderId': 3936576836, 'side': 'BUY', 'price': '698.8', 'qty': '0.037', 'realizedPnl': '-0.41440000', 'marginAsset': 'USDT', 'quoteQty': '25.8556', 'commission': '0.00002977', 'commissionAsset': 'BNB', 'time': 1677110489921, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32995842, 'orderId': 3936576836, 'side': 'BUY', 'price': '698.8', 'qty': '1.016', 'realizedPnl': '-11.37920000', 'marginAsset': 'USDT', 'quoteQty': '709.9808', 'commission': '0.00081748', 'commissionAsset': 'BNB', 'time': 1677110489921, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32995843, 'orderId': 3936576836, 'side': 'BUY', 'price': '698.9', 'qty': '0.034', 'realizedPnl': '-0.38420000', 'marginAsset': 'USDT', 'quoteQty': '23.7626', 'commission': '0.00002736', 'commissionAsset': 'BNB', 'time': 1677110489921, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32995844, 'orderId': 3936576836, 'side': 'BUY', 'price': '698.9', 'qty': '0.038', 'realizedPnl': '-0.42940000', 'marginAsset': 'USDT', 'quoteQty': '26.5582', 'commission': '0.00003057', 'commissionAsset': 'BNB', 'time': 1677110489921, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}, {'symbol': 'DEFIUSDT', 'id': 32995845, 'orderId': 3936576836, 'side': 'BUY', 'price': '699', 'qty': '0.012', 'realizedPnl': '-0.13680000', 'marginAsset': 'USDT', 'quoteQty': '8.3880', 'commission': '0.00000965', 'commissionAsset': 'BNB', 'time': 1677110489921, 'positionSide': 'SHORT', 'buyer': True, 'maker': False}]
    # if len(data_all) > 0:
    #     format = data_all[0]
    # keys = format.keys()
    # df = pd.DataFrame.from_records(data, index=keys)

    df = pd.DataFrame.from_records(data_all)
    # df = df[(df.realizedPnl != '0')]

    df['sid'] = df.apply(lambda x: x['symbol'] + '_' + str(x['orderId']), axis=1)
    df['realizedPnl_f'] = df.apply(lambda x: float(x['realizedPnl']), axis=1)
    df['commission_f'] = df.apply(lambda x: float(x['commission']), axis=1)
    df['qty_f'] = df.apply(lambda x: float(x['qty']), axis=1)
    df['quoteQty_f'] = df.apply(lambda x: float(x['quoteQty']), axis=1)
    df['datetime'] = df.apply(lambda x: str(dt.datetime.fromtimestamp(float(x['time']) / 1000).strftime('%Y-%m-%d %H:%M:%S')), axis=1)
    df['date'] = df.apply(lambda x: str(dt.datetime.fromtimestamp(float(x['time']) / 1000).strftime('%Y-%m-%d')), axis=1)

    df['sum_realizedPnl'] = df.groupby(['sid'])['realizedPnl_f'].transform('sum')
    df['sum_commission'] = df.groupby(['sid'])['commission_f'].transform('sum')



    df['sum_qty'] = df.groupby(['sid'])['qty_f'].transform('sum')
    df['sum_quoteQty'] = df.groupby(['sid'])['quoteQty_f'].transform('sum')
    print(df)

    # df = df.drop_duplicates(subset=['sid']).drop('symbol', 1)
    df = df.drop_duplicates(subset=['sid'])
    print(df)
    return df


result_df = account_trades_df(symbols)
result_df = result_df[(result_df.realizedPnl != '0')]
result_df.sort_values(by='time', ascending=False, inplace=True)  # https://sparkbyexamples.com/pandas/sort-pandas-dataframe-by-date/

print(result_df)

# from here, statistics
result_df['pnl_win'] = result_df.apply(lambda x: x['sum_realizedPnl'] if float(x['sum_realizedPnl']) > 0 else 0, axis=1)
result_df['pnl_loss'] = result_df.apply(lambda x: x['sum_realizedPnl'] if float(x['sum_realizedPnl']) <= 0 else 0, axis=1)
result_df['pnl_winmark'] = result_df.apply(lambda x: 1 if float(x['sum_realizedPnl']) > 0 else 0, axis=1)

# print(result_df)
total_realizedPnl_sum = result_df['sum_realizedPnl'].sum()
total_commission = result_df['sum_commission'].sum()
total_pnl_win_sum = result_df['pnl_win'].sum()
total_pnl_loss_sum = result_df['pnl_loss'].sum()

print(result_df)

trade_cnt = len(result_df)
trade_win_cnt = result_df['pnl_winmark'].sum()

if trade_cnt:
    winrate = trade_win_cnt/trade_cnt
    print('winrate: %s' % str(winrate))

print('total_realizedPnl_sum: %s' % str(total_realizedPnl_sum))
print('total_pnl_win_sum: %s' % str(total_pnl_win_sum))
print('total_pnl_loss_sum: %s' % str(total_pnl_loss_sum))

print('total_commission: %s' % str(total_commission))



