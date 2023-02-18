#!/usr/bin/env python
import logging
from binancefutures.um_futures import UMFutures
from binancefutures.lib.utils import config_logging
from binancefutures.error import ClientError

config_logging(logging, logging.DEBUG)

key = "IkzH8WHKl0lGzOSqiZZ4TnAyKnDpqnC9Xi31kzrRNpwJCp28gP8AuWDxntSqWdrn"
secret = "FwKTmQ2RWSiECMfhZOaY7Hed45JuXqlEPno2xiLGgCzloLq4NMMcmusG6gtMCKa5"

um_futures_client = UMFutures(key=key, secret=secret)

def get_i_r(list, key, value):
    r = [[i, x] for i, x in enumerate(list) if x[key] == value]
    if len(r) == 1:
        return r[0][0], r[0][1]
    return None, None


def filtered_order_after_1dayago(response, status=None):
    response_filtered = []
    if len(response) > 0:
        response_filled_trades = [x for x in response if x['status'] == status]
        for x in response_filled_trades:
            dt_object = datetime.fromtimestamp(x['time'] / 1000)
            ini_time_for_now = datetime.now()
            datetime_1days_ago = ini_time_for_now - timedelta(days=1)
            if x['time']/1000 >= datetime_1days_ago.timestamp():
                x['time_date'] = dt_object
                response_filtered.append(x)
    return response_filtered

import time
from datetime import datetime, timedelta

try:
    response = um_futures_client.get_all_orders(symbol="WAVESBUSD", recvWindow=6000)
    logging.info(response)

    filtered_order_after_1dayago(response, status='FILLED')

except ClientError as error:
    logging.error(
        "Found error. status: {}, error code: {}, error message: {}".format(
            error.status_code, error.error_code, error.error_message
        )
    )
pass
# success when new order
# DEBUG:urllib3.connectionpool:https://fapi.binance.com:443 "GET /fapi/v1/allOrders?symbol=WOOUSDT&recvWindow=2000&timestamp=1674267096006&signature=a0bf28eedeaa9bff39f94bd66cdb88bd1abebd2cb1f2ce1869848f2ee032ed05 HTTP/1.1" 200 None
# DEBUG:root:raw response from server:[{"orderId":1298843319,"symbol":"WOOUSDT","status":"CANCELED","clientOrderId":"web_st_NH7H4YnClZWvykx","price":"0.19054","avgPrice":"0.00000","origQty":"2099","executedQty":"0","cumQuote":"0","timeInForce":"GTC","type":"LIMIT","reduceOnly":false,"closePosition":false,"side":"SELL","positionSide":"SHORT","stopPrice":"0","workingType":"CONTRACT_PRICE","priceProtect":false,"origType":"LIMIT","time":1674233712731,"updateTime":1674235391024},{"orderId":1300638258,"symbol":"WOOUSDT","status":"NEW","clientOrderId":"wave1111","price":"0.21254","avgPrice":"0.00000","origQty":"355","executedQty":"0","cumQuote":"0","timeInForce":"GTC","type":"LIMIT","reduceOnly":false,"closePosition":false,"side":"SELL","positionSide":"SHORT","stopPrice":"0","workingType":"CONTRACT_PRICE","priceProtect":false,"origType":"LIMIT","time":1674266989020,"updateTime":1674266989020}]
# INFO:root:[{'orderId': 1298843319, 'symbol': 'WOOUSDT', 'status': 'CANCELED', 'clientOrderId': 'web_st_NH7H4YnClZWvykx', 'price': '0.19054', 'avgPrice': '0.00000', 'origQty': '2099', 'executedQty': '0', 'cumQuote': '0', 'timeInForce': 'GTC', 'type': 'LIMIT', 'reduceOnly': False, 'closePosition': False, 'side': 'SELL', 'positionSide': 'SHORT', 'stopPrice': '0', 'workingType': 'CONTRACT_PRICE', 'priceProtect': False, 'origType': 'LIMIT', 'time': 1674233712731, 'updateTime': 1674235391024}, {'orderId': 1300638258, 'symbol': 'WOOUSDT', 'status': 'NEW', 'clientOrderId': 'wave1111', 'price': '0.21254', 'avgPrice': '0.00000', 'origQty': '355', 'executedQty': '0', 'cumQuote': '0', 'timeInForce': 'GTC', 'type': 'LIMIT', 'reduceOnly': False, 'closePosition': False, 'side': 'SELL', 'positionSide': 'SHORT', 'stopPrice': '0', 'workingType': 'CONTRACT_PRICE', 'priceProtect': False, 'origType': 'LIMIT', 'time': 1674266989020, 'updateTime': 1674266989020}]


# after cancel_order
# EBUG:urllib3.connectionpool:https://fapi.binance.com:443 "GET /fapi/v1/allOrders?symbol=WOOUSDT&recvWindow=2000&timestamp=1674267424873&signature=bafb43eef0c7121d747d05f764214cee9108d62cf252efdd5754f7cd2d43c705 HTTP/1.1" 200 None
# DEBUG:root:raw response from server:[{"orderId":1298843319,"symbol":"WOOUSDT","status":"CANCELED","clientOrderId":"web_st_NH7H4YnClZWvykx","price":"0.19054","avgPrice":"0.00000","origQty":"2099","executedQty":"0","cumQuote":"0","timeInForce":"GTC","type":"LIMIT","reduceOnly":false,"closePosition":false,"side":"SELL","positionSide":"SHORT","stopPrice":"0","workingType":"CONTRACT_PRICE","priceProtect":false,"origType":"LIMIT","time":1674233712731,"updateTime":1674235391024},{"orderId":1300638258,"symbol":"WOOUSDT","status":"CANCELED","clientOrderId":"wave1111","price":"0.21254","avgPrice":"0.00000","origQty":"355","executedQty":"0","cumQuote":"0","timeInForce":"GTC","type":"LIMIT","reduceOnly":false,"closePosition":false,"side":"SELL","positionSide":"SHORT","stopPrice":"0","workingType":"CONTRACT_PRICE","priceProtect":false,"origType":"LIMIT","time":1674266989020,"updateTime":1674267383844}]
# INFO:root:[{'orderId': 1298843319, 'symbol': 'WOOUSDT', 'status': 'CANCELED', 'clientOrderId': 'web_st_NH7H4YnClZWvykx', 'price': '0.19054', 'avgPrice': '0.00000', 'origQty': '2099', 'executedQty': '0', 'cumQuote': '0', 'timeInForce': 'GTC', 'type': 'LIMIT', 'reduceOnly': False, 'closePosition': False, 'side': 'SELL', 'positionSide': 'SHORT', 'stopPrice': '0', 'workingType': 'CONTRACT_PRICE', 'priceProtect': False, 'origType': 'LIMIT', 'time': 1674233712731, 'updateTime': 1674235391024}, {'orderId': 1300638258, 'symbol': 'WOOUSDT', 'status': 'CANCELED', 'clientOrderId': 'wave1111', 'price': '0.21254', 'avgPrice': '0.00000', 'origQty': '355', 'executedQty': '0', 'cumQuote': '0', 'timeInForce': 'GTC', 'type': 'LIMIT', 'reduceOnly': False, 'closePosition': False, 'side': 'SELL', 'positionSide': 'SHORT', 'stopPrice': '0', 'workingType': 'CONTRACT_PRICE', 'priceProtect': False, 'origType': 'LIMIT', 'time': 1674266989020, 'updateTime': 1674267383844}]
