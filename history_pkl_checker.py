import pickle
open_order_history = [
    # {'id':'timestamp', 'symbol':'SYMBOLUSDT', 'wavepattern': wavepattern, 'entry':'10000', 'target':'10381', 'status':'NEW or DRAW or TAKE_PROFIT or DONE',  'data': [{limit_result}, {sl_result}, {tp_result}], 'position':[]}
    # {'id':'1234567890.1234', 'symbol': 'BTCUSDT', 'wavepattern': wavepattern, 'entry':'10000', 'target':'10381', 'status':'NEW' 'data': [{'orderId': 1300759837, 'symbol': 'WOOUSDT', 'status': 'NEW', 'clientOrderId': 'waveshortlimit001', 'price': '0.21310', 'avgPrice': '0.00000', 'origQty': '30', 'executedQty': '0', 'cumQty': '0', 'cumQuote': '0', 'timeInForce': 'GTC', 'type': 'LIMIT', 'reduceOnly': False, 'closePosition': False, 'side': 'SELL', 'positionSide': 'SHORT', 'stopPrice': '0', 'workingType': 'CONTRACT_PRICE', 'priceProtect': False, 'origType': 'LIMIT', 'updateTime': 1674269613350}, {'orderId': 1300759838, 'symbol': 'WOOUSDT', 'status': 'NEW', 'clientOrderId': 'waveshortlimit001sl', 'price': '0', 'avgPrice': '0.00000', 'origQty': '30', 'executedQty': '0', 'cumQty': '0', 'cumQuote': '0', 'timeInForce': 'GTC', 'type': 'STOP_MARKET', 'reduceOnly': True, 'closePosition': False, 'side': 'BUY', 'positionSide': 'SHORT', 'stopPrice': '0.22103', 'workingType': 'CONTRACT_PRICE', 'priceProtect': False, 'origType': 'STOP_MARKET', 'updateTime': 1674269613350}]}
]
import os
def load_history_pkl():
    try:
        with open('oohistory_17532.pkl', 'rb') as f:
            h = pickle.load(f)
            print('load_history_pk:' + str(h))
            return h
    except Exception as e:
        print(e)
        try:
            os.remove("open_order_history.pkl")
        except Exception as e:
            print(e)
        return []
    return []

open_order_history = load_history_pkl()
print(len(open_order_history))
print(str(open_order_history))