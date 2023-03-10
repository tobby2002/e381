{
  "default": {
    "version": "0.51",
    "descrition": "045_MTF_TIME_OUT",
    "_exchange": "bybit_usdt_perp, binance_usdt_perp, binance_busd_perp binance_usdt_busd_perp",
    "exchange": "BINANCE",
    "exchange_symbol": "binance_usdt_busd_perp",
    "futures": true,
    "_trade_type": "[long, short] or [long] or [short]",
    "type": ["long", "short"],
    "maxleverage": 100,
    "qtyrate": 0.25,
    "walletrate": 0.15,

    "_c_risk_beyond_desc": ["0.10: 10%,   0.0010: 0.1%"],
    "c_risk_beyond_flg": true,
    "c_risk_beyond_max": 0.13,
    "c_risk_beyond_min": 0.0005,

    "c_time_beyond_flg": false,
    "c_time_beyond_rate": 0.500,
    "o_fibo": 118.0,

    "_period": ["period_days_ago:30 --> 30 days ago", "period_days_ago_till:0 --> now", "period_interval:1 per 1day"],
    "timeframe": [15],
    "period_days_ago": 5,
    "period_days_ago_till": 0,
    "period_interval": 2,
    "window": 720,
    "symbol_duplicated": false,

    "symbol_random": true,
    "symbol_last":false,
    "symbol_length":false,
    "round_trip_flg": false,
    "round_trip_count": 1,
    "compounding": true,

    "_fractal_count": ["5: default, 7, 9 , 11"],
    "_fcnt": [11, 31, 51, 71],
    "fcnt": [31, 51, 71],
    "loop_count": 1,
    "up_to_count": 3,
    "condi_same_date": true,

    "_fee": "bybit_usdt_perp: -0.00025  0.0006    binance_futures: 0.0002    0.0004             0.01 --> 1%, 0.001 --> 0.1%, 0.0001 -> 0.01%   fee_taker:0.0004 시장가, fee_maker: 0.0002 지정가",
    "fee": 0.0000,
    "fee_limit": 0.00018,
    "fee_sl": 0.00036,
    "tp_type": "maker",
    "fee_tp": 0.00018,
    "fee_slippage": 0.00010,

    "high_target": 1.0,
    "low_target": 1.0,
    "low_target_w2": false,

    "h_fibo_rate": [0, 11.8, 23.6, 38.2, 50, 61.8, 76.4, 88.2, 100, -0.75],
    "h_fibo": 100.0,
    "l_fibo": 100.0,
    "h_fibo_rate_ex": [0.01, 0.03, 0.351, 0, 0.058, 0.381, 0.11800000000000, 0.125000000, 0.1910000000000000000, 0.21400000000000, 0.236, 0.382, 0.5, 0.618, 0.764, 0.882, 0.1, 1.618, 2.618],
    "entry_fibo": 0.045,
    "target_fibo": 0.381,
    "sl_fibo": -0.005,

    "profit_long": 0.02,
    "profit_short": 0.02,
    "stop_long": 0.005,
    "stop_short": 0.005,

    "intersect_idx": false,
    "plotview": true,
    "printout": false,
    "multi_process": true,
    "seed": 100,
    "init_running_trade": true,
    "reset_leverage": false
  },
  "basic": {
    "secret_key": "knwlJpZVdWLy20iRnHInxyRi2SXHknbj0tKTO9vJqi7NOno30fDO2y2zYNPyvYZq",
    "secret_value": "ZKlLtBwjRVI2QfQTNyH0vnchxRuTTsHDLZkcbA3JK9Z1ieYvbJeZgVSi8oyA17rE"
  },
  "futures": {
    "secret_key": "FwqJ8hQV79mUbfsUe4BuSuU9G3bmzq5lL3XaKDdRqvI8fYbPOQMU2UGJv4oM1KC0",
    "secret_value": "8yLlpdL0gvNsxbOkhQ1ZEywMY01Sr6y97YJt2vvWPwg4HL9U2MhoHdAut3btA3Wc"
  }
}
