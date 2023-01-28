from binancefutures.api import API


class UMFutures(API):
    def __init__(self, key=None, secret=None, **kwargs):
        if "base_url" not in kwargs:
            kwargs["base_url"] = "https://fapi.binance.com"
        super().__init__(key, secret, **kwargs)

    # MARKETS
    from binancefutures.um_futures.market import ping
    from binancefutures.um_futures.market import time
    from binancefutures.um_futures.market import exchange_info
    from binancefutures.um_futures.market import depth
    from binancefutures.um_futures.market import trades
    from binancefutures.um_futures.market import historical_trades
    from binancefutures.um_futures.market import agg_trades
    from binancefutures.um_futures.market import klines
    from binancefutures.um_futures.market import continuous_klines
    from binancefutures.um_futures.market import index_price_klines
    from binancefutures.um_futures.market import mark_price_klines
    from binancefutures.um_futures.market import mark_price
    from binancefutures.um_futures.market import funding_rate
    from binancefutures.um_futures.market import ticker_24hr_price_change
    from binancefutures.um_futures.market import ticker_price
    from binancefutures.um_futures.market import book_ticker
    from binancefutures.um_futures.market import open_interest
    from binancefutures.um_futures.market import open_interest_hist
    from binancefutures.um_futures.market import top_long_short_position_ratio
    from binancefutures.um_futures.market import long_short_account_ratio
    from binancefutures.um_futures.market import top_long_short_account_ratio
    from binancefutures.um_futures.market import taker_long_short_ratio
    from binancefutures.um_futures.market import blvt_kline
    from binancefutures.um_futures.market import index_info
    from binancefutures.um_futures.market import asset_Index

    # ACCOUNT(including orders and trades)
    from binancefutures.um_futures.account import change_position_mode
    from binancefutures.um_futures.account import get_position_mode
    from binancefutures.um_futures.account import change_multi_asset_mode
    from binancefutures.um_futures.account import get_multi_asset_mode
    from binancefutures.um_futures.account import new_order
    from binancefutures.um_futures.account import new_order_test
    from binancefutures.um_futures.account import new_batch_order
    from binancefutures.um_futures.account import query_order
    from binancefutures.um_futures.account import cancel_order
    from binancefutures.um_futures.account import cancel_open_orders
    from binancefutures.um_futures.account import cancel_batch_order
    from binancefutures.um_futures.account import countdown_cancel_order
    from binancefutures.um_futures.account import get_open_orders
    from binancefutures.um_futures.account import get_orders
    from binancefutures.um_futures.account import get_all_orders
    from binancefutures.um_futures.account import balance
    from binancefutures.um_futures.account import account
    from binancefutures.um_futures.account import change_leverage
    from binancefutures.um_futures.account import change_margin_type
    from binancefutures.um_futures.account import modify_isolated_position_margin
    from binancefutures.um_futures.account import get_position_margin_history
    from binancefutures.um_futures.account import get_position_risk
    from binancefutures.um_futures.account import get_account_trades
    from binancefutures.um_futures.account import get_income_history
    from binancefutures.um_futures.account import leverage_brackets
    from binancefutures.um_futures.account import adl_quantile
    from binancefutures.um_futures.account import force_orders
    from binancefutures.um_futures.account import api_trading_status
    from binancefutures.um_futures.account import commission_rate
    from binancefutures.um_futures.account import download_transactions_asyn
    from binancefutures.um_futures.account import aysnc_download_info

    # STREAMS
    from binancefutures.um_futures.data_stream import new_listen_key
    from binancefutures.um_futures.data_stream import renew_listen_key
    from binancefutures.um_futures.data_stream import close_listen_key

    # PORTFOLIO MARGIN
    from binancefutures.um_futures.portfolio_margin import pm_exchange_info
