from binancefutures.api import API


class CMFutures(API):
    def __init__(self, key=None, secret=None, **kwargs):
        if "base_url" not in kwargs:
            kwargs["base_url"] = "https://dapi.binance.com"
        super().__init__(key, secret, **kwargs)

    # MARKETS
    from binancefutures.cm_futures.market import ping
    from binancefutures.cm_futures.market import time
    from binancefutures.cm_futures.market import exchange_info
    from binancefutures.cm_futures.market import depth
    from binancefutures.cm_futures.market import trades
    from binancefutures.cm_futures.market import historical_trades
    from binancefutures.cm_futures.market import agg_trades
    from binancefutures.cm_futures.market import klines
    from binancefutures.cm_futures.market import continuous_klines
    from binancefutures.cm_futures.market import index_price_klines
    from binancefutures.cm_futures.market import mark_price_klines
    from binancefutures.cm_futures.market import mark_price
    from binancefutures.cm_futures.market import funding_rate
    from binancefutures.cm_futures.market import ticker_24hr_price_change
    from binancefutures.cm_futures.market import ticker_price
    from binancefutures.cm_futures.market import book_ticker
    from binancefutures.cm_futures.market import open_interest
    from binancefutures.cm_futures.market import open_interest_hist
    from binancefutures.cm_futures.market import top_long_short_account_ratio
    from binancefutures.cm_futures.market import top_long_short_position_ratio
    from binancefutures.cm_futures.market import long_short_account_ratio
    from binancefutures.cm_futures.market import taker_long_short_ratio
    from binancefutures.cm_futures.market import basis

    # ACCOUNT(including orders and trades)
    from binancefutures.cm_futures.account import change_position_mode
    from binancefutures.cm_futures.account import get_position_mode
    from binancefutures.cm_futures.account import new_order
    from binancefutures.cm_futures.account import modify_order
    from binancefutures.cm_futures.account import new_batch_order
    from binancefutures.cm_futures.account import modify_batch_order
    from binancefutures.cm_futures.account import order_modify_history
    from binancefutures.cm_futures.account import query_order
    from binancefutures.cm_futures.account import cancel_order
    from binancefutures.cm_futures.account import cancel_open_orders
    from binancefutures.cm_futures.account import cancel_batch_order
    from binancefutures.cm_futures.account import countdown_cancel_order
    from binancefutures.cm_futures.account import get_open_orders
    from binancefutures.cm_futures.account import get_orders
    from binancefutures.cm_futures.account import get_all_orders
    from binancefutures.cm_futures.account import balance
    from binancefutures.cm_futures.account import account
    from binancefutures.cm_futures.account import change_leverage
    from binancefutures.cm_futures.account import change_margin_type
    from binancefutures.cm_futures.account import modify_isolated_position_margin
    from binancefutures.cm_futures.account import get_position_margin_history
    from binancefutures.cm_futures.account import get_position_risk
    from binancefutures.cm_futures.account import get_account_trades
    from binancefutures.cm_futures.account import get_income_history
    from binancefutures.cm_futures.account import leverage_brackets
    from binancefutures.cm_futures.account import adl_quantile
    from binancefutures.cm_futures.account import force_orders
    from binancefutures.cm_futures.account import commission_rate

    # STREAMS
    from binancefutures.cm_futures.data_stream import new_listen_key
    from binancefutures.cm_futures.data_stream import renew_listen_key
    from binancefutures.cm_futures.data_stream import close_listen_key

    # PORTFOLIO MARGIN
    from binancefutures.cm_futures.portfolio_margin import pm_exchange_info
