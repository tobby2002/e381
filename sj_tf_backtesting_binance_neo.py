import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import yfinance as yf
import warnings
# set the style and ignore warnings
plt.style.use('seaborn-colorblind')
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
from termcolor import colored
import datetime
import pyfolio as pf
import backtrader as bt
from backtrader.feeds import PandasData
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.models import Sequential
from talib import RSI, BBANDS, MACD
from binancefutures.um_futures import UMFutures


um_futures_client = UMFutures(key=futures_secret_key, secret=futures_secret_value)
statistics = list()

def run(ticker, model_name, df, history_window=50, N=1, epochs=50):
    df_origin = df.copy().set_index(['date'])
    stock = df.set_index(['date'])
    # del stock['date']
    close = stock.close
    # calculate daily log returns and market direction
    stock['returns'] = np.log(close / close.shift(1))
    stock.dropna(inplace=True)
    stock['direction'] = np.sign(stock['returns']).astype(int)
    print(stock.head(3))

    # visualize the closing price and daily returns
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
    ax[0].plot(stock.close, label=f'{ticker} Close')
    ax[0].set(title=f'{ticker} Closing Price_0', ylabel='Price')
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(stock['returns'], label='Daily Returns')
    ax[1].set(title=f'{ticker} Daily Returns_0', ylabel='Returns')
    ax[1].grid(True)
    plt.legend()

    plt.tight_layout();
    plt.savefig('images/chart1', dpi=300)
    plt.show()

    ### Feature Engineering
    # define the number of lags
    # compute lagged log returns
    cols = []
    for lag in range(1, history_window + 1, 1):
        col = f'rtn_lag{lag}'
        stock[col] = stock['returns'].shift(lag)
        cols.append(col)
    stock.head(2)

    # RSI - Relative Strenght Index
    stock['rsi'] = RSI(stock.close)
    cols.append('rsi')
    #
    # # Bollinger Bands
    # high, mid, low = BBANDS(stock.close, timeperiod=20)
    # stock = stock.join(pd.DataFrame({'bb_high': high, 'bb_low': low}, index=stock.index))
    # cols.append('bb_high')
    # cols.append('bb_low')
    #
    # # Compute Moving Average Convergence/ Divergence
    stock['macd'] = MACD(stock.close)[0]
    cols.append('macd')

    # # let's look at the head and tail of our dataframe
    # stock.head().append(stock.tail())

    ### Build and Apply the Model
    # print(cols)

    # split the dataset in training and test datasets
    train, test = train_test_split(stock.dropna(), test_size=0.3, shuffle=False)

    # sort the data on date index
    train = train.copy().sort_index()
    test = test.copy().sort_index()

    # # view train dataset
    # train.tail()

    # # view test dataset
    # test.tail()
    # define a function to create the deep neural network model
    def create_model():
        # seed 고정 start
        import os
        SEED = 42
        os.environ['PYTHONHASHSEED'] = str(SEED)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'

        import random
        random.seed(SEED)

        tf.random.set_seed(SEED)
        np.random.seed(SEED)
        # seed 고정 end
        # tf.random.set_seed(SEED)
        # np.random.seed(SEED)

        model = Sequential()
        model.add(Dense(64 * N, activation='relu', input_dim=len(cols)))
        model.add(Dense(64 * N, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam',  # 'rmsprop',
                      metrics=['accuracy'])
        return model

    # normalized the training dataset
    mu, std = train.mean(), train.std()
    train_ = (train - mu) / mu.std()

    # create the model
    model = create_model()

    # map market direction of (1,-1) to (1,0)
    train['direction_'] = np.where(train['direction'] > 0, 1, 0)

    if is_load:
        try:
            model = tf.keras.models.load_model(model_name)
        except Exception as e:
            print(e, ', Now making new %s' % model_name)
            pass

    if is_train:
        # train_.head()
        # fit the model for training dataset
        r = model.fit(train_[cols], train['direction_'], epochs=epochs * N, verbose=False)
        if is_save:
            model.save(model_name)

    # normalized the test dataset
    mu, std = test.mean(), test.std()
    test_ = (test - mu) / std

    # map market direction of (1,-1) to (1,0)
    test['direction_'] = np.where(test['direction'] > 0, 1, 0)

    # evaluate the model with test dataset
    model.evaluate(test_[cols], test['direction_'])

    # predict the direction and map it (1,0)
    pred = np.where(model.predict(test_[cols]) > 0.5, 1, 0)
    pred[:10].flatten()

    # based on prediction calculate the position for strategy
    test['position_strategy'] = np.where(pred > 0, 1, -1)

    # calculate daily returns for the strategy
    test['strategy_return'] = test['position_strategy'] * test['returns']

    # test.head()

    # calculate total return and std. deviation of each strategy
    print('\nTotal Returns:')
    print(test[['returns', 'strategy_return']].sum().apply(np.exp))
    print('\nAnnual Volatility:')
    print(test[['returns', 'strategy_return']].std() * 252 ** 0.5)

    # number of trades over time for the strategy
    print('Number of trades = ', (test['position_strategy'].diff() != 0).sum())

    # plot cumulative returns
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(14, 6))
    ax.plot(test.returns.cumsum().apply(np.exp), label=ticker + ' Buy and Hold')
    ax.plot(test.strategy_return.cumsum().apply(np.exp), label='Strategy')
    ax.set(title=ticker + ' Buy and Hold vs. Strategy_1', ylabel='Cumulative Returns')
    ax.grid(True)
    ax.legend()
    plt.savefig('images/chart2');
    plt.show()

    ### Backtesting using Backtrader

    # backtesting start and end dates
    start = test.index[0]
    end = test.index[-1]
    print(start)
    print(end)

    # for signal dataname

    prices = df_origin
    prices.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume', }, inplace=True)
    prices.head(3)

    # add the predicted column to data_signal dataframe. This will be used as signal for buy or sell
    predictions = test.strategy_return
    predictions = pd.DataFrame(predictions)
    predictions.rename(columns={'strategy_return': 'predicted'}, inplace=True)
    prices = predictions.join(prices, how='right').dropna()
    print(prices.head(2))

    prices[['predicted']].sum().apply(np.exp)

    OHLCV = ['open', 'high', 'low', 'close', 'volume']

    # class to define the columns we will provide
    class SignalData(PandasData):
        """
        Define pandas DataFrame structure
        """
        cols = OHLCV + ['predicted']

        # create lines
        lines = tuple(cols)

        # define parameters
        params = {c: -1 for c in cols}
        params.update({'datetime': None})
        params = tuple(params.items())

    # Strategy:
    # 1.	Buy when the predicted value is +1 and sell (only if stock is in possession) when the predicted value is -1.
    # 2.	All-in strategy—when creating a buy order, buy as many shares as possible.
    # 3.	Short selling is not allowed

    # define backtesting strategy class
    class MLStrategy(bt.Strategy):
        params = dict(
        )

        def __init__(self):
            # keep track of open, close prices and predicted value in the series
            self.data_predicted = self.datas[0].predicted
            self.data_open = self.datas[0].open
            self.data_close = self.datas[0].close

            # keep track of pending orders/buy price/buy commission
            self.order = None
            self.price = None
            self.comm = None

        # logging function
        def log(self, txt):
            '''Logging function'''
            dt = self.datas[0].datetime.date(0).isoformat()
            print(f'{dt}, {txt}')

        def log(self, txt, send_telegram=False, color=None):
            value = datetime.datetime.now()
            if len(self) > 0:
                value = self.data0.datetime.datetime()
            if color:
                txt = colored(txt, color)
            print('[%s] %s' % (value.strftime("%d-%m-%y %H:%M"), txt))

        def notify_order(self, order):
            if order.status in [order.Submitted, order.Accepted]:
                # order already submitted/accepted - no action required
                return

            # report executed order
            if order.status in [order.Completed]:
                if order.isbuy():
                    self.log(
                        f'BUY EXECUTED --- Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f},Commission: {order.executed.comm:.2f}'
                    )
                    self.price = order.executed.price
                    self.comm = order.executed.comm
                else:
                    self.log(
                        f'SELL EXECUTED --- Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f},Commission: {order.executed.comm:.2f}'
                    )

            # report failed order
            elif order.status in [order.Canceled, order.Margin,
                                  order.Rejected]:
                self.log('Order Failed')

            # set no pending order
            self.order = None

        def notify_trade(self, trade):
            date = self.data.datetime.datetime()
            if trade.isclosed:
                print(
                    '{} [T]     TradeId: {}, Symbol: {}, Close Price: {}, Profit, Gross {}, Net {}, Comm {}, Cash {}, Value {}'.format(
                        date, trade.tradeid, trade.getdataname(), trade.price, round(trade.pnl, 2),
                        round(trade.pnlcomm, 2),
                        round(trade.commission, 2),
                        self.broker.get_cash(),
                        self.broker.get_value(),
                    ))

            if not trade.isclosed:
                return
            # self.log(f'OPERATION RESULT --- Gross: {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}')

            if trade.isopen:
                pass

            color = 'green'
            if trade.pnl < 0:
                color = 'red'

            # self.log(colored('OPERATION PROFIT, GROSS %.2f, NET %.2f' % (trade.pnl, trade.pnlcomm), color), True)
            # print('               %s OPERATION PROFIT, GROSS %.2f, NET %.2f' % (color, trade.pnl, trade.pnlcomm))

        # We have set cheat_on_open = True.This means that we calculated the signals on day t's close price,
        # but calculated the number of shares we wanted to buy based on day t+1's open price.
        def next_open(self):
            if not self.position:
                if self.data_predicted > 0:
                    # calculate the max number of shares ('all-in')
                    size = int(self.broker.getcash() / self.datas[0].open)
                    size = int(size*0.99) # by neo
                    # buy order
                    #                 self.log(f'BUY CREATED --- Size: {size}, Cash: {self.broker.getcash():.2f}, Open: {self.data_open[0]}, Close: {self.data_close[0]}')
                    self.buy(size=size)
            else:
                if self.data_predicted < 0:
                    # sell order
                    #                 self.log(f'SELL CREATED --- Size: {self.position.size}')
                    self.sell(size=self.position.size)

    # instantiate SignalData class
    data = SignalData(dataname=prices)

    # instantiate Cerebro, add strategy, data, initial cash, commission and pyfolio for performance analysis
    cerebro = bt.Cerebro(stdstats=False, cheat_on_open=True)
    cerebro.addstrategy(MLStrategy)
    cerebro.adddata(data, name=ticker)
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.0004)
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')

    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="ta")
    cerebro.addanalyzer(bt.analyzers.SQN, _name="sqn")

    # run the backtest
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    results = cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    max_dd = results[0].analyzers.drawdown.get_analysis()["max"]["moneydown"]
    print('Final max_dd : %.2f' % max_dd)

    def print_sqn(analyzer):
        sqn = round(analyzer.sqn, 2)
        print('SQN: {}'.format(sqn))

    def print_trade_analysis(analyzer):
        # Get the results we are interested in
        if not analyzer.get("total"):
            return

        total_open = analyzer.total.open
        total_closed = analyzer.total.closed
        total_won = analyzer.won.total
        total_lost = analyzer.lost.total
        total_win_rate = round(total_won / total_closed, 2)
        win_streak = analyzer.streak.won.longest
        lose_streak = analyzer.streak.lost.longest
        pnl_net = round(analyzer.pnl.net.total, 2)
        strike_rate = round((total_won / total_closed) * 2)

        # Designate the rows
        h1 = ['Total Open', 'Total Closed', 'Total Won', 'Total Lost', 'Total Winrate']
        h2 = ['Strike Rate', 'Win Streak', 'Losing Streak', 'PnL Net', ' ']
        r1 = [total_open, total_closed, total_won, total_lost, total_win_rate]
        r2 = [strike_rate, win_streak, lose_streak, pnl_net, ' ']

        statistics.append(pnl_net)
        print('##' * 20)
        print('####### statistics:', len(statistics), sum(statistics) / len(statistics), statistics)
        print('##' * 20)

        # Check which set of headers is the longest.
        if len(h1) > len(h2):
            header_length = len(h1)
        else:
            header_length = len(h2)

        # Print the rows
        print_list = [h1, r1, h2, r2]
        row_format = "{:<15}" * (header_length + 1)
        print("Trade Analysis Results:")
        for row in print_list:
            print(row_format.format('', *row))

    for r in results:
        try:
            print_sqn(r.analyzers.sqn.get_analysis())
            print_trade_analysis(r.analyzers.ta.get_analysis())
        except Exception as e:
            print(str(e))
    # cerebro.plot(style='bar', numfigs=1, volume=False)
    print(ticker, cols)

    try:
        # Extract inputs for pyfolio
        strat = results[0]
        pyfoliozer = strat.analyzers.getbyname('pyfolio')
        returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
        returns.name = 'Strategy'
        returns.head(2)

        # get benchmark returns
        benchmark_rets = stock['returns']
        benchmark_rets.index = benchmark_rets.index.tz_localize('UTC')
        benchmark_rets = benchmark_rets.filter(returns.index)
        benchmark_rets.name = ticker + '-Ticker'
        benchmark_rets.head(2)

        # get performance statistics for strategy
        # pf.show_perf_stats(returns)   by neo cause of error

        # plot performance for strategy vs benchmark
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 9), constrained_layout=True)
        axes = ax.flatten()

        # pf.plot_drawdown_periods(returns=returns, ax=axes[0])
        axes[0].grid(True)
        pf.plot_rolling_returns(returns=returns, factor_returns=benchmark_rets, ax=axes[1], title='Strategy vs %s-Ticker' % ticker)
        axes[1].grid(True)
        pf.plot_drawdown_underwater(returns=returns, ax=axes[2])
        axes[2].grid(True)
        pf.plot_rolling_sharpe(returns=returns, ax=axes[3])
        axes[3].grid(True)
        fig.suptitle(ticker + '_3', fontsize=16, y=0.990)

        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('images/chart3', dpi=300)
        plt.show()

        # plot performance for strategy vs benchmark
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 9), constrained_layout=True)
        axes = ax.flatten()

        pf.plot_rolling_beta(returns=returns, factor_returns=benchmark_rets, ax=axes[0])
        axes[0].grid(True)

        pf.plot_rolling_volatility(returns=returns, factor_returns=benchmark_rets, ax=axes[1])
        axes[1].grid(True)

        pf.plot_annual_returns(returns=returns, ax=axes[2])
        axes[2].grid(True)

        pf.plot_monthly_returns_heatmap(returns=returns, ax=axes[3], )
        fig.suptitle(ticker + '_chart_4', fontsize=16, y=1.0)

        plt.tight_layout()
        plt.savefig('images/chart4', dpi=300)
        plt.show()
    except Exception as e:
        print('Exception 101', str(e))

def get_fetch_dohlcv(symbol,
                     interval=None,
                     limit=500):
    datalist = um_futures_client.klines(symbol, interval, limit=limit)

    if datalist:
        D = pd.DataFrame(datalist)
        D.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades',
                     'taker_base_vol', 'taker_quote_vol', 'is_best_match']
        # D['open_date_time'] = [datetime.datetime.fromtimestamp(x / 1000).strftime('%Y-%m-%d %H:%M:%S') for x in D.open_time]
        D['open_date_time'] = [datetime.datetime.fromtimestamp(x / 1000) for x in D.open_time]

        D['symbol'] = symbol
        D = D[['symbol', 'open_date_time', 'open', 'high', 'low', 'close', 'volume', 'num_trades', 'taker_base_vol',
               'taker_quote_vol']]
        D.rename(columns={
            "open_date_time": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }, inplace=True)
        new_names = ['date', 'open', 'high', 'low', 'close', 'volume']
        # D['date'] = D['date'].astype(str)
        D['open'] = D['open'].astype(float)
        D['high'] = D['high'].astype(float)
        D['low'] = D['low'].astype(float)
        D['close'] = D['close'].astype(float)
        D['volume'] = D['volume'].astype(float)
        D = D[new_names]
    return D


def show_symbol(symbol, interval='1h', limit=500):
    from binance.client import Client
    api_key = ""
    secret_key = "t"
    client = Client(api_key, secret_key)
    candles = client.get_klines(symbol=symbol, interval=interval, limit=limit);
    df = pd.DataFrame(candles)
    df.columns = ['Open time', 'open', 'high', 'low', 'close', 'volume', 'Close time', 'Quote asset volume',
                  'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Can be ignored']
    print(len(candles))
    c = candles[499]  # time-open-high-low-close-volume-close_time
    print(c)
    price_raw = np.array([float(candles[i][4]) for i in range(500)])
    print(price_raw)
    # Fetching opening time from candlesticks data
    time = np.array([int(candles[i][0]) for i in range(500)])
    # Converting time to HH:MM:SS format
    # t = np.array([datetime.datetime.fromtimestamp(time[i]/1000).strftime('%H:%M:%S') for i in range(500)])
    t = np.array([datetime.datetime.fromtimestamp(time[i] / 1000).strftime('%Y-%m-%d %H:%M:%S') for i in range(500)])
    print(price_raw.shape)
    plt.figure(figsize=(8, 5))
    plt.xlabel("Time Step")
    plt.ylabel("Bitcoin Price $")
    plt.plot(price_raw)
    plt.show()


if __name__ == '__main__':

    # tickers = ['BTCUSDT', 'ETHUSDT', 'ONEUSDT', 'ICPUSDT']  # 값이 작은 것은 안됨
    # tickers = ['ICPUSDT']
    # tickers = ['BTCUSDT', 'ETHUSDT']
    # tickers = ['ICPUSDT']

    exchange_info = um_futures_client.exchange_info()
    symbols_binance_futures_USDT = []
    symbols_binance_futures_BUSD = []
    symbols_binance_futures_USDT_BUSD = []
    symbols_binance_futures = []
    if exchange_info:
        for s in exchange_info['symbols']:
            if s['contractType'] == 'PERPETUAL' and s['symbol'][-4:] == 'USDT':
                symbols_binance_futures_USDT.append(s['symbol'])
                symbols_binance_futures_USDT_BUSD.append(s['symbol'])
            elif s['contractType'] == 'PERPETUAL' and s['symbol'][-4:] == 'BUSD':
                symbols_binance_futures_BUSD.append(s['symbol'])
                symbols_binance_futures_USDT_BUSD.append(s['symbol'])
            symbols_binance_futures.append(s['symbol'])

    tickers = symbols_binance_futures_USDT[:20]
    print('symbols:', tickers)

    N = 4
    epochs = 200
    history_window = 5
    interval = '2h'
    is_union = False

    is_load = True
    is_train = True
    is_save = True

    start = datetime.datetime(2010, 1, 1)
    end = datetime.datetime(2023, 8, 1)

    # show_symbol('BTCUSDT', interval='1h', limit=500)

    def loop():
        l = 1
        s = 1
        while True:
            for ticker in tickers:
                df = get_fetch_dohlcv(ticker, interval=interval, limit=1000)
                if is_union:
                    model_name = 'model_usdt_union_N_%s_E_%s_W_%s_%s.h5' % (N, epochs, history_window, interval)
                else:
                    model_name = 'model_usdt_N_%s_E_%s_W_%s_%s_%s.h5' % (N, epochs, history_window, ticker, interval)
                run(ticker, model_name, df, history_window, N, epochs)
                print('##### [s=%s]' % str(s))
                s += 1
            print('##### [l=%s]' % str(1))
            l += 1

    loop()