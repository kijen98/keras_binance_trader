import os
import gc
import time
import math
import csv
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from binance.client import Client

X_len = 128
time_windows = [1,2,4,8,16]
date_length = (X_len-1) * time_windows[-1] + 1

def get_close_data(symbol):
    client = Client()
    now = client.get_server_time()['serverTime'] - 1000*60 # 1분 전의 ticker를 얻어야 함
    since = now - date_length * 1000 * 60
    klines = client.futures_historical_klines(symbol, Client.KLINE_INTERVAL_1MINUTE, since, now)
    close_data = np.array(klines, dtype=np.float64)[-date_length:,4:5]

    return close_data

def get_X_data(close_data):
    mul_rate = 100
    X_data = np.zeros((1, X_len, len(time_windows)))
    for i_time_window in range(len(time_windows)):
        time_window = time_windows[i_time_window]
        hour_data = close_data[-(X_len-1)*time_window-1::time_window,0] # (X_len,)
        today_value = hour_data[-1]
        
        rate_data = np.log(hour_data/today_value) * mul_rate / (math.log(time_window,2)+1)
        X_data[0,:,i_time_window] = rate_data
        
    return X_data

def floor_digit(val, digits):
    val *= 10 ** (digits)
    return '{1:.{0}f}'.format(digits, math.floor(val) / 10 ** digits)

class StockTrader_Real:
    def __init__(self, predictor, ticker, apiKey, secret):
        self.predictor = predictor # 예측모델
        self.coin = ticker[0]
        self.stable = ticker[1]
        self.symbol = self.coin + self.stable
        self.margin_rate = 3.0
        self.limit_rate = 0.54 / 100
        th = 0.0
        self.thresholds = [th,-th]

        self.apiKey = apiKey
        self.secret = secret
        
        self.now = datetime.fromtimestamp(Client().get_server_time()['serverTime']/1000)
        self.binance = Client(self.apiKey, self.secret)
        self.InitAccountInfo()
        self.max_position_money = self.money_total
        
        seperator = 'data/log/'
        self.log_file = save_dir + seperator + self.coin + "_log.csv"
    
    def InitAccountInfo(self):
        info = self.binance.futures_account()
        asset = next(asset for asset in info['assets'] if asset['asset']==self.stable)
        self.money = float(asset['availableBalance'])
        self.initial_money = float(asset['crossWalletBalance'])
        
        position = next(position for position in info['positions'] if position['symbol']==self.symbol)
        self.positionAmt = float(position["positionAmt"])
        self.amount = self.positionAmt
        self.entry_price = float(position['entryPrice'])
        
        self.money_total = float(asset['marginBalance'])
    
    def createOrder(self, order_side, amount, value):
        try:
            if amount > 28.0:
                self.binance.futures_create_order(
                    symbol=self.symbol,
                    isIsolated='FALSE',
                    side=order_side,
                    type="LIMIT",
                    timeInForce = 'GTC',
                    quantity = floor_digit(amount, 0),
                    price = floor_digit(value, 5),
                )
            return True
        except Exception as e:
            print(datetime.utcnow(), end=": createOrder Error = ")
            print(e)
            return False
    
    def cancelOpenOrders(self):
        return self.binance.futures_cancel_all_open_orders(symbol=self.symbol)
    
    def buyAmount(self,amount_rate):
        self.cancelOpenOrders()
        buy_value = self.close_value * (1 - self.limit_rate)
        
        available_money = self.money_total * self.margin_rate - self.amount * buy_value
        if available_money > self.money_total * self.margin_rate * 0.1:
            buy_amount = (available_money / buy_value) * amount_rate
            order_success = self.createOrder("BUY", buy_amount, buy_value)
            if order_success == False:
                num_divide = 4
                small_amount = buy_amount / num_divide
                for i_divide in range(num_divide):
                    order_success = self.createOrder("BUY", small_amount, buy_value)
    
    def sellAmount(self,amount_rate):
        self.cancelOpenOrders()
        sell_value = self.close_value * (1 + self.limit_rate)
        
        available_money = self.money_total * self.margin_rate + self.amount * sell_value
        if available_money > self.money_total * self.margin_rate * 0.1:
            sell_amount = (available_money / sell_value) * amount_rate
            order_success = self.createOrder("SELL", sell_amount, sell_value)
            if order_success == False:
                num_divide = 4
                small_amount = sell_amount / num_divide
                for i_divide in range(num_divide):
                    order_success = self.createOrder("SELL", small_amount, sell_value)
    
    def performAct(self):
        # 초기화
        close_data = get_close_data(self.symbol)
        X_data = get_X_data(close_data)
        self.rate_tomorrow = self.predictor.predict(X_data, verbose=0)[0,0]
        self.close_value = close_data[-1,0]
        
        self.binance = Client(self.apiKey, self.secret)
        self.InitAccountInfo()
        
        # 청산 방지
        buy_orderbook = self.binance.futures_order_book(symbol=self.symbol)['bids']
        buy_value = float(buy_orderbook[0][0])
        sell_orderbook = self.binance.futures_order_book(symbol=self.symbol)['asks']
        sell_value = float(sell_orderbook[0][0])
        
        if self.amount * sell_value > self.money_total * (self.margin_rate * 2.0):
            self.sellAmount(0.5)
        elif -1 * self.amount * buy_value > self.money_total * (self.margin_rate * 2.0):
            self.buyAmount(0.5)
        
        elif self.rate_tomorrow > self.thresholds[0]:
            self.buyAmount(1.0)
                
        elif self.rate_tomorrow < self.thresholds[1]:
            self.sellAmount(1.0)
        
        self.InitAccountInfo()
        
        self.now = datetime.fromtimestamp(Client().get_server_time()['serverTime']/1000)
    
    def printState(self):
        with open(self.log_file, "a") as f:
            log = str(self.now) + "," + str(self.money) + "," + \
            str(self.amount) + "," + \
            str(self.money_total) + "," + str(self.rate_tomorrow) + "," + \
            str(self.close_value) + "\n"
            f.write(log)
            f.flush()
            f.close()

save_dir = "./"
predictor = load_model('./predictor.h5',compile=False)

apiKey = "Binance apikey"
secret = 'Binance secret'

ticker = ('DOGE','USDT')
auto_trader = StockTrader_Real(predictor, ticker, apiKey, secret)

seconds = np.arange(60)
start_second = 1

while True:
    try:
        gc.collect()
        now = datetime.fromtimestamp(Client().get_server_time()['serverTime']/1000).second
        wait_second = seconds[start_second - now]
        time.sleep(wait_second)
        
        auto_trader.performAct()
        auto_trader.printState()
    except Exception as e:
        print(datetime.utcnow(), end=": While Error = ")
        time.sleep(30)
        print(e)