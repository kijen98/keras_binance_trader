import numpy as np

def get_real_pred(predictor, dataloader, symbol):
    test_data = dataloader.test_datas[symbol]
    batch_size = 1024
    X_data = np.zeros(((batch_size,) + dataloader.X_shape)) 
    y_data = np.zeros(((batch_size,) + dataloader.y_shape))
    
    y_datas = []
    y_preds = []
    period_ix = np.arange(0, len(test_data) - dataloader.start_diff - 1)
    for i in range(0, period_ix[-1], batch_size):
        ix = period_ix[i:i+batch_size]
        dataloader.load_Xy(test_data[:,3], ix+dataloader.start_diff, X_data[:len(ix)], y_data[:len(ix)])
        y_pred = predictor.predict(X_data[:len(ix)], verbose=0)
        y_datas.append(y_data[:len(ix)].copy())
        y_preds.append(y_pred[:len(ix)].copy())
    y_data = np.concatenate(y_datas, axis=0)
    y_pred = np.concatenate(y_preds, axis=0)
    
    return y_data, y_pred

# 부모 Trading 시뮬레이터
class Trader:
    def __init__(self, ohlcv_data, y_pred):
        self.today = 0
        self.money = 0 # 예수금
        self.amount = 0 # 보유량
        self.money_total = self.money # 순 자산 가치
        self.value_today = 0 # 오늘 가격
        self.y_today = 0 # 오늘 예측 값
        
        start_diff = len(ohlcv_data) - len(y_pred) - 1
        self.ohlcv_data = ohlcv_data[start_diff:]
        self.y_pred = y_pred

        self.value_log = []
        self.trade_log = []
        self.long_win_log = []
        self.short_win_log = []

    def day_init(self):
        self.value_today = self.ohlcv_data[self.today,3]
        self.y_today = self.y_pred[self.today]

        self.money_total = self.money + self.amount * self.value_today
        if self.money_total < 0: # 청산
            self.money = 0;
            self.amount = 0
            self.money_total = 0

    def day_start(self, start_date):
        self.today = start_date
        self.money = 10000000
        self.day_init()
        self.value_log.append(self.value_today)
        self.trade_log.append(0)
   
    def print_state(self):
        print("today: %s" % self.today)
        print("money: %s" % self.money)
        print("amount: %s" % self.amount)
        print("value_today: %s" % self.value_today)
        print("money_total: %s" % self.money_total)
        
    def get_yield(self):
        period = self.y_pred.shape[0]
        self.day_start(0)

        money_list = []
        value_list = []
        for i in range(period):
            self.day_init()
            self.perform_act()
            money_list.append(self.money_total)
            value_list.append(self.value_today)
            self.today += 1

        money_list = np.array(money_list)
        value_list = np.array(value_list)
        trade_num = [len(self.long_win_log), len(self.short_win_log)]
        win_rate = [np.mean(self.long_win_log) * 100, np.mean(self.short_win_log) * 100]
        trading_yield = money_list[len(money_list)-1]/money_list[0] * 100

        trading_info = [self.trade_log, trade_num, win_rate, trading_yield]

        return trading_info, money_list, value_list

# 선물 시장가 트레이더
class TraderFutureMarket(Trader):
    def __init__(self, ohlcv_data, y_pred, trading_conditions):
        super().__init__(ohlcv_data, y_pred)
        self.fee_rate, self.margin_rate, self.threshold = trading_conditions
        self.position_rate = 0.0

    def buy(self, amount_rate):
        self.money_total = self.money + self.amount * self.value_today
        available_money = self.money_total * self.margin_rate - self.amount * self.value_today

        if available_money > self.money_total * self.margin_rate * 0.05:
            #print("BUY: " + str(self.value_today))
            buy_amount = available_money / self.value_today * amount_rate
            self.money -= buy_amount * self.value_today
            self.amount += buy_amount * (1 - self.fee_rate)

    def sell(self, amount_rate):
        self.money_total = self.money + self.amount * self.value_today
        available_money = self.money_total * self.margin_rate + self.amount * self.value_today

        if available_money > self.money_total * self.margin_rate * 0.05:
            #print("SELL: " + str(self.value_today))
            sell_amount = available_money / self.value_today * amount_rate
            self.money += sell_amount * self.value_today * (1 - self.fee_rate)
            self.amount -= sell_amount

    def perform_act(self):
        if self.amount * self.value_today > self.money_total * (self.margin_rate * 2.0):
            self.sell(0.5)
        elif -1 * self.amount * self.value_today > self.money_total * (self.margin_rate * 2.0):
            self.buy(0.5)

        if self.y_today > self.threshold[0]:
            self.buy(1.0)

            self.trade_log.append(1)
            self.append_win_log()
            
        elif self.y_today < self.threshold[1]:
            self.sell(1.0)

            self.trade_log.append(-1)
            self.append_win_log()

        else:
            self.trade_log.append(self.trade_log[-1])
    
    def append_win_log(self):
        if self.trade_log[-1] != self.trade_log[-2]:
            self.value_log.append(self.value_today)
            if self.value_log[-1] > self.value_log[-2]:
                if self.trade_log[-2] == 1:
                    self.long_win_log.append(1)
                elif self.trade_log[-2] == -1:
                    self.short_win_log.append(0)
            else:
                if self.trade_log[-2] == 1:
                    self.long_win_log.append(0)
                elif self.trade_log[-2] == -1:
                    self.short_win_log.append(1)

# 선물 지정가 트레이더
class TraderFutureLimit(Trader):
    def __init__(self, ohlcv_data, y_pred, trading_conditions):
        super().__init__(ohlcv_data, y_pred)
        self.fee_rate, self.margin_rate, self.threshold, self.limit_rate = trading_conditions
        self.position_rate = 0.0

    def buy(self, amount_rate):
        if self.amount <= 0:
            self.low_tomorrow = self.ohlcv_data[self.today+1, 2]
            self.limit_value = self.value_today * (1 - self.limit_rate)
            self.money_total = self.money + self.amount * self.limit_value
            available_money = self.money_total * self.margin_rate - self.amount * self.limit_value
            if self.limit_value >= self.low_tomorrow and available_money > self.money_total * self.margin_rate * 0.1:
                buy_amount = available_money / self.limit_value * amount_rate
                #print("BUY: " + str(self.limit_value), buy_amount)
                self.money -= buy_amount * self.limit_value
                self.amount += buy_amount * (1 - self.fee_rate)

                self.append_win_log(1)

    def sell(self, amount_rate):
        if self.amount >= 0:
            self.high_tomorrow = self.ohlcv_data[self.today+1, 1]
            self.limit_value = self.value_today * (1 + self.limit_rate)
            self.money_total = self.money + self.amount * self.limit_value
            available_money = self.money_total * self.margin_rate + self.amount * self.limit_value
            if self.limit_value <= self.high_tomorrow and available_money > self.money_total * self.margin_rate * 0.1:
                sell_amount = available_money / self.limit_value * amount_rate
                #print("SELL: " + str(self.limit_value), sell_amount)
                self.money += sell_amount * self.limit_value * (1 - self.fee_rate)
                self.amount -= sell_amount

                self.append_win_log(-1)

    def perform_act(self):
        risk_rate = self.amount * self.value_today / (self.money_total * self.margin_rate)
        if risk_rate > 2:
            self.sell(0.5)
        elif risk_rate < -2:
            self.buy(0.5)
        
        elif self.y_today > self.threshold[0]:
            self.buy(1.0)

        elif self.y_today < self.threshold[1]:
            self.sell(1.0)
        
        self.money_total = self.money + self.amount * self.value_today
    
    def append_win_log(self, log):
        self.trade_log.append(log)
        self.value_log.append(self.limit_value)
        if self.value_log[-1] > self.value_log[-2]:
            if self.trade_log[-2] == 1:
                self.long_win_log.append(1)
            elif self.trade_log[-2] == -1:
                self.short_win_log.append(0)
        else:
            if self.trade_log[-2] == 1:
                self.long_win_log.append(0)
            elif self.trade_log[-2] == -1:
                self.short_win_log.append(1)