import math
import random
import numpy as np

# 데이터 로드
def load_data(TRAIN_DIR, TEST_DIR):
    FILE_TAIL = "_minute.npy"

    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'YFIIUSDT', 'XRPUSDT', 'MATICUSDT', 'AVAXUSDT', 'ADAUSDT', 'NEARUSDT', 'YFIUSDT', 'FTMUSDT', 'DOGEUSDT', 'RUNEUSDT', 'SANDUSDT', 'AXSUSDT', 'OGUSDT', 'WAVESUSDT', 'TRXUSDT', 'ZILUSDT', 'LINKUSDT', 'DOTUSDT', 'BCHUSDT', 'CELRUSDT', 'LTCUSDT', 'MANAUSDT', 'WNXMUSDT', 'ONEUSDT', 'AAVEUSDT', 'ATOMUSDT', 'THETAUSDT', 'VETUSDT', 'UNIUSDT', 'RSRUSDT', 'FILUSDT', 'BELUSDT', 'EGLDUSDT', 'LUNAUSDT', 'STORJUSDT', 'ROSEUSDT', 'SNXUSDT', 'ARPAUSDT', 'ETCUSDT', 'UNFIUSDT', 'ALGOUSDT', 'HNTUSDT', 'DUSKUSDT', 'CRVUSDT', 'GRTUSDT', 'OGNUSDT', 'COMPUSDT', 'FTTUSDT', 'UMAUSDT', 'LRCUSDT', 'SUSHIUSDT', 'BLZUSDT', 'BATUSDT', 'XMRUSDT', 'EOSUSDT', 'TRBUSDT', 'KAVAUSDT', 'ASRUSDT', 'XLMUSDT', '1INCHUSDT', 'CHZUSDT', 'SUNUSDT', 'KNCUSDT', 'BALUSDT', 'JSTUSDT', 'ZECUSDT', 'ENJUSDT', 'HBARUSDT', 'FLMUSDT', 'WINUSDT', 'ANTUSDT', 'XTZUSDT', 'CHRUSDT', 'FETUSDT', 'IOTXUSDT', 'GTOUSDT', 'IOSTUSDT', 'MKRUSDT', 'ALPHAUSDT', 'SXPUSDT', 'HOTUSDT', 'NEOUSDT', 'OMGUSDT', 'DASHUSDT', 'OCEANUSDT', 'REPUSDT', 'ANKRUSDT', 'KSMUSDT', 'BTSUSDT', 'BANDUSDT', 'SRMUSDT', 'CTSIUSDT', 'REEFUSDT', 'COTIUSDT', 'RENUSDT', 'RVNUSDT', 'DENTUSDT', 'QTUMUSDT', 'MTLUSDT', 'SKLUSDT', 'IOTAUSDT', 'TFUELUSDT', 'MBLUSDT', 'ZRXUSDT', 'PAXGUSDT', 'TCTUSDT', 'MDTUSDT', 'COCOSUSDT', 'INJUSDT', 'COSUSDT', 'BNTUSDT', 'ONTUSDT', 'JUVUSDT', 'WINGUSDT', 'ATMUSDT', 'DNTUSDT', 'AKROUSDT', 'CVCUSDT', 'XEMUSDT', 'ZENUSDT', 'ICXUSDT', 'TROYUSDT', 'ORNUSDT', 'WRXUSDT', 'NKNUSDT', 'CTKUSDT', 'XVSUSDT', 'TOMOUSDT', 'MFTUSDT', 'PSGUSDT', 'CTXCUSDT', 'NBSUSDT', 'MITHUSDT', 'KEYUSDT', 'VTHOUSDT', 'PNTUSDT', 'DGBUSDT', 'UTKUSDT', 'DREPUSDT', 'AIONUSDT', 'RLCUSDT', 'OXTUSDT', 'PERLUSDT', 'STXUSDT', 'NULSUSDT', 'HARDUSDT', 'LTOUSDT', 'SCUSDT', 'DIAUSDT', 'KMDUSDT', 'STPTUSDT', 'WTCUSDT', 'FIOUSDT', 'DATAUSDT', 'IRISUSDT', 'WANUSDT', 'BEAMUSDT', 'NMRUSDT', 'HIVEUSDT', 'STRAXUSDT', 'ONGUSDT', 'VITEUSDT', 'STMXUSDT', 'DOCKUSDT', 'AVAUSDT', 'LSKUSDT', 'DCRUSDT', 'ARDRUSDT', 'FUNUSDT', 'BCCUSDT', 'VENUSDT', 'PAXUSDT', 'BCHABCUSDT', 'BCHSVUSDT', 'BTTUSDT', 'NANOUSDT', 'ERDUSDT', 'NPXSUSDT', 'STORMUSDT', 'HCUSDT', 'MCOUSDT', 'BULLUSDT', 'BEARUSDT', 'ETHBULLUSDT', 'ETHBEARUSDT', 'EOSBULLUSDT', 'EOSBEARUSDT', 'XRPBULLUSDT', 'XRPBEARUSDT', 'STRATUSDT', 'BNBBULLUSDT', 'BNBBEARUSDT', 'XZCUSDT', 'GXSUSDT', 'LENDUSDT',]
    symbols = [symbol.replace("USDT","") for symbol in symbols]
    print("DATA kind:", len(symbols))

    train_datas = {}
    test_datas = {}
    for symbol in symbols:
        train_datas[symbol] = np.load(TRAIN_DIR + symbol + FILE_TAIL)
        if len(train_datas[symbol]) > 0:
            train_datas[symbol] = train_datas[symbol][:,1:5]
        
        test_datas[symbol] = np.load(TEST_DIR + symbol + FILE_TAIL)
        if len(test_datas[symbol]) > 0:
            test_datas[symbol] = test_datas[symbol][:,1:5]

    sorted_train_datas = sorted(train_datas.items(), key=lambda item: item[1].shape[0], reverse=True)
    symbols = [symbol for symbol, data in sorted_train_datas]
    
    for symbol in symbols:
        train_datas[symbol] = train_datas.pop(symbol)
        test_datas[symbol] = test_datas.pop(symbol)
        print(symbol, train_datas[symbol].shape, test_datas[symbol].shape)
    
    return train_datas, test_datas

class DataLoader_origin:
    def __init__(self, datas, data_shapes, data_infos):
        self.train_datas, self.test_datas = datas
        self.symbols = list(self.train_datas.keys())
        self.X_shape, self.y_shape = data_shapes
        self.batch_size, self.X_len, self.y_len, self.time_windows, self.y_windows = data_infos
        self.start_diff = (self.X_len-1) * self.time_windows[-1]
        
        # train_ixs 생성
        total_size = 0
        self.train_ixs = {}
        for symbol in self.symbols:
            train_i = self.train_datas[symbol].shape[0] - self.start_diff - self.y_windows[-1] * self.y_len
            self.train_ixs[symbol] = np.arange(0, train_i)
            total_size += len(self.train_ixs[symbol])
            print(symbol, self.train_ixs[symbol].shape)
        print("total_size:", total_size)
        
        eval_symbol = 'BTC'
        # 배치 데이터
        self.X_data = np.zeros(((self.batch_size,) + self.X_shape)) 
        self.y_data = np.zeros(((self.batch_size,) + self.y_shape))
        
        # eval 데이터
        eval_size = 1024 * 10
        self.X_train = np.zeros(((eval_size,) + self.X_shape)); 
        self.y_train = np.zeros(((eval_size,) + self.y_shape))
        self.X_test = np.zeros(((eval_size,) + self.X_shape)); 
        self.y_test = np.zeros(((eval_size,) + self.y_shape))

        # train eval 초기화
        train_data = self.train_datas[eval_symbol]
        ix = np.arange(len(self.X_train))
        self.load_Xy(train_data[:, 3], ix+self.start_diff, self.X_train, self.y_train)

        # test eval 초기화
        test_data = self.test_datas[eval_symbol]
        ix = np.arange(len(self.X_test))
        self.load_Xy(test_data[:, 3], ix+self.start_diff, self.X_test, self.y_test)
    
    def load_Xy(self, data, ix, X_data, y_data):
        mul_rate = 100
        for i_batch in range(len(ix)):
            idx = ix[i_batch]
            if X_data is not None:
                for i_time_window, time_window in enumerate(self.time_windows):
                    hour_data = data[idx-(self.X_len-1)*time_window:idx+1:time_window] # (X_len,)
                    rate_data = np.log(hour_data/hour_data[-1]) * mul_rate / (math.log(time_window,2)+1)
                    
                    X_data[i_batch,:,i_time_window] = rate_data

            if y_data is not None:
                for i_y_window, y_window in enumerate(self.y_windows):
                    hour_data = np.mean(data[idx+1:idx+1+y_window])
                    rate_data = np.log(hour_data/data[idx]) * mul_rate
                    
                    y_data[i_batch,i_y_window] = rate_data
        
        return True

    def get_train_data(self, symbol=None):
        batch_ix = 0
        while batch_ix < self.batch_size:
            if symbol == None:
                random_symbol = random.choice(self.symbols)
            else:
                random_symbol = symbol
            train_data = self.train_datas[random_symbol]
            train_ix = self.train_ixs[random_symbol]
            if len(train_ix) > 0:
                ix = np.random.randint(len(train_ix))
                ix = train_ix[np.arange(ix, ix+1)]
                self.load_Xy(train_data[:,3], ix+self.start_diff, 
                             self.X_data[batch_ix:batch_ix+1], self.y_data[batch_ix:batch_ix+1])
                batch_ix += 1
        
        return self.X_data, self.y_data

# y값의 크기에 따라 패턴으로 나누고, 같은 양만큼 훈련 시키기
class DataLoader:
    def __init__(self, datas, data_shapes, data_infos):
        self.train_datas, self.test_datas = datas
        self.symbols = list(self.train_datas.keys())
        self.X_shape, self.y_shape = data_shapes
        self.batch_size, self.X_len, self.y_len, self.time_windows, self.y_windows = data_infos
        #self.start_diff = (self.X_len-1) * self.time_windows[-1]
        self.start_diff = self.X_len * self.time_windows[-1]
        
        # train_ixs 생성
        total_size = 0
        self.train_ixs = {}
        for symbol in self.symbols:
            train_i = self.train_datas[symbol].shape[0] - self.start_diff - self.y_windows[-1] * self.y_len
            self.train_ixs[symbol] = np.arange(0, train_i)
            total_size += len(self.train_ixs[symbol])
            #print(symbol, self.train_ixs[symbol].shape)
        print("total_size:", total_size)
        
        eval_symbol = 'BTC'
        # 배치 데이터
        self.X_data = np.zeros(((self.batch_size,) + self.X_shape)) 
        self.y_data = np.zeros(((self.batch_size,) + self.y_shape))
        
        # eval 데이터
        eval_size = 1024 * 10
        self.X_train = np.zeros(((eval_size,) + self.X_shape)); 
        self.y_train = np.zeros(((eval_size,) + self.y_shape))
        self.X_test = np.zeros(((eval_size,) + self.X_shape)); 
        self.y_test = np.zeros(((eval_size,) + self.y_shape))

        # train eval 초기화
        train_data = self.train_datas[eval_symbol]
        ix = np.arange(len(self.X_train))
        self.load_Xy(train_data[:, 3], ix+self.start_diff, self.X_train, self.y_train)

        # test eval 초기화
        test_data = self.test_datas[eval_symbol]
        ix = np.arange(len(self.X_test))
        self.load_Xy(test_data[:, 3], ix+self.start_diff, self.X_test, self.y_test)
        
        # 패턴 종류에 따라 train_ix 분류
        y_datas = {}
        for symbol in self.symbols:
            train_data = self.train_datas[symbol]
            ix = self.train_ixs[symbol]
            if len(ix) > 0:
                ma_data = np.convolve(train_data[ix[0]+self.start_diff+1:,3], np.ones(self.y_windows[-1]), 'valid') / self.y_windows[-1]
                today_data = train_data[ix[0]+self.start_diff:ix[-1]+self.start_diff+1,3]
                y_data = np.log(ma_data[:len(today_data)]/today_data) * 100
                y_datas[symbol] = y_data
            else:
                y_datas[symbol] = np.array([])
        
        pattern_size = 10
        y_patterns = {}
        for symbol in self.symbols:
            y_data = y_datas[symbol]
            y_abs = np.abs(y_data)
            y_pattern = []
            for i_pattern in range(pattern_size):
                abs_min = i_pattern * 1 / pattern_size
                abs_max = (i_pattern + 1) * 1 / pattern_size
                if i_pattern == pattern_size - 1:
                    y_pattern.append(np.where(y_abs > abs_min)[0])
                else:
                    y_pattern.append(np.where((y_abs >= abs_min) & (y_abs < abs_max))[0])
            y_patterns[symbol] = y_pattern
            print(symbol, len(y_data))
            size = 0
            for pattern in y_pattern:
                print(len(pattern), end=", ")
                size += len(pattern)
            print(size, len(y_data))
        self.y_patterns = y_patterns
            
    def load_Xy(self, data, ix, X_data, y_data):
        mul_rate = 100
        for i_batch in range(len(ix)):
            idx = ix[i_batch]
            if X_data is not None:
                for i_time_window, time_window in enumerate(self.time_windows):
                    hour_data = data[idx-(self.X_len-1)*time_window:idx+1:time_window] # (X_len,)
                    rate_data = np.log(hour_data/hour_data[-1]) * mul_rate / (math.log(time_window,2)+1)
                    
                    X_data[i_batch,:,i_time_window] = rate_data

            if y_data is not None:
                for i_y_window, y_window in enumerate(self.y_windows):
                    hour_data = np.mean(data[idx+1:idx+1+y_window])
                    rate_data = np.log(hour_data/data[idx]) * mul_rate
                    
                    y_data[i_batch,i_y_window] = rate_data
        return True

    def get_train_data(self, symbol=None):
        batch_ix = 0
        while batch_ix < self.batch_size:
            if symbol == None:
                random_symbol = random.choice(self.symbols)
            else:
                random_symbol = symbol
            train_data = self.train_datas[random_symbol]
            train_ix = self.train_ixs[random_symbol]
            y_pattern = self.y_patterns[random_symbol]
            if len(train_ix) > 0:
                random_pattern = random.choice(y_pattern)
                if len(random_pattern) > 0:
                    ix = random.choice(random_pattern)
                    ix = train_ix[ix:ix+1]
                    self.load_Xy(train_data[:,3], ix+self.start_diff, 
                                 self.X_data[batch_ix:batch_ix+1], self.y_data[batch_ix:batch_ix+1])
                    batch_ix += 1
        
        return self.X_data, self.y_data
