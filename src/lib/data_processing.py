import os
import math
import random
from glob import glob

import numpy as np

# 데이터 로드
def load_data(TRAIN_DIR, TEST_DIR):
    datas = []
    for file_dirs in [glob(f"{TRAIN_DIR}/*.*"), glob(f"{TEST_DIR}/*.*")]:
        data = {}
        for file_dir in file_dirs:
            ticker = file_dir.split("\\")[-1].split("_")[0]
            data[ticker] = np.load(file_dir)[:,1:5]
        datas.append(data)
        
    train_datas, test_datas = datas
    sorted_train_datas = sorted(train_datas.items(), key=lambda item: item[1].shape[0], reverse=True)
    symbols = [symbol for symbol, data in sorted_train_datas]
    print("DATA kind:", len(symbols))
    
    for symbol in symbols:
        if symbol in train_datas:
            train_datas[symbol] = train_datas.pop(symbol)
            print(symbol, train_datas[symbol].shape)
        if symbol in test_datas:
            test_datas[symbol] = test_datas.pop(symbol)
            #print(symbol, test_datas[symbol].shape)
    
    return train_datas, test_datas

class DataLoader:
    def __init__(self, datas, data_shapes, data_infos):
        self.train_datas, self.test_datas = datas
        self.symbols = list(self.train_datas.keys())
        self.X_shape, self.y_shape = data_shapes
        self.batch_size, self.X_len, self.y_len, self.time_windows, self.y_windows = data_infos
        self.start_diff = (self.X_len-1) * self.time_windows[-1]
        
        # train_ixs 생성
        total_size = 0
        self.valid_symbols = []
        self.train_ixs = {}
        for symbol in self.symbols:
            train_i = self.train_datas[symbol].shape[0] - self.start_diff - self.y_windows[-1] * self.y_len
            self.train_ixs[symbol] = np.arange(0, train_i)
            total_size += len(self.train_ixs[symbol])
            print(symbol, self.train_ixs[symbol].shape)
            if len(self.train_ixs[symbol]) > 0:
                self.valid_symbols.append(symbol)
        print("total_size:", total_size)
        print("valid_symbols:", len(self.valid_symbols))
        
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
    
    # 기본형
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