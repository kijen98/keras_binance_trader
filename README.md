# keras_binance_trader
Binance cryptocurrency auto trading system with keras prediction model

## 1. Data_Downloader
You can get cryptocurrency stock price or candle data of stock price from Binance.

In the case of Binance, you can choose between spot and future markets.


## 2. Train_and_Test

You can load ohlc data, train a model, and calculate yield with virtual trading.

It can be simulated by setting market trading and limit trading, and it is possible to find the threshold that shows the best yield.

More than 100 symbol data was used for actual training, and it was not uploaded to github due to capacity issues.


## 3. Trader

Using docker, collect candle data every minute, use the model to decide whether to trade, implement an automatic trading bot.

For it to work, you need to enter Binance's apikey and secret in Trader.py.

Build the docker image with ./build.sh, start the Trading container with ./run.sh, and stop and delete the container with ./stop.sh.

You can check docker logs of running Trading container with ./logs.

Plot_money_list.ipynb can read the log of the data folder and visualize the current yield.
