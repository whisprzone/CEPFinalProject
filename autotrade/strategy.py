import requests
import math
import requests
import tensorflow as tf
import numpy as np
import pandas as pd
from transformers import BertTokenizer
import backtrader as bt
from backtrader_plotting import Bokeh
import datetime
import json 
import random
import bokeh
import app
from backtrader_plotting.schemes import Tradimo

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
testing_period = [2015, 2019]
global stock_name 
stock_name = app.stock_name

financial_news = open("/Users/surya/OneDrive/Programming/autotrade/seconddataset.txt", "r")
lines = financial_news.readlines()
modified_lines = []
for i in range(1, len(lines)):
    line = lines[i]
    temp_line = line.strip("\n").split(',', 2)
    del temp_line[0]
    temp_line[1] = temp_line[1].replace('"', '')
    modified_lines.append(temp_line)

lines_dataframe = pd.DataFrame(modified_lines, columns = ['sentiment', 'news'])

print(lines_dataframe)
lines_dataframe = lines_dataframe.sample(frac = 1)
lines_dataframe.to_csv(r'test.csv')
#newsapi = NewsApiClient(api_key = 'c914785892784aed8d6dbb973b4b6c1e')

sentiment = lines_dataframe['sentiment'] 
news = lines_dataframe['news'] 

combined_content_data = []
counter = 0
for i in lines_dataframe['news']:
    counter += 1
    encoding = tokenizer.encode_plus(i, 
            max_length=300,
            padding='max_length',
            add_special_tokens=True, 
            return_token_type_ids=False, 
            return_tensors='tf')
    combined_content_data.append(tf.convert_to_tensor(encoding["input_ids"]))

combined_sentiment_data = tf.keras.utils.to_categorical(np.array(sentiment), num_classes = 3)

class SentimentAnalyzer():
    def __init__(self, combined_content_data, combined_sentiment_data):
        '''
        Params: combined_content_data, combined_sentiment_data
        __init__ returns the model for use in other subfunctions and inherits all data
        '''
        self.combined_content_data = combined_content_data
        self.combined_sentiment_data = combined_sentiment_data
        x_train = tf.convert_to_tensor(combined_content_data[0:int(0.7 * len(combined_content_data))]) # tensor values of encoded string (train)
        x_test = np.array(combined_content_data[int(0.7 * len(combined_content_data) + 1):len(combined_content_data)]) # test values (string) 
        y_train = np.array(combined_sentiment_data[0:int(0.7 * len(combined_sentiment_data))]) # sentiment value (train)
        y_test = np.array(combined_sentiment_data[int(0.7 * len(combined_sentiment_data) + 1):len(combined_content_data)]) # sentiment value (test)

        inputs = tf.keras.layers.Input(shape = (1,300))
        lstm = tf.keras.layers.LSTM(60)(inputs)
        flatten = tf.keras.layers.Flatten()(lstm)
        dense1 = tf.keras.layers.Dense(200)(flatten)
        dense2 = tf.keras.layers.Dense(50)(dense1)
        outputs = tf.keras.layers.Dense(3, activation = 'softmax')(dense2)
        model = tf.keras.Model(inputs = inputs, outputs = outputs, name = "sentiment_model")
        
        model.summary()
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
            loss = "categorical_crossentropy",
            metrics = ["mae", "acc"])
        model.fit(x = x_train,
            y = y_train,
            batch_size = 32,
            epochs = 150)
        z = model.evaluate(x = x_test,
            y = y_test,
            verbose = 1)
        self.model = model

    def get_all_news(self):
        
        url = "https://contextualwebsearch-websearch-v1.p.rapidapi.com/api/Search/NewsSearchAPI"
        # get_all_news 
        bottom_date = datetime.date(2015, 1, 1)
        top_date = datetime.date(2019, 12, 31)
        titles = {}
        querystring = {
                "fromPublishedDate": str(bottom_date.year) + '-' + str(bottom_date.month) + '-' + str(bottom_date.day),
                "toPublishedDate": str(top_date.year) + '-' + str(top_date.month) + '-' + str(top_date.day),
                "autoCorrect": "false",
                "pageNumber": 1, 
                "pageSize": "50", 
                "q": stock_name,
                "safeSearch": "false"
            }
        headers = {
                "x-rapidapi-host": "contextualwebsearch-websearch-v1.p.rapidapi.com",
                "x-rapidapi-key": "b8d8c22ee5msh9eefd4554791cd5p134bd9jsn6c20af9894ff"
            }

        response = requests.request("GET", url, headers = headers, params = querystring)
        all_news = json.loads(response.text)
        num_iters = int(all_news["totalCount"] / 50)
        print(num_iters)
        for x in range(1, num_iters):
            
            querystring = {
                "fromPublishedDate": str(bottom_date.year) + '-' + str(bottom_date.month) + '-' + str(bottom_date.day),
                "toPublishedDate": str(top_date.year) + '-' + str(top_date.month) + '-' + str(top_date.day),
                "autoCorrect": "false",
                "pageNumber": x, 
                "pageSize": "50", 
                "q": stock_name,
                "safeSearch": "false"
            }
            headers = {
                    "x-rapidapi-host": "contextualwebsearch-websearch-v1.p.rapidapi.com",
                    "x-rapidapi-key": "b8d8c22ee5msh9eefd4554791cd5p134bd9jsn6c20af9894ff"
                }


            response = requests.request("GET", url, headers = headers, params = querystring)
            all_news = json.loads(response.text)
            all_values = dict(all_news)['value']
            for i in range(0, len(all_values)):
                titles[all_values[i]["datePublished"]] = all_values[i]["title"]
        print(titles)
        return titles

    def return_prediction_value(self):
        headline = self.get_all_news()
        prediction_values = {}
        for key, value in headline.items():
            tensor = tf.convert_to_tensor(tokenizer.encode_plus(value, 
                max_length=300,
                padding='max_length',
                add_special_tokens=True, 
                return_token_type_ids=False, 
                return_tensors='tf')['input_ids'])
            tensor = np.reshape(tensor, (1, 1, 300))
            tensor = tf.cast(tensor, tf.float32)
            prediction = list(self.model.predict(np.array(tensor)))
            split_key = key.split("T")
            year_month_day = split_key[0].split("-")
            imprecise_datetime = datetime.date(int(year_month_day[0]), int(year_month_day[1]), int(year_month_day[2]))
            prediction_values[imprecise_datetime] = prediction
        
        
        return prediction_values

class SentimentBollingerStrategy(bt.Strategy):
    params = (
            ('period', 25),
            ('devfactor', 2.0),
            ('size', 20),
            ('stop_loss', 0.4)
        )

    def log(self, message):
        # simple logging function for messages
        dt = self.datas[0].datetime.date(0)
        print(str(dt) + ", " + message)

    def __init__(self):
        # market closing data 
        self.dataclose = self.datas[0].close

        self.order = None
        self.buyprice = None
        self.buycomm = None

        self.BOLLBands = bt.indicators.BollingerBands(
                period = self.params.period,
                devfactor = self.params.devfactor
            )
        
        self.combined_content_data = combined_content_data
        self.combined_sentiment_data = combined_sentiment_data

        self.analyzer = SentimentAnalyzer(self.combined_content_data, self.combined_sentiment_data)
        self.prediction_values = self.analyzer.return_prediction_value()
        
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # order is accepted
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                        'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm: %2f' %
                        (order.executed.price,
                        order.executed.value,
                        order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            
            else:
                
                self.log(
                        'SELL EXECUTED, Price: %2f, Cost: %2f, Comm: %2f' %
                        (order.executed.price,
                        order.executed.value,
                        order.executed.comm))
                    
        elif order.status in [order.Canceled, order.Margin, order.Rejected]: # unsuccessful order for whatever reason e.g. broker refusals or connectivity issues
            self.log('Order canceled/margin/rejected')


    def notify_trade(self, trade):

        if not trade.isclosed:
            return

        self.log(
                'OPERATION PROFIT, GROSS %2f, NET %.2F' %
                (trade.pnl, trade.pnlcomm))

    def next(self):
        # log the closing price --> we will make our moves partially based on this
        self.log('Close: ' + str(self.dataclose[0]))
        date = self.datas[0].datetime.date(0)
        valid_time_period = datetime.timedelta(14) # check news from 14 days before
        bottom_date = date - valid_time_period
        
        prediction_values = self.prediction_values
        all_dates = prediction_values.keys()
        relevant_predictions = {}
        for i in all_dates:
            if i <= date and i >= bottom_date:
                relevant_predictions[i] = prediction_values[i]
        total_sentiment = [0, 0, 0] 
        for key, value in relevant_predictions.items(): 
            total_sentiment[0] += value[0][0]
            total_sentiment[1] += value[0][1]
            total_sentiment[2] += value[0][2]

        self.log("news values: " + str(total_sentiment))
    
        # check if we are in the market
        if not self.position:

            if self.dataclose[0] >= self.BOLLBands.lines.top[0]:

                if total_sentiment.index(max(total_sentiment), 0, 3) == 0:
                    diff = total_sentiment[0] - total_sentiment[1]
                    if diff != 0:
                        volume = 100 / diff
                    
                    else:
                        
                        if total_sentiment[0] != 0:
                            volume = 100 / total_sentiment[0]
                        
                        else:
                            volume = 20

                else: # less unlikely scenario
                    if total_sentiment[1] != 0:
                        volume = 100 / total_sentiment[1]
                    
                    else:
                        volume = 20

                self.log("SELL CREATE, %.2F" % self.dataclose[0])
                self.order = self.sell_bracket(limitprice = self.BOLLBands.lines.bot[0], 
                    price = self.dataclose[0],
                    stopprice = self.dataclose[0] + (self.p.stop_loss * self.dataclose[0]),
                    size = volume) 
                
        else:
            if self.dataclose[0] <= self.BOLLBands.lines.bot[0]:

                if total_sentiment.index(max(total_sentiment), 0, 3) == 0:
                    diff = total_sentiment[0] - total_sentiment[2]
                    if diff != 0:
                        volume = int(100 / diff)

                    else:
                        if total_sentiment[0] != 0:
                            volume = 100 / total_sentiment[0]
                        else:
                            volume = 20

                else: # most unlikely scenario
                    if total_sentiment[2] != 0:
                        volume = 100 / total_sentiment[2]

                    else:
                        volume = 20

                self.log("BUY CREATE, %.2f" % self.dataclose[0])
                self.order = self.buy_bracket(limitprice = self.BOLLBands.lines.top[0],
                    price = self.dataclose[0],
                    stopprice = self.dataclose[0] - (self.p.stop_loss * self.dataclose[0]),
                    size = self.p.size)



data = bt.feeds.YahooFinanceData(
        dataname = stock_name, 
        fromdate = datetime.datetime(2015, 1, 1),
        todate = datetime.datetime(2019, 12, 31),
        reverse = False
    )

cerebro = bt.Cerebro(stdstats = False)
cerebro.addobserver(bt.observers.BuySell)

starting_portfolio_value = 100000
cerebro.addstrategy(SentimentBollingerStrategy)
cerebro.adddata(data)
cerebro.broker.setcash(100000)
cerebro.run()

print("Final portfolio value: " + str(cerebro.broker.getvalue()))
final_portfolio_value = cerebro.broker.getvalue()
