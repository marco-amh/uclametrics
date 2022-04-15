# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 21:16:10 2021

@author: marco
"""

import os
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import quandl
import tweepy as tw
import re
import collections
import nltk 
from nltk.corpus import stopwords
from textblob import TextBlob
from wordcloud import WordCloud
from pytrends.request import TrendReq
from pytrends import dailydata
import datetime
from datetime import date
import seaborn as sns
os.chdir('C://Users//marco//Desktop//Projects//Puller')
cwd = os.getcwd()
print("Current working directory: {0}".format(cwd))

class Puller:
    def Banxico(serie,name, plot):
        API_key_banxico = "bafc26d5c3c13e17b465d9550a5f38ca5f888a4dc8c6661ef3658720ba1702a9"
        URL = "https://www.banxico.org.mx/SieAPIRest/service/v1/series/"
        parameters = str(serie) + '/datos/?token=' + str(API_key_banxico)
        PATH =  str(URL) + parameters
        r = requests.get(PATH)
        data =r.text
        data = json.loads(data)

        list_a = data["bmx"]["series"][0]['datos']
        for i in list_a:
            if i['dato'] == "N/E":
                pass
            else:
                nuevo_val = float(i['dato'].replace(",",""))
                i['dato'] = nuevo_val

        data = []
        date = []

        for i in range(0,len(list_a)):
            if list_a[i]['dato'] == "N/E":
                pass
            else:
                date.append(list_a[i]['fecha'])
                data.append(list_a[i]['dato'])

        df = pd.DataFrame(date)
        df['Date'] = pd.DataFrame(date)
        df[name] = pd.DataFrame(data)
        df = df.set_index('Date')
        df = df.drop(columns=[0])
        df.index = pd.to_datetime(df.index, format = '%d/%m/%Y').strftime('%Y-%m-%d')
        df[name] = df[name].astype('float')
        if plot == True:
            plt.figure(figsize=(16,8))
            sns.set_style('ticks')
            line, = plt.plot(df.index,df[name], lw=2, linestyle='-', color='b')
            sns.despine()
            plt.gca().set(title=name, xlabel = 'Date', ylabel = name)
            plt.xticks(np.arange(0, len(df), step=round(len(df)*.05)), rotation=90)
            plt.show() #plot
    
        return(df)


# https://www.banxico.org.mx/SieAPIRest/service/v1/doc/catalogoSeries

    def Inegi(serie,name, plot):
        API_key_inegi = "36c55a30-5b09-f211-164e-06da067a41af"
        URL = "https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml/INDICATOR/"
        parameters = str(serie) + '/es/0700/false/BIE/2.0/' + str(API_key_inegi) + "?type=jsonp"
        PATH =  str(URL) + parameters
        r = requests.get(PATH)
        data = r.text
        data = json.loads(data)
        list_a = data['Series'][0]['OBSERVATIONS']
        data = []
        date = []
        for i in range(0,len(list_a)):
            date.append(list_a[i]['TIME_PERIOD'])
            data.append(list_a[i]['OBS_VALUE'])
    
        df = pd.DataFrame(date)
        df['Date'] = pd.DataFrame(date)
        df[name] = pd.DataFrame(data)
        df = df.set_index('Date')
        df = df.drop(columns=[0])
        df.index = pd.to_datetime(df.index, format = '%Y/%m').strftime('%Y-%m-%d')
        df.sort_index(ascending=True, inplace=True)
        df[name] = df[name].astype('float')  
        if plot == True:
            plt.figure(figsize=(16,8))
            sns.set_style('ticks')
            line, = plt.plot(df.index,df[name], lw=2, linestyle='-', color='b')
            sns.despine()
            plt.gca().set(title=name, xlabel = 'Date', ylabel = name)
            plt.xticks(np.arange(0, len(df), step=round(len(df)*.05)), rotation=90)
            plt.show() #plot
        return(df)


# https://www.inegi.org.mx/servicios/api_indicadores.html

    def Fred(serie,name, plot):
        API_key_fred = "1d956f22f710e02c387275323899460a"
        URL = "https://api.stlouisfed.org/fred/series/observations?series_id="
        parameters = str(serie) + '&api_key=' + str(API_key_fred) + "&file_type=json"
        PATH =  str(URL) + parameters
        r = requests.get(PATH)
        data = r.text
        data = json.loads(data)
        list_a = data['observations']
        data = []
        date = []
        for i in range(0,len(list_a)):
            date.append(list_a[i]['date'])
            data.append(list_a[i]['value'])
    
        df = pd.DataFrame(date)
        df['Date'] = pd.DataFrame(date)
        df[name] = pd.DataFrame(data)
        df = df.set_index('Date')
        df = df.drop(columns=[0])
        df.index = pd.to_datetime(df.index, format = '%Y-%m-%d').strftime('%Y-%m-%d')
        df.sort_index(ascending=True, inplace=True)
        df[name] = df[name].astype('float')
        if plot == True:
            plt.figure(figsize=(16,8))
            sns.set_style('ticks')
            line, = plt.plot(df.index,df[name], lw=2, linestyle='-', color='b')
            sns.despine()
            plt.gca().set(title=name, xlabel = 'Date', ylabel = name)
            plt.xticks(np.arange(0, len(df), step=round(len(df)*.05)), rotation=90)
            plt.show() #plot
        return(df)



    def IMF(serie,name, plot):
        API_key_imf = 'CompactData/IFS/'
        URL = 'http://dataservices.imf.org/REST/SDMX_JSON.svc/'
        parameters = str(API_key_imf) + str(serie)
        PATH =  str(URL) + parameters
        r = requests.get(PATH)
        data = r.text
        data = json.loads(data)
        data = data['CompactData']['DataSet']['Series']
        data_list = [[obs.get('@TIME_PERIOD'), obs.get('@OBS_VALUE')]
                     for obs in data['Obs']]
        df = pd.DataFrame(data_list, columns=['Date', name])
        df = df.set_index('Date')
        df.index = pd.to_datetime(df.index, format = '%Y-%m').strftime('%Y-%m-%d')
        df[name] = df[name].astype('float')
        if plot == True:
            plt.figure(figsize=(16,8))
            sns.set_style('ticks')
            line, = plt.plot(df.index,df[name], lw=2, linestyle='-', color='b')
            sns.despine()
            plt.gca().set(title=name, xlabel = 'Date', ylabel = name)
            plt.xticks(np.arange(0, len(df), step=round(len(df)*.05)), rotation=90)
            plt.show() #plot
        return(df)


# http://www.bd-econ.com/imfapi1.html

    def WTO(serie,country,name, plot):
        API_key_wto_a = 'cc4f1b42693846f1b0435a77ee994e67'
        API_key_wto_b = '846439f545694810a3484917176c8df6'
        URL = 'https://api.wto.org/timeseries/v1/data?i='
        parameters = str(serie) + '&r=' + str(country) + '&ps=all&max=5000&fmt=json&mode=full&lang=1&meta=false&subscription-key=' + str(API_key_wto_a)
        PATH =  str(URL) + parameters
        r = requests.get(PATH)
        data = r.text
        data = json.loads(data)
        data_list = [[obs.get('Period'), obs.get('Year'),obs.get('Value')]
                     for obs in data['Dataset']]
        df = pd.DataFrame(data_list, columns=['Month','Year', name])
        df['Date'] = df['Year'].astype(str) + '-' + df['Month'] + '-01'
        df = df.set_index('Date')
        df.index = pd.to_datetime(df.index, format = '%Y-%B-%d').strftime('%Y-%m-%d')
        df.sort_index(ascending=True, inplace=True)
        df[name] = df[name].astype('float')
        df= df.drop(['Month', 'Year'], axis = 1)
        if plot == True:
            plt.figure(figsize=(16,8))
            sns.set_style('ticks')
            line, = plt.plot(df.index,df[name], lw=2, linestyle='-', color='b')
            sns.despine()
            plt.gca().set(title=name, xlabel = 'Date', ylabel = name)
            plt.xticks(np.arange(0, len(df), step=round(len(df)*.05)), rotation=90)
            plt.show() #plot
        return(df)

# https://apiportal.wto.org/query-builder#


    def WB(serie,country, name, plot):
        URL = 'http://api.worldbank.org/v2/country/'
        parameters = str(country) + '/indicator/' + str(serie) + '?format=json'
        PATH =  str(URL) + parameters
        r = requests.get(PATH)
        data = r.text
        data = json.loads(data)
        data = data[1]
        data_list = [[obs.get('date'), obs.get('value')]
                     for obs in data]
        df = pd.DataFrame(data_list, columns=['Date', name])
        df['Date'] = df['Date'] + '-12'
        df = df.set_index('Date')
        df.index = pd.to_datetime(df.index, format = '%Y-%m').strftime('%Y-%m-%d')
        df.sort_index(ascending=True, inplace=True)
        df[name] = df[name].astype('float')
        if plot == True:
            plt.figure(figsize=(16,8))
            sns.set_style('ticks')
            line, = plt.plot(df.index,df[name], lw=2, linestyle='-', color='r')
            sns.despine()
            plt.gca().set(title=name, xlabel = 'Date', ylabel = name)
            plt.xticks(np.arange(0, len(df), step=round(len(df)*.05)), rotation=90)
            plt.show() #plot
        return(df)



# https://data.worldbank.org/indicator

    def yfin(serie, name, plot):
        df = yf.download(tickers = serie, period ="max", interval = "1mo")['Close'].dropna().to_frame()
        df.rename(columns={df.columns[0]: name }, inplace=True)
        df.index = pd.to_datetime(df.index, format = '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
        if plot == True:
            plt.figure(figsize=(16,8))
            sns.set_style('ticks')
            line, = plt.plot(df.index,df[name], lw=2, linestyle='-', color='b')
            sns.despine()
            plt.gca().set(title=name, xlabel = 'Date', ylabel = name)
            plt.xticks(np.arange(0, len(df), step=round(len(df)*.05)), rotation=90)
            plt.show() #plot
            return(df)

#!pip install yfinance --upgrade --no-cache-dir 

# quandl
    def qndl(serie, name, plot):
        quandl.ApiConfig.api_key = "K9ZEBqduJFAH2WG_JAVg"#API key
        df = quandl.get(serie)
        df.rename(columns={df.columns[0]: name }, inplace=True)
        df.index = pd.to_datetime(df.index, format = '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
        if plot == True:
            plt.figure(figsize=(16,8))
            sns.set_style('ticks')
            line, = plt.plot(df.index,df[name], lw=2, linestyle='-', color='b')
            sns.despine()
            plt.gca().set(title=name, xlabel = 'Date', ylabel = name)
            plt.xticks(np.arange(0, len(df), step=round(len(df)*.05)), rotation=90)
            plt.show() #plot
        return(df)




# twitter
#https://www.youtube.com/watch?v=ujId4ipkBio
    def tuiti(search_term, mode):
        consumer_key = "e1FYKkEzVeg6TYtP7Ta4etfTC"
        consumer_secret = "l8Uw6fDdMdxSOXzEsGNQXreSLNBsyKAb2jeXaNGUKYqgNPjE4V"
        access_token ="558587999-zwP08BA5Mq2UVlJZ58pUEfRgKUmZlIWUJXnym6mq"
        access_token_secret = "MCLClo4NyXjK9GstJuZRnX2Tidn1yGKasjxakbFFdQ5YT"
    
        auth = tw.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tw.API(auth, wait_on_rate_limit = True)
    
        # Stopwords
        nltk.download('stopwords')
        stop_words = set(stopwords.words('spanish'))
        #list(stop_words)[0:10]
        # Regular expressions subtitution
        def keep_text(txt):
            return " ".join(re.sub("(@[^0-9A-Za-z \t]) | (\w+:\/\/\S+)","",txt).split())
        # Create subjectivite function
        def getSubjectivity(text):
            return TextBlob(text).translate(from_lang='es', to='en').sentiment.subjectivity
        # Create polarity
        def getPolarity(text):
            return TextBlob(text).translate(from_lang='es', to='en').sentiment.polarity
        # Analysis of tweets
        def getAnalysis(score):
            if score > 0:
                return 'Negative'
            elif score == 0:
                return 'Neutral'
            else:
                return 'Positive'
    
        if mode == "account":
            tweets = tw.Cursor(api.user_timeline,
                               screen_name = search_term,
                               tweet_mode = 'extended'
                               ).items(49)
        
            df = pd.DataFrame([tweet.full_text for tweet in tweets], columns=['Tweets'])
            df['Tweets'] = df['Tweets'].apply(keep_text)
            #df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)
            df['Subjectivity'] = [TextBlob(i).translate(from_lang='es', to='en').sentiment.subjectivity for i in df['Tweets']]
            #df['Polarity'] = df['Tweets'].apply(getPolarity)
            df['Polarity'] = [TextBlob(i).translate(from_lang='es', to='en').sentiment.polarity for i in df['Tweets']]
            df['Analysis'] = df['Polarity'].apply(getAnalysis)
        
            ptweets = df[df.Analysis== 'Positive']
            ptweets = ptweets['Tweets']
            round((ptweets.shape[0]/df.shape[0])*100,1)
        
            # Plot the word cloud
            pt = ' '.join([twts for twts in df['Tweets']])
            wordCloud = WordCloud(width=500,height=300,random_state=92,max_font_size=119).generate(pt)
            plt.imshow(wordCloud, interpolation = 'bilinear')
            plt.axis('off')
            plt.show()
            # Scatterplot
            plt.figure(figsize=(8,6))
            plt.scatter(df['Polarity'], df['Subjectivity'], color='Blue')
            plt.title('Sentiment Analysis')
            plt.xlabel('Polarity')
            plt.ylabel('Subjectivity')
            plt.show()
            # Bar plot
            plt.title('Sentiment Analysis')
            plt.xlabel('Sentiment')
            plt.ylabel('Counts')
            df['Analysis'].value_counts().plot(kind='bar')
            plt.show()
                
            return(df)
        elif mode == "word":
            tweets = tw.Cursor(api.search,
                               q = search_term,
                               lang = 'es',
                               since = '2010-01-01'
                               ).items(100)
        
            all_tweets = [tweet.text for tweet in tweets]
            clean_tweets = [keep_text(tweet) for tweet in all_tweets]
            cleaner_tweets = [tweet.lower().split() for tweet in clean_tweets]
            all_words = [item for sublist in cleaner_tweets for item in sublist]
            all_words = [x for x in all_words if x not in stop_words]
            counts = collections.Counter(all_words)
            print(counts.most_common(15))
            return(counts)



#sub = TextBlob('la cama es redonda').translate(from_lang='es', to='en').sentiment.subjectivity
#pol = TextBlob('la cama es redonda').translate(from_lang='es', to='en').sentiment.polarity

#googletrends
#!pip install pytrends --upgrade --no-cache-dir 

    def gulutrend(serie,plot, freq):
        ptrends = TrendReq(hl = 'es-MX', 
                           tz = 360, 
                           timeout = 10.25, 
                           #proxies=['https://34.203.233.13:80',],
                           retries = 2,
                           backoff_factor = 0.1)
    
        if freq == 'weekly':
            today = date.today() # todays date
            delta = datetime.timedelta(365 * 5,0,0) # 5 year lag
            # create a time frame
            timeframes = []
    
            for i in range(1,5): ### Change to be dynamic ###
                lagged = (today - (i * delta) ).strftime("%Y-%m-%d")
                nonlagged = (today - ((i-1) * delta) ).strftime("%Y-%m-%d")
                timeframes = timeframes + [ lagged + " " + nonlagged ] # Add in each time frame to a df

            df = pd.DataFrame() #initialize a dataframe
   
            for tf in timeframes: # loop through the timeframes
                ptrends.build_payload([serie], 
                                      cat = 0, 
                                      timeframe = tf,
                                      geo = 'MX',
                                      gprop = '')
                print(tf)
                df_temp = pd.DataFrame(ptrends.interest_over_time()) #Do the request to get df_temp
        
                if df.shape != (0,0) and df_temp.shape != (0,0):  #Check thet the df aren't empty
                    factor = (df.iloc[0,0] / df_temp.iloc[-1,0])  #Make scaling factor
                    df_temp = df_temp * factor                    #Rewrite df_temp
        
                df = pd.concat([ df_temp, df]) 
    
            df.index = pd.to_datetime(df.index, format = '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')

            if plot == True:
                plt.figure(figsize=(16,8))
                sns.set_style('ticks')
                line, = plt.plot(df.index,df[serie], lw=2, linestyle='-', color='b')
                sns.despine()
                plt.gca().set(title=serie, xlabel = 'Date', ylabel = serie)
                plt.xticks(np.arange(0, len(df), step=round(len(df)*.05)), rotation=90)
                plt.show() #plot
    
        elif freq == 'monthly':
            ptrends.build_payload([serie],cat = 0,timeframe = 'all' ,geo = 'MX',gprop = '')
            df = pd.DataFrame(ptrends.interest_over_time())
            df.index = pd.to_datetime(df.index, format = '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
        
            if plot == True:
                plt.figure(figsize=(16,8))
                sns.set_style('ticks')
                line, = plt.plot(df.index,df[serie], lw=2, linestyle='-', color='b')
                sns.despine()
                plt.gca().set(title=serie, xlabel = 'Date', ylabel = serie)
                plt.xticks(np.arange(0, len(df), step=round(len(df)*.05)), rotation=90)
                plt.show() #plot

        elif freq == 'daily':
            today = date.today()
            df = dailydata.get_daily_data(serie, 2004, 1, today.year, today.month, geo = 'MX')
            df.index = pd.to_datetime(df.index, format = '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
            df = df.iloc[:,4:5] 
        
            if plot == True:
                plt.figure(figsize=(16,8))
                sns.set_style('ticks')
                line, = plt.plot(df.index,df[serie], lw=2, linestyle='-', color='b')
                sns.despine()
                plt.gca().set(title=serie, xlabel = 'Date', ylabel = serie)
                plt.xticks(np.arange(0, len(df), step=round(len(df)*.05)), rotation=90)
                plt.show() #plot
            
        return df.iloc[:,0:1]





#df = Puller.Banxico(serie="SR15059", name="Incertidumbre_pol_interna", plot=True)
#Puller.Inegi('444557','Economically Active Population', plot=True)
#Puller.Fred('INDPRO', 'Industrial_Production', plot=True)
#Puller.IMF('M.MX.PCPI_IX', 'Consumer_Price_Index_in_Mexico', plot=True)
#Puller.WTO('ITS_MTV_MX','484','Total_Merchandise_Exports_Mexico', plot=True)
#Puller.WB('NY.GDP.MKTP.KN', 'ar','Population_GB', plot=True)
#Puller.yfin(serie= 'qqq', name='Google', plot = True)
#Puller.qndl(serie= 'CHRIS/CBOE_VX1', name='Oil', plot = True)
#Puller.tuiti(search_term ='marcovaas', mode = 'account')
#kw_list = ["incertidumbre"] # Choose your Keywords
#Puller.gulutrend(serie = kw_list[0], plot=True, freq='monthly')


