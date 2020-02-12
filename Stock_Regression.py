import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # disable GPU
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import numpy as np
import seaborn as sns

# processing / validation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# keras/tf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout

# models
from sklearn import svm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import tree
from sklearn.ensemble import VotingRegressor

# metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# constant seed for reproducibility
SEED = 111 
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# cpu workers
WORKERS = 6 


tickers = ["ASIANPAINT.NS","ADANIPORTS.NS","AXISBANK.NS","BAJAJ-AUTO.NS",
           "BAJFINANCE.NS","BAJAJFINSV.NS","BPCL.NS","BHARTIARTL.NS",
           "INFRATEL.NS","CIPLA.NS","COALINDIA.NS","DRREDDY.NS","EICHERMOT.NS",
           "GAIL.NS","GRASIM.NS","HCLTECH.NS","HDFCBANK.NS","HEROMOTOCO.NS",
           "HINDALCO.NS","HINDPETRO.NS","HINDUNILVR.NS","HDFC.NS","ITC.NS",
           "ICICIBANK.NS","IBULHSGFIN.NS","IOC.NS","INDUSINDBK.NS","INFY.NS",

           
           ]

df = pd.DataFrame() 
attempt = 0
drop = []
while len(tickers) != 0 and attempt <= 5:
    tickers = [j for j in tickers if j not in drop] 
    for i in range(len(tickers)):
        try:
            temp = web.get_data_yahoo(tickers[i],datetime.date.today() - datetime.timedelta(365*1), # reduce delta
                                      datetime.date.today())
            
            temp.dropna(inplace = True)
            df[tickers[i]] = temp["Adj Close"]
            drop.append(tickers[i])       
        except:
            print(tickers[i]," :failed to fetch data...retrying")
            continue
    attempt+=1
    

df.rename( {"ASIANPAINT.NS":"target"},axis=1, inplace=True)
df.dropna(inplace=True, axis=1, how='any')


X = df.drop("target", axis=1).values
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=SEED)
print(f"X_train shape:{X_train.shape}")

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

dnnReg = Sequential()

dnnReg.add(Dense(100, activation="relu"))
dnnReg.add(Dense(100, activation="relu"))
dnnReg.add(Dense(100, activation="relu"))
dnnReg.add(Dense(1))

dnnReg.compile(optimizer="adam", loss="mse")

dnnReg.fit(x=X_train_scaled, y=y_train, epochs=10,validation_data=(X_test_scaled,y_test),use_multiprocessing=True, workers=WORKERS)
pd.DataFrame(dnnReg.history.history).plot()

gboostReg = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.01, max_depth=1,  loss='ls', random_state=SEED).fit(X_train, y_train)
gboostPredds = gboostReg.predict(X_test)
print(mean_squared_error(y_test, gboostPredds))
print(r2_score(y_test, gboostPredds))