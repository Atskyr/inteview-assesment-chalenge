#!/usr/bin/env python
# coding: utf-8

# In[63]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import regression
import datetime
#d = datetime.datetime.strptime('2011-06-09', '%Y-%m-%d')
#d.strftime('%b %d,%Y')


sns.set()
plt.style.use('seaborn-whitegrid')

data = pd.read_csv(r"C:\Users/Hp/Desktop/challenge/trades.csv")
print("thetyperofdata is: ",data.shape,"\n")
print(data.head())

data.dropna()
plt.figure(figsize=(10, 4))
plt.title("IOTA/BTC")
plt.xlabel("timestamp")
plt.ylabel("price")
plt.plot(data["price"])
plt.show()


from autots import AutoTS
model = AutoTS(forecast_length=1, frequency='infer', ensemble='simple', drop_data_older_than_periods=200)
model = model.fit(data, date_col='timestamp', value_col='price', id_col=None)
 
prediction = model.predict()
forecast = prediction.forecast
print("IOTA/BTC .predict is ")
print(forecast)


# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import regression
import datetime
#d = datetime.datetime.strptime('2011-06-09', '%Y-%m-%d')
#d.strftime('%b %d,%Y')


sns.set()
plt.style.use('seaborn-whitegrid')

data = pd.read_csv(r"C:\Users/Hp/Desktop/challenge/trades.csv")
#print("Shape of Dataset is: ",data.shape,"\n")
#print(data)
data1['timestamp'] = pd.to_datetime(data1['timestamp'])

data = data1[~(data1['timestamp'] > '2020-07-28 17:53:05')]

print(res)


# In[40]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import regression
import datetime
#d = datetime.datetime.strptime('2011-06-09', '%Y-%m-%d')
#d.strftime('%b %d,%Y')


sns.set()
plt.style.use('seaborn-whitegrid')

data = pd.read_csv(r"C:\Users/Hp/Desktop/challenge/trades.csv")

#data['timestamp'] = pd.to_datetime(data['timestamp'])
#value = input("Please enter date with this form --->yyyy-mm-dd hh:mm:ss")
 
#print(f'You entered {value}')
#data = data1[~(data1['timestamp'] > value)]
data = data[~(data['timestamp'] > '2020-07-29 17:53:05')]


print("thetyperofdata is: ",data.shape,"\n")
print(data.head())

data.dropna()
plt.figure(figsize=(10, 4))
plt.title("IOTA/BTC")
plt.xlabel("timestamp")
plt.ylabel("price")
plt.plot(data["price"])
plt.show()


from autots import AutoTS
model = AutoTS(forecast_length=1, frequency='infer', ensemble='simple', drop_data_older_than_periods=200)
model = model.fit(data, date_col='timestamp', value_col='price', id_col=None)
 
prediction = model.predict()
forecast = prediction.forecast
print("IOTA/BTC .predict is ")
print(forecast)
print(forecast)
print(forecast)
print(forecast)

forecastnumber = forecast[forecast.columns[-1]]
length = len(forecastnumber)
forecastnumber = forecastnumber[length -1]
forecastnumber= float(forecastnumber)

#forecastnumber = forecast[forecast.columns[ -1:, -1:]]
#forecastnumber = int(forecastnumber)
print(forecastnumber)


# 

# In[64]:


print(forecastnumber)



data = pd.read_csv(r"C:\Users/Hp/Desktop/challenge/trades.csv")
  
df = pd.DataFrame(data,columns=['timestamp','price',])

df.plot(x ='timestamp', y='price', kind = 'line')



data['timestamp'] = pd.to_datetime(data['timestamp'])
#value = input("Please enter date with this form --->yyyy-mm-dd hh:mm:ss")
 
#print(f'You entered {value}')
#data = data1[~(data1['timestamp'] > value)]
data = data[~(data['timestamp'] > '2020-07-29 17:53:05')]

ask1= int(forecastnumber())
bid1= int(forecastnumber())

asksum = ask1 * 1.02
askdif = asksum - ask1
bidsum = bid1 - askdif

asksum1 = int(asksum)
bidsum1 = int(bidsum)
#print(askdif)
plt.axhline(y = asksum1, xmin=0.93, xmax=1.5, color = 'r', linestyle = '-')
plt.axhline(y = bidsum1, xmin=0.93, color = 'g', linestyle = '-')


plt.show()


# In[65]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import regression
import datetime
#d = datetime.datetime.strptime('2011-06-09', '%Y-%m-%d')
#d.strftime('%b %d,%Y')


sns.set()
plt.style.use('seaborn-whitegrid')

data = pd.read_csv(r"C:\Users/Hp/Desktop/challenge/trades.csv")

data['timestamp'] = pd.to_datetime(data['timestamp'])
#value = input("Please enter date with this form --->yyyy-mm-dd hh:mm:ss")
 
#print(f'You entered {value}')
#data = data1[~(data1['timestamp'] > value)]
data = data[~(data['timestamp'] > '2020-07-29 17:53:05')]


print("thetyperofdata is: ",data.shape,"\n")
print(data.head())

data.dropna()
plt.figure(figsize=(10, 4))
plt.title("IOTA/BTC")
plt.xlabel("timestamp")
plt.ylabel("price")
plt.plot(data["price"])
plt.show()


from autots import AutoTS
model = AutoTS(forecast_length=1, frequency='infer', ensemble='simple', drop_data_older_than_periods=200)
model = model.fit(data, date_col='timestamp', value_col='price', id_col=None)
 
prediction = model.predict()
forecast = prediction.forecast
print("IOTA/BTC .predict is ")
print(forecast)
print(forecast)
print(forecast)
print(forecast)

forecastnumber = forecast[forecast.columns[-1]]
length = len(forecastnumber)
forecastnumber = forecastnumber[length -1]

#forecastnumber = forecast[forecast.columns[ -1:, -1:]]
#forecastnumber = int(forecastnumber)
float(forecastnumber)

print(str(forecastnumber))
print(float(forecastnumber))
print("%.9f" % (forecastnumber))

numberwearelookingfor = ("%.9f" % (forecastnumber))

ask1= numberwearelookingfor
#bid1= int(forecastnumber())
print(numberwearelookingfor)



data = pd.read_csv(r"C:\Users/Hp/Desktop/challenge/trades.csv")
  
df = pd.DataFrame(data,columns=['timestamp','price',])

df.plot(x ='timestamp', y='price', kind = 'line')



data['timestamp'] = pd.to_datetime(data['timestamp'])
#value = input("Please enter date with this form --->yyyy-mm-dd hh:mm:ss")
 
#print(f'You entered {value}')
#data = data1[~(data1['timestamp'] > value)]
data = data[~(data['timestamp'] > '2020-07-29 17:53:05')]





ask1= float(numberwearelookingfor)
bid1= float(numberwearelookingfor)
ask3= float(numberwearelookingfor)

asksum = ask1 * 1.05
askdif = asksum - ask1
bidsum = bid1 - askdif

asksum1 = float(asksum)
bidsum1 = float(bidsum)
#print(askdif)

#asksum2 = ask1 * 1.047
#askdif2 = asksum2 - ask1
#bidsum2 = bid1 - askdif2

#asksum2 = float(asksum2)
#bidsum2 = float(bidsum2)



#asksum3 = ask1 * 1.045
#askdif3 = asksum3 - ask1
#bidsum3 = bid1 - askdif3

#asksum3 = float(asksum2)
#bidsum3 = float(bidsum2)


#plt.axhline(y = asksum3, xmin=0.93, xmax=1.5, color = 'y', linestyle = '-')
plt.axhline(y = asksum1, xmin=0.93, xmax=1.5, color = 'r', linestyle = '-')
plt.axhline(y = bidsum1, xmin=0.93, color = 'g', linestyle = '-')
#level2
#plt.axhline(y = asksum2, xmin=0.93, xmax=1.5, color = 'r', linestyle = '-')
#plt.axhline(y = bidsum2, xmin=0.93, color = 'g', linestyle = '-')
#level3
#plt.axhline(y = asksum3, xmin=0.93, xmax=1.5, color = 'r', linestyle = '-')
#plt.axhline(y = bidsum3, xmin=0.93, color = 'g', linestyle = '-')


plt.show()


# In[49]:


im = float("8.99284722486562e-02")

print("%.17f" % (im))


# In[ ]:




data = pd.read_csv(r"C:\Users/Hp/Desktop/challenge/trades.csv")
  
df = pd.DataFrame(data,columns=['timestamp','price',])

df.plot(x ='timestamp', y='price', kind = 'line')



data['timestamp'] = pd.to_datetime(data['timestamp'])
#value = input("Please enter date with this form --->yyyy-mm-dd hh:mm:ss")
 
#print(f'You entered {value}')
#data = data1[~(data1['timestamp'] > value)]
data = data[~(data['timestamp'] > '2020-07-28 17:53:05')]





ask1= int(forecastnumber())
bid1= int(forecastnumber())

asksum = ask1 * 1.02
askdif = asksum - ask1
bidsum = bid1 - askdif

asksum1 = int(asksum)
bidsum1 = int(bidsum)
#print(askdif)
plt.axhline(y = asksum1, xmin=0.93, xmax=1.5, color = 'r', linestyle = '-')
plt.axhline(y = bidsum1, xmin=0.93, color = 'g', linestyle = '-')


plt.show()

