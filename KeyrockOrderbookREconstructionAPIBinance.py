#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

r = requests.get("https://api.binance.com/api/v3/depth",
                 params=dict(symbol="IOTABTC"))
results = r.json()

frames = {side: pd.DataFrame(data=results[side], columns=["price", "quantity"],
                             dtype=float)
          for side in ["bids", "asks"]}

frames_list = [frames[side].assign(side=side) for side in frames]
data = pd.concat(frames_list, axis="index", 
                 ignore_index=True, sort=True)

price_summary = data.groupby("side").price.describe()
price_summary.to_markdown()

fig, ax = plt.subplots()

ax.set_title(f"Last update: {t} (ID: {last_update_id})")

sns.scatterplot(x="price", y="quantity", hue="side", data=data, ax=ax)

ax.set_xlabel("Price")
ax.set_ylabel("Quantity")

plt.show()

fig, ax = plt.subplots()

ax.set_title(f"Last update: {t} (ID: {last_update_id})")

sns.histplot(x="price", hue="side", binwidth=binwidth, data=data, ax=ax)
sns.rugplot(x="price", hue="side", data=data, ax=ax)

plt.show()


fig, ax = plt.subplots()

ax.set_title(f"Last update: {t} (ID: {last_update_id})")

sns.histplot(x="price", weights="quantity", hue="side", binwidth=binwidth, data=data, ax=ax)
sns.scatterplot(x="price", y="quantity", hue="side", data=data, ax=ax)

ax.set_xlabel("Price")
ax.set_ylabel("Quantity")

plt.show()


fig, ax = plt.subplots()

ax.set_title(f"Last update: {t} (ID: {last_update_id})")

sns.ecdfplot(x="price", weights="quantity", stat="count", complementary=True, data=frames["bids"], ax=ax)
sns.ecdfplot(x="price", weights="quantity", stat="count", data=frames["asks"], ax=ax)
sns.scatterplot(x="price", y="quantity", hue="side", data=data, ax=ax)

ax.set_xlabel("Price")
ax.set_ylabel("Quantity")

plt.show()


# In[2]:



fig, ax = plt.subplots()

ax.set_title(f"Last update: {t} (ID: {last_update_id})")

sns.ecdfplot(x="price", weights="quantity", stat="count", complementary=True, data=frames["bids"], ax=ax)
sns.ecdfplot(x="price", weights="quantity", stat="count", data=frames["asks"], ax=ax)
sns.scatterplot(x="price", y="quantity", hue="side", data=data, ax=ax)

ax.set_xlabel("Price")
ax.set_ylabel("Quantity")

plt.show()


# In[ ]:




