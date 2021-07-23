#!/usr/bin/env python
# coding: utf-8

# # Set up environment

# In[65]:


# bring in libraries
import sys
import sklearn
import numpy as np
import pandas as pd
import os
import seaborn as sns
import datetime
get_ipython().run_line_magic('matplotlib', 'inline')

# to make this notebook's output stable across runs
np.random.seed(42)

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "C:\\Users\\forbej06\\OneDrive - Kingfisher PLC\\dev\\Jupyter"
MODEL = "linear regression"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", MODEL)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# ## Import data set

# In[66]:


df = pd.read_excel('C:\\Users\\forbej06\\OneDrive - Kingfisher PLC\\Reports\\Trading\\BANNER WEEKLY EXEC PACK INPUT SHEET for REPORTING.xlsx', sheet_name='Jupyter')
df = df.loc[df['Banner'] == 'B&Q'].dropna()
df['finWeek'] = df['FY'].map(str) + '-' + df['Week'].map(str)

df['finWeek'] = pd.to_datetime(df.finWeek.add('-0'), format='%Y-%W-%w', infer_datetime_format=True)
df = df.set_index('finWeek')
df # weekly trading data


# ## Visualise Data

# In[67]:


df.hist(bins=50, figsize=(20,15))
save_fig('hist')
plt.show


# In[68]:


sns.heatmap(df.corr())


# ## Define X and y

# In[69]:


X = df.iloc[:,14:17] # Realised Part. %, Sessions & Orders (GA)
y = df.iloc[:,13] # Realised Digital Sales
X,y.head()


# In[70]:


plt.xlabel('Week')
plt.ylabel('Web Sales')
y.plot()
plt.xticks(rotation = 45)


# # Split data into training set & test set

# ## Train the model on the training set

# In[71]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[72]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)


# ## Predict test set results

# In[73]:


y_pred = lin_reg.predict(X_test)
print(y_pred)


# In[74]:


print(lin_reg.coef_)


# In[75]:


print(lin_reg.intercept_)


# ## Evaluate the model

# In[76]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[77]:


# Plot the results


# In[78]:


plt.plot(y_test, y_pred, "b.")
plt.plot(X, y, "r.")
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Digital Sales Forecast Trajectory')
plt.legend(['prediction', 'history'])
save_fig('DIY Sales Forecast')
plt.show()


# ## Predicted values

# In[79]:


pred_y_df = pd.DataFrame({'Actual Value':y_test, 'Predicted Value':y_pred})


# In[80]:


pred_y_df.sort_index()  # test set prediction


# ## Bootstrap

# ## Re-evaluate

# # ARIMA Model

# ## Make the time series stationary

# In[81]:


# rolling statistics (method 1)
moving_avg = y.rolling(window=13).mean()
plt.plot(ts_log, label='log')
plt.plot(moving_avg, color='red', label='moving_avg')
plt.legend(loc='lower right', bbox_to_anchor=(0.67, 0.95), fancybox=True, shadow=True)
plt.xticks(rotation=45)


# In[82]:


# plot dickey-fuller test (method 2)

from statsmodels.tsa.stattools import adfuller
def test_stationarity(y):
    
    # determining rolling statistics
    rollMean = y.rolling(window=13).mean()
    rollStd = y.rolling(window=13).std()
# plot rolling statistics
    plt.plot(y, color='blue', label='Original')
    plt.plot(rollMean, color='red', label='Rolling Mean')
    plt.plot(rollStd, color='black', label='Rolling Std')
    plt.legend(loc='best', shadow=True)
    plt.title('Rolling Mean & Standard Deviation')
    plt.xlabel('Week')
    plt.ylabel('Web Sales')
    plt.xticks(rotation=45)
    plt.show()
    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller test:')
    dfTest  = adfuller(y, autolag='AIC')
    dfOutput = pd.Series(dfTest[0:4], index=['Test Statistic', 'p-value',
                        '#Lags Used', 'Number of Observations Used'])
    for key, value in dfTest[4].items():
        dfOutput['Critical Value (%s)'%key] = value
    print(dfOutput)
    
test_stationarity(y)


# ### Converting series to Stationary
# 

# In[83]:


y.head()


# In[84]:


y_diff = y.diff(periods=1)


# In[85]:


y_diff = y_diff[1:]
y_diff.head()


# In[86]:


from statsmodels.graphics.tsaplots import plot_acf
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
ax1.plot(y)
ax1.set_title('Original')
ax1.tick_params('x', labelrotation=45)
ax2.plot(y_diff)
ax2.set_title('Difference')
ax2.tick_params('x', labelrotation=45)
plot_acf(y_diff);


# 

# In[87]:


# plot difference for stationarity
y_diff.plot()


# In[88]:


from statsmodels.tsa.arima.model import ARIMA
# Fit ARIMA model to data - p=periods taken for autoregressive model, d=Integrated order/difference, q=periods in moving average model
# Fit model
model = ARIMA(y, order=(2,1,0))
model_fit = model.fit()
# Summary of fit model
print(model_fit.summary())
# Line plot of residules
residules = pd.DataFrame(model_fit.resid)
residules.plot()
plt.show()
# Density plot of residules
residules.plot(kind='kde')
plt.show()
# summary stats of residules
print(residules.describe())

