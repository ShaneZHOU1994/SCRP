import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


### CoSupply data preprocessing :
# hr = pd.read_csv('CoSupplyChain/DataCoSupplyChainDataset.csv')
# # hr.info()
# # hrs = hr.select_dtypes('number')
# hrs = hr[['Days for shipping (real)', 'Days for shipment (scheduled)', 'Benefit per order', 'Sales per customer',
#           'Order Item Discount', 'Order Item Discount Rate', 'Order Item Product Price', 'Order Item Profit Ratio',
#           'Order Item Quantity', 'Sales', 'Order Item Total', 'Order Profit Per Order', 'Product Price',
#           'Shipping Mode', 'order date (DateOrders)', 'shipping date (DateOrders)', 'Delivery Status',
#           'Late_delivery_risk']]
# hrs.info()
#
# print(hrs['Shipping Mode'].unique())
# hrs['Shipping Mode'].replace('Standard Class', 2, inplace=True)
# hrs['Shipping Mode'].replace('Second Class', 1, inplace=True)
# hrs['Shipping Mode'].replace('First Class', 0, inplace=True)
# hrs['Shipping Mode'].replace('Same Day', -1, inplace=True)
#
# print(hrs['Delivery Status'].unique())
# hrs['Delivery Status'].replace('Advance shipping', 0, inplace=True)
# hrs['Delivery Status'].replace('Shipping on time', 1, inplace=True)
# hrs['Delivery Status'].replace('Shipping canceled', -1, inplace=True)
# hrs['Delivery Status'].replace('Late delivery', 2, inplace=True)
#
# hrs['order date (DateOrders)'] = pd.to_datetime(hrs.loc[:, 'order date (DateOrders)'])
# hrs['shipping date (DateOrders)'] = pd.to_datetime(hrs.loc[:, 'shipping date (DateOrders)'])
# print(hrs.loc[:5, ['order date (DateOrders)', 'shipping date (DateOrders)']])
#
# hrs.to_csv('CoSupplyChain/PrepDataset.csv', sep=',', encoding='utf-8')

# ### model training :
# hr = pd.read_csv('CoSupplyChain/PrepDataset.csv')
# hr.info()
# X = hr.drop(['order date (DateOrders)', 'shipping date (DateOrders)',
#              'Delivery Status', 'Late_delivery_risk'], axis=1)
# y = hr['Late_delivery_risk']

### SCRM data preprocessing :
hr = pd.read_csv('SCRM/SCRM_timeSeries_2018_test.csv', parse_dates=['Timestamp'], infer_datetime_format=True)
# hr['Datetime_parsed'] = pd.to_datetime(hr['Timestamp'], format='%Y/%m/%d', infer_datetime_format=True)
print(hr.dtypes)
hr.Timestamp = hr.Timestamp.dt.to_period('T')
print(hr.head(20))
print('\n'+'-'*20)
print(hr.describe())

hrt = hr.copy()
y = hrt.Total_Cost.dropna() #SCMstability_category
dp = DeterministicProcess(index=y.index, constant=True, order=2, drop=True)
X = dp.in_sample()
# X = hrt.drop(columns=['Timestamp', 'SCMstability_category'], axis=1).fillna(0.0)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

## It will be easier for us later if we split the date index instead of the dataframe directly.
idx_train, idx_test = train_test_split(y.index, test_size=0.1, shuffle=False)
X_train, X_test = X.loc[idx_train, :], X.loc[idx_test, :]
y_train, y_test = y.loc[idx_train], y.loc[idx_test]

## Fit trend model
model = LinearRegression(fit_intercept=False)
model.fit(X_train, y_train)

## Make predictions
y_fit = pd.DataFrame(model.predict(X_train), index=y_train.index)
y_pred = pd.DataFrame(model.predict(X_test), index=y_test.index)

## Plot
axs = y_train.plot(color='0.25', subplots=True, sharex=True)
axs = y_test.plot(color='0.25', subplots=True, sharex=True, ax=axs)
axs = y_fit.plot(color='C0', subplots=True, sharex=True, ax=axs)
axs = y_pred.plot(color='C3', subplots=True, sharex=True, ax=axs)
for ax in axs: ax.legend([])
_ = plt.suptitle("Trends")
plt.show()


# model = DecisionTreeClassifier(max_depth=4).fit(X_train, y_train)
# model = xgb.XGBClassifier(max_depth=4, n_estimators=100).fit(X_train, y_train)
# print("Accuracy == ", accuracy_score(y_test, model.predict(X_test)))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass
