import numpy as np
import pandas as pd
from funcs import *
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


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
n0=100000
n1=400000
nstep=100000
hr = pd.read_csv('SCRM/SCRM_timeSeries_2018_train.csv', parse_dates=['Timestamp'], infer_datetime_format=True,
                 skiprows=range(1, n0), nrows=n1)
hrt = pd.read_csv('SCRM/SCRM_timeSeries_2018_train.csv', parse_dates=['Timestamp'], infer_datetime_format=True,
                  skiprows=range(1, n0+n1), nrows=nstep)
# hr['Datetime_parsed'] = pd.to_datetime(hr['Timestamp'], format='%Y/%m/%d', infer_datetime_format=True)
print(hr.dtypes)
hr.Timestamp = hr.Timestamp.dt.to_period('H')
# print(hr.head(20))
# print('\n'+'-'*20)
# print(hr.describe())

window = 0
y_train = hr.Total_Cost.fillna(0).shift(-window).fillna(0) #SCMstability_category
dp = DeterministicProcess(index=y_train.index, constant=True, order=0, drop=True)#,
                          # seasonal=True, additional_terms=[CalendarFourier(freq='S', order=4)])
X_train = dp.in_sample()
y_test = hrt.Total_Cost.fillna(0).shift(-window).fillna(0)
X_test = dp.out_of_sample(steps=nstep)
y_test.index = X_test.index  # Align test target with test data 'future'

## It will be easier for us later if we split the date index instead of the dataframe directly.
# idx_train, idx_test = train_test_split(y.index, test_size=0.2, shuffle=False)
# X_train, X_test = X.loc[idx_train, :], X.loc[idx_test, :]
# y_train, y_test = y.loc[idx_train], y.loc[idx_test]

## Fit trend model
model1 = LinearRegression(fit_intercept=False)
model1.fit(X_train, y_train)

## Make predictions
y_fit = pd.DataFrame(model1.predict(X_train), index=y_train.index)
y_pred = pd.DataFrame(model1.predict(X_test), index=y_test.index)

## Plot
# axs = y_train.plot(color='0.25', subplots=True, sharex=True)
# axs = y_test.plot(color='0.25', subplots=True, sharex=True, ax=axs)
# axs = y_fit.plot(color='C0', subplots=True, sharex=True, ax=axs)
# axs = y_pred.plot(color='C3', subplots=True, sharex=True, ax=axs)
# for ax in axs: ax.legend([])
# _ = plt.suptitle("Trends")
#
# # plot_periodogram(hr.Total_Cost)
# plt.show()



## The `stack` method converts column labels to row labels, pivoting from wide format to long
# X = hr.drop(columns=['Timestamp', 'SCMstability_category', 'Total_Cost'], axis=1).fillna(0.0) # pivot dataset wide to long

## Make lag features
nlag = 4
X = make_lags(hr.drop(columns=['Timestamp', 'SCMstability_category', 'Total_Cost'], axis=1), lags=nlag).fillna(0.0)
Xt = make_lags(hrt.drop(columns=['Timestamp', 'SCMstability_category', 'Total_Cost'], axis=1), lags=nlag).fillna(0.0)
# X = pd.concat([make_lags(pd.DataFrame(hr.Total_Cost.fillna(0), columns=['Total_Cost']), lags=nlag),
#                hr.drop(columns=['Timestamp', 'SCMstability_category', 'Total_Cost'], axis=1)], axis=1)
# Xt = pd.concat([make_lags(pd.DataFrame(hrt.Total_Cost.fillna(0), columns=['Total_Cost']), lags=nlag),
#                hrt.drop(columns=['Timestamp', 'SCMstability_category', 'Total_Cost'], axis=1)], axis=1)
X_train, y_train = X.align(y_train, join='inner', axis=0)
X_test = Xt
X_test.index = y_test.index
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


## Pivot wide to long (stack) and convert DataFrame to Series (squeeze)
y_fit = y_fit.squeeze()    # trend from training set
y_pred = y_pred.squeeze()  # trend from test set

## Learn residual
y_res = y_train - y_fit
model2 = XGBRegressor(max_depth=5, n_estimators=100)
model2.fit(X_train, y_res)

# Add the predicted residuals onto the predicted trends
y_fit_boosted = model2.predict(X_train) + y_fit
y_pred_boosted = model2.predict(X_test) + y_pred
print("Train R^2 : ", r2_score(y_train, y_fit_boosted))
print("Test R^2 : ", r2_score(y_test, y_pred_boosted))
print("Train MAE :", mean_absolute_error(y_train,y_fit_boosted))
print("Test MAE :", mean_absolute_error(y_test,y_pred_boosted))
print("Train RMSE:",np.sqrt(mean_squared_error(y_train, y_fit_boosted)))
print("Test RMSE:",np.sqrt(mean_squared_error(y_test, y_pred_boosted)))

## Plot
axs = y_train.plot(color='0.25', subplots=True, sharex=True)
axs = y_test.plot(color='0.25', subplots=True, sharex=True, ax=axs)
axs = y_fit_boosted.plot(color='C0', subplots=True, sharex=True, ax=axs)
axs = y_pred_boosted.plot(color='C3', subplots=True, sharex=True, ax=axs)
for ax in axs: ax.legend([])
_ = plt.suptitle("Trends combine Residuals")
plt.show()


# model1 = DecisionTreeClassifier(max_depth=4).fit(X_train, y_train)
# model2 = XGBClassifier(max_depth=4, n_estimators=100).fit(X_train, y_train)
# print("Accuracy == ", accuracy_score(y_test, model1.predict(X_test)))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass
