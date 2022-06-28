import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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

hr = pd.read_csv('CoSupplyChain/PrepDataset.csv')
hr.info()
X = hr.drop(['order date (DateOrders)', 'shipping date (DateOrders)',
             'Delivery Status', 'Late_delivery_risk'], axis=1)
y = hr['Late_delivery_risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model = xgb.XGBClassifier(max_depth=4, n_estimators=100).fit(X_train, y_train)
print("Accuracy == ", accuracy_score(y_test, model.predict(X_test)))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass
