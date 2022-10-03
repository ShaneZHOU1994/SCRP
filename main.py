import numpy as np
import pandas as pd
from funcs import *
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score


def read_data_simu():
    hr = pd.read_excel('Simu_w34.xlsx', header=[0,1,2,3], nrows=34, keep_default_na=False)
    hr.replace('', float('NaN'), inplace=True)
    hr.dropna(how='all', axis=1, inplace=True)

    ## remove unnamed levels :
    columns = pd.DataFrame(hr.columns.tolist())
    for i in range(len(columns.columns)):
        columns.loc[columns[i].str.startswith('Unnamed:'), i] = '_'
    hr.columns = pd.MultiIndex.from_tuples(columns.to_records(index=False).tolist())
    hr.columns = hr.columns.remove_unused_levels()
    ## expand multiindex :
    hr.columns = [' '.join(col).strip() for col in hr.columns.values if col != '_']
    cols = hr.columns
    return hr


def calcul_risk_simu(hr):
    ## Calculate Risk Indices :
    ## Factory F1 and F2:
    hr['CSR_F1'] = (hr['F1 Sale Price w1 BJ']*hr['F1 Sale w1 BJ']+hr['F1 Sale Price w2 BJ']*hr['F1 Sale w2 BJ']
                    +hr['F1 Sale Price w3 BJ']*hr['F1 Sale w3 BJ']+hr['F1 Sale Price w1 PO']*hr['F1 Sale w1 PO']
                    +hr['F1 Sale Price w2 PO'] * hr['F1 Sale w2 PO']+hr['F1 Sale Price w3 PO']*hr['F1 Sale w3 PO'])\
                   /(hr['F1 Receive SA1 B']*hr['F1 Buy Price SA1 B']+hr['F1 Receive SA1 P']*hr['F1 Buy Price SA1 P']
                     +hr['F1 Receive SA2 B']*hr['F1 Buy Price SA2 B']+hr['F1 Receive SA2 P']*hr['F1 Buy Price SA2 P']
                     +hr['F1 Receive SB1 PE']*hr['F1 Buy Price SB1 PE']+hr['F1 Receive SB1 O']*hr['F1 Buy Price SB1 O']
                     +hr['F1 Receive SB2 PE']*hr['F1 Buy Price SB2 PE']+hr['F1 Receive SB2 O']*hr['F1 Buy Price SB2 O']) - 1
    hr['PPR_F1'] = (hr['F1 Receive SA1 B']/hr['F1 Demand SA1 B']+hr['F1 Receive SA1 P']/hr['F1 Demand SA1 P']
                   +hr['F1 Receive SA2 B']/hr['F1 Demand SA2 B']+hr['F1 Receive SA2 P']/hr['F1 Demand SA2 P']
                   +hr['F1 Receive SB1 PE']/hr['F1 Demand SB1 PE']+hr['F1 Receive SB1 O']/hr['F1 Demand SB1 O']
                   +hr['F1 Receive SB2 PE']/hr['F1 Demand SB2 PE']+hr['F1 Receive SB2 O']/hr['F1 Demand SB2 O'])/8
    hr['OR_F1'] = (hr['F1 Product _ BJ']/(hr['F1 Order w1 BJ']+hr['F1 Order w2 BJ']+hr['F1 Order w3 BJ'])
                   +hr['F1 Product _ PO']/(hr['F1 Order w1 PO']+hr['F1 Order w2 PO']+hr['F1 Order w3 PO']))/2
    hr['OCR_F1'] = (hr['F1 Order w1 BJ'].shift(1).fillna(0)+hr['F1 Order w2 BJ'].shift(1).fillna(0)
                    +hr['F1 Order w3 BJ'].shift(1).fillna(0)+hr['F1 Order w1 PO'].shift(1).fillna(0)
                    +hr['F1 Order w2 PO'].shift(1).fillna(0)+hr['F1 Order w3 PO'].shift(1).fillna(0)) / \
                   (hr['F1 Order w1 BJ']+hr['F1 Order w2 BJ']+hr['F1 Order w3 BJ']\
                   +hr['F1 Order w1 PO']+hr['F1 Order w2 PO']+hr['F1 Order w3 PO']) - 1
    hr['SSR_F1'] = (hr['F1 Sale w1 BJ']+hr['F1 Sale w2 BJ']+hr['F1 Sale w3 BJ']+hr['F1 Sale w1 PO']
                    +hr['F1 Sale w2 PO']+hr['F1 Sale w3 PO']) / (hr['F1 Demand Stock BJ']+hr['F1 Demand Stock PO'])
    t_es = 3
    hr['ITO_F1'] = (hr['F1 Sale Price w1 BJ']*hr['F1 Sale w1 BJ']+hr['F1 Sale Price w2 BJ']*hr['F1 Sale w2 BJ']
                    +hr['F1 Sale Price w3 BJ']*hr['F1 Sale w3 BJ']+hr['F1 Sale Price w1 PO']*hr['F1 Sale w1 PO']
                    +hr['F1 Sale Price w2 PO'] * hr['F1 Sale w2 PO']+hr['F1 Sale Price w3 PO']*hr['F1 Sale w3 PO'])
    for i in range(1, t_es+1):
        hr['ITO_F1'] += (hr['F1 Sale Price w1 BJ'].shift(i).fillna(0)*hr['F1 Sale w1 BJ'].shift(i).fillna(0)
                    +hr['F1 Sale Price w2 BJ'].shift(i).fillna(0)*hr['F1 Sale w2 BJ'].shift(i).fillna(0)
                    +hr['F1 Sale Price w3 BJ'].shift(i).fillna(0)*hr['F1 Sale w3 BJ'].shift(i).fillna(0)
                    +hr['F1 Sale Price w1 PO'].shift(i).fillna(0)*hr['F1 Sale w1 PO'].shift(i).fillna(0)
                    +hr['F1 Sale Price w2 PO'].shift(i).fillna(0)*hr['F1 Sale w2 PO'].shift(i).fillna(0)
                    +hr['F1 Sale Price w3 PO'].shift(i).fillna(0)*hr['F1 Sale w3 PO'].shift(i).fillna(0))
    hr['ITO_F1'] /= (hr['F1 Demand Stock BJ']*(hr['F1 Sale Price w1 BJ']+hr['F1 Sale Price w2 BJ']+hr['F1 Sale Price w3 BJ'])/3
                     +hr['F1 Demand Stock PO']*(hr['F1 Sale Price w1 PO']+hr['F1 Sale Price w2 PO']+hr['F1 Sale Price w3 PO'])/3
                     +hr['F1 Demand Stock BJ'].shift(t_es).fillna(0)*(hr['F1 Sale Price w1 BJ'].shift(t_es).fillna(0)
                    +hr['F1 Sale Price w2 BJ'].shift(t_es).fillna(0)+hr['F1 Sale Price w3 BJ'].shift(t_es).fillna(0))/3
                     +hr['F1 Demand Stock PO'].shift(t_es).fillna(0)*(hr['F1 Sale Price w1 PO'].shift(t_es).fillna(0)
                    +hr['F1 Sale Price w2 PO'].shift(t_es).fillna(0)+hr['F1 Sale Price w3 PO'].shift(t_es).fillna(0))/3)/2

    # display(hr)
    X = hr.drop(['CSR_F1','PPR_F1','OR_F1','OCR_F1','SSR_F1','ITO_F1'], axis=1)
    y = (hr['CSR_F1']>=(hr['CSR_F1'].mean())).astype(int)
    return X, y


def prep_data_CoSupply():
    ## CoSupply data preprocessing :
    hr = pd.read_csv('CoSupplyChain/DataCoSupplyChainDataset.csv')
    # hr.info()
    # hrs = hr.select_dtypes('number')
    hrs = hr[['Days for shipping (real)', 'Days for shipment (scheduled)', 'Benefit per order', 'Sales per customer',
              'Order Item Discount', 'Order Item Discount Rate', 'Order Item Product Price', 'Order Item Profit Ratio',
              'Order Item Quantity', 'Sales', 'Order Item Total', 'Order Profit Per Order', 'Product Price',
              'Shipping Mode', 'Customer Segment', 'Market',
              'order date (DateOrders)', 'shipping date (DateOrders)', 'Delivery Status', 'Late_delivery_risk']]
    hrs.info()

    ## convert categorical to numerical :
    hrs['Shipping Mode'].replace(list(hrs['Shipping Mode'].unique()), list(range(len(hrs['Shipping Mode'].unique()))), inplace=True)
    hrs['Delivery Status'].replace(list(hrs['Delivery Status'].unique()), list(range(len(hrs['Delivery Status'].unique()))), inplace=True)
    hrs['Customer Segment'].replace(list(hrs['Customer Segment'].unique()), list(range(len(hrs['Customer Segment'].unique()))), inplace=True)
    hrs['Market'].replace(list(hrs['Market'].unique()), list(range(len(hrs['Market'].unique()))), inplace=True)
    ## convert timestamp to datetime variables :
    hrs['order date (DateOrders)'] = pd.to_datetime(hrs.loc[:, 'order date (DateOrders)'])
    hrs['shipping date (DateOrders)'] = pd.to_datetime(hrs.loc[:, 'shipping date (DateOrders)'])
    # print(hrs.loc[:5, ['order date (DateOrders)', 'shipping date (DateOrders)']])

    hrs.to_csv('CoSupplyChain/PrepDataset.csv', sep=',', encoding='utf-8')
    return hrs


def read_data_CoSupply(nskip=10000, nrow=10000):
    hr = pd.read_csv('CoSupplyChain/PrepDataset.csv', skiprows=range(1, nskip), nrows=nrow)
    hr.pop(hr.columns[0])
    # hr.info()
    # print(hr.describe(include=['category', np.object]))
    hr['order date (DateOrders)'] = pd.to_datetime(hr.loc[:, 'order date (DateOrders)'])
    hr['shipping date (DateOrders)'] = pd.to_datetime(hr.loc[:, 'shipping date (DateOrders)'])
    hr['order date (DateOrders)'] = hr['order date (DateOrders)'].dt.to_period('D')
    hr['shipping date (DateOrders)'] = hr['shipping date (DateOrders)'].dt.to_period('D')

    hr.sort_values(by=['order date (DateOrders)'])
    list_cat = ['Shipping Mode', 'Delivery Status', 'Customer Segment', 'Market', 'Late_delivery_risk']
    list_date = ['order date (DateOrders)', 'shipping date (DateOrders)']
    agg_func = {}
    for feat in hr.columns:
        if feat in list_cat:
            agg_func.update({feat: lambda x: x.value_counts().idxmax()})
        elif feat not in list_date:
            agg_func.update({feat: np.mean})
    hr = hr.groupby(['order date (DateOrders)']).agg(agg_func)

    X = hr.drop(['Days for shipping (real)', 'Days for shipment (scheduled)', 'Delivery Status', 'Late_delivery_risk'], axis=1)
    y = hr['Late_delivery_risk']
    return X, y


def xgb_clf_cosupply(X, y, nlag=4, nlead=4):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    ## Make lag features
    X_train = make_lags(X_train, lags=nlag).fillna(0.0)
    X_test = make_lags(X_test, lags=nlag).fillna(0.0)
    y_train = y_train.shift(-nlead).dropna()
    y_test = y_test.shift(-nlead).dropna()
    X_train, y_train = X_train.align(y_train, join='inner', axis=0)
    X_test, y_test = X_test.align(y_test, join='inner', axis=0)
    ## Learn model
    model = XGBClassifier(objective='binary:logistic', max_depth=4, n_estimators=30)#'multi:softprob'
    model.fit(X_train, y_train)
    ## Add the predicted residuals onto the predicted trends
    y_fit = model.predict(X_train)
    y_pred = model.predict(X_test)
    p_fit = model.predict_proba(X_train)
    p_pred = model.predict_proba(X_test)
    e_train = pd.get_dummies(y_train).astype('float32')
    e_test = pd.get_dummies(y_test)
    e_test = e_test.reindex(columns=e_train.columns.to_list()).fillna(0).astype('float32')
    y_train = np.array(y_train).ravel()
    y_test = np.array(y_test).ravel()
    return y_train, y_test, y_fit, y_pred, e_train, e_test, p_fit, p_pred


def read_data_SCRM(n0=1, n1=500000, nstep=100000):
    ## SCRM data preprocessing :
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
    return hr, hrt, nstep


def trend_model_SCRM(hr, hrt, nstep, nlead=4):
    y_train = hr.Total_Cost.fillna(0).shift(-nlead).fillna(0) #SCMstability_category
    dp = DeterministicProcess(index=y_train.index, constant=True, order=0, drop=True)#,
                              # seasonal=True, additional_terms=[CalendarFourier(freq='S', order=4)])
    X_train = dp.in_sample()
    y_test = hrt.Total_Cost.fillna(0).shift(-nlead).fillna(0)
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
    axs = y_train.plot(color='0.25', subplots=True, sharex=True)
    axs = y_test.plot(color='0.25', subplots=True, sharex=True, ax=axs)
    axs = y_fit.plot(color='C0', subplots=True, sharex=True, ax=axs)
    axs = y_pred.plot(color='C3', subplots=True, sharex=True, ax=axs)
    for ax in axs: ax.legend([])
    _ = plt.suptitle("Trends")
    # plot_periodogram(hr.Total_Cost)
    plt.show()
    return X_train, y_train, X_test, y_test, model1, y_fit, y_pred


def xgb_reg_SCRM(hr, hrt, y_train, y_test, y_fit, y_pred, nlag=4):
    ## Make lag features
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
    ## Add the predicted residuals onto the predicted trends
    y_fit_boosted = model2.predict(X_train) + y_fit
    y_pred_boosted = model2.predict(X_test) + y_pred
    return y_fit_boosted, y_pred_boosted


def xgb_clf_SCRM(hr, hrt, nlag=4, nlead=4):
    ## Make lag features
    X_train = make_lags(hr.drop(columns=['Timestamp', 'SCMstability_category'], axis=1), lags=nlag).fillna(0.0)
    X_test = make_lags(hrt.drop(columns=['Timestamp', 'SCMstability_category'], axis=1), lags=nlag).fillna(0.0)
    y_train = hr['SCMstability_category'].shift(-nlead).dropna()
    y_test = hrt['SCMstability_category'].shift(-nlead).dropna()
    X_train, y_train = X_train.align(y_train, join='inner', axis=0)
    X_test, y_test = X_test.align(y_test, join='inner', axis=0)
    ## Learn model
    model = XGBClassifier(objective='multi:softprob', max_depth=6, n_estimators=100)
    model.fit(X_train, y_train)
    ## Add the predicted residuals onto the predicted trends
    y_fit = model.predict(X_train)
    y_pred = model.predict(X_test)
    p_fit = model.predict_proba(X_train)
    p_pred = model.predict_proba(X_test)
    e_train = pd.get_dummies(y_train).astype('float32')
    e_test = pd.get_dummies(y_test)
    e_test = e_test.reindex(columns=e_train.columns.to_list()).fillna(0).astype('float32')
    y_train = np.array(y_train).ravel()
    y_test = np.array(y_test).ravel()
    return y_train, y_test, y_fit, y_pred, e_train, e_test, p_fit, p_pred


def eval_reg_result(y_train, y_test, y_fit_boosted, y_pred_boosted):
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


def eval_clf_result(y_train, y_test, y_fit, y_pred, e_train, e_test, p_fit, p_pred):
    print("Train Balanced Accuracy: ", balanced_accuracy_score(y_train, y_fit))
    print("Test Balanced Accuracy: ", balanced_accuracy_score(y_test, y_pred))
    print("Train Log Loss: ", log_loss(e_train, p_fit))
    print("Test Log Loss: ", log_loss(e_test, p_pred))
    # print("Train ROC AUC score: ", roc_auc_score(e_train, p_fit, multi_class='raise'))
    # print("Test ROC AUC score: ", roc_auc_score(e_test, p_pred, multi_class='raise'))
    ## Plot
    ntrain = len(y_train)
    ntest = len(y_test)
    fig, ax = plt.subplots()
    ax.plot(range(ntrain), y_train, 'k-')
    ax.plot(range(ntrain), y_fit, 'b-')
    ax.plot(np.arange(ntrain, ntrain+ntest), y_test, 'k-')
    ax.plot(np.arange(ntrain, ntrain+ntest), y_pred, 'r-')
    plt.legend(['GT train', 'pred train', 'GT test', 'pred test'])
    plt.title("Risk Category Classification")
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # hr = read_data_simu()
    # X, y = calcul_risk_simu(hr)
    # y_train, y_test, y_fit, y_pred, e_train, e_test, p_fit, p_pred = xgb_clf_cosupply(X, y, nlag=3, nlead=1)
    # eval_clf_result(y_train, y_test, y_fit, y_pred, e_train, e_test, p_fit, p_pred)
    # pass

    ## CoSupply dataset:
    # prep_data_CoSupply()
    X, y = read_data_CoSupply(nskip=10000, nrow=2000)
    y_train, y_test, y_fit, y_pred, e_train, e_test, p_fit, p_pred = xgb_clf_cosupply(X, y, nlag=5, nlead=1)
    eval_clf_result(y_train, y_test, y_fit, y_pred, e_train, e_test, p_fit, p_pred)

    ## SCRM dataset:
    # hr, hrt, nstep = read_data_SCRM(n0=300000, n1=50000, nstep=10000)
    ## reg model predict total cost:
    # X_train, y_train, X_test, y_test, model1, y_fit, y_pred = trend_model_SCRM(hr, hrt, nstep, nlead=10)
    # y_fit_boosted, y_pred_boosted = xgb_reg_SCRM(hr, hrt, y_train, y_test, y_fit, y_pred, nlag=10)
    # eval_reg_result(y_train, y_test, y_fit_boosted, y_pred_boosted)
    ## clf model predict risk category:
    # y_train, y_test, y_fit, y_pred, e_train, e_test, p_fit, p_pred = xgb_clf_SCRM(hr, hrt, nlag=10, nlead=10)
    # eval_clf_result(y_train, y_test, y_fit, y_pred, e_train, e_test, p_fit, p_pred)
