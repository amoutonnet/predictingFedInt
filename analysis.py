import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.utils import resample
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import GridSearchCV
import sys
import time
import warnings
from tqdm import tqdm
import seaborn as sns
import pickle as pkl
warnings.filterwarnings("ignore")


def get_data(split_idx='2013-01-01', n=3, to_keep_y='fedEffectiveInterest'):
    try:
        X = pd.read_csv('Datasets/X_n_%d.csv' % n, index_col=0, parse_dates=True)
        y = pd.read_csv('Datasets/y_n_%d.csv' % n, index_col=0, parse_dates=True)
    except FileNotFoundError:
        data_df = pd.read_csv('final_data.csv')
        data_df["date"] = data_df["year"].map(str) + data_df["period"]
        data_df["date"] = data_df["date"].apply(lambda s: dt.datetime.strptime(s, '%YM%m'))
        data_df = data_df.set_index('date')
        data_df = data_df.drop(['period', 'year', 'cpi', 'cpiExclEnergy', 'cpiEnergy'], axis=1)
        data_df = data_df.loc[(data_df.index >= '2000-01-01') & (data_df.index < '2019-01-01')]

        def process_time_series(df):
            y = df.iloc[n:, :2]
            columns = []
            for i in range(1, n+1):
                columns = [col+'_t-%d' % i for col in df.columns] + columns
            X = pd.DataFrame(columns=columns)
            for idx, dtstamp in enumerate(df.index[n:]):
                features = df.loc[(df.index >= df.index[idx]) & (df.index < dtstamp)]
                X.loc[dtstamp] = features.values.flatten()
            X.to_csv('Datasets/X_n_%d.csv' % n)
            y.to_csv('Datasets/y_n_%d.csv' % n)
            return X, y
        X, y = process_time_series(data_df)
    X = pd.get_dummies(X)
    y = y[[to_keep_y]]
    return X.loc[:split_idx], y.loc[:split_idx], X.loc[split_idx:], y.loc[split_idx:]


def get_osr2(y_test, y_train, y_pred):
    return 1-sum(np.square(np.array(y_test).reshape(-1)-np.array(y_pred).reshape(-1)))/sum(np.square(np.array(y_test).reshape(-1)-np.mean(np.array(y_train))))


def reg_mod(type_of_mod, X_train, y_train, X_test, y_test, ax=None, alpha=0.95, get_osr=False, cross_val=True):
    if type_of_mod == 'Decision Tree':
        if cross_val:
            param_grid = {
                'min_samples_leaf': list(range(5,16,2)),
                'max_depth': list(range(2,9,1))
            }
            grid_search = GridSearchCV(estimator = DecisionTreeRegressor(random_state=42), param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)
            grid_search.fit(X_train, y_train.values.ravel())
            print(grid_search.best_params_)
            time.sleep(1)
            mod = grid_search.best_estimator_
        else:
            mod = DecisionTreeRegressor(random_state=42)
    elif type_of_mod == 'Random Forest':
        if cross_val:
            param_grid = {
                'max_features': list(range(1,16,2)),
                'n_estimators': [200,400,600,800,1000],
            }
            grid_search = GridSearchCV(estimator = RandomForestRegressor(random_state=42), param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)
            grid_search.fit(X_train, y_train.values.ravel())
            print(grid_search.best_params_)
            time.sleep(1)
            mod = grid_search.best_estimator_
        else:
            mod = RandomForestRegressor(random_state=42)
    elif type_of_mod == 'Linear Regression':
        mod = linear_model.LinearRegression()
    else:
        return
    n_iterations = 500
    n_size = len(X_train)
    osr2_stats = np.empty((n_iterations,))
    if not get_osr:
        preds_stats = np.empty((len(y_test), n_iterations))
        r2_stats = np.empty((n_iterations,))
    print('Processing bootstrap for ' + type_of_mod)
    for i in tqdm(range(n_iterations)):
        boot_x, boot_y = resample(X_train, y_train, replace=True, n_samples=n_size)
        mod.fit(boot_x, boot_y)
        y_pred = mod.predict(X_test).reshape(-1)
        preds_stats[:, i] = y_pred
        osr2_stats[i] = get_osr2(y_test, y_train, y_pred)
        r2_stats[i] = mod.score(boot_x, boot_y)

    if not get_osr:
        preds_stats = np.sort(preds_stats)
        preds_lower = preds_stats[:, int(n_iterations*(1-alpha))]
        preds_upper = preds_stats[:, int(n_iterations*alpha)]
        r2_stats = np.sort(r2_stats)
        r2_lower = r2_stats[int(n_iterations*(1-alpha))]
        r2_upper = r2_stats[int(n_iterations*alpha)]

    osr2_stats = np.sort(osr2_stats)
    osr2_lower = osr2_stats[int(n_iterations*(1-alpha))]
    osr2_upper = osr2_stats[int(n_iterations*alpha)]

    mod.fit(X_train, y_train)
    y_pred = mod.predict(X_test)
    if not get_osr:
        return (y_pred, preds_lower, preds_upper), (mod.score(X_train, y_train), r2_lower, r2_upper), (get_osr2(y_test, y_train, y_pred), osr2_lower, osr2_upper)
    else:
        return [get_osr2(y_test, y_train, y_pred), osr2_lower, osr2_upper]


def class_mod(type_of_mod, X_train, y_train, X_test, y_test, alpha=0.95, get_acc=False, cross_val=False, bootstrap=True):
    labels=['DOWN', 'SAME', 'UP']
    if type_of_mod == 'LDA':
        mod = LinearDiscriminantAnalysis()
    elif type_of_mod == 'Random Forest':
        if cross_val:
            param_grid = {
                'max_features': list(range(1,16,2)),
                'n_estimators': [200,400,600,800,1000],
            }
            grid_search = GridSearchCV(estimator = RandomForestClassifier(random_state=42), param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)
            grid_search.fit(X_train, y_train.values.ravel())
            print(grid_search.best_params_)
            time.sleep(1)
            mod = grid_search.best_estimator_
        else:
            mod = RandomForestClassifier(random_state=42)
    elif type_of_mod == 'Decision Tree':
        if cross_val:
            param_grid = {
                'min_samples_leaf': list(range(5,16,2)),
                'max_depth': list(range(2,9,1))
            }
            grid_search = GridSearchCV(estimator = DecisionTreeClassifier(random_state=42), param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)
            grid_search.fit(X_train, y_train.values.ravel())
            print(grid_search.best_params_)
            time.sleep(1)
            mod = grid_search.best_estimator_
        else:
            mod = DecisionTreeClassifier(random_state=42)
    else:
        return
    n_iterations = 500
    acc_stats = np.zeros((n_iterations,))
    if not get_acc:
        tpr_stats = np.zeros((3, n_iterations))
        fpr_stats = np.zeros((3, n_iterations))
    if bootstrap:
        print('Processing bootstrap for ' + type_of_mod)
        bootmodel = BaggingRegressor(mod, 
                         n_estimators=n_iterations,
                         bootstrap=True)
        bootmodel.fit(X_train, y_train)
        for i, m in tqdm(zip(range(n_iterations), bootmodel.estimators_), total=n_iterations):
            y_pred = m.predict(X_test).reshape(-1)
            acc_stats[i] = metrics.accuracy_score(y_test, y_pred)
            if not get_acc:
                confusion_matrix = metrics.confusion_matrix(y_test, y_pred, labels=labels)
                fp = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
                fn = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
                tp = np.diag(confusion_matrix)
                tn = confusion_matrix.sum() - (fp + fn + tp)
                tpr_stats[:, i] = tp/(tp+fn)
                fpr_stats[:, i] = fp/(fp+tn)

        acc_stats = np.sort(acc_stats)
        acc_lower = acc_stats[int(n_iterations*(1-alpha))]
        acc_upper = acc_stats[int(n_iterations*alpha)]

        if not get_acc:
            tpr_stats = np.sort(tpr_stats)
            tpr_lower = tpr_stats[:, int(n_iterations*(1-alpha))]
            tpr_upper = tpr_stats[:, int(n_iterations*alpha)]

            fpr_stats = np.sort(fpr_stats)
            fpr_lower = fpr_stats[:, int(n_iterations*(1-alpha))]
            fpr_upper = fpr_stats[:, int(n_iterations*alpha)]
    else:
        tpr_lower = np.zeros(3)
        tpr_upper = np.zeros(3)
        fpr_lower = np.zeros(3)
        fpr_upper = np.zeros(3)
        acc_lower = 0
        acc_upper = 1
    mod.fit(X_train, y_train)
    y_pred = mod.predict(X_test).reshape(-1)
    if not get_acc:
        confusion_matrix = metrics.confusion_matrix(y_test, y_pred, labels=labels)
        fp = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
        fn = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
        tp = np.diag(confusion_matrix)
        tn = confusion_matrix.sum() - (fp + fn + tp)
        tpr = {'tpr': list(tp/(tp+fn)), 'tpr_lower': list(tpr_lower), 'tpr_upper': list(tpr_upper)}
        fpr = {'fpr': list(fp/(fp+tn)), 'fpr_lower': list(fpr_lower), 'fpr_upper': list(fpr_upper)}
        return (metrics.accuracy_score(y_test, y_pred), acc_lower, acc_upper), pd.DataFrame(tpr, index=labels), pd.DataFrame(fpr, index=labels)
    else:
        return [metrics.accuracy_score(y_test, y_pred), acc_lower, acc_upper]

def vif_analysis(cols_not_to_analyze=None):
    data_df = pd.read_csv('final_data.csv')
    data_df["date"] = data_df["year"].map(str) + data_df["period"]
    data_df["date"] = data_df["date"].apply(lambda s: dt.datetime.strptime(s, '%YM%m'))
    data_df = data_df.set_index('date')
    data_df = data_df.drop(['period', 'year'], axis=1)
    df = data_df.loc[(data_df.index >= '2000-01-01') & (data_df.index < '2019-01-01')]
    if cols_not_to_analyze is None:
        df_vif = df
    else:
        df_vif = df[df.columns.difference(cols_not_to_analyze)]
    plt.figure(figsize=(9, 7))
    sns.heatmap(df_vif.corr(), cmap='coolwarm', linewidth=.5)
    df_vif['intercept'] = 1
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(df_vif.values, i) for i in range(df_vif.shape[1])]
    vif["features"] = df_vif.columns
    print(vif.iloc[:-1, :])
    plt.title('Correlation matrix')
    plt.tight_layout()
    plt.show()


def classif_pb_valn():
    done = True
    ns = list(range(1, 9))
    if not done:
        type_of_mods = ['Decision Tree', 'Random Forest', 'LDA']
        res = {}
        for mod in type_of_mods:
            res[mod] = []
        for n in ns:
            print('\nn=%d' % n)
            X_train, y_train, X_test, y_test = get_data(n=n, to_keep_y='fedEffIntBehaviour')
            for mod in type_of_mods:
                res[mod] += [class_mod(mod, X_train, y_train, X_test, y_test, get_acc=True)]
        for k in res.keys():
            res[k] = np.array(res[k])
        with open('data_class.pkl', 'wb') as f:
            pkl.dump(res, f)
    else:
        with open('data_class.pkl', 'rb') as f:
            res = pkl.load(f)
    colors = ['b', 'r', 'g', 'k']
    res['LDA'][5, 2] = res['LDA'][5, 0]
    res['LDA'][7, 2] = res['LDA'][7, 0]
    plt.figure()
    for k, c in zip(res.keys(), colors):
        plt.plot(ns, res[k][:, 0], label=k)
        plt.fill_between(ns, res[k][:, 1], res[k][:, 2], color=c, alpha=.1, label='95% CI ' + k)
    plt.xlabel('n')
    plt.ylabel('Accuracy')
    plt.ylim([0, 100])
    plt.legend()
    plt.show()


def classif_pb(nb_time_step_to_analyze=6):
    X_train, y_train, X_test, y_test = get_data(n=nb_time_step_to_analyze, to_keep_y='fedEffIntBehaviour')
    type_of_mods = ['Random Forest', 'Decision Tree', 'LDA']
    _, ax = plt.subplots(nrows=1, ncols=len(type_of_mods))
    color = ['tomato', 'gold', 'gold', 'gold', 'aqua', 'aqua', 'aqua']
    for i in range(len(type_of_mods)):
        height = []
        upper = []
        lower = []
        xticks_name = []
        t = type_of_mods[i]
        (acc, acc_lower, acc_upper), tpr, fpr = class_mod(t, X_train, y_train, X_test, y_test, get_acc=False, cross_val=True, bootstrap=True)
        height += [100*acc]
        upper += [min(100, 100*acc_upper)]
        lower += [max(0, 100*acc_lower)]
        xticks_name += ['Accuracy']
        for j in tpr.index:
            height += [100*tpr.loc[j, 'tpr']]
            upper += [min(100, 100*tpr.loc[j, 'tpr_upper'])]
            lower += [max(0, 100*tpr.loc[j, 'tpr_lower'])]
            xticks_name += ['TPR ' + str(j).lower()]
        for j in fpr.index:
            height += [100*fpr.loc[j, 'fpr']]
            upper += [min(100, 100*fpr.loc[j, 'fpr_upper'])]
            lower += [max(0, 100*fpr.loc[j, 'fpr_lower'])]
            xticks_name += ['FPR ' + str(j).lower()]
        ax[i].bar(xticks_name, height, color=color)
        err = np.concatenate([np.array(lower).reshape((1, -1)), np.array(upper).reshape((1, -1))], axis=0)
        for e in range(len(err[0])):
            err[0][e] = height[e]-err[0][e]
        for e in range(len(err[1])):
            err[1][e] = err[1][e]-height[e]
        for label, h in zip(list(range(len(xticks_name))), height):
            ax[i].annotate('%.2f%%' % h, (label-0.5, h+0.5))
        ax[i].errorbar(xticks_name, height, yerr=err, fmt='none', markersize=8, capsize=10, ecolor='g', alpha=.4, label='95% CI')
        ax[i].set_ylim([-5, 105])
        ax[i].set_title(t)
        ax[i].set_xticklabels(xticks_name, rotation=45)
        ax[i].legend()
    plt.show()


def reg_pb_plot(nb_time_step_to_analyze=1):
    X_train, y_train, X_test, y_test = get_data(n=nb_time_step_to_analyze)
    type_of_mods = ['Decision Tree', 'Random Forest', 'Linear Regression']
    _, ax = plt.subplots(nrows=1, ncols=len(type_of_mods), sharex=True)
    for i in range(len(type_of_mods)):
        t = type_of_mods[i]
        y_train.plot(ax=ax[i], c='c')
        y_test.plot(ax=ax[i], c='m')
        (preds, preds_lower, preds_upper), (r2, r2_lower, r2_upper), (osr2, osr2_lower, osr2_upper) = reg_mod(t, X_train, y_train, X_test, y_test, ax[i], cross_val=False)
        y_pred = pd.DataFrame(preds, index=y_test.index, columns=['fedInterestPreds'])
        y_pred.plot(ax=ax[i], c='r')
        ax[i].fill_between(y_test.index, preds_lower, preds_upper, color='b', alpha=.1)
        ax[i].set_title('%s\nR2 = %.3f | CI=[%.3f, %.3f]\nOSR2 = %.3f | CI=[%.3f, %.3f]' % (t, r2, r2_lower, r2_upper, osr2, osr2_lower, osr2_upper))
        ax[i].legend(["Training", "Testing", "Prediction", "95% Confidence Interval"])
    plt.show()


def reg_pb_valn():
    done = True
    ns = list(range(1, 9))
    if not done:
        type_of_mods = ['Decision Tree', 'Random Forest', 'Linear Regression']
        res = {}
        for mod in type_of_mods:
            res[mod] = []
        for n in ns:
            print('\nn=%d' % n)
            X_train, y_train, X_test, y_test = get_data(n=n)
            for mod in type_of_mods:
                res[mod] += [reg_mod(mod, X_train, y_train, X_test, y_test, get_osr=True)]
        for k in res.keys():
            res[k] = np.array(res[k])
        with open('data.pkl', 'wb') as f:
            pkl.dump(res, f)
    else:
        with open('data.pkl', 'rb') as f:
            res = pkl.load(f)
    colors = ['b', 'r', 'g', 'k']
    plt.figure(figsize=(8, 6))
    for k, c in zip(res.keys(), colors):
        plt.plot(ns, res[k][:, 0], label=k)
        plt.fill_between(ns, res[k][:, 1], res[k][:, 2], color=c, alpha=.1, label='95% CI ' + k)
    plt.xlabel('n')
    plt.ylabel('OSR2')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()


def baseline_timeseries():
    data_df = pd.read_csv('final_data.csv')
    data_df["date"] = data_df["year"].map(str) + data_df["period"]
    data_df["date"] = data_df["date"].apply(lambda s: dt.datetime.strptime(s, '%YM%m'))
    data_df = data_df.set_index('date')
    data_df = data_df.drop(['period', 'year'], axis=1)
    tr_data_df = data_df.loc[(data_df.index < '2013-01-01')]
    te_data_df = data_df.loc[(data_df.index >= '2013-01-01')]
    X_train = np.array([i for i in range(len(tr_data_df.index))]).reshape(-1, 1)
    y_train = np.array([i for i in tr_data_df['fedEffectiveInterest']])
    X_test = np.array([i for i in range(len(tr_data_df.index), len(tr_data_df.index)+len(te_data_df.index))]).reshape(-1, 1)
    y_test = np.array([i for i in te_data_df['fedEffectiveInterest']])
    lmod = linear_model.LinearRegression()
    lmod.fit(X_train, y_train)
    y_pred = lmod.predict(X_test)
    plt.figure()
    plt.plot(X_train.reshape(-1), y_train, 'c', label='Training set')
    plt.plot(X_test.reshape(-1), y_test, 'r', label='Testing set')
    plt.plot(X_test.reshape(-1), y_pred, 'b', label='Predictions')
    plt.title('Linear Trend Model R2 = %.2f | OSR2=%.2f' % (lmod.score(X_train, y_train), get_osr2(y_test, y_train, y_pred)))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    vif_analysis(['fedEffIntBehaviour', 'date', 'fedEffectiveInterest', 'potus', 'houseOfRep', 'fedChair', 'cpi', 'cpiExclEnergy', 'cpiEnergy'])
    reg_pb_valn()
    classif_pb_valn()
    classif = False
    classif_pb()
    reg_pb_plot()
    