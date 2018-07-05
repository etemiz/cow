import warnings
import copy
from bayes_opt import BayesianOptimization
import traceback
from pprint import pprint
import xgboost
import numpy as np
import time
import threading
import pandas as pd
from sklearn import svm
from sklearn import neighbors, svm, ensemble, linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA, TruncatedSVD, FactorAnalysis


class Cow():
    def __init__(self):
        self.xs = {}
        self.ys = {}
        self.model = None
        self.thread = None
        self.scalerx = StandardScaler()
        self.scalery = StandardScaler()
        self.suicide = False
        self.idx = 0


    def think(self):
        while True and not self.suicide:
            model_tmp = svm.SVR(epsilon=1)
            df = self.scalerx.fit_transform(pd.DataFrame(self.xs).T.astype(np.float32))
            ser = self.scalery.fit_transform(pd.Series(self.ys).values.astype(np.float32).reshape(-1,1))
            model_tmp.fit(df,ser.reshape(-1))
            self.model = model_tmp
            count_sleep = np.minimum(60,len(self.ys))
            while count_sleep > 0 and not self.suicide:
                time.sleep(1)
                count_sleep -= 1


    def start(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self.think)
            self.thread.start()


    def feed(self, xs, ys):
        self.idx += 1
        self.xs[self.idx] = xs
        self.ys[self.idx] = ys

        self.start()


    def milk(self,xs):
        if self.model is None:
            return None
        else:
            df = pd.DataFrame({self.idx+1:xs}).T.astype(np.float32)
            df= self.scalerx.transform(df)
            prediction = self.model.predict(df)
            prediction = self.scalery.inverse_transform(prediction)
            return prediction


class DecomposingCow():
    def __init__(self,dec_comps):
        self.decomposer = None
        self.dec_comps = dec_comps
        self.xs_not_decomposed = {}
        self.ys_not_decomposed = {}
        self.model = None
        self.thread = None
        self.scalerx = StandardScaler()
        self.scalery = StandardScaler()
        self.suicide = False
        self.idx = 0


    def think(self):
        while True and not self.suicide:
            time.sleep(0.2)
            df = pd.DataFrame(self.xs_not_decomposed).T
            xs = self.scalerx.fit_transform(df.astype(np.float32))
            ser = self.scalery.fit_transform(pd.Series(self.ys_not_decomposed).values.astype(np.float32).reshape(-1,1))

            dec_tmp = FactorAnalysis(self.dec_comps)
            xs = dec_tmp.fit_transform(xs)
            self.decomposer = dec_tmp

            model_tmp = svm.SVR()
            model_tmp.fit(xs,ser.reshape(-1))
            self.model = model_tmp
            count_sleep = np.minimum(60,len(self.ys_not_decomposed))
            while count_sleep > 0 and not self.suicide:
                time.sleep(1)
                count_sleep -= 1


    def start(self):
        # if self.thread is None:
        #     self.thread = threading.Thread(target=self.think)
        #     self.thread.start()
        pass


    def feed(self,xs,ys):
        self.idx += 1
        self.xs_not_decomposed[self.idx] = xs
        self.ys_not_decomposed[self.idx] = ys

        self.start()


    def milk(self,xs):
        if self.model is None:
            return None
        else:
            xs_new = pd.DataFrame({self.idx+1:xs}).T.astype(np.float32)
            xs_new = self.scalerx.transform(xs_new)
            xs_new = self.decomposer.transform(xs_new)
            prediction = self.model.predict(xs_new)
            prediction = self.scalery.inverse_transform(prediction)
            return prediction

    def save(self,fn):
        pd.DataFrame(self.xs_not_decomposed).to_msgpack(fn+'.X.msg')
        pd.Series(self.ys_not_decomposed).to_msgpack(fn+'.Y.msg')

    def load(self,fn):
        self.xs_not_decomposed = pd.read_msgpack(fn+'.X.msg')
        self.xs_not_decomposed = self.xs_not_decomposed.to_dict()
        self.ys_not_decomposed = pd.read_msgpack(fn+'.Y.msg')
        self.idx = self.ys_not_decomposed.index.max()
        self.ys_not_decomposed = self.ys_not_decomposed.to_dict()

        self.start()


class ForgettingCow(DecomposingCow):
    def __init__(self, dec_comps, mem):
        self.mem = mem
        self.features_not_dropped = []
        super(ForgettingCow,self).__init__(dec_comps)

    def think(self):
        while True and not self.suicide:
            if len(self.xs_not_decomposed) >= 1:
                time.sleep(0.2)
                try:
                    df = pd.DataFrame(self.xs_not_decomposed).T.iloc[-self.mem:]
                    df = df.dropna(axis=1)
                    self.features_not_dropped = list(df.columns)
                    scaler_temp = StandardScaler()
                    xs = scaler_temp.fit_transform(df.astype(np.float32))
                    self.scalerx = scaler_temp
                    scaler_temp = StandardScaler()
                    ser = scaler_temp.fit_transform(
                        pd.Series(self.ys_not_decomposed).values[-self.mem:].astype(np.float32).reshape(-1, 1))
                    self.scalery = scaler_temp
                except ValueError:
                    print(traceback.format_exc())
                    pprint(pd.DataFrame(self.xs_not_decomposed).T.iloc[-self.mem:])
                    pprint(pd.Series(self.ys_not_decomposed).values[-self.mem:])
                    time.sleep(5)
                    continue

                dec_tmp = FactorAnalysis(self.dec_comps)
                xs = dec_tmp.fit_transform(xs)
                self.decomposer = dec_tmp

                model_tmp = svm.SVR()
                model_tmp.fit(xs, ser.reshape(-1))
                self.model = model_tmp
                count_sleep = np.minimum(60, len(self.ys_not_decomposed))
                while count_sleep > 0 and not self.suicide:
                    time.sleep(1)
                    count_sleep -= 1
            else:
                time.sleep(5)


    def milk(self,xs):
        if self.model is None:
            return None
        else:
            xs_new = {k:xs[k] for k in self.features_not_dropped if k in xs}
            xs_new = pd.DataFrame({self.idx+1:xs_new}).T.astype(np.float32)
            xs_new = self.scalerx.transform(xs_new)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                xs_new = self.decomposer.transform(xs_new)
            prediction = self.model.predict(xs_new)
            prediction = self.scalery.inverse_transform(prediction)
            return prediction


class OptimizingCow(ForgettingCow):
    def __init__(self, dec_comps, mem):
        self.best_score = -np.inf
        self.best_pars = {}
        self.model_func = None
        self.best_model_func = None
        super(OptimizingCow,self).__init__(dec_comps, mem)

    def geometric_error(self, y_true, y_pred):
        return np.prod(np.maximum(y_pred, y_true) / np.minimum(y_pred, y_true))

    def score(self, **pars):
        xs_all = copy.deepcopy(self.xs_not_decomposed)
        ys_all = copy.deepcopy(self.ys_not_decomposed)
        steps = 7
        size = np.minimum(self.mem // 5, len(xs_all))
        validation_size = size // steps
        validation_start = -validation_size
        validation_end = len(xs_all)
        train_start = validation_start - size

        df = pd.DataFrame(xs_all).T
        df = df.dropna(axis=1)
        non_na_cols = list(df.columns)

        cv_score = -1.0
        while validation_start > -size:
            df = pd.DataFrame(xs_all).T.iloc[train_start:validation_start]
            df = df[non_na_cols]
            scaler_temp_xs = StandardScaler()
            xs = scaler_temp_xs.fit_transform(df.astype(np.float32))
            scaler_temp_ys = StandardScaler()
            ys = scaler_temp_ys.fit_transform(
                pd.Series(ys_all).values[train_start:validation_start].astype(np.float32).reshape(-1, 1))

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                dec_tmp = FactorAnalysis(self.dec_comps)
                xs = dec_tmp.fit_transform(xs)

            if self.model_func == svm.SVR:
                pars['gamma'] = pars['gamma'] / pars['C']
            if self.model_func == ensemble.AdaBoostRegressor:
                pars['gamma'] = pars['gamma'] / pars['C']
            elif self.model_func == xgboost.XGBRegressor:
                pars['n_estimators'] = int(pars['n_estimators'])
                pars['max_depth'] = int(pars['max_depth'])
                pars['min_child_weight'] = int(pars['min_child_weight'])
                pars['n_jobs'] = 2
            elif self.model_func == neighbors.KNeighborsRegressor:
                pars['n_neighbors'] = int(pars['n_neighbors'])
                pars['weights'] = 'distance'
                pars['algorithm'] = 'ball_tree'
                pars['p'] = int(pars['p'])

            if self.model_func == neighbors.KNeighborsRegressor and pars['n_neighbors'] > len(xs):
                cv_score *= 1.1
            else:
                if self.model_func == ensemble.AdaBoostRegressor:
                    model_tmp = ensemble.AdaBoostRegressor(svm.SVR(**pars),n_estimators=5)
                else:
                    model_tmp = self.model_func(**pars)
                model_tmp.fit(xs, ys.reshape(-1))

                df_test = pd.DataFrame(xs_all).T.iloc[validation_start:validation_end]
                df_test = df_test[non_na_cols]
                xs_test = scaler_temp_xs.transform(df_test.astype(np.float32))

                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    xs_test = dec_tmp.transform(xs_test)
                y_hat = model_tmp.predict(xs_test)
                yhat_it = scaler_temp_ys.inverse_transform(y_hat.reshape(-1,1)).reshape(-1)

                ys_test = pd.Series(ys_all).values[validation_start:validation_end].astype(np.float32).reshape(-1)

                cv_score *= float(self.geometric_error(yhat_it, ys_test) ** (1/validation_size))

            validation_start -= validation_size
            validation_end -= validation_size
        return cv_score


    def think(self):
        while True and not self.suicide:
            if len(self.xs_not_decomposed) >= 10:
                time.sleep(0.2)

                if self.model is None:
                    self.model_func = svm.SVR
                else:
                    self.model_func = np.random.choice([svm.SVR,xgboost.XGBRegressor,neighbors.KNeighborsRegressor,
                                                        ensemble.AdaBoostRegressor])

                if self.model_func == svm.SVR:
                    bo = BayesianOptimization(self.score,
                                              {'C': (10 ** -3, 10 ** 4),
                                               'gamma': (10 ** -3, 10 ** 3),
                                               'epsilon': (0.01, 3),
                                               },verbose=0)
                elif self.model_func == xgboost.XGBRegressor:
                    bo = BayesianOptimization(self.score,
                                              {'n_estimators': (2, 20),
                                               'learning_rate': (0.1, 1),
                                               'max_depth': (1, 3),
                                               'min_child_weight': (1.0, 3.0),
                                               'gamma': (0, 0.2),
                                               'subsample': (0.4, 1),
                                               'colsample_bytree': (0.4, 1),
                                               'colsample_bylevel': (0.4, 1),
                                               'reg_alpha': (0.0, 1),
                                               'reg_lambda': (0.0, 1),
                                               },verbose=0)
                elif self.model_func == neighbors.KNeighborsRegressor:
                    bo = BayesianOptimization(self.score,
                                              {'n_neighbors': (5, 50),
                                               'leaf_size': (1, 10),
                                               'p': (1, 20),
                                               },verbose=0)
                elif self.model_func == ensemble.AdaBoostRegressor:
                    bo = BayesianOptimization(self.score,
                                              {'C': (10 ** -3, 10 ** 4),
                                               'gamma': (10 ** -3, 10 ** 3),
                                               'epsilon': (0.01, 3),
                                               }, verbose=0)
                try:
                    if self.model is None:
                        bo.maximize(n_iter=1)
                    else:
                        bo.maximize(n_iter=2)

                    df = pd.DataFrame(self.xs_not_decomposed).T.iloc[-self.mem:]
                    df = df.dropna(axis=1)
                    scaler_temp_xs = StandardScaler()
                    xs = scaler_temp_xs.fit_transform(df.astype(np.float32))
                    scaler_temp_ys = StandardScaler()
                    ys = scaler_temp_ys.fit_transform(
                        pd.Series(self.ys_not_decomposed).values[-self.mem:].astype(np.float32).reshape(-1, 1))

                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore')
                        dec_tmp = FactorAnalysis(self.dec_comps)
                        xs = dec_tmp.fit_transform(xs)

                    self.best_score *= 1.01
                    if bo.res['max']['max_val'] > self.best_score:
                        self.best_score = bo.res['max']['max_val']
                        print('OptimizingCow new best_score',self.best_score,self.model_func,list(df.columns))

                        pars = bo.res['max']['max_params']
                        if self.model_func == svm.SVR:
                            pars['gamma'] = pars['gamma'] / pars['C']
                        if self.model_func == ensemble.AdaBoostRegressor:
                            pars['gamma'] = pars['gamma'] / pars['C']
                        elif self.model_func == xgboost.XGBRegressor:
                            pars['n_estimators'] = int(pars['n_estimators'])
                            pars['max_depth'] = int(pars['max_depth'])
                            pars['min_child_weight'] = int(pars['min_child_weight'])
                            pars['n_jobs'] = 2
                        elif self.model_func == neighbors.KNeighborsRegressor:
                            pars['n_neighbors'] = int(pars['n_neighbors'])
                            pars['weights'] = 'distance'
                            pars['algorithm'] = 'ball_tree'
                            pars['p'] = int(pars['p'])
                        self.best_pars = copy.deepcopy(pars)
                        self.best_model_func = self.model_func
                    else:
                        print('OptimizingCow refitting')

                    if self.best_model_func == ensemble.AdaBoostRegressor:
                        model_tmp = ensemble.AdaBoostRegressor(svm.SVR(**self.best_pars),n_estimators=5)
                    else:
                        model_tmp = self.best_model_func(**self.best_pars)
                    model_tmp.fit(xs, ys.reshape(-1))

                    self.features_not_dropped = list(df.columns)
                    self.scalerx = scaler_temp_xs
                    self.scalery = scaler_temp_ys
                    self.decomposer = dec_tmp
                    self.model = model_tmp
                except:
                    print(traceback.format_exc())

                count_sleep = 10
                while count_sleep > 0 and not self.suicide:
                    time.sleep(1)
                    count_sleep -= 1
            else:
                time.sleep(5)

    def save(self,fn):
        pd.Series(self.best_pars).to_msgpack(fn+'.pars.msg')
        super(OptimizingCow,self).save(fn)

    def load(self,fn):
        self.best_pars = pd.read_msgpack(fn+'.pars.msg').to_dict()
        super(OptimizingCow,self).load(fn)


# cow = OptimizingCow(1,100)
# cow.feed({'a':3, 'b':5},4)
# cow.feed({'a':4, 'b':1},2.5)
# cow.feed({'a':1, 'b':1},1)
# cow.feed({'a':9, 'b':1},5)
# cow.feed({'a':2, 'b':7},4.5)
# cow.feed({'a':2, 'b':5},3.5)
# cow.feed({'a':2, 'b':6},4)
# cow.feed({'a':6, 'b':3},4.5)
# cow.feed({'a':6, 'b':4},5)
# cow.feed({'a':2, 'b':2},2)
# time.sleep(100)
# print(cow.milk({'a':0,'b':6}))
# cow.suicide = True