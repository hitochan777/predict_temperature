#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.linear_model import Ridge, Lasso, BayesianRidge
from sklearn.metrics import mean_squared_error

def make_data_frames(is_training=True, base="data/", shift_max_width = 0):
    runType = "train" if is_training else "test"
    years = [0,1,2,3,4] if is_training else [5,6,7,8,9]

    data_paths = {
        "train": "Train_Feature.tsv",
        "test": "Test_Feature.tsv",
        "loc": "Location.tsv"
    }
    # df.loc[:,"year"].drop_duplicates().tolist()
    data_frames = []
    data_frame = pd.read_csv(base+data_paths[runType], sep='\t')
    for year in years:
        df = data_frame[data_frame["year"] == year]
        df = df.loc[:, ['%s_place%d' % (s,i) for i in range(11) for s in ["prep","temp","sun"]]]
        for width in range(shift_max_width):
            df = pd.concat([df, df.shift(width+1)], axis=1)
       
        data_frames.append(df) 

    data_frames = pd.concat(data_frames, axis=0)
    return data_frames 

def main(tuning=False):
    TEMPERATURE_TRAIN_TARGET_PATH = 'data/Temperature_Train_Target.dat.tsv'
    TEST_SIZE = 0.2
    RANDOM_STATE = 0
    WIDTH = 3
    y = np.loadtxt(TEMPERATURE_TRAIN_TARGET_PATH)
    X = make_data_frames(is_training=True, shift_max_width=WIDTH).values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    imp = Imputer(strategy='mean', axis=0)
    alpha = 5
    if tuning:
        imp.fit(X_train)
        X_train = imp.transform(X_train)
        X_val = imp.transform(X_val)

        reg = Ridge(alpha=alpha)
        # reg = BayesianRidge()
        # reg.fit(X_train, y_train, np.ones((X_train.shape[0],))*np.arange(X_train.shape[0])**2)
        reg.fit(X_train, y_train)

        y_val_pred = reg.predict(X_val)
        error = mean_squared_error(y_val, y_val_pred)
        print("MSE:", error)
    else:
        imp.fit(X)
        X = imp.transform(X)
        reg_submit = Ridge(alpha=alpha)
        # reg_submit = BayesianRidge()
        reg_submit.fit(X, y)
        X_test = make_data_frames(is_training=False, shift_max_width=WIDTH).values
        X_test = imp.transform(X_test)
        y_test_pred = reg_submit.predict(X_test)
        SUBMIT_PATH = 'submission.dat'
        np.savetxt(SUBMIT_PATH, y_test_pred, fmt='%.10f')

if __name__ == "__main__":
    main(tuning=True)
