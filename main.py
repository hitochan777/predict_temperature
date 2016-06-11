#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error

def main(tuning=False):
    train_data_paths = ['data/Temperature_Train_Feature.tsv', 'data/Precipitation_Train_Feature.tsv', 'data/SunDuration_Train_Feature.tsv']
    TEMPERATURE_TRAIN_TARGET_PATH = 'data/Temperature_Train_Target.dat.tsv'
    TEST_SIZE = 0.2
    RANDOM_STATE = 0
    data_frames = []
    for path in train_data_paths:
        data_frame = pd.read_csv(path, sep='\t')
        data_frame = data_frame.loc[:, ['place%d' % i for i in range(11)]]
        data_frames.append(data_frame)

    y = np.loadtxt(TEMPERATURE_TRAIN_TARGET_PATH)
    
    X = pd.concat(data_frames, axis=1).values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    imp = Imputer(strategy='mean', axis=0)
    alpha = 0.5
    if tuning:
        imp.fit(X_train)
        X_train = imp.transform(X_train)
        X_val = imp.transform(X_val)

        reg = Ridge(alpha=alpha)
        # reg.fit(X_train, y_train, np.ones((X_train.shape[0],))*np.arange(X_train.shape[0])**2)
        reg.fit(X_train, y_train)

        y_val_pred = reg.predict(X_val)
        error = mean_squared_error(y_val, y_val_pred)
        print("MSE:", error)
    else:
        imp.fit(X)
        X = imp.transform(X)
        reg_submit = Ridge(alpha=alpha)
        reg_submit.fit(X, y)
        train_data_paths = ['data/Temperature_Test_Feature.tsv', 'data/Precipitation_Test_Feature.tsv',
                            'data/SunDuration_Test_Feature.tsv']
        data_frames = []
        for path in train_data_paths:
            data_frame = pd.read_csv(path, sep='\t')
            data_frame = data_frame.loc[:, ['place%d' % i for i in range(11)]]
            data_frames.append(data_frame)

        X_test = pd.concat(data_frames, axis=1)
        X_test = imp.transform(X_test)
        y_test_pred = reg_submit.predict(X_test)
        SUBMIT_PATH = 'submission.dat'
        np.savetxt(SUBMIT_PATH, y_test_pred, fmt='%.10f')

if __name__ == "__main__":
    main(tuning=True)
