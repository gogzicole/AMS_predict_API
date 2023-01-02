# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


def model_train(data, prepped_data, month):

    df = data.sort_values('month')
    #replace wrongly imputted depot data with the mode
    df['depot'] = np.where(df.depot == 'FUNT','IWO',df.depot)
    #drop region and year data as they have zero variance (a single values for all observations)
    


    u = month
    if u == 4:
        # select data for july, august, september, october and November for training
        df_slice = df[(df['month']==1) | (df['month']==2) | (df['month']==3)].reset_index(drop=True)
    elif u == 5:
        # select data for july, august, september, october and November for training
        df_slice = df[(df['month']==2) | (df['month']==3) | (df['month']==4)].reset_index(drop=True)
    elif u == 6:
        # select data for july, august, september, october and November for training
        df_slice = df[(df['month']==3) | (df['month']==4) | (df['month']==5)].reset_index(drop=True)
    elif u == 7:
        # select data for july, august, september, october and November for training
        df_slice = df[(df['month']==4) | (df['month']==5) | (df['month']==6)].reset_index(drop=True)
    elif u == 8:
        # select data for july, august, september, october and November for training
        df_slice = df[(df['month']==5)| (df['month']==6) | (df['month']==7)].reset_index(drop=True)
    elif u == 9:
        # select data for july, august, september, october and November for training
        df_slice = df[(df['month']==6) | (df['month']==7) | (df['month']==8)].reset_index(drop=True)
    elif u == 10:
        # select data for july, august, september, october and November for training
        df_slice = df[(df['month']==7) | (df['month']==8) | (df['month']==9)].reset_index(drop=True)
    elif u == 11:
        # select data for july, august, september, october and November for training
        df_slice = df[(df['month']==8) | (df['month']==9) | (df['month']==10)].reset_index(drop=True)
    elif u == 12:
        # select data for july, august, september, october and November for training
        df_slice = df[(df['month']==9) | (df['month']==10) | (df['month']==11)].reset_index(drop=True)
    else:
        df_slice = df[df['month'] != u]


    x = df_slice.drop(columns = ['ams'])
    y = df_slice['ams']

    pred_data = pd.DataFrame.from_dict([prepped_data])
    new_x = pd.concat([pred_data,x]).reset_index(drop=True)

    #Encode labels for depot and item_no
    
    le_dep = LabelEncoder()
    le_item = LabelEncoder()
    new_x['depot'] = le_dep.fit_transform(new_x['depot'])
    new_x['item_no'] = le_item.fit_transform(new_x['item_no'])

    pred_x = new_x.iloc[0,:]
    X = new_x.iloc[1:,:]

    #perform train test split for crossvalidation
    x_train,x_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state=42)

    # instantiate several models to test which one fits the data
    lr = RandomForestRegressor(random_state = 43)
    model_lr = lr.fit(x_train, y_train)

    #evaluate random_forest
    pred_lr = model_lr.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred_lr))
    r2score = r2_score(y_test, pred_lr)

    result_dict = {"Random Forest RMSE: NT": rmse,
                    "Accuracy": r2score}

    lr1 = RandomForestRegressor()
    lr1.fit(X, y)
    result = lr1.predict([pred_x])

    return result_dict, result[0].tolist()
