from tqdm import tqdm
import numpy as np
import pandas as pd
from functions import *


if __name__ == '__main__':
    df_train = pd.read_csv("data/no1_train.csv")
    df_validation = pd.read_csv("data/no1_validation.csv")
    # df_test = pd.read_csv("data/no1_test.csv")

    percent = 0.005
    upper_bound = df_train.y.quantile(1 - percent / 2)
    lower_bound = df_train.y.quantile(percent / 2)

    print("Upper bound:", upper_bound)
    print("Lower bound:", lower_bound)

    df_train = clean_data(df_train, upper_bound, lower_bound)
    df_validation = clean_data(df_validation, upper_bound, lower_bound)

    column_indcies = ['hydro', 'micro', 'thermal', 'wind', 'total',
                      'sys_reg', 'flow', 'Minute sine', 'Minute cosine', 'Hour sine',
                      'Hour cosine', 'Day sine', 'Day cosine', 'Month sine', 'Month cosine',
                      'Week sine', 'Week cosine', 'prev_y_288', 'prev_y_1',
                      'y']  ## y should be last column

    lags = [1, 12 * 24]
    df_train = add_all_features(df_train, lags, struc_imbalance=True)
    df_validation = add_all_features(df_validation, lags, struc_imbalance=True)
    # df_test = add_all_features(df_test, lags, struc_imbalance=True)

    ## Selectes only the wanted features
    train_data = df_train[column_indcies]
    validation_data = df_validation[column_indcies]

    generator = WindowGenerator(train_data=train_data[-12*24*60:],
                                validation_data=validation_data,
                                test_data=None,
                                target="y",
                                n_input=12 * 24, n_output=1, shift=1, num_predictions=12 * 2)

    model = build_model(generator.getInputShape(), load_prev_model=False)
    # model = build_model(load_prev_model="model_weights/lstm_512_epochs20_alldata.h5")
    model = fit_and_plot(model, generator.getTrainData(), generator.getValidationData(), epochs=5)
    generator.predict_and_plot(model, data_raw=validation_data, start_positions=[0, 100, 200, 300, 400, 500, 600, 700],
                               save_plot=False)




