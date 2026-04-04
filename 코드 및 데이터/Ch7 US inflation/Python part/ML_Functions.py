import pandas as pd
import numpy as np
import xgboost as xgb
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import date
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import ElasticNetCV, LassoLarsIC, RidgeCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import warnings
from statsmodels.stats.sandwich_covariance import cov_hac, cov_white_simple
from scipy.stats import norm
import time
from itertools import combinations
from scipy.stats import t
from statsmodels.api import OLS, add_constant
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=UserWarning)

scaler = MinMaxScaler()
scaler_demean = StandardScaler(with_mean=True, with_std=False)
scaler_std = StandardScaler(with_mean=True, with_std=True)

#### Normalize and Embed function
def embed(x, dimension=1):
    n, d = x.shape
    if dimension < 1 or dimension > n:
        raise ValueError("Invalid embedding dimension")
    return np.hstack([x[i:n - dimension + i + 1, :] for i in reversed(range(dimension))])

def normalize_columns(df):
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

def denormalize(val, minval, maxval):
    return val * (maxval - minval) + minval

def normalize_columns_std(df):
    return pd.DataFrame(scaler_std.fit_transform(df), columns=df.columns)

def denormalize_std(val, meanval, stdval):
    return val * stdval + meanval

#### Functions
#### Random Walk
def rw_rolling_window(Y, npred, horizon, indice=0):
    save_pred = []

    for i in range(npred, horizon - 1, -1):
        Y_window = Y.iloc[(npred - i):(len(Y) - i), :]
        pred = Y_window.iloc[-1, 0]
        save_pred.append(pred)
        print(f"rw_iter {npred - i} horizon {horizon}", end='\r', flush=True)

    real = Y.iloc[:, indice].values
    pred_series = np.array([np.nan] * (len(real) - len(save_pred)) + save_pred)

    plt.plot(real, label="Actual")
    plt.plot(pred_series, color="red", label="Forecast")
    plt.legend()
    plt.show()

    rmse = np.sqrt(mean_squared_error(real[-len(save_pred):], save_pred))
    mae = mean_absolute_error(real[-len(save_pred):], save_pred)
    errors = {"rmse": rmse, "mae": mae}

    return {"pred": save_pred, "errors": errors}

#### AR
def run_ar(Y, horizon, lag, type="fixed"):
    Y2 = Y.iloc[:, 0].copy()
    lags = lag + horizon - 1
    aux = pd.concat([Y2.shift(i) for i in range(1, lags + 1)], axis=1)
    aux = aux.dropna().reset_index(drop=True)
    Y2_clean = Y2.dropna().reset_index(drop=True)

    Xin = aux[:-horizon].to_numpy()
    Xout = aux.iloc[-horizon].to_numpy()
    y = Y2_clean[lags:-horizon].to_numpy()
    X = Xin
    X_out = Xout.reshape(1, -1)

    if type == "fixed":
        model = LinearRegression().fit(X, y)
        coef = np.concatenate(([model.intercept_], model.coef_))
    elif type == "bic":
        best_bic = np.inf
        best_model = None
        ar_coef = None
        for i in range(1, X.shape[1] + 1):
            m = LinearRegression().fit(X[:, :i], y)
            pred = m.predict(X[:, :i])
            residual = y - pred
            n = len(y)
            k = i + 1
            sse = np.sum(residual ** 2)
            bic = n * np.log(sse / n) + k * np.log(n)
            if bic < best_bic:
                best_bic = bic
                best_model = m
                ar_coef = np.concatenate(([m.intercept_], m.coef_[:i]))
        coef = np.zeros(X.shape[1] + 1)
        coef[:len(ar_coef)] = ar_coef
        model = best_model
    else:
        raise ValueError("type must be either 'fixed' or 'bic'")

    pred = model.predict(X_out)
    return model, pred[0]

def ar_rolling_window(Y, npred, lag = 1, horizon=1, type="fixed"):
    save_pred = []
    for i in range(npred, horizon - 1, -1):
        Y_window = Y.iloc[(npred - i):(len(Y) - i), :]
        _, pred = run_ar(Y_window, horizon, lag, type)
        save_pred.append(pred)
        print(f"ar_iter {npred - i} horizon {horizon}", end='\r', flush=True)

    real = Y.values[:, 0]
    pred_series = np.concatenate([np.full(len(real) - len(save_pred), np.nan), save_pred])

    rmse = np.sqrt(mean_squared_error(real[-len(save_pred):], save_pred))
    mae = mean_absolute_error(real[-len(save_pred):], save_pred)

    # Plot
    plt.plot(real, label="Actual")
    padding = [np.nan] * (len(real) - len(pred_series))
    plt.plot(padding + list(pred_series), label="Forecast", color="red")
    plt.legend()
    plt.title("Rolling Window Forecast")
    plt.show()

    return {
        "pred": save_pred,
        "errors": {"rmse": rmse, "mae": mae}}

# Shrinkage Method: LASSO, Ridge, Elastic_net CV
def run_shrinkage_cv(Y, horizon, alpha=1.0):
    Y2 = Y.copy()
    pca = PCA(n_components=4)
    standard_Y2 = scaler_demean.fit_transform(Y2)
    scores = pca.fit_transform(standard_Y2)
    Y2 = pd.concat([Y2, pd.DataFrame(scores, index=Y2.index)], axis=1)
    Y2.columns = Y2.columns.astype(str)
    Y3 = normalize_columns_std(Y2).to_numpy()

    aux = embed(Y3, 4)

    Xin = aux[:-horizon]
    Xout = aux[-1]

    y = Y2.iloc[-Xin.shape[0]:, 0]
    X = Xin
    X_out = Xout

    if alpha==0.0: 
        n_lambdas = 100
        lambda_grid = np.exp(np.linspace(np.log(100.0), np.log(0.00001), n_lambdas))
        model = RidgeCV(alphas=lambda_grid, fit_intercept=True).fit(X, y)
    else :
        model = ElasticNetCV(l1_ratio=alpha, n_alphas=100, fit_intercept=True).fit(X, y)

    pred = model.predict(X_out.reshape(1, -1))
    return model, pred
    
def shrinkage_cv_rolling_window(Y, npred, horizon, alpha=1.0):
    save_model = []
    save_pred = []

    for i in range(npred, horizon - 1, -1):
        Y_window = Y.iloc[(npred - i):(len(Y) - i), :]
        model, pred = run_shrinkage_cv(Y_window, horizon, alpha)
        save_model.append(model)
        save_pred.append(pred[0])
        print(f"shrink_iter {npred - i} horizon {horizon}", end='\r', flush=True)

    real = Y.values[:, 0]
    pred_series = np.concatenate([np.full(len(real) - len(save_pred), np.nan), save_pred])

    rmse = np.sqrt(mean_squared_error(real[-len(save_pred):], save_pred))
    mae = mean_absolute_error(real[-len(save_pred):], save_pred)

    # Plot
    plt.plot(real, label="Actual")
    padding = [np.nan] * (len(real) - len(pred_series))
    plt.plot(padding + list(pred_series), label="Forecast", color="red")
    plt.legend()
    plt.title("Rolling Window Forecast")
    plt.show()

    return {
        "pred": save_pred,
        "errors": {"rmse": rmse, "mae": mae},
        "model" : save_model}

### CSR
def csr(x: pd.DataFrame, y: pd.Series, K=None, k=4, fixed_controls=None):
    if not isinstance(x, pd.DataFrame):
        x = pd.DataFrame(x)
    if x.shape[1] < 2:
        raise ValueError("Only one variable in x, csr is senseless in this case.")
    
    n, p = x.shape
    if K is None:
        K = min(20, p)

    if fixed_controls is not None:
        if isinstance(fixed_controls[0], str):
            fixed_idx = [x.columns.get_loc(fc) for fc in fixed_controls]
        else:
            fixed_idx = fixed_controls
        w = x.iloc[:, fixed_idx]
        nonw = list(set(range(p)) - set(fixed_idx))
    else:
        w = None
        fixed_idx = []
        nonw = list(range(p))

    stats = []
    for i in nonw:
        xi = x.iloc[:, i]
        X_design = add_constant(pd.concat([w, xi], axis=1)) if w is not None else add_constant(xi)
        model = OLS(y, X_design).fit()
        tval = np.abs(model.tvalues[-1])
        stats.append((tval, i))
    
    top_k_idx = [i for _, i in sorted(stats, key=lambda x: -x[0])[:K]]

    subset_coefs = []
    subset_consts = []
    x_columns = x.columns.tolist()

    for subset in combinations(top_k_idx, k):
        cols = list(subset)
        if fixed_controls is not None:
            cols = fixed_idx + cols
        X_subset = x.iloc[:, cols]
        X_design = add_constant(X_subset)
        model = OLS(y, X_design).fit()
        coef_full = np.zeros(p)
        param_dict = dict(zip(X_subset.columns, model.params[1:]))
        for col in X_subset.columns:
            coef_full[x.columns.get_loc(col)] = param_dict[col]
        subset_coefs.append(coef_full)
        subset_consts.append(model.params[0])

    X_full = add_constant(x)
    subset_coefs_arr = np.array(subset_coefs)
    subset_consts_arr = np.array(subset_consts).reshape(-1, 1)
    coef_matrix = np.hstack([subset_consts_arr, subset_coefs_arr])

    fitted_values_all = X_full @ coef_matrix.T
    fitted_values = np.mean(fitted_values_all, axis=1)
    residuals = y - fitted_values

    result = {
        "coefficients": pd.DataFrame(coef_matrix, columns=["intercept"] + x_columns),
        "fitted_values": fitted_values,
        "residuals": residuals
    }
    return result

def runcsr(Y, horizon):
    Y2 = Y.copy()
    pca = PCA(n_components=4)
    standard_Y2 = scaler_demean.fit_transform(Y2)
    scores = pca.fit_transform(standard_Y2)
    Y2 = pd.concat([Y2, pd.DataFrame(scores, index=Y2.index)], axis=1).to_numpy()

    aux = embed(Y2, 4)

    Xin = aux[:-horizon]
    Xout = aux[-1]

    y = Y2[-Xin.shape[0]:, 0]
    X = Xin
    X_out = Xout

    X_df = pd.DataFrame(X, columns=[f'lag{i}' for i in range(X.shape[1])])
    y_series = pd.Series(y)

    result = csr(X_df, y_series, K=min(20, X.shape[1]), k=4)

    avg_coef = result['coefficients'].mean()
    intercept = avg_coef['intercept']
    coefs = avg_coef.drop('intercept').values

    pred = intercept + np.dot(X_out, coefs)
    return pred

def csr_rolling_window(Y, npred, horizon=1):
    save_pred = []
    for i in range(npred, horizon - 1, -1):
        Y_window = Y.iloc[(npred - i):(len(Y) - i), :]
        pred = runcsr(Y_window, horizon)
        save_pred.append(pred)
        print(f"csr_iter {npred - i} horizon {horizon}", end='\r', flush=True)

    real = Y.values[:, 0]
    pred_series = np.concatenate([np.full(len(real) - len(save_pred), np.nan), save_pred])

    rmse = np.sqrt(mean_squared_error(real[-len(save_pred):], save_pred))
    mae = mean_absolute_error(real[-len(save_pred):], save_pred)

    # Plot
    plt.plot(real, label="Actual")
    padding = [np.nan] * (len(real) - len(pred_series))
    plt.plot(padding + list(pred_series), label="Forecast", color="red")
    plt.legend()
    plt.title("Rolling Window Forecast")
    plt.show()

    return {
        "pred": save_pred,
        "errors": {"rmse": rmse, "mae": mae}}

### Target factor
def run_fact(Y, horizon):
    Y2 = Y.copy()
    n_components = min(4, Y2.shape[0], Y2.shape[1])
    if n_components >= 1:
        pca = PCA(n_components=n_components)
        standard_Y2 = scaler_demean.fit_transform(Y2)
        scores = pca.fit_transform(standard_Y2)
        Y2 = pd.concat([Y2.reset_index(drop=True), pd.DataFrame(scores)], axis=1)

    Y_matrix = Y2.to_numpy().astype(float)
    aux = embed(Y_matrix, 4)
    
    Xin = aux[:-horizon]
    Xout = aux[-1]
    y = Y2.iloc[3 + horizon:, 0].to_numpy().astype(float)
    X = Xin
    X_out = Xout

    best_bic = np.inf
    best_model = None
    best_coef = None

    for i in range(5, min(21, X.shape[1] + 1), 5):
        model = LinearRegression().fit(X[:, :i], y)
        pred = model.predict(X[:, :i])
        residual = y - pred
        n = len(y)
        k = i + 1
        sse = np.sum(residual ** 2)
        bic = n * np.log(sse / n) + k * np.log(n)
        if bic < best_bic:
            best_bic = bic
            best_model = model
            best_coef = np.concatenate(([model.intercept_], model.coef_))

    coef = np.zeros(X.shape[1] + 1)
    coef[:len(best_coef)] = best_coef
    prediction = np.dot(np.concatenate(([1], X_out)), coef)

    return best_model, float(prediction)

def baggit(mat, pre_testing="individual", fixed_controls=None, t_stat=1.96, ngroups=10):
    y = mat.iloc[:, 0].values.astype(float)
    X = mat.iloc[:, 1:].values.astype(float)

    if pre_testing == "individual":
        store_t = np.zeros(X.shape[1])
        if fixed_controls:
            for i in fixed_controls:
                store_t[i] = 999.0
            
            remaining = sorted(list(set(range(X.shape[1])) - set(fixed_controls)))
            for i in remaining:
                idx = [i] + fixed_controls
                model = LinearRegression().fit(X[:, idx], y)
                store_t[i] = model.coef_[0]
        else:
            for i in range(X.shape[1]):
                model = LinearRegression().fit(X[:, [i]], y)
                store_t[i] = model.coef_[0]
        
        selected = np.where(np.abs(store_t) > t_stat)[0]
    else:
        selected = np.arange(X.shape[1])

    return selected

def run_tfact(Y, horizon):
    y = Y.iloc[:, 0]
    X = Y.iloc[:, 1:]
    
    y_lags = []
    for i in range(5):
        y_lags.append(y.shift(i))
    
    y_mat = pd.concat(y_lags, axis=1).dropna()
    x_mat = X.iloc[4:, :].reset_index(drop=True)
    mat = pd.concat([y_mat.reset_index(drop=True), x_mat], axis=1)
    
    selected_indices = baggit(mat, pre_testing="individual", fixed_controls=[0, 1, 2, 3])
    
    selected_x_indices = [i - 5 for i in selected_indices if i >= 5]
    
    selection_mask = [False] * Y.shape[1]
    selection_mask[0] = True 
    for idx in selected_x_indices:
        if idx + 1 < len(selection_mask):
            selection_mask[idx + 1] = True
            
    Y2 = Y.iloc[:, selection_mask]
    model, pred = run_fact(Y2, horizon=horizon)
    
    return model, pred

def tfact_rolling_window(Y, npred, horizon=1):
    save_pred = []
    for i in range(npred, horizon - 1, -1):
        Y_window = Y.iloc[(npred - i):(len(Y) - i), :]
        _, pred = run_tfact(Y_window, horizon)
        save_pred.append(pred)
        print(f"tfact_iter {npred - i} horizon {horizon}", end='\r', flush=True)

    real = Y.iloc[:, 0].to_numpy().astype(float)
    save_pred = np.array(save_pred).astype(float)
    pred_series = np.concatenate([np.full(len(real) - len(save_pred), np.nan), save_pred])

    rmse = np.sqrt(mean_squared_error(real[-len(save_pred):], save_pred))
    mae = mean_absolute_error(real[-len(save_pred):], save_pred)

    plt.plot(real, label="Actual")
    padding = [np.nan] * (len(real) - len(pred_series))
    plt.plot(padding + list(pred_series), label="Forecast", color="red")
    plt.legend()
    plt.title("Rolling Window Forecast")
    plt.show()

    return {
        "pred": save_pred,
        "errors": {"rmse": rmse, "mae": mae}
    }

# Random forest
def run_rf(Y, horizon, n_estimators=100):
    Y2 = Y.copy()
    pca = PCA(n_components=4)
    standard_Y2 = scaler_demean.fit_transform(Y2)
    scores = pca.fit_transform(standard_Y2)
    Y2 = pd.concat([Y2, pd.DataFrame(scores, index=Y2.index)], axis=1).to_numpy()

    aux = embed(Y2, 4)

    Xin = aux[:-horizon]
    Xout = aux[-1]

    y = Y2[-Xin.shape[0]:, 0]
    X = Xin
    X_out = Xout

    model = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=42)
    model.fit(X, y)

    pred = model.predict(X_out.reshape(1, -1))
    return {"model": model, "pred": pred}

def rf_rolling_window(Y, npred, horizon=1, n_estimators=100):
    save_pred = []
    save_importance = []

    for i in range(npred, horizon - 1, -1):
        Y_window = Y.iloc[(npred - i):(len(Y) - i), :].copy()
        result = run_rf(Y_window, horizon, n_estimators)
        save_pred.append(result["pred"][0])
        save_importance.append(result["model"].feature_importances_)
        print(f"rf_iter {npred - i} horizon {horizon}", end='\r', flush=True)

    real = Y.values[:, 0]
    pred_series = np.concatenate([np.full(len(real) - len(save_pred), np.nan), save_pred])

    rmse = np.sqrt(mean_squared_error(real[-len(save_pred):], save_pred))
    mae = mean_absolute_error(real[-len(save_pred):], save_pred)

    # Plot
    plt.plot(real, label="Actual")
    padding = [np.nan] * (len(real) - len(pred_series))
    plt.plot(padding + list(pred_series), label="Forecast", color="red")
    plt.legend()
    plt.title("Rolling Window Forecast")
    plt.show()

    return {
        "pred": save_pred,
        "errors": {"rmse": rmse, "mae": mae},
        "save_importance": save_importance}

# XGBoost
def run_xgb(Y, horizon):
    Y2 = Y.copy()
    pca = PCA(n_components=4)
    standard_Y2 = scaler_demean.fit_transform(Y2)
    scores = pca.fit_transform(standard_Y2)
    Y2 = pd.concat([Y2, pd.DataFrame(scores, index=Y2.index)], axis=1).to_numpy()

    aux = embed(Y2, 4)

    Xin = aux[:-horizon]
    Xout = aux[-1]

    y = Y2[-Xin.shape[0]:, 0]
    X = Xin
    X_out = Xout

    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        colsample_bylevel=2/3,
        subsample=1,
        max_depth=4,
        min_child_weight=len(X)/200,
        n_jobs=-1,
        verbosity=0,
        random_state=42
    )

    model.fit(X, y)
    pred = model.predict(X_out.reshape(1, -1))
    return model, pred

def xgb_rolling_window(y, npred, horizon):
    save_pred = []

    for i in range(npred, horizon - 1, -1):
        y_window = y.iloc[(npred - i):(len(y) - i), :]
        _, pred = run_xgb(y_window, horizon)
        save_pred.append(pred[0])
        print(f"xgb_iter {npred - i} horizon {horizon}", end='\r', flush=True)

    real = y.values[:, 0]
    pred_series = np.concatenate([np.full(len(real) - len(save_pred), np.nan), save_pred])

    rmse = np.sqrt(mean_squared_error(real[-len(save_pred):], save_pred))
    mae = mean_absolute_error(real[-len(save_pred):], save_pred)

    # Plot
    plt.plot(real, label="Actual")
    padding = [np.nan] * (len(real) - len(pred_series))
    plt.plot(padding + list(pred_series), label="Forecast", color="red")
    plt.legend()
    plt.title("Rolling Window Forecast")
    plt.show()

    return {
        "pred": save_pred,
        "errors": {"rmse": rmse, "mae": mae}
    }

#### Neural Net
def run_nn(Y, horizon):
    Y2 = Y.copy()
    pca = PCA(n_components=4)
    standard_Y2 = scaler_demean.fit_transform(Y2)
    scores = pca.fit_transform(standard_Y2)
    Y2 = pd.concat([Y2, pd.DataFrame(scores, index=Y2.index)], axis=1)
    Y2.columns = Y2.columns.astype(str)
    Y3 = normalize_columns_std(Y2).to_numpy()

    aux = embed(Y3, 4)

    Xin = aux[:-horizon]
    Xout = aux[-1]

    y = Y3[-Xin.shape[0]:, 0]
    X = Xin
    X_out = Xout

    model = Sequential([
        Dense(32, activation='relu', input_dim=X.shape[1]),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(), loss='mse')
    model.fit(X, y, epochs=100, verbose=0)

    pred = model.predict(X_out.reshape(1, -1), verbose=0)
    meanval = Y2.iloc[:, 0].mean()
    stdval = Y2.iloc[:, 0].std()
    pred = denormalize_std(pred, meanval, stdval)
    return model, pred[0][0]

def nn_rolling_window(Y, npred, horizon):
    save_pred = []
    for i in range(npred, horizon - 1, -1):
        Y_window = Y.iloc[(npred - i):(len(Y) - i), :]
        _, pred = run_nn(Y_window, horizon=horizon)
        save_pred.append(pred)
        print(f"nn_iter {npred - i} horizon {horizon}", end='\r', flush=True)

    real = Y.values[:, 0]
    pred_series = np.concatenate([np.full(len(real) - len(save_pred), np.nan), save_pred])

    rmse = np.sqrt(mean_squared_error(real[-len(save_pred):], save_pred))
    mae = mean_absolute_error(real[-len(save_pred):], save_pred)

    # Plot
    plt.plot(real, label="Actual")
    padding = [np.nan] * (len(real) - len(pred_series))
    plt.plot(padding + list(pred_series), label="Forecast", color="red")
    plt.legend()
    plt.title("Rolling Window Forecast")
    plt.show()

    return {
        "pred": save_pred,
        "errors": {"rmse": rmse, "mae": mae}}

# LSTM
def run_single_lstm(Y, horizon, batch_size=30, unit_n=32, epochs=100):
    Y2 = Y.copy()
    pca = PCA(n_components=4)
    standard_Y2 = scaler_demean.fit_transform(Y2)
    scores = pca.fit_transform(standard_Y2)
    Y2 = pd.concat([Y2, pd.DataFrame(scores, index=Y2.index)], axis=1)
    Y2.columns = Y2.columns.astype(str)
    Y3 = normalize_columns(Y2).to_numpy()

    aux = embed(Y3, 4)

    Xin = aux[:-horizon]
    Xout = aux[-1]

    y = Y3[-Xin.shape[0]:, 0]
    X = Xin
    X_out = Xout

    X2 = np.zeros_like(X)
    for i in range(Y3.shape[1]):
        X2[:, 4 * i + 0] = X[:, i + 3 * Y3.shape[1]]
        X2[:, 4 * i + 1] = X[:, i + 2 * Y3.shape[1]]
        X2[:, 4 * i + 2] = X[:, i + 1 * Y3.shape[1]]
        X2[:, 4 * i + 3] = X[:, i + 0 * Y3.shape[1]]

    X_out2 = np.zeros_like(X_out)
    for i in range(Y3.shape[1]):
        X_out2[4 * i + 0] = X_out[i + 3 * Y3.shape[1]]
        X_out2[4 * i + 1] = X_out[i + 2 * Y3.shape[1]]
        X_out2[4 * i + 2] = X_out[i + 1 * Y3.shape[1]]
        X_out2[4 * i + 3] = X_out[i + 0 * Y3.shape[1]]

    X_arr = X2.reshape((X2.shape[0], 4, Y3.shape[1]))
    X_out_arr = X_out2.reshape((1, 4, Y3.shape[1]))

    model = Sequential()
    model.add(LSTM(units=unit_n, input_shape=(4, Y3.shape[1]), stateful=False))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_arr, y, epochs=epochs, batch_size=batch_size, shuffle=False, verbose=0)

    pred = model.predict(X_out_arr)
    minval = Y2.iloc[:, 0].min()
    maxval = Y2.iloc[:, 0].max()
    pred = denormalize(pred, minval, maxval)

    return {"model": model, "pred": pred}

def rolling_window_lstm_single(Y, npred, horizon=1, batch=30, unit=32):
    save_pred = []

    for i in range(npred, horizon - 1, -1):
        Y_window = Y.iloc[(npred - i):(len(Y) - i), :].copy()
        result = run_single_lstm(Y_window, horizon, batch, unit)
        save_pred.append(result["pred"][0][0])
        print(f"lstm_iter {npred - i} horizon {horizon}", end='\r', flush=True)

    real = Y.values[:, 0]
    pred_series = np.concatenate([np.full(len(real) - len(save_pred), np.nan), save_pred])

    rmse = np.sqrt(mean_squared_error(real[-len(save_pred):], save_pred))
    mae = mean_absolute_error(real[-len(save_pred):], save_pred)

    # Plot
    plt.plot(real, label="Actual")
    padding = [np.nan] * (len(real) - len(pred_series))
    plt.plot(padding + list(pred_series), label="Forecast", color="red")
    plt.legend()
    plt.title("Rolling Window Forecast")
    plt.show()

    return {
        "pred": save_pred,
        "errors": {"rmse": rmse, "mae": mae}
        }

# Shrinkage Method: LASSO IC
def run_lasso_ic(Y, horizon):
    Y2 = Y.copy()
    pca = PCA(n_components=4)
    standard_Y2 = scaler_demean.fit_transform(Y2)
    scores = pca.fit_transform(standard_Y2)
    Y2 = pd.concat([Y2, pd.DataFrame(scores, index=Y2.index)], axis=1)
    Y2.columns = Y2.columns.astype(str)
    Y_matrix = Y2.to_numpy()

    aux = embed(Y_matrix, 4)

    Xin = aux[:-horizon]
    Xout = aux[-1]

    y = Y2.iloc[-Xin.shape[0]:, 0]
    X = Xin
    X_out = Xout

    model = LassoLarsIC(criterion='aic', fit_intercept=True).fit(X, y)

    pred = model.predict(X_out.reshape(1, -1))
    return model, pred

def lasso_ic_rolling_window(Y, npred, horizon):
    save_model = []
    save_pred = []

    for i in range(npred, horizon - 1, -1):
        Y_window = Y.iloc[(npred - i):(len(Y) - i), :]
        model, pred = run_lasso_ic(Y_window, horizon)
        save_model.append(model)
        save_pred.append(pred[0])
        print(f"shrink_iter {npred - i} horizon {horizon}", end='\r', flush=True)

    real = Y.values[:, 0]
    pred_series = np.concatenate([np.full(len(real) - len(save_pred), np.nan), save_pred])

    rmse = np.sqrt(mean_squared_error(real[-len(save_pred):], save_pred))
    mae = mean_absolute_error(real[-len(save_pred):], save_pred)

    # Plot
    plt.plot(real, label="Actual")
    padding = [np.nan] * (len(real) - len(pred_series))
    plt.plot(padding + list(pred_series), label="Forecast", color="red")
    plt.legend()
    plt.title("Rolling Window Forecast")
    plt.show()

    return {
        "pred": save_pred,
        "errors": {"rmse": rmse, "mae": mae},
        "model" : save_model}

### GW test
def gw_test(x, y, p, T, tau, method="HAC", alternative="two.sided"):
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    p = np.asarray(p).flatten()

    if len(x) != len(y):
        raise ValueError("Length of x and y must match.")
    
    if tau < 1:
        raise ValueError("Predictive horizon tau must be >= 1.")

    l1 = np.abs(x - p)
    l2 = np.abs(y - p)
    diff = l1 - l2
    q = len(l1)
    m = T - q
    n = T - tau - m + 1
    delta = np.mean(diff)

    X = np.ones((q, 1))
    model = sm.OLS(diff, X).fit()

    if tau == 1:
        stat = model.tvalues[0]
        method_used = "Standard Statistic Simple Regression Estimator"
    else:
        if method == "HAC":
            cov = cov_hac(model, nlags=tau)
            method_used = "HAC Covariance matrix Estimation"
        elif method == "NeweyWest":
            cov = cov_hac(model, nlags=tau)
            method_used = "Newey-West HAC Covariance matrix Estimation"
        elif method == "Andrews":
            cov = cov_hac(model, nlags=tau, kernel="bartlett")
            method_used = "Kernel-based HAC Covariance matrix Estimation"
        elif method == "LumleyHeagerty":
            cov = cov_white_simple(model)
            method_used = "Lumley HAC Covariance matrix Estimation"
        else:
            raise ValueError("Invalid method selected.")
        se = np.sqrt(cov[0, 0])
        stat = delta / se

    if alternative == "two.sided":
        pval = 2 * norm.cdf(-abs(stat))
    elif alternative == "less":
        pval = norm.cdf(stat)
    elif alternative == "greater":
        pval = norm.sf(stat)
    else:
        raise ValueError("Invalid alternative hypothesis")

    return {
        "statistic": stat,
        "p_value": pval,
        "method": method_used,
        "alternative": alternative
    }

### Model Confidence Set Test
def boot_block(n, v, k):
    start_indexes = np.random.randint(0, n - k, size=v + 1)
    blocks = np.concatenate([np.arange(p, p + k) for p in start_indexes])
    return blocks[:n]

def compute_d_b_i_mean(d_dict_list, model_names):
    M = len(model_names)
    result = []
    for row_dict in d_dict_list:
        mean_vals = []
        for i in range(M):
            other_vals = [row_dict[f"{model_names[i]}.{model_names[j]}"] for j in range(M) if i != j]
            mean_vals.append(np.mean(other_vals))
        result.append(mean_vals)
    return np.array(result)

def MCSprocedure_py(Loss_df, alpha=0.15, B=5000, statistic="Tmax", min_k=3, verbose=True):
    start_time = time.time()
    Loss_df = Loss_df.copy()
    Loss_df.columns = [col.replace(".", "_") for col in Loss_df.columns]
    M_start = Loss_df.shape[1]
    
    while True:
        M = Loss_df.shape[1]
        model_names = Loss_df.columns.tolist()
        diffs = {}
        for i in model_names:
            for j in model_names:
                if i != j:
                    diffs[f"{i}.{j}"] = Loss_df[i] - Loss_df[j]
        d = pd.concat(diffs, axis=1)

        d_ij_mean = d.mean()
        d_i_mean = pd.Series({i: np.mean([d_ij_mean[f"{i}.{j}"] for j in model_names if j != i]) for i in model_names})

        k = max(min_k, 3)
        n = d.shape[0]
        v = int(np.ceil(n / k))
        indexes_b = [boot_block(n, v, k) for _ in range(B)]

        d_ij_avg_resampled = pd.DataFrame([d.iloc[idx].mean() for idx in indexes_b], columns=d.columns)
        d_b_i_mean = compute_d_b_i_mean(d_ij_avg_resampled.to_dict(orient="records"), model_names)
        d_b_i_mean_df = pd.DataFrame(d_b_i_mean, columns=model_names)

        d_b_i_var = d_b_i_mean_df.sub(d_i_mean.values).pow(2).sum() / B
        d_ij_var = d_ij_avg_resampled.sub(d_ij_mean).pow(2).sum() / B

        if isinstance(d_ij_var, (float, int)):
            d_ij_var = pd.Series([d_ij_var]*len(d_ij_mean), index=d_ij_mean.index)

        TR = np.max(np.abs(d_ij_mean) / np.sqrt(d_ij_var))
        TM = np.max(d_i_mean / np.sqrt(d_b_i_var))

        Tb_R = d_ij_avg_resampled.sub(d_ij_mean).div(np.sqrt(d_ij_var)).abs().max(axis=1)
        Tb_M = d_b_i_mean_df.sub(d_i_mean).div(np.sqrt(d_b_i_var)).max(axis=1)

        Pr = np.mean(TR < Tb_R)
        Pm = np.mean(TM < Tb_M)

        v_i_M = d_i_mean / np.sqrt(d_b_i_var)
        v_i_R = pd.Series({i: np.max(d_ij_mean[[f"{i}.{j}" for j in model_names if j != i]] / 
                                     np.sqrt(d_ij_var[[f"{i}.{j}" for j in model_names if j != i]]))
                          for i in model_names})

        matrix_show = pd.DataFrame(index=model_names)
        matrix_show["v_M"] = v_i_M
        matrix_show["v_R"] = v_i_R
        matrix_show["Loss"] = Loss_df.mean()

        if statistic == "Tmax":
            p2test = Pm
        elif statistic == "TR":
            p2test = Pr
        else:
            raise ValueError("Invalid statistic type")

        if p2test > alpha or np.all(d_ij_var == 0):
            return {
                "SSM": matrix_show.sort_values("Loss"),
                "statistic": statistic,
                "p_value": p2test,
                "n_eliminated": M_start - M,
                "elapsed_time": time.time() - start_time
            }

        if statistic == "Tmax":
            eliminate = v_i_M.idxmax()
        else:
            eliminate = v_i_R.idxmax()

        if verbose:
            print(f"Model {eliminate} eliminated at p = {p2test:.4f}")

        Loss_df.drop(columns=eliminate, inplace=True)

        if Loss_df.shape[1] == 1:
            return {
                "SSM": pd.DataFrame({
                    "v_M": v_i_M.drop(eliminate),
                    "v_R": v_i_R.drop(eliminate),
                    "Loss": Loss_df.mean()
                }),
                "statistic": statistic,
                "p_value": p2test,
                "n_eliminated": M_start - 1,
                "elapsed_time": time.time() - start_time
            }
        