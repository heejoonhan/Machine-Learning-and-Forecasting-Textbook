import warnings
import numpy as np
import cvxpy as cp
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin

warnings.filterwarnings("ignore", category=UserWarning)

scaler = MinMaxScaler()
scaler_std = StandardScaler()

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

class AdaptiveShrinkage(BaseEstimator, RegressorMixin):
    def __init__(self, lambda1=0.1, alpha=0.5, adaptive_weights=None, 
                 fit_intercept=True, tol=1e-8, criterion='cv', solver='OSQP'):
        self.lambda1 = lambda1
        self.alpha = alpha
        self.criterion = criterion
        self.adaptive_weights = adaptive_weights
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.solver = solver

    def _solve_internal(self, X, y, l_val, penalty_weights):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        
        beta = cp.Variable(n_features)
        intercept = cp.Variable() if self.fit_intercept else 0
        
        loss = (0.5 / n_samples) * cp.sum_squares(y - (X @ beta + intercept))
        l1_pen = self.alpha * cp.norm1(cp.multiply(penalty_weights, beta))
        l2_pen = (1 - self.alpha) * 0.5 * cp.sum_squares(beta)
        
        objective = cp.Minimize(loss + l_val * (l1_pen + l2_pen))
        prob = cp.Problem(objective)
        
        try:
            prob.solve(solver=self.solver, verbose=False)
            
            if beta.value is None:
                return np.zeros(n_features), 0.0
            
            res_beta = np.array(beta.value).flatten()
            res_beta[np.abs(res_beta) < self.tol] = 0
            res_intercept = float(intercept.value) if self.fit_intercept else 0.0
            return res_beta, res_intercept
        except Exception:
            return np.zeros(n_features), 0.0

    def fit(self, X, y, lambda_grid=None):
        X, y = check_X_y(X, y)
        n_features = X.shape[1]

        if self.adaptive_weights is not None:
            aw = np.asarray(self.adaptive_weights).flatten()
            penalty_weights = 1.0 / (np.abs(aw) + 1e-6)
        else:
            penalty_weights = np.ones(n_features)

        if lambda_grid is not None:
            best_lambda = self.lambda1
            self.best_score_ = np.inf
            
            for l_cand in lambda_grid:
                if self.criterion == 'cv':
                    tscv = TimeSeriesSplit(n_splits=5)
                    scores = []
                    for train_idx, test_idx in tscv.split(X):
                        b, i = self._solve_internal(X[train_idx], y[train_idx], l_cand, penalty_weights)
                        pred = X[test_idx] @ b + i
                        scores.append(np.mean((y[test_idx] - pred)**2))
                    current_score = np.mean(scores) if scores else np.inf
                else:
                    b, i = self._solve_internal(X, y, l_cand, penalty_weights)
                    current_score = self._get_ic_value(X, y, b, i, self.criterion)
                
                if current_score < self.best_score_:
                    self.best_score_ = current_score
                    best_lambda = l_cand
            
            self.lambda1 = best_lambda

        self.coef_, self.intercept_ = self._solve_internal(X, y, self.lambda1, penalty_weights)
        return self

    def _get_ic_value(self, X, y, beta, intercept, criterion):
        n_samples = X.shape[0]
        mse = np.mean((y - (X @ beta + intercept))**2)
        k = np.sum(np.abs(beta) > self.tol) + (1 if self.fit_intercept else 0)
        if mse <= 1e-15: return -np.inf
        if criterion == 'aic':
            return n_samples * np.log(mse) + 2 * k
        return n_samples * np.log(mse) + k * np.log(n_samples)

    def predict(self, X):
        check_is_fitted(self)
        X = np.asarray(X)
        return X @ self.coef_ + self.intercept_

def run_adaptshrink(Y, horizon, criterion, alpha=1.0):
    Y2 = Y.copy()
    pca = PCA(n_components=4)
    standard_Y2 = scaler_std.fit_transform(Y2)
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

    panalties = np.logspace(2, -4, 30)
    init_model = LassoCV(alphas=panalties, fit_intercept=True).fit(Xin, y)
    init_coef = init_model.coef_

    penalty = 1 / (np.abs(init_coef) + 1 / np.sqrt(len(y)))

    model = AdaptiveShrinkage(alpha=alpha, adaptive_weights=penalty, criterion=criterion)
    model.fit(X, y, lambda_grid=panalties)

    pred = model.predict(X_out.reshape(1, -1))
    return model, pred
    
from joblib import Parallel, delayed

def adaptshrinkage_cv_rolling_window(Y, npred, horizon, criterion="cv", alpha=1.0):
    window_list = []
    for i in range(npred, horizon - 1, -1):
        Y_window = Y.iloc[(npred - i):(len(Y) - i), :]
        window_list.append(Y_window)

    print(f"Starting parallel processing for {len(window_list)} windows...")

    results = Parallel(n_jobs=-1)(
        delayed(run_adaptshrink)(W, horizon, criterion, alpha) 
        for W in window_list
    )

    save_model = [res[0] for res in results]
    save_pred = [res[1][0] for res in results]

    real = Y.values[:, 0]
    
    pred_series = np.full(len(real), np.nan)
    pred_series[-len(save_pred):] = save_pred

    rmse = np.sqrt(mean_squared_error(real[-len(save_pred):], save_pred))
    mae = mean_absolute_error(real[-len(save_pred):], save_pred)

    plt.figure(figsize=(10, 6))
    plt.plot(real, label="Actual", alpha=0.7)
    plt.plot(pred_series, label="Forecast", color="red", linewidth=1.5)
    plt.legend()
    plt.title(f"Rolling Window Forecast (RMSE: {rmse:.4f})")
    plt.show()

    return {
        "pred": save_pred,
        "errors": {"rmse": rmse, "mae": mae},
        "model": save_model
    }

