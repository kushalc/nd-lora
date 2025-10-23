import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import json
import os
import pandas as pd

def parametric_fit(param_list, p_list, loss_list):
    param_list = np.asarray(param_list).reshape((-1, ))
    loss_list = np.asarray(loss_list).reshape((-1, ))
    p_list = np.asarray(p_list).reshape((-1, ))

    def huber_loss(y_true, y_pred, delta=0.001):
        error = y_true - y_pred
        is_small_error = np.abs(error) <= delta
        squared_loss = np.square(error) / 2
        linear_loss = delta * (np.abs(error) - delta / 2)
        return np.where(is_small_error, squared_loss, linear_loss).sum()
    
    def pred_loss(params):
        E, A, alpha, k = params
        return E + (A * 1e9 / (param_list * (np.log(p_list) * k + 1))) ** alpha

    def objective_function(params):
        pred = pred_loss(params)
        return huber_loss(np.log(loss_list), np.log(pred))

    best_param = None
    best_func = 1000000
    for E in [-1, -0.5, 0]:
        for log_A in [-4, -2, 0, 2, 4]:
            for alpha in [0, 0.5, 1, 1.5, 2]:
                for k in [0.2, 0.4, 0.6, 0.8]:
                    initial_params = [np.exp(E), np.exp(log_A), alpha, k]
                    bounds = [(1e-8, None), (1e-8, None), (1e-8, None), (1e-8, None)]
                    result = minimize(objective_function, initial_params, method='L-BFGS-B', bounds=bounds)
                    if result.fun < best_func:
                        best_param = result.x
                        best_func = result.fun
    print(f"{result = }")
    print(f"{best_param = }")
    print(f"{best_func = }")

    pred_key = "$\\mathcal L_{\\text{pred}}$"
    true_key = "$\\mathcal L_{\\text{true}}$"
    df = pd.DataFrame({
        "$P$": p_list,
        "Parameters (Non-Embedding)": param_list,
        pred_key: pred_loss(best_param),
        true_key: loss_list,
        "Error": pred_loss(best_param) - loss_list
    })
    df['Parameters (Non-Embedding)'] = df['Parameters (Non-Embedding)'].apply(lambda x: f"{x:,}")
    r2 = r2_score(df[true_key].to_numpy().reshape(-1, 1), df[pred_key].to_numpy().reshape(-1, 1))

    print(df.to_latex(float_format=lambda x: f"{x:.4f}", index=False, column_format='rrrrr'))
    print(f"{r2 = }")


if __name__ == "__main__":

    params = [
        [535813376, 693753856, 1088376320, 1571472384, 2774773760, 4353203200], 
        [538195842, 696738818, 1092762882, 1577522690, 2784937986, 4368529922],
        [540577412, 699722756, 1097148164, 1583571460, 2795100164, 4383854084],
        [545340552, 705690632, 1105918728, 1595669000, 2815424520, 4414502408],
    ]

    stack_loss = [
        [1.1722, 1.1496, 1.1131, 1.0817, 1.0451, 1.0213], # 1.0006], # P1 
        [1.1507, 1.1262, 1.094, 1.0623, 1.0244, 1.0025], # P2
        [1.1354, 1.1124, 1.0808, 1.049, 1.0126, 0.9906], # P4
        [1.1231, 1.0997, 1.0688, 1.0383, 1.0016, 0.9794], # P8
    ]

    pile_loss = [
        [2.1113, 2.0671, 2.0027, 1.9539, 1.8876, 1.8451], # P1
        [2.0772, 2.0363, 1.973, 1.9266, 1.861, 1.8137], # P2
        [2.0544, 2.0128, 1.9509, 1.904, 1.8394, 1.7938], # P4
        [2.0364, 1.9933, 1.9318, 1.8856, 1.8218, 1.7772], # P8
    ]

    p = [
        [1] * 6,
        [2] * 6,
        [4] * 6,
        [8] * 6,
    ]
    
    print("=" * 10 + " Stack-V2 Python " + "=" * 10)
    parametric_fit(params, p, stack_loss)
    print("=" * 10 + " Pile " + "=" * 10)
    parametric_fit(params, p, pile_loss)