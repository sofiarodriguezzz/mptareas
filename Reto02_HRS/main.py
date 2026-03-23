"""
Reto 02 - Gradiente Descendente desde Cero
Modelado Predictivo 2026

Ejecuta el pipeline completo:
  1. Gradiente descendente 1D
  2. Experimentos con learning rates
  3. Gradiente descendente 2D
  4. Generacion del CSV de resultados
  5. Bonus: SGD con mini-batches
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

# FUNCIONES OBJETIVO


def f_1d(x):
    return (x - 3) ** 2 + 5

def df_1d(x):
    return 2 * (x - 3)

def f_2d(x, y):
    return x**2 + y**2 - 4*x - 2*y + 5

def grad_2d(x, y):
    return np.array([2*x - 4, 2*y - 2])


# PARTE 1: GRADIENTE DESCENDENTE 1D

def gradiente_descendente_1d(x_inicial, learning_rate, max_iter=1000, tolerancia=1e-6):
    x_actual = x_inicial
    historial_x = [x_inicial]
    historial_f = [f_1d(x_inicial)]
    convergido = False

    for _ in range(max_iter):
        g = df_1d(x_actual)
        x_nuevo = x_actual - learning_rate * g
        historial_x.append(x_nuevo)
        historial_f.append(f_1d(x_nuevo))
        if abs(x_nuevo - x_actual) < tolerancia:
            convergido = True
            x_actual = x_nuevo
            break
        x_actual = x_nuevo

    return {
        "x_final": x_actual,
        "f_final": f_1d(x_actual),
        "iteraciones": len(historial_x) - 1,
        "convergido": convergido,
        "historial_x": historial_x,
        "historial_f": historial_f,
    }



# PARTE 3: GRADIENTE DESCENDENTE 2D


def gradiente_descendente_2d(x_inicial, y_inicial, learning_rate, max_iter=1000, tolerancia=1e-6):
    x_actual, y_actual = x_inicial, y_inicial
    historial_x = [x_inicial]
    historial_y = [y_inicial]
    historial_f = [f_2d(x_inicial, y_inicial)]
    convergido = False

    for _ in range(max_iter):
        g = grad_2d(x_actual, y_actual)
        x_nuevo = x_actual - learning_rate * g[0]
        y_nuevo = y_actual - learning_rate * g[1]
        historial_x.append(x_nuevo)
        historial_y.append(y_nuevo)
        historial_f.append(f_2d(x_nuevo, y_nuevo))
        if np.linalg.norm(g) < tolerancia:
            convergido = True
            x_actual, y_actual = x_nuevo, y_nuevo
            break
        x_actual, y_actual = x_nuevo, y_nuevo

    return {
        "x_final": x_actual,
        "y_final": y_actual,
        "f_final": f_2d(x_actual, y_actual),
        "iteraciones": len(historial_x) - 1,
        "convergido": convergido,
        "historial_x": historial_x,
        "historial_y": historial_y,
        "historial_f": historial_f,
    }



# BONUS: SGD CON MINI-BATCHES


def sgd_regresion_lineal(X, y, learning_rate=0.01, epochs=50, batch_size=32):
    n = len(X)
    w, b = 0.0, 0.0
    historial_loss, historial_w, historial_b = [], [], []

    for _ in range(epochs):
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            batch_idx = indices[start : start + batch_size]
            X_b, y_b = X[batch_idx], y[batch_idx]
            m = len(X_b)
            error = y_b - (w * X_b + b)
            dw = -2 / m * np.sum(error * X_b)
            db = -2 / m * np.sum(error)
            w -= learning_rate * dw
            b -= learning_rate * db

        mse = np.mean((y - (w * X + b)) ** 2)
        historial_loss.append(mse)
        historial_w.append(w)
        historial_b.append(b)

    return {
        "w_final": w,
        "b_final": b,
        "historial_loss": historial_loss,
        "historial_w": historial_w,
        "historial_b": historial_b,
    }



# PIPELINE PRINCIPAL


def main():
    print("=" * 60)
    print("  RETO 02 - GRADIENTE DESCENDENTE DESDE CERO")
    print("  Modelado Predictivo 2026")
    print("=" * 60)

    # --- Parte 1 ---
    print("\n[1/4] Gradiente Descendente 1D")
    res1d = gradiente_descendente_1d(x_inicial=-2.0, learning_rate=0.1, max_iter=100)
    print(f"      x final = {res1d['x_final']:.6f}  (esperado: 3.0)")
    print(f"      f(x)    = {res1d['f_final']:.6f}  (esperado: 5.0)")
    print(f"      iters   = {res1d['iteraciones']}  |  convergido: {res1d['convergido']}")

    # --- Parte 2 ---
    print("\n[2/4] Experimentos con Learning Rates")
    lrs = [0.001, 0.01, 0.1, 0.5, 0.9, 1.0, 1.5]
    print(f"  {'LR':>6}  {'x_final':>10}  {'f_final':>10}  {'iters':>6}  {'conv':>5}")
    print("  " + "-" * 48)
    for lr in lrs:
        r = gradiente_descendente_1d(x_inicial=-2.0, learning_rate=lr, max_iter=200)
        print(f"  {lr:>6.3f}  {r['x_final']:>10.4f}  {r['f_final']:>10.4f}  {r['iteraciones']:>6}  {str(r['convergido']):>5}")

    # --- Parte 3 ---
    print("\n[3/4] Gradiente Descendente 2D")
    res2d = gradiente_descendente_2d(x_inicial=-1.0, y_inicial=4.0, learning_rate=0.1, max_iter=100)
    print(f"      (x,y) final = ({res2d['x_final']:.6f}, {res2d['y_final']:.6f})")
    print(f"      f(x,y)      = {res2d['f_final']:.6f}  (esperado: 0.0)")
    print(f"      iters       = {res2d['iteraciones']}  |  convergido: {res2d['convergido']}")

    # --- Parte 4: CSV ---
    print("\n[4/4] Generando CSV de experimentos...")
    filas = []

    for lr in [0.001, 0.01, 0.1, 0.5, 0.9, 1.0, 1.5]:
        r = gradiente_descendente_1d(x_inicial=-2.0, learning_rate=lr, max_iter=200)
        filas.append({"learning_rate": lr, "dimension": "1D", "x_inicial": -2.0,
                      "y_inicial": float("nan"), "x_final": r["x_final"],
                      "y_final": float("nan"), "valor_minimo": r["f_final"],
                      "iteraciones": r["iteraciones"], "convergido": r["convergido"]})

    for pt in [(-1.0, 4.0), (5.0, -1.0), (0.0, 0.0)]:
        for lr in [0.001, 0.01, 0.1, 0.5]:
            r = gradiente_descendente_2d(x_inicial=pt[0], y_inicial=pt[1],
                                         learning_rate=lr, max_iter=500)
            filas.append({"learning_rate": lr, "dimension": "2D", "x_inicial": pt[0],
                          "y_inicial": pt[1], "x_final": r["x_final"],
                          "y_final": r["y_final"], "valor_minimo": r["f_final"],
                          "iteraciones": r["iteraciones"], "convergido": r["convergido"]})

    df = pd.DataFrame(filas)
    df.to_csv("experimentos_gd.csv", index=False)
    print(f"      Guardado: experimentos_gd.csv  ({len(df)} filas)")

    # --- Bonus ---
    print("\n[Bonus] SGD con mini-batches")
    n_datos = 200
    X_sgd = np.random.uniform(0, 10, n_datos)
    y_sgd = 3 * X_sgd + 7 + np.random.normal(0, 2, n_datos)
    res_sgd = sgd_regresion_lineal(X_sgd, y_sgd, learning_rate=0.01, epochs=50, batch_size=32)
    print(f"      w = {res_sgd['w_final']:.4f}  (real: 3.0)")
    print(f"      b = {res_sgd['b_final']:.4f}  (real: 7.0)")
    print(f"      MSE final = {res_sgd['historial_loss'][-1]:.4f}")

    print("\n" + "=" * 60)
    print("  Listo. Revisa experimentos_gd.csv para los resultados.")
    print("=" * 60)


if __name__ == "__main__":
    main()
