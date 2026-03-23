# Reto 02 — Gradiente Descendente desde Cero

**Materia:** Modelado Predictivo 2026 · IPN. Huerta Rodríguez Sofía

---

## Descripción

Implementación del algoritmo de **gradiente descendente** desde cero en Python, sin usar librerías de ML. El reto explora cómo este algoritmo —el corazón del Machine Learning moderno— encuentra el mínimo de una función siguiendo la pendiente paso a paso.

Se trabajan dos funciones objetivo:

| Función | Mínimo |
|---------|--------|
| f(x) = (x − 3)² + 5 | x = 3, f = 5 |
| f(x, y) = x² + y² − 4x − 2y + 5 | (x, y) = (2, 1), f = 0 |

---

## Estructura del proyecto

```
reto_02/
├── main.py                                      # Pipeline completo ejecutable
├── requirements.txt                             # Dependencias
├── reto_02_gradiente_descendente_COMPLETO.ipynb # Notebook con análisis y gráficas
├── experimentos_gd.csv                          # Resultados de todos los experimentos
└── README.md
```

---

## Partes del reto

| Parte | Descripción | Puntos |
|-------|-------------|--------|
| 1 | Gradiente descendente en 1D | 25 |
| 2 | Experimentación con learning rates | 25 |
| 3 | Gradiente descendente en 2D | 25 |
| 4 | Análisis y generación de CSV | 25 |
| Bonus | SGD con mini-batches para regresión lineal | +15 |

---

## Resultados principales

### Learning rates — qué pasa con cada uno

| LR | Comportamiento | ¿Converge? |
|----|---------------|------------|
| 0.001 | Pasos minúsculos, apenas avanza en 200 iters | No |
| 0.01 | Converge pero lento | No (200 iters insuficientes) |
| 0.1 | Buen balance velocidad/precisión | Sí (63 iters) |
| 0.5 | Óptimo para esta función — llega en 2 iters | Sí (2 iters) |
| 0.9 | Converge pero oscila más | Sí (73 iters) |
| 1.0 | Oscila entre −2 y 8 eternamente | No |
| 1.5 | Diverge, los valores explotan | No |

### Gradiente descendente 2D
Desde tres puntos iniciales distintos (−1,4), (5,−1) y (0,0), el algoritmo converge siempre al mismo mínimo **(2, 1)** con LR=0.1 en ~70 iteraciones. Las funciones convexas garantizan un único mínimo global.

### Bonus — SGD
Con 200 datos generados con la relación `y = 3x + 7 + ruido`, el SGD con mini-batches de 32 recupera los parámetros reales:
- **w ≈ 3.0** (real: 3.0)
- **b ≈ 7.0** (real: 7.0)

El mini-batch de 32 resultó el mejor balance: curva de pérdida suave y convergencia rápida.

---

## Cómo ejecutar

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Correr el pipeline completo
python main.py

# 3. O abrir el notebook para ver gráficas y análisis
jupyter notebook reto_02_gradiente_descendente_COMPLETO.ipynb
```

---

## Conclusión

El gradiente descendente es la base de todo el ML moderno. Cada vez que se entrena una red neuronal, un modelo de lenguaje o un sistema de recomendación, hay una variante de este mismo algoritmo corriendo por debajo. La elección del learning rate es crítica: demasiado pequeño y tarda siglos, demasiado grande y diverge.
