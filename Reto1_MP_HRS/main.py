import seaborn as sns
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def clasificador_humano(bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g):
    """
    Clasificador basado en reglas humanas para identificar la especie de pingüino.
    """

    # Regla 1: Gentoo tiene aletas más largas
    if flipper_length_mm > 210:
        return "Gentoo"

    # Regla 2: Chinstrap tiene picos más largos
    elif bill_length_mm > 45:
        return "Chinstrap"

    # Regla 3: Si no cumple las anteriores
    else:
        return "Adelie"


def main():

    # 1. Cargar dataset
    df = sns.load_dataset("penguins").dropna().reset_index(drop=True)

    # 2. Seleccionar variables
    X = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
    y = df['species']

    # 3. División entrenamiento / prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # CLASIFICADOR HUMANO

    pred_humano = X_test.apply(
        lambda row: clasificador_humano(
            row['bill_length_mm'],
            row['bill_depth_mm'],
            row['flipper_length_mm'],
            row['body_mass_g']
        ),
        axis=1
    )

    acc_humano = accuracy_score(y_test, pred_humano)

    print("Accuracy clasificador humano:", acc_humano)

    # MODELO MACHINE LEARNING

    modelo_ml = DecisionTreeClassifier(random_state=42)

    modelo_ml.fit(X_train, y_train)

    pred_ml = modelo_ml.predict(X_test)

    acc_ml = accuracy_score(y_test, pred_ml)

    print("Accuracy modelo ML:", acc_ml)


if __name__ == "__main__":
    main()