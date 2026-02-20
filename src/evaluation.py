"""
Funciones de evaluación de modelos.

Funciones extraídas del notebook 03_model_training_and_evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model, X_test, y_test, model_name, figsize=(16, 14)):
    """
    Evalúa un modelo de clasificación con reporte de clasificación y
    matriz de confusión normalizada estilizada.

    Parámetros:
        model: Modelo o pipeline entrenado con método predict() y atributo classes_.
        X_test: Datos de prueba (features).
        y_test: Etiquetas reales de prueba.
        model_name (str): Nombre del modelo para los títulos de las gráficas.
        figsize (tuple): Tamaño de la figura para la matriz de confusión.

    Retorna:
        dict: Diccionario con 'y_pred', 'classification_report' y 'confusion_matrix'.
    """
    # Predecir
    y_pred = model.predict(X_test)

    # Reporte de clasificación
    print(f"\n--- Reporte de Clasificación ({model_name}) ---")
    report = classification_report(y_test, y_pred)
    print(report)

    # Matriz de confusión
    print(f"\n--- Matriz de Confusión ({model_name}) ---")
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    labels = (np.asarray(["{0:.1%}\n({1})".format(pct, count)
                          for pct, count in zip(cm_normalized.flatten(), cm.flatten())])
              ).reshape(cm.shape)

    plt.figure(figsize=figsize)
    sns.heatmap(cm_normalized,
                annot=labels,
                fmt='',
                cmap='Blues',
                xticklabels=model.classes_,
                yticklabels=model.classes_)
    plt.title(f'Matriz de Confusión Normalizada - {model_name}',
              fontsize=18, fontweight='bold', pad=20)
    plt.ylabel('Clase Real', fontsize=14)
    plt.xlabel('Clase Predicha', fontsize=14)
    plt.show()

    return {
        'y_pred': y_pred,
        'classification_report': report,
        'confusion_matrix': cm,
    }
