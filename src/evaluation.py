"""
Funciones de evaluación de modelos.

Funciones extraídas del notebook 03_model_training_and_evaluation.
Separación de concerns: cálculo de métricas vs. visualización.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)


def compute_metrics(model, X_test, y_test):
    """
    Calcula métricas de clasificación para un modelo entrenado.

    Parámetros:
        model: Modelo o pipeline entrenado con método predict().
        X_test: Datos de prueba (features).
        y_test: Etiquetas reales de prueba.

    Retorna:
        dict: Diccionario con 'y_pred', 'classification_report' (str),
              'classification_report_dict' (dict) y 'confusion_matrix'.
    """
    y_pred = model.predict(X_test)
    report_str = classification_report(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    logger.info("Métricas calculadas. Accuracy: %.4f", report_dict['accuracy'])

    return {
        'y_pred': y_pred,
        'classification_report': report_str,
        'classification_report_dict': report_dict,
        'confusion_matrix': cm,
    }


def compute_train_test_gap(model, X_train, y_train, X_test, y_test, metric='accuracy'):
    """
    Calcula y reporta explícitamente el gap entre rendimiento en train vs test.

    Útil para detectar sobreajuste: un gap alto indica que el modelo
    memoriza los datos de entrenamiento en lugar de generalizar.

    Parámetros:
        model: Modelo o pipeline entrenado con método score().
        X_train: Features de entrenamiento.
        y_train: Target de entrenamiento.
        X_test: Features de prueba.
        y_test: Target de prueba.
        metric (str): Nombre de la métrica para el reporte.

    Retorna:
        dict: Diccionario con 'train_score', 'test_score', 'gap' y 'is_overfitting'.
    """
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    gap = train_score - test_score

    result = {
        'train_score': train_score,
        'test_score': test_score,
        'gap': gap,
        'is_overfitting': gap > 0.05,
    }

    logger.info("--- Análisis de Gap Train/Test (%s) ---", metric)
    logger.info("Score de entrenamiento: %.4f", train_score)
    logger.info("Score de prueba:        %.4f", test_score)
    logger.info("Gap (train - test):     %.4f", gap)

    if result['is_overfitting']:
        logger.warning(
            "⚠  Gap significativo detectado (%.4f > 0.05). "
            "El modelo podría estar sobreajustado.", gap
        )
    else:
        logger.info("✓  El gap train/test es aceptable.")

    return result


def plot_confusion_matrix(cm, class_names, model_name, figsize=(16, 14)):
    """
    Genera una visualización de la matriz de confusión normalizada.

    Parámetros:
        cm (np.ndarray): Matriz de confusión (de confusion_matrix).
        class_names (array-like): Nombres de las clases.
        model_name (str): Nombre del modelo para el título.
        figsize (tuple): Tamaño de la figura.

    Retorna:
        matplotlib.figure.Figure: La figura generada.
    """
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    labels = (np.asarray([f"{pct:.1%}\n({count})"
                          for pct, count in zip(cm_normalized.flatten(), cm.flatten())])
              ).reshape(cm.shape)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm_normalized,
                annot=labels,
                fmt='',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax)
    ax.set_title(f'Matriz de Confusión Normalizada - {model_name}',
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel('Clase Real', fontsize=14)
    ax.set_xlabel('Clase Predicha', fontsize=14)

    return fig


def evaluate_model(model, X_test, y_test, model_name, figsize=(16, 14)):
    """
    Evalúa un modelo de clasificación con reporte de clasificación y
    matriz de confusión normalizada estilizada.

    Esta función mantiene compatibilidad con el uso anterior, combinando
    compute_metrics() y plot_confusion_matrix().

    Parámetros:
        model: Modelo o pipeline entrenado con método predict() y atributo classes_.
        X_test: Datos de prueba (features).
        y_test: Etiquetas reales de prueba.
        model_name (str): Nombre del modelo para los títulos de las gráficas.
        figsize (tuple): Tamaño de la figura para la matriz de confusión.

    Retorna:
        dict: Diccionario con 'y_pred', 'classification_report' y 'confusion_matrix'.
    """
    metrics = compute_metrics(model, X_test, y_test)

    logger.info("\n--- Reporte de Clasificación (%s) ---\n%s",
                model_name, metrics['classification_report'])

    plot_confusion_matrix(
        metrics['confusion_matrix'],
        model.classes_,
        model_name,
        figsize=figsize,
    )
    plt.show()

    return metrics
