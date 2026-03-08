"""
Excepciones personalizadas para el Detector de Intrusiones.

Proporciona una jerarquía clara de errores para mejor depuración
y manejo de errores específicos del dominio.
"""


class DetectorError(Exception):
    """Excepción base para todos los errores del Detector de Intrusiones."""
    pass


class ModelNotFoundError(DetectorError, FileNotFoundError):
    """El archivo del modelo no fue encontrado en disco."""
    pass


class InvalidDataError(DetectorError, ValueError):
    """Los datos de entrada no tienen el formato esperado."""
    pass


class PipelineError(DetectorError, RuntimeError):
    """Error durante la ejecución de un paso del pipeline."""
    pass


class DriftDetectedError(DetectorError):
    """Se detectó drift significativo en los datos o predicciones."""
    pass
