"""
Utilidades para generar y manejar datos en tutoriales de Meta-Learning
"""

import numpy as np
import torch
from typing import Tuple, List, Optional, Dict
from torch.utils.data import Dataset
import random


def create_sine_task(
    amplitude: Optional[float] = None,
    phase: Optional[float] = None,
    k_shot: int = 10,
    q_query: int = 10,
    x_range: Tuple[float, float] = (-5.0, 5.0),
    noise_std: float = 0.0
) -> Dict[str, torch.Tensor]:
    """
    Crea una tarea de regresión de funciones sinusoidales

    Útil para demostrar Meta-Learning en problemas de regresión.
    Cada tarea es una función seno con amplitud y fase aleatorias.

    Args:
        amplitude: Amplitud de la función seno (aleatorio si None)
        phase: Fase de la función seno (aleatorio si None)
        k_shot: Número de ejemplos de soporte (entrenamiento)
        q_query: Número de ejemplos de consulta (test)
        x_range: Rango de valores x
        noise_std: Desviación estándar del ruido gaussiano

    Returns:
        Diccionario con:
            - 'x_support': Input de soporte [k_shot, 1]
            - 'y_support': Output de soporte [k_shot, 1]
            - 'x_query': Input de consulta [q_query, 1]
            - 'y_query': Output de consulta [q_query, 1]
            - 'amplitude': Amplitud usada
            - 'phase': Fase usada
    """
    if amplitude is None:
        amplitude = np.random.uniform(0.1, 5.0)
    if phase is None:
        phase = np.random.uniform(0, np.pi)

    # Generar puntos x
    x_support = np.random.uniform(x_range[0], x_range[1], size=(k_shot, 1))
    x_query = np.random.uniform(x_range[0], x_range[1], size=(q_query, 1))

    # Calcular y = A * sin(x + φ) + ruido
    y_support = amplitude * np.sin(x_support + phase)
    y_query = amplitude * np.sin(x_query + phase)

    if noise_std > 0:
        y_support += np.random.normal(0, noise_std, y_support.shape)
        y_query += np.random.normal(0, noise_std, y_query.shape)

    return {
        'x_support': torch.FloatTensor(x_support),
        'y_support': torch.FloatTensor(y_support),
        'x_query': torch.FloatTensor(x_query),
        'y_query': torch.FloatTensor(y_query),
        'amplitude': amplitude,
        'phase': phase
    }


def create_classification_task(
    n_way: int = 5,
    k_shot: int = 5,
    q_query: int = 15,
    img_size: int = 28,
    n_channels: int = 1
) -> Dict[str, torch.Tensor]:
    """
    Crea una tarea sintética de clasificación N-way K-shot

    Args:
        n_way: Número de clases
        k_shot: Número de ejemplos por clase en el conjunto de soporte
        q_query: Número de ejemplos por clase en el conjunto de consulta
        img_size: Tamaño de la imagen (cuadrada)
        n_channels: Número de canales (1 para gris, 3 para RGB)

    Returns:
        Diccionario con:
            - 'x_support': [n_way * k_shot, n_channels, img_size, img_size]
            - 'y_support': [n_way * k_shot]
            - 'x_query': [n_way * q_query, n_channels, img_size, img_size]
            - 'y_query': [n_way * q_query]
    """
    # Generar imágenes sintéticas (patrones aleatorios)
    x_support = torch.randn(n_way * k_shot, n_channels, img_size, img_size)
    x_query = torch.randn(n_way * q_query, n_channels, img_size, img_size)

    # Generar etiquetas
    y_support = torch.arange(n_way).repeat_interleave(k_shot)
    y_query = torch.arange(n_way).repeat_interleave(q_query)

    # Mezclar
    support_perm = torch.randperm(n_way * k_shot)
    query_perm = torch.randperm(n_way * q_query)

    return {
        'x_support': x_support[support_perm],
        'y_support': y_support[support_perm],
        'x_query': x_query[query_perm],
        'y_query': y_query[query_perm]
    }


def sample_batch_tasks(
    task_generator: callable,
    batch_size: int,
    **task_kwargs
) -> List[Dict]:
    """
    Genera un batch de tareas

    Args:
        task_generator: Función que genera una tarea
        batch_size: Número de tareas a generar
        **task_kwargs: Argumentos para task_generator

    Returns:
        Lista de diccionarios de tareas
    """
    return [task_generator(**task_kwargs) for _ in range(batch_size)]


class MetaLearningDataset(Dataset):
    """
    Dataset base para Meta-Learning
    """

    def __init__(
        self,
        task_generator: callable,
        n_tasks: int,
        **task_kwargs
    ):
        """
        Args:
            task_generator: Función que genera tareas
            n_tasks: Número total de tareas
            **task_kwargs: Argumentos para task_generator
        """
        self.task_generator = task_generator
        self.n_tasks = n_tasks
        self.task_kwargs = task_kwargs

    def __len__(self):
        return self.n_tasks

    def __getitem__(self, idx):
        """Genera una tarea al vuelo"""
        return self.task_generator(**self.task_kwargs)


class OmniglotNWayKShot:
    """
    Generador de tareas N-way K-shot para Omniglot

    Nota: Este es un placeholder. En tutoriales posteriores
    implementaremos la carga real de Omniglot.
    """

    def __init__(
        self,
        n_way: int = 5,
        k_shot: int = 1,
        q_query: int = 15,
        img_size: int = 28
    ):
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.img_size = img_size

    def sample_task(self) -> Dict[str, torch.Tensor]:
        """Genera una tarea N-way K-shot"""
        return create_classification_task(
            n_way=self.n_way,
            k_shot=self.k_shot,
            q_query=self.q_query,
            img_size=self.img_size,
            n_channels=1
        )


def create_toy_regression_task(
    func_type: str = 'sine',
    k_shot: int = 10,
    q_query: int = 10
) -> Dict[str, torch.Tensor]:
    """
    Crea tareas de regresión de juguete con diferentes tipos de funciones

    Args:
        func_type: Tipo de función ('sine', 'linear', 'quadratic')
        k_shot: Ejemplos de soporte
        q_query: Ejemplos de consulta

    Returns:
        Diccionario con datos de la tarea
    """
    x_support = torch.FloatTensor(k_shot, 1).uniform_(-5, 5)
    x_query = torch.FloatTensor(q_query, 1).uniform_(-5, 5)

    if func_type == 'sine':
        amp = np.random.uniform(0.1, 5.0)
        phase = np.random.uniform(0, np.pi)
        y_support = amp * torch.sin(x_support + phase)
        y_query = amp * torch.sin(x_query + phase)

    elif func_type == 'linear':
        slope = np.random.uniform(-3, 3)
        intercept = np.random.uniform(-5, 5)
        y_support = slope * x_support + intercept
        y_query = slope * x_query + intercept

    elif func_type == 'quadratic':
        a = np.random.uniform(-1, 1)
        b = np.random.uniform(-2, 2)
        c = np.random.uniform(-5, 5)
        y_support = a * x_support**2 + b * x_support + c
        y_query = a * x_query**2 + b * x_query + c

    else:
        raise ValueError(f"Tipo de función desconocido: {func_type}")

    return {
        'x_support': x_support,
        'y_support': y_support,
        'x_query': x_query,
        'y_query': y_query
    }


def split_support_query(
    x: torch.Tensor,
    y: torch.Tensor,
    k_shot: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Divide datos en conjunto de soporte y consulta

    Args:
        x: Features [N, ...]
        y: Labels [N]
        k_shot: Número de ejemplos de soporte por clase

    Returns:
        x_support, y_support, x_query, y_query
    """
    unique_classes = torch.unique(y)

    x_support_list = []
    y_support_list = []
    x_query_list = []
    y_query_list = []

    for cls in unique_classes:
        cls_mask = y == cls
        cls_x = x[cls_mask]
        cls_y = y[cls_mask]

        # Shuffle
        perm = torch.randperm(len(cls_x))
        cls_x = cls_x[perm]
        cls_y = cls_y[perm]

        # Split
        x_support_list.append(cls_x[:k_shot])
        y_support_list.append(cls_y[:k_shot])
        x_query_list.append(cls_x[k_shot:])
        y_query_list.append(cls_y[k_shot:])

    x_support = torch.cat(x_support_list, dim=0)
    y_support = torch.cat(y_support_list, dim=0)
    x_query = torch.cat(x_query_list, dim=0)
    y_query = torch.cat(y_query_list, dim=0)

    return x_support, y_support, x_query, y_query


def set_seed(seed: int = 42):
    """
    Establece la semilla para reproducibilidad

    Args:
        seed: Valor de la semilla
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
