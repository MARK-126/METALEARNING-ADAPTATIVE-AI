"""
Utilidades para visualización en tutoriales de Meta-Learning
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches


# Configurar estilo por defecto
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_learning_curves(
    curves: Dict[str, List[float]],
    title: str = "Curvas de Aprendizaje",
    xlabel: str = "Iteraciones",
    ylabel: str = "Loss",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
):
    """
    Grafica múltiples curvas de aprendizaje para comparación

    Args:
        curves: Diccionario {nombre: lista_de_valores}
        title: Título del gráfico
        xlabel: Etiqueta del eje X
        ylabel: Etiqueta del eje Y
        figsize: Tamaño de la figura
        save_path: Ruta para guardar la figura (opcional)
    """
    plt.figure(figsize=figsize)

    for name, values in curves.items():
        plt.plot(values, label=name, linewidth=2)

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_few_shot_results(
    x_support: np.ndarray,
    y_support: np.ndarray,
    x_query: np.ndarray,
    y_query: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    title: str = "Few-Shot Learning Task",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
):
    """
    Visualiza una tarea de Few-Shot Learning con predicciones

    Args:
        x_support: Puntos de soporte [N, 1]
        y_support: Valores de soporte [N, 1]
        x_query: Puntos de consulta [M, 1]
        y_query: Valores verdaderos de consulta [M, 1]
        y_pred: Predicciones (opcional) [M, 1]
        title: Título del gráfico
        figsize: Tamaño de la figura
        save_path: Ruta para guardar
    """
    plt.figure(figsize=figsize)

    # Convertir a numpy si es necesario
    if isinstance(x_support, torch.Tensor):
        x_support = x_support.detach().cpu().numpy()
    if isinstance(y_support, torch.Tensor):
        y_support = y_support.detach().cpu().numpy()
    if isinstance(x_query, torch.Tensor):
        x_query = x_query.detach().cpu().numpy()
    if isinstance(y_query, torch.Tensor):
        y_query = y_query.detach().cpu().numpy()
    if y_pred is not None and isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Plot soporte
    plt.scatter(x_support, y_support, c='blue', s=100, marker='o',
                label='Support Set', edgecolors='black', linewidth=1.5, zorder=3)

    # Plot consulta verdadera
    plt.scatter(x_query, y_query, c='green', s=80, marker='s',
                label='True Query', alpha=0.6, edgecolors='black', linewidth=1, zorder=2)

    # Plot predicciones si están disponibles
    if y_pred is not None:
        plt.scatter(x_query, y_pred, c='red', s=80, marker='^',
                    label='Predicted Query', alpha=0.7, edgecolors='black', linewidth=1, zorder=2)

        # Líneas conectando predicción con verdadero valor
        for i in range(len(x_query)):
            plt.plot([x_query[i], x_query[i]], [y_query[i], y_pred[i]],
                     'r--', alpha=0.3, linewidth=1)

    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str = "Embedding Space",
    method: str = 'tsne',
    figsize: Tuple[int, int] = (10, 10),
    save_path: Optional[str] = None
):
    """
    Visualiza embeddings en 2D usando t-SNE o PCA

    Args:
        embeddings: Embeddings [N, D]
        labels: Etiquetas [N]
        title: Título
        method: 'tsne' o 'pca'
        figsize: Tamaño de la figura
        save_path: Ruta para guardar
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    # Convertir a numpy si es necesario
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Reducir dimensionalidad
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    else:
        raise ValueError(f"Método desconocido: {method}")

    embeddings_2d = reducer.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=figsize)

    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        mask = labels == label
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                    c=[color], label=f'Class {label}', s=50, alpha=0.7,
                    edgecolors='black', linewidth=0.5)

    plt.xlabel(f'{method.upper()} Component 1', fontsize=12)
    plt.ylabel(f'{method.upper()} Component 2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_meta_learning_comparison(
    traditional_ml: List[float],
    transfer_learning: List[float],
    meta_learning: List[float],
    title: str = "Comparación: ML Tradicional vs Transfer Learning vs Meta-Learning",
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
):
    """
    Compara curvas de aprendizaje de diferentes paradigmas

    Args:
        traditional_ml: Loss de ML tradicional
        transfer_learning: Loss de Transfer Learning
        meta_learning: Loss de Meta-Learning
        title: Título
        figsize: Tamaño
        save_path: Ruta para guardar
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Gráfico 1: Curvas completas
    ax1.plot(traditional_ml, label='ML Tradicional', linewidth=2, color='red')
    ax1.plot(transfer_learning, label='Transfer Learning', linewidth=2, color='orange')
    ax1.plot(meta_learning, label='Meta-Learning', linewidth=2, color='green')
    ax1.set_xlabel('Iteraciones', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Curvas de Aprendizaje Completas', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Gráfico 2: Zoom en las primeras iteraciones
    zoom_range = min(50, len(traditional_ml))
    ax2.plot(traditional_ml[:zoom_range], label='ML Tradicional', linewidth=2, color='red')
    ax2.plot(transfer_learning[:zoom_range], label='Transfer Learning', linewidth=2, color='orange')
    ax2.plot(meta_learning[:zoom_range], label='Meta-Learning', linewidth=2, color='green')
    ax2.set_xlabel('Iteraciones', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title(f'Zoom: Primeras {zoom_range} Iteraciones', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_task_distribution(
    tasks: List[Dict],
    title: str = "Distribución de Tareas",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
):
    """
    Visualiza la distribución de múltiples tareas

    Args:
        tasks: Lista de diccionarios de tareas
        title: Título
        figsize: Tamaño
        save_path: Ruta para guardar
    """
    n_tasks = len(tasks)
    cols = 4
    rows = (n_tasks + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if n_tasks > 1 else [axes]

    for idx, task in enumerate(tasks):
        ax = axes[idx]

        x_support = task['x_support']
        y_support = task['y_support']

        if isinstance(x_support, torch.Tensor):
            x_support = x_support.detach().cpu().numpy()
        if isinstance(y_support, torch.Tensor):
            y_support = y_support.detach().cpu().numpy()

        ax.scatter(x_support, y_support, c='blue', s=50, alpha=0.6)
        ax.set_title(f'Task {idx + 1}', fontsize=10)
        ax.grid(True, alpha=0.3)

    # Ocultar ejes sobrantes
    for idx in range(n_tasks, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Matriz de Confusión",
    figsize: Tuple[int, int] = (8, 7),
    save_path: Optional[str] = None
):
    """
    Visualiza matriz de confusión

    Args:
        y_true: Etiquetas verdaderas
        y_pred: Etiquetas predichas
        class_names: Nombres de las clases
        title: Título
        figsize: Tamaño
        save_path: Ruta para guardar
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=class_names, yticklabels=class_names)

    plt.xlabel('Predicción', fontsize=12)
    plt.ylabel('Verdadero', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_gradient_flow(named_parameters, save_path: Optional[str] = None):
    """
    Visualiza el flujo de gradientes en la red

    Args:
        named_parameters: model.named_parameters()
        save_path: Ruta para guardar
    """
    ave_grads = []
    max_grads = []
    layers = []

    for n, p in named_parameters:
        if p.requires_grad and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu().item())
            max_grads.append(p.grad.abs().max().cpu().item())

    plt.figure(figsize=(14, 6))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color="c", label="Max Gradient")
    plt.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.5, lw=1, color="b", label="Mean Gradient")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001)
    plt.xlabel("Capas", fontsize=12)
    plt.ylabel("Gradiente Promedio", fontsize=12)
    plt.title("Flujo de Gradientes", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
