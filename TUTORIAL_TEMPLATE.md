# 📚 Plantilla Estándar para Tutoriales Expandidos

**Basado en**: Coursera Deep Learning Specialization (W2A1 como referencia)
**Objetivo**: Tutoriales completos de 1500-2500 líneas con nivel profesional

---

## 🎯 Estructura Estándar de Cada Tutorial

### 1. **Header y Objetivos** (Celda Markdown)

```markdown
# 🎓 Tutorial XX: [Título Descriptivo]

## [Subtítulo que explica el concepto]

[Párrafo introductorio de 2-3 líneas explicando qué aprenderás]

By the end of this tutorial, you'll be able to:

- ✅ [Objetivo específico 1]
- ✅ [Objetivo específico 2]
- ✅ [Objetivo específico 3]
- ✅ [Objetivo específico 4]
- ✅ [Objetivo específico 5]

[Imagen conceptual o diagrama]

<caption><center> <u><b>Figure 1</b></u>: [Descripción de la imagen] </center></caption>

**Important Note on Submission**: [Si hay autograder, instrucciones]
```

---

### 2. **Tabla de Contenidos** (Celda Markdown)

```markdown
## Table of Contents
- [1 - Packages](#1)
- [2 - Theoretical Background](#2)
    - [2.1 - Conceptual Introduction](#2-1)
    - [2.2 - Mathematical Formulation](#2-2)
    - [2.3 - Comparison with Other Methods](#2-3)
- [3 - Dataset and Setup](#3)
- [4 - Exercise 1 - [Nombre]](#ex-1)
- [5 - Exercise 2 - [Nombre]](#ex-2)
- [6 - Exercise 3 - [Nombre]](#ex-3)
- [7 - Complete Implementation](#7)
- [8 - Training and Evaluation](#8)
- [9 - Experiments and Comparisons](#9)
    - [9.1 - Baseline](#9-1)
    - [9.2 - With Optimization](#9-2)
    - [9.3 - Comparison](#9-3)
- [10 - Visualizations](#10)
- [11 - Summary](#11)
```

---

### 3. **Imports y Setup** (Celda Código)

```python
# Versión del tutorial
### v1.0

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import sys
sys.path.append('..')

from utils.test_utils import print_success, print_hint, HintSystem, run_test
from utils.data_utils import create_sine_task, set_seed
from utils.visualization import plot_learning_curves, plot_embeddings

# Configuración de matplotlib
%matplotlib inline
plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Autoreload para desarrollo
%load_ext autoreload
%autoreload 2

# Reproducibilidad
set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Setup completo!")
print(f"   Device: {device}")
print(f"   PyTorch: {torch.__version__}")
```

---

### 4. **Sección Teórica Expandida** (3-5 Celdas Markdown)

#### 4.1 - Introducción Conceptual

```markdown
<a name='2'></a>
## 2 - Theoretical Background

<a name='2-1'></a>
### 2.1 - Conceptual Introduction

[Explicación en lenguaje simple con analogías]

**The Core Idea:**

> [Quote destacando la idea principal]

**Intuition:**

[Ejemplo concreto con emojis o diagramas]

<img src="images/concept.png" style="width:600px;height:300px;">
<caption><center> <u><b>Figure 2</b></u>: [Descripción] </center></caption>
```

#### 4.2 - Formulación Matemática

```markdown
<a name='2-2'></a>
### 2.2 - Mathematical Formulation

Formally, [nombre del método] works as follows:

**Step 1:** [Descripción del paso 1]

$$[Ecuación LaTeX]$$

**Step 2:** [Descripción del paso 2]

$$[Ecuación LaTeX]$$

**Step 3:** [Descripción del paso 3]

$$[Ecuación LaTeX]$$

where:
- $\alpha$ is [descripción]
- $\theta$ represents [descripción]
- $\mathcal{L}$ is [descripción]
```

#### 4.3 - Comparación con Otros Métodos

```markdown
<a name='2-3'></a>
### 2.3 - Comparison with Other Methods

<table>
<tr>
    <td><b>Aspect</b></td>
    <td><b>Method A</b></td>
    <td><b>Method B (Ours)</b></td>
    <td><b>Method C</b></td>
</tr>
<tr>
    <td>Complexity</td>
    <td>O(N²)</td>
    <td>O(N)</td>
    <td>O(N log N)</td>
</tr>
<tr>
    <td>Memory</td>
    <td>High</td>
    <td>Low</td>
    <td>Medium</td>
</tr>
<tr>
    <td>Performance</td>
    <td>Good</td>
    <td>Better</td>
    <td>Best</td>
</tr>
</table>
```

---

### 5. **Ejercicios (3-5 por tutorial)**

Cada ejercicio debe seguir este patrón:

```markdown
<a name='ex-1'></a>
### Exercise 1 - [Nombre de la función]

[Explicación de qué debe implementar el estudiante]

**Instructions:**
1. [Paso 1 detallado]
2. [Paso 2 detallado]
3. [Paso 3 detallado]

**Mathematical Reference:**
$$[Ecuación que deben implementar]$$

**Hints:**
- [Hint sobre estructura]
- [Hint sobre funciones útiles]
```

```python
# GRADED FUNCTION: nombre_funcion

def nombre_funcion(param1, param2):
    """
    [Docstring detallado]

    Arguments:
    param1 -- [descripción], shape (dimensions)
    param2 -- [descripción], shape (dimensions)

    Returns:
    output -- [descripción], shape (dimensions)
    """

    # TODO: Implementa [paso 1]
    # (approx. 1-2 lines)
    # variable_name = ...
    # YOUR CODE STARTS HERE


    # YOUR CODE ENDS HERE

    # TODO: Implementa [paso 2]
    # (approx. 2-3 lines)
    # YOUR CODE STARTS HERE


    # YOUR CODE ENDS HERE

    return output

# Sistema de hints
hints = HintSystem([
    "Hint 1: Usa np.dot() para multiplicación matricial",
    "Hint 2: No olvides normalizar con np.sum()",
    "Hint 3: La shape final debe ser (m, n)"
])

# Mostrar hints si es necesario
# hints.show_hint()
```

```python
# Test automatizado
params = test_function_case()
result = nombre_funcion(params['param1'], params['param2'])

# Verificaciones detalladas
assert result.shape == (expected_shape), f"Wrong shape: {result.shape} != {expected_shape}"
assert np.allclose(result[0, 0], expected_value), f"Wrong value at [0,0]"
assert np.all(result >= 0), "All values should be non-negative"

print_success("✅ Exercise 1 passed! Great job!")

# Test adicional si existe
test_nombre_funcion(nombre_funcion)
```

---

### 6. **Training Loop Completo**

```python
def train_model(model, data_loader, optimizer, criterion, num_epochs=1000):
    """
    [Docstring completo]
    """
    losses = []
    accuracies = []

    model.train()

    for epoch in tqdm(range(num_epochs), desc="Training"):
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0

        for batch_x, batch_y in data_loader:
            # Move to device
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Forward
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            with torch.no_grad():
                predictions = outputs.argmax(dim=1)
                acc = (predictions == batch_y).float().mean().item()

            epoch_loss += loss.item()
            epoch_acc += acc
            num_batches += 1

        # Average metrics
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches

        losses.append(avg_loss)
        accuracies.append(avg_acc)

        # Print progress
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")

    return losses, accuracies
```

---

### 7. **Experimentos y Comparaciones**

```markdown
<a name='9'></a>
## 9 - Experiments and Comparisons

Now let's compare different configurations and see which one works best!

<a name='9-1'></a>
### 9.1 - Baseline (Method A)
```

```python
# Train baseline
print("🔵 Training Baseline...\n")
baseline_params = model(X_train, Y_train, optimizer="method_a")
baseline_acc = evaluate(X_test, Y_test, baseline_params)

# Visualize
plot_decision_boundary(baseline_params, X_test, Y_test)
plt.title("Baseline Method A")
plt.show()

print(f"Baseline Accuracy: {baseline_acc:.2%}")
```

```markdown
<a name='9-2'></a>
### 9.2 - Improved Method (Method B)
```

```python
# Train improved
print("🟢 Training Improved Method...\n")
improved_params = model(X_train, Y_train, optimizer="method_b")
improved_acc = evaluate(X_test, Y_test, improved_params)

# Visualize
plot_decision_boundary(improved_params, X_test, Y_test)
plt.title("Improved Method B")
plt.show()

print(f"Improved Accuracy: {improved_acc:.2%}")
```

```markdown
<a name='9-3'></a>
### 9.3 - Comparison

<table>
<tr>
    <td><b>Method</b></td>
    <td><b>Accuracy</b></td>
    <td><b>Training Time</b></td>
    <td><b>Convergence Speed</b></td>
</tr>
<tr>
    <td>Baseline (A)</td>
    <td>85%</td>
    <td>Fast</td>
    <td>Slow</td>
</tr>
<tr>
    <td>Improved (B)</td>
    <td>94%</td>
    <td>Medium</td>
    <td>Fast</td>
</tr>
</table>

**Key Observations:**
- ✅ Method B achieves 9% higher accuracy
- ✅ Converges faster despite slightly longer per-iteration time
- ✅ More stable training (less oscillation)
```

---

### 8. **Visualizaciones Múltiples**

```python
# Visualización 1: Learning Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Loss curve
ax1.plot(losses, alpha=0.3, label='Raw')
ax1.plot(np.convolve(losses, np.ones(50)/50, mode='valid'),
         linewidth=2, label='Smoothed')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Accuracy curve
ax2.plot(accuracies, alpha=0.3, label='Raw')
ax2.plot(np.convolve(accuracies, np.ones(50)/50, mode='valid'),
         linewidth=2, label='Smoothed')
ax2.axhline(y=1/n_classes, color='r', linestyle='--', label='Random Baseline')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Accuracy')
ax2.set_title('Training Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

```python
# Visualización 2: Embeddings con t-SNE
from sklearn.manifold import TSNE

# Extract embeddings
model.eval()
with torch.no_grad():
    embeddings = model.encoder(X_test.to(device)).cpu().numpy()

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                     c=Y_test, cmap='viridis', alpha=0.6)
plt.colorbar(scatter)
plt.title('t-SNE Visualization of Embeddings')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.grid(True, alpha=0.3)
plt.show()
```

```python
# Visualización 3: Ejemplos de predicciones
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i, ax in enumerate(axes.flat):
    idx = np.random.randint(len(X_test))

    # Plot image
    ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')

    # Predict
    pred = model.predict(X_test[idx:idx+1])
    true = Y_test[idx]

    # Title with color coding
    color = 'green' if pred == true else 'red'
    ax.set_title(f'Pred: {pred}, True: {true}', color=color)
    ax.axis('off')

plt.tight_layout()
plt.show()
```

---

### 9. **Resumen y Conclusiones**

```markdown
<a name='11'></a>
## 11 - Summary

<font color='blue'>

**What you should remember:**
- ✅ [Punto clave 1]
- ✅ [Punto clave 2]
- ✅ [Punto clave 3]
- ✅ [Punto clave 4]

</font>

### When to use this method:

**✅ Good for:**
- [Caso de uso 1]
- [Caso de uso 2]
- [Caso de uso 3]

**❌ Not recommended for:**
- [Caso 1]
- [Caso 2]

### Performance Summary:

[Tabla o bullets con métricas finales]

### Next Steps:

Now that you've mastered [tema], you're ready to move on to [siguiente tema]!

**References:**
- Original Paper: [Link]
- PyTorch Docs: [Link]
- Additional Resources: [Link]

---

## 🎉 Congratulations!

You've successfully completed Tutorial [XX] on [Tema]!

**You now know how to:**
- [Skill 1]
- [Skill 2]
- [Skill 3]

Keep up the great work! 🚀
```

---

## 📏 Métricas de Calidad por Tutorial

Cada tutorial expandido debe cumplir:

| Métrica | Mínimo | Ideal |
|---------|--------|-------|
| **Líneas totales** | 1,200 | 1,500-2,500 |
| **Celdas** | 50 | 70-100 |
| **Ejercicios con TODOs** | 3 | 4-6 |
| **Tests automatizados** | 3 | 4-6 |
| **Ecuaciones LaTeX** | 5 | 8-12 |
| **Imágenes/Diagramas** | 3 | 5-8 |
| **Visualizaciones de resultados** | 3 | 5-7 |
| **Comparaciones experimentales** | 2 | 3-4 |
| **Secciones principales** | 8 | 10-12 |

---

## 🎨 Elementos Visuales Requeridos

Cada tutorial debe incluir:

1. **Diagrama conceptual** del algoritmo
2. **Gráfica de loss/accuracy** durante training
3. **Comparación visual** con baseline (side by side)
4. **t-SNE o PCA** de embeddings (si aplica)
5. **Decision boundaries** o predicciones visuales
6. **Tabla comparativa** con otros métodos
7. **Ejemplos de datos** del dataset usado

---

## ✅ Checklist de Completitud

Antes de considerar un tutorial "completo", verificar:

- [ ] Tiene tabla de contenidos con anchors
- [ ] Teoría explicada en 3 niveles: conceptual, matemático, intuitivo
- [ ] Al menos 3 ejercicios implementados
- [ ] Todos los ejercicios tienen tests automatizados
- [ ] Sistema de hints funcional
- [ ] Training loop completo con progress bar
- [ ] Al menos 2 experimentos/comparaciones
- [ ] Mínimo 3 visualizaciones diferentes
- [ ] Sección "What you should remember"
- [ ] Referencias a papers originales
- [ ] Código ejecuta sin errores
- [ ] Output de celdas guardado (para referencia)
- [ ] Comentarios explicativos en código complejo
- [ ] Docstrings completos en todas las funciones

---

## 📝 Notas de Implementación

### Para mantener consistencia:

1. **Nomenclatura**:
   - Funciones: `snake_case`
   - Clases: `PascalCase`
   - Constantes: `UPPER_CASE`
   - Variables privadas: `_leading_underscore`

2. **Estilo de código**:
   - Máximo 88 caracteres por línea
   - 2 líneas en blanco entre funciones
   - Docstrings formato Google

3. **Comentarios**:
   - `# TODO:` para ejercicios del estudiante
   - `# YOUR CODE STARTS HERE` y `# YOUR CODE ENDS HERE`
   - Comentarios explicativos arriba del código, no al lado

4. **Mensajes de salida**:
   - `✅` para éxito
   - `❌` para error
   - `⚠️` para advertencia
   - `💡` para hints
   - `🔵`/`🟢`/`🟡` para diferentes métodos en comparaciones

---

## 🚀 Orden de Implementación Recomendado

1. **Tutorial 03** - Prototypical Networks (más fundamental)
2. **Tutorial 03b** - Datasets Reales (necesario para aplicar)
3. **Tutorial 04** - MAML (algoritmo importante)
4. **Tutorial 03c** - Matching Networks (comparación métrica)
5. **Tutorial 01** - Introducción (overview general)
6. **Tutorial 02** - Curvas de Aprendizaje (fundamentos)
7. **Tutorial 06** - Meta-RL (aplicación importante)
8. **Tutorial 05** - Meta-Learning con Memoria (más avanzado)
9. **Tutorial 07** - Skill Discovery (aplicación específica)
10. **Tutorial 08** - OOD Generalization (robustez)
11. **Tutorial 09** - Proyecto Final (integrador)
12. **Tutorial 10** (opcional) - Temas avanzados

Este orden asegura que los tutoriales fundamentales estén listos primero.
