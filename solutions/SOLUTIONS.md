# 📝 Soluciones de los Ejercicios

Este archivo contiene las soluciones completas de todos los ejercicios de los tutoriales.

**⚠️ IMPORTANTE**: Intenta resolver los ejercicios por tu cuenta primero! Solo consulta estas soluciones si estás completamente bloqueado.

---

## 📘 Tutorial 01: Introducción al Meta-Learning

### Ejercicio 1: Modelo Simple de Regresión

```python
class SimpleRegressionModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=40, output_dim=1):
        super(SimpleRegressionModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

### Ejercicio 2: Entrenamiento Simple

```python
def train_on_task(model, task, n_steps=100, lr=0.01):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    x_support = task['x_support']
    y_support = task['y_support']

    losses = []

    for step in range(n_steps):
        optimizer.zero_grad()
        predictions = model(x_support)
        loss = criterion(predictions, y_support)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses
```

---

## 📊 Tutorial 02: Curvas de Aprendizaje

### Ejercicio 1: Función de Evaluación

```python
def evaluate_model(model, task):
    model.eval()

    x_query = task['x_query']
    y_query = task['y_query']

    criterion = nn.MSELoss()

    with torch.no_grad():
        predictions = model(x_query)
        loss = criterion(predictions, y_query)

    return loss.item()
```

### Ejercicio 2: Adaptación con Curva

```python
def adapt_model_with_curve(model, task, n_steps=50, lr=0.01):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    x_support = task['x_support']
    y_support = task['y_support']

    query_losses = []

    # Evaluación inicial
    initial_loss = evaluate_model(model, task)
    query_losses.append(initial_loss)

    for step in range(n_steps):
        model.train()

        optimizer.zero_grad()
        predictions = model(x_support)
        loss = criterion(predictions, y_support)
        loss.backward()
        optimizer.step()

        # Evaluar en query
        query_loss = evaluate_model(model, task)
        query_losses.append(query_loss)

    return query_losses
```

### Ejercicio 3: Análisis Cuantitativo

```python
def compare_paradigms(traditional, transfer, meta, steps_list=[1, 5, 10, 20, 50]):
    results = {}

    for steps in steps_list:
        trad_loss = traditional[steps]
        trans_loss = transfer[steps]
        meta_loss = meta[steps]

        transfer_improvement = (trad_loss - trans_loss) / trad_loss * 100
        meta_improvement = (trad_loss - meta_loss) / trad_loss * 100

        results[steps] = {
            'traditional': trad_loss,
            'transfer': trans_loss,
            'meta': meta_loss,
            'transfer_improvement': transfer_improvement,
            'meta_improvement': meta_improvement
        }

    return results
```

---

## 🎯 Tutorial 03: Prototypical Networks

### Ejercicio 1: Red de Embeddings

```python
class EmbeddingNetwork(nn.Module):
    def __init__(self, input_channels=1, embedding_dim=64):
        super(EmbeddingNetwork, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, embedding_dim, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Block 4
        x = self.conv4(x)
        x = F.relu(x)

        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        return x
```

### Ejercicio 2: Calcular Prototipos

```python
def compute_prototypes(embeddings, labels, n_way):
    prototypes = []

    for c in range(n_way):
        # Encontrar embeddings de la clase c
        class_mask = (labels == c)
        class_embeddings = embeddings[class_mask]

        # Calcular el prototipo (promedio)
        prototype_c = class_embeddings.mean(dim=0)
        prototypes.append(prototype_c)

    # Apilar todos los prototipos
    prototypes = torch.stack(prototypes)

    return prototypes
```

### Ejercicio 3: Distancias Euclidianas

```python
def euclidean_distances(queries, prototypes):
    # Método 1: Usando torch.cdist (más simple)
    distances = torch.cdist(queries, prototypes)

    # Método 2: Manual (para entender la implementación)
    # n_queries = queries.size(0)
    # n_prototypes = prototypes.size(0)
    # queries_expanded = queries.unsqueeze(1).expand(n_queries, n_prototypes, -1)
    # prototypes_expanded = prototypes.unsqueeze(0).expand(n_queries, n_prototypes, -1)
    # distances = (queries_expanded - prototypes_expanded).pow(2).sum(dim=2).sqrt()

    return distances
```

### Ejercicio 4: Método classify

```python
def classify(self, support_x, support_y, query_x, n_way):
    # Obtener logits
    logits = self.forward(support_x, support_y, query_x, n_way)

    # Convertir a probabilidades
    probabilities = F.softmax(logits, dim=1)

    # Predicciones
    predictions = logits.argmax(dim=1)

    return predictions, probabilities
```

---

## 🌟 Tutorial 04: MAML

### Ejercicio 1: Inner Loop

```python
def inner_loop(model, task, inner_lr=0.01, inner_steps=5):
    # Crear copia del modelo
    adapted_model = deepcopy(model)

    # Optimizador para la copia
    optimizer = optim.SGD(adapted_model.parameters(), lr=inner_lr)
    criterion = nn.MSELoss()

    x_support = task['x_support']
    y_support = task['y_support']

    # Entrenar por inner_steps
    for step in range(inner_steps):
        optimizer.zero_grad()
        predictions = adapted_model(x_support)
        loss = criterion(predictions, y_support)
        loss.backward()
        optimizer.step()

    return adapted_model
```

### Ejercicio 2: MAML Completo (Simplificado)

```python
def train_maml_simple(model, n_iterations=1000, meta_batch_size=4,
                      inner_lr=0.01, outer_lr=0.001, inner_steps=5):
    meta_optimizer = optim.Adam(model.parameters(), lr=outer_lr)
    criterion = nn.MSELoss()

    meta_losses = []

    for iteration in range(n_iterations):
        meta_loss = 0.0
        meta_optimizer.zero_grad()

        # Batch de tareas
        for _ in range(meta_batch_size):
            # Sample tarea
            task = create_sine_task(k_shot=10, q_query=10)

            # Inner loop: adaptar modelo
            adapted_model = inner_loop(model, task, inner_lr, inner_steps)

            # Evaluar en query set
            query_pred = adapted_model(task['x_query'])
            query_loss = criterion(query_pred, task['y_query'])

            meta_loss += query_loss

        # Outer loop: actualizar modelo original
        meta_loss = meta_loss / meta_batch_size
        meta_loss.backward()
        meta_optimizer.step()

        meta_losses.append(meta_loss.item())

        if (iteration + 1) % 100 == 0:
            print(f\"Iteración {iteration+1}/{n_iterations} - Meta-Loss: {meta_loss.item():.4f}\")\

    return meta_losses
```

---

## 🧠 Tutorial 05: Meta-Learning con Memoria

### Optimizador LSTM

```python
class LSTMOptimizer(nn.Module):
    def __init__(self, input_size=1, hidden_size=20):
        super(LSTMOptimizer, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, gradients, hidden=None):
        # Procesar secuencia de gradientes
        lstm_out, hidden = self.lstm(gradients, hidden)

        # Generar actualizaciones
        updates = self.fc(lstm_out)

        return updates, hidden
```

---

## 🔥 Consejos para Aprender

1. **No copies y pegues**: Escribe el código tú mismo para entender mejor
2. **Experimenta**: Modifica parámetros y observa qué pasa
3. **Debuggea**: Usa print() y visualizaciones para entender el flujo
4. **Compara**: Ejecuta tu código y el de las soluciones lado a lado
5. **Pregunta**: Si algo no tiene sentido, investiga más en la documentación

---

## 📚 Recursos Adicionales

- **PyTorch Docs**: https://pytorch.org/docs/
- **Papers with Code**: https://paperswithcode.com/task/meta-learning
- **Meta-Learning Survey**: https://arxiv.org/abs/1810.03548

---

¡Sigue practicando y construyendo! 🚀
