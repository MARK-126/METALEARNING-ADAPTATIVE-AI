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

## 📦 Tutorial 03b: Datasets Reales (Omniglot)

### Ejercicio: N-way K-shot Sampler

```python
class OmniglotNWayKShot:
    def __init__(self, dataset, n_way=5, k_shot=1, q_query=15):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query

        # Organizar dataset por clases
        self.class_to_indices = self._organize_by_class()
        self.classes = list(self.class_to_indices.keys())

    def _organize_by_class(self):
        """Organiza los índices del dataset por clase."""
        from collections import defaultdict

        class_to_indices = defaultdict(list)

        for idx, (img, label) in enumerate(self.dataset):
            class_to_indices[label].append(idx)

        return class_to_indices

    def sample_episode(self):
        """Genera un episodio N-way K-shot."""
        import random

        # 1. Seleccionar n_way clases aleatorias
        selected_classes = random.sample(self.classes, self.n_way)

        support_x = []
        support_y = []
        query_x = []
        query_y = []

        # 2. Para cada clase, seleccionar k_shot + q_query ejemplos
        for new_label, class_id in enumerate(selected_classes):
            # Obtener índices de esta clase
            class_indices = self.class_to_indices[class_id]

            # Seleccionar k_shot + q_query ejemplos aleatorios
            selected_indices = random.sample(
                class_indices,
                self.k_shot + self.q_query
            )

            # Los primeros k_shot van a support, el resto a query
            for i, idx in enumerate(selected_indices):
                img, _ = self.dataset[idx]

                if i < self.k_shot:
                    support_x.append(img)
                    support_y.append(new_label)
                else:
                    query_x.append(img)
                    query_y.append(new_label)

        # 3. Convertir a tensores
        support_x = torch.stack(support_x)
        support_y = torch.LongTensor(support_y)
        query_x = torch.stack(query_x)
        query_y = torch.LongTensor(query_y)

        return support_x, support_y, query_x, query_y
```

---

## 🎯 Tutorial 03c: Matching Networks

### Ejercicio 1: Similitud Coseno

```python
def cosine_similarity(x1, x2):
    """Calcula similitud coseno entre dos conjuntos de vectores."""
    # Normalizar vectores
    x1_norm = F.normalize(x1, p=2, dim=1)
    x2_norm = F.normalize(x2, p=2, dim=1)

    # Producto punto (matriz de similitudes)
    similarities = torch.matmul(x1_norm, x2_norm.T)

    return similarities
```

### Ejercicio 2: Mecanismo de Atención

```python
def compute_attention(query_embeddings, support_embeddings, temperature=1.0):
    """Calcula pesos de atención entre queries y support set."""
    # Calcular similitudes coseno
    similarities = cosine_similarity(query_embeddings, support_embeddings)

    # Aplicar temperatura y softmax
    attention = F.softmax(similarities / temperature, dim=1)

    return attention
```

### Ejercicio 3: Matching Network Forward

```python
def forward(self, support_x, support_y, query_x, n_way):
    """Forward pass usando atención."""
    # 1. Codificar support y query
    support_embeddings = self.encoder(support_x)
    query_embeddings = self.encoder(query_x)

    # 2. Calcular atención
    attention = compute_attention(query_embeddings, support_embeddings)

    # 3. Convertir support_y a one-hot
    support_one_hot = F.one_hot(support_y, n_way).float()

    # 4. Weighted vote: cada query vota por clases basándose en atención
    logits = torch.matmul(attention, support_one_hot)

    return logits
```

---

## 🧩 Tutorial 07: Skill Discovery

### Código Conceptual Completo

```python
class SkillPolicy(nn.Module):
    """Política condicionada en un skill z."""

    def __init__(self, state_dim, n_skills=4, n_actions=4, hidden=64):
        super().__init__()
        self.n_skills = n_skills

        self.fc1 = nn.Linear(state_dim + n_skills, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_actions)

    def forward(self, state, skill):
        # Concatenar estado con skill one-hot
        skill_onehot = F.one_hot(skill, self.n_skills).float()
        x = torch.cat([state, skill_onehot], dim=-1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_logits = self.fc3(x)

        return F.softmax(action_logits, dim=-1)

# Uso:
env = SimpleGridWorld(size=5)
policy = SkillPolicy(state_dim=25, n_skills=4, n_actions=4)

# Rollout con skill específico
state = env.reset()
skill = 0  # Usar skill 0

for t in range(10):
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    skill_tensor = torch.LongTensor([skill])

    with torch.no_grad():
        action_probs = policy(state_tensor, skill_tensor)
        action = torch.multinomial(action_probs, 1).item()

    state, _, _ = env.step(action)
```

---

## 🌍 Tutorial 08: Generalización OOD

### Transformaciones OOD

```python
def create_ood_mnist(mnist_dataset, transform_type='rotate'):
    """Crea versión OOD de MNIST aplicando transformaciones."""

    if transform_type == 'rotate':
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=45),
            transforms.ToTensor()
        ])

    elif transform_type == 'noise':
        def add_noise(img):
            return torch.clamp(img + torch.randn_like(img) * 0.3, 0, 1)
        transform = add_noise

    elif transform_type == 'blur':
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.GaussianBlur(kernel_size=5),
            transforms.ToTensor()
        ])

    elif transform_type == 'invert':
        def invert(img):
            return 1.0 - img
        transform = invert

    return transform


# Evaluación de robustez OOD
def evaluate_ood_robustness(model, dataset, ood_transforms):
    """Evalúa robustez del modelo bajo diferentes shifts."""
    results = {}

    for transform_name, transform_fn in ood_transforms.items():
        correct = 0
        total = 0

        for img, label in dataset:
            # Aplicar transform OOD
            img_ood = transform_fn(img)

            # Predicción
            with torch.no_grad():
                pred = model(img_ood.unsqueeze(0).to(device)).argmax().item()

            correct += (pred == label)
            total += 1

            if total >= 1000:  # Evaluar en subset
                break

        results[transform_name] = correct / total

    return results
```

---

## 🎓 Tutorial 09: Proyecto Final

### Soluciones de los TODOs Principales

#### TODO 1: Cargar Omniglot

```python
import torchvision.transforms as transforms

# Background set para training
omniglot_train = torchvision.datasets.Omniglot(
    root='../datasets',
    background=True,
    download=True,
    transform=transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
)

# Evaluation set para testing
omniglot_test = torchvision.datasets.Omniglot(
    root='../datasets',
    background=False,
    download=True,
    transform=transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
)
```

#### TODO 2: Implementar Prototypical Networks

```python
class PrototypicalNetwork(nn.Module):
    def __init__(self, input_channels=1, embedding_dim=64):
        super().__init__()

        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 4
            nn.Conv2d(64, embedding_dim, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, support_x, support_y, query_x, n_way):
        # Encode
        support_embeddings = self.encoder(support_x)
        query_embeddings = self.encoder(query_x)

        # Compute prototypes
        prototypes = []
        for c in range(n_way):
            class_mask = (support_y == c)
            class_embeddings = support_embeddings[class_mask]
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)

        prototypes = torch.stack(prototypes)

        # Compute distances
        distances = torch.cdist(query_embeddings, prototypes)

        # Convert to logits (negative distances)
        logits = -distances

        return logits
```

#### TODO 4: Training Loop

```python
def train_few_shot(model, sampler, n_episodes=1000, lr=0.001):
    """Entrena un modelo de few-shot learning."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    losses = []
    accuracies = []

    model.train()

    for episode in tqdm(range(n_episodes)):
        # Sample episodio
        support_x, support_y, query_x, query_y = sampler.sample_episode()

        # Move to device
        support_x = support_x.to(device)
        support_y = support_y.to(device)
        query_x = query_x.to(device)
        query_y = query_y.to(device)

        # Forward
        logits = model(support_x, support_y, query_x, n_way=sampler.n_way)
        loss = criterion(logits, query_y)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        acc = (logits.argmax(dim=1) == query_y).float().mean().item()

        losses.append(loss.item())
        accuracies.append(acc)

    return losses, accuracies
```

#### TODO 5: Evaluación Estándar

```python
def evaluate_few_shot(model, sampler, n_episodes=600):
    """Evalúa el modelo siguiendo el protocolo estándar."""
    model.eval()
    accuracies = []

    with torch.no_grad():
        for _ in tqdm(range(n_episodes)):
            # Sample episodio
            support_x, support_y, query_x, query_y = sampler.sample_episode()

            # Move to device
            support_x = support_x.to(device)
            support_y = support_y.to(device)
            query_x = query_x.to(device)
            query_y = query_y.to(device)

            # Forward
            logits = model(support_x, support_y, query_x, n_way=sampler.n_way)

            # Accuracy
            acc = (logits.argmax(dim=1) == query_y).float().mean().item()
            accuracies.append(acc)

    # Calcular estadísticas
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    ci95 = 1.96 * std_acc / np.sqrt(n_episodes)

    return mean_acc, ci95
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
- **Omniglot Dataset**: https://github.com/brendenlake/omniglot

---

¡Sigue practicando y construyendo! 🚀
