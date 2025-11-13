# 🔧 Troubleshooting Guide

Esta guía te ayudará a resolver los problemas más comunes al trabajar con los tutoriales.

---

## 📦 Problemas de Instalación

### ❌ "ModuleNotFoundError: No module named 'torch'"

**Problema**: PyTorch no está instalado.

**Solución**:
```bash
pip install torch torchvision torchaudio
```

Para GPU con CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

### ❌ "ModuleNotFoundError: No module named 'higher'"

**Problema**: La librería `higher` no está instalada (necesaria para MAML).

**Solución**:
```bash
pip install higher
```

**Alternativa**: Si no puedes instalar `higher`, puedes usar la versión simplificada de MAML que no requiere esta librería.

---

### ❌ "jupyter: command not found"

**Problema**: Jupyter no está instalado.

**Solución**:
```bash
pip install jupyter notebook
```

---

## 💾 Problemas con Datasets

### ❌ "RuntimeError: Dataset not found..."

**Problema**: Omniglot no se descargó correctamente.

**Solución**:
1. Verifica tu conexión a internet
2. Intenta descargar manualmente:
```python
import torchvision

# Forzar descarga
torchvision.datasets.Omniglot(
    root='./datasets',
    background=True,
    download=True
)
```

3. Si sigue fallando, descarga manualmente desde: https://github.com/brendenlake/omniglot

---

### ❌ "Disk space error" al descargar datasets

**Problema**: No hay suficiente espacio en disco.

**Solución**:
- Omniglot: ~32 MB
- Mini-ImageNet: ~6 GB

Libera espacio o cambia el directorio de descarga:
```python
omniglot_train = torchvision.datasets.Omniglot(
    root='/path/to/large/disk',  # Cambia la ruta
    background=True,
    download=True
)
```

---

## 🧠 Problemas de Memoria

### ❌ "CUDA out of memory"

**Problema**: La GPU no tiene suficiente memoria.

**Soluciones**:

1. **Reducir batch size**:
```python
# En lugar de:
sampler = OmniglotNWayKShot(dataset, n_way=20, k_shot=5, q_query=50)

# Usa:
sampler = OmniglotNWayKShot(dataset, n_way=5, k_shot=1, q_query=15)
```

2. **Usar CPU en lugar de GPU**:
```python
device = torch.device('cpu')  # Forzar CPU
```

3. **Limpiar cache de CUDA**:
```python
import torch
torch.cuda.empty_cache()
```

4. **Gradient accumulation** (para entrenamiento):
```python
# Acumular gradientes cada N pasos
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

### ❌ "RuntimeError: [enforce fail at alloc_cpu.cpp:...] out of memory"

**Problema**: Se acabó la RAM del sistema.

**Soluciones**:
1. Reducir tamaño de episodios
2. No cargar todo el dataset en memoria a la vez
3. Usar `num_workers=0` en DataLoader

---

## 🐛 Errores Comunes en Código

### ❌ "RuntimeError: mat1 and mat2 shapes cannot be multiplied"

**Problema**: Dimensiones incompatibles en operaciones matriciales.

**Debugging**:
```python
# Agrega prints para verificar shapes
print(f"support_x shape: {support_x.shape}")
print(f"support_embeddings shape: {support_embeddings.shape}")
print(f"prototypes shape: {prototypes.shape}")
```

**Causas comunes**:
- Olvidaste `Flatten()` después de capas convolucionales
- Input no tiene la dimensión de batch
- Transpusiste incorrectamente

**Solución**: Verifica que las dimensiones coincidan:
```python
# Si x es [batch, features] y weight es [features, output]
output = torch.matmul(x, weight.T)  # Nota la transpuesta
```

---

### ❌ "IndexError: Target is out of bounds"

**Problema**: Las etiquetas están fuera del rango esperado.

**Debugging**:
```python
print(f"Labels: {support_y}")
print(f"Unique labels: {torch.unique(support_y)}")
print(f"n_way: {n_way}")
print(f"Max label: {support_y.max()}, should be < {n_way}")
```

**Solución**: Asegúrate de que las labels estén re-mapeadas a 0..n_way-1:
```python
# Re-mapear labels
for new_label, class_id in enumerate(selected_classes):
    # Usa new_label (0, 1, 2...) en lugar de class_id original
    support_y.append(new_label)
```

---

### ❌ "ValueError: Expected input batch_size to match target batch_size"

**Problema**: Batch sizes de input y target no coinciden.

**Solución**:
```python
# Verifica shapes
print(f"logits shape: {logits.shape}")  # Debe ser [batch, n_classes]
print(f"labels shape: {labels.shape}")   # Debe ser [batch]

# Si labels tiene dimensión extra:
labels = labels.squeeze()  # Remove extra dimensions
```

---

## ✅ Problemas con Tests

### ❌ "AssertionError" en tests pero el código parece correcto

**Debugging Step-by-Step**:

1. **Verifica el error exacto**:
```python
try:
    run_test(test_function, "Test Name")
except AssertionError as e:
    print(f"Error detail: {e}")
    import traceback
    traceback.print_exc()
```

2. **Ejecuta el test manualmente**:
```python
# En lugar de usar run_test(), ejecuta directamente
model = PrototypicalNetwork()
x = torch.randn(5, 1, 28, 28)
output = model.encoder(x)
print(f"Output shape: {output.shape}")  # Debugging
```

3. **Compara con soluciones**:
```bash
# Revisa solutions/SOLUTIONS.md
# Compara tu implementación línea por línea
```

---

### ❌ Test pasa pero la visualización se ve mal

**Posibles causas**:
- Modelo no está entrenado (inicialización aleatoria)
- Learning rate demasiado alto/bajo
- No suficientes iteraciones de entrenamiento

**Solución**:
```python
# Verifica que el modelo esté entrenando
losses = train_model(model, ...)
plt.plot(losses)
plt.show()

# El loss debe bajar consistentemente
# Si no baja, ajusta hyperparámetros
```

---

## 🔄 Problemas de Entrenamiento

### ❌ "Loss no baja" / "Accuracy se queda en random"

**Checklist**:

1. ✅ **¿Llamaste a `optimizer.zero_grad()`?**
```python
for episode in range(n_episodes):
    optimizer.zero_grad()  # ← IMPORTANTE!
    loss = ...
    loss.backward()
    optimizer.step()
```

2. ✅ **¿El loss se calcula correctamente?**
```python
# Verifica que loss no sea NaN o Inf
print(f"Loss: {loss.item()}")
assert not torch.isnan(loss), "Loss is NaN!"
```

3. ✅ **¿Learning rate apropiado?**
```python
# Prueba valores diferentes
lr_values = [0.1, 0.01, 0.001, 0.0001]
```

4. ✅ **¿Datos normalizados?**
```python
# Para imágenes, usar ToTensor() normaliza a [0, 1]
transform = transforms.Compose([
    transforms.ToTensor(),  # Normaliza a [0, 1]
])
```

---

### ❌ "Loss explota" / "Loss = NaN"

**Causas**:
- Learning rate demasiado alto
- Gradientes explotan
- División por cero

**Soluciones**:
```python
# 1. Reducir learning rate
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Más bajo

# 2. Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 3. Verificar divisiones
denominator = denominator + 1e-8  # Evitar división por cero
```

---

## 📊 Problemas con Visualizaciones

### ❌ "Matplotlib plots no se muestran"

**Solución**:
```python
import matplotlib.pyplot as plt
%matplotlib inline  # En Jupyter

# O al final de cada plot:
plt.show()
```

---

### ❌ "Imágenes se ven muy oscuras/claras"

**Problema**: Rango de valores incorrecto.

**Solución**:
```python
# Para imágenes [0, 1]:
plt.imshow(img.squeeze(), cmap='gray', vmin=0, vmax=1)

# Para imágenes normalizadas [-1, 1]:
img_display = (img + 1) / 2  # Convertir a [0, 1]
plt.imshow(img_display.squeeze(), cmap='gray')
```

---

## 🚀 Optimización y Performance

### ❌ "Entrenamiento muy lento"

**Soluciones**:

1. **Usar GPU**:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device)
```

2. **Reducir resolución de imágenes**:
```python
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # En lugar de 84x84
    transforms.ToTensor(),
])
```

3. **Usar DataLoader con workers**:
```python
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # Parallel loading
    pin_memory=True  # Faster GPU transfer
)
```

4. **Mixed precision training**:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data in dataloader:
    with autocast():
        output = model(data)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## 🔍 Debugging General

### Técnicas Útiles:

1. **Print shapes everywhere**:
```python
def forward(self, x):
    print(f"Input: {x.shape}")
    x = self.conv1(x)
    print(f"After conv1: {x.shape}")
    x = self.pool(x)
    print(f"After pool: {x.shape}")
    return x
```

2. **Usa `ipdb` para debugging interactivo**:
```python
import ipdb

# En el punto que quieres inspeccionar:
ipdb.set_trace()  # El código se pausará aquí
```

3. **Verifica gradientes**:
```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm().item()}")
```

4. **Guarda checkpoints frecuentes**:
```python
# Guardar cada 100 episodios
if episode % 100 == 0:
    torch.save({
        'episode': episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, f'checkpoint_ep{episode}.pth')
```

---

## 📚 Recursos Adicionales

Si el problema persiste:

1. **Foros y Comunidades**:
   - PyTorch Forums: https://discuss.pytorch.org/
   - Stack Overflow: Tag `pytorch`, `meta-learning`
   - Reddit: r/MachineLearning

2. **Documentación**:
   - PyTorch Docs: https://pytorch.org/docs/
   - PyTorch Tutorials: https://pytorch.org/tutorials/

3. **Abre un Issue**:
   - Si encuentras un bug en los tutoriales, abre un issue en el repositorio con:
     - Código que reproduce el problema
     - Error completo
     - Versiones de las librerías (`pip list`)

---

## ✉️ ¿Aún Tienes Problemas?

Si ninguna de estas soluciones funciona:

1. Verifica que tienes las versiones correctas:
```bash
python --version  # 3.8+
pip list | grep torch  # 2.0+
```

2. Intenta en un entorno limpio:
```bash
python -m venv fresh_env
source fresh_env/bin/activate
pip install -r requirements.txt
```

3. Consulta las soluciones completas en `solutions/SOLUTIONS.md`

---

**¡Suerte con tu aprendizaje! 🚀**
