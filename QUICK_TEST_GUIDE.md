# ⚡ Guía Rápida de Verificación

**Tiempo estimado**: 30-45 minutos
**Propósito**: Verificar que las soluciones funcionan correctamente

---

## 🚀 Opción 1: Test Rápido (10 minutos)

### Paso 1: Instalar dependencias mínimas

```bash
pip install torch torchvision numpy matplotlib tqdm
```

### Paso 2: Ejecutar script de tests unitarios

```bash
cd /ruta/a/METALEARNING-ADAPTATIVE-AI
python test_solutions.py
```

**✅ Resultado esperado**:
```
============================================================
  Verificación de Soluciones - Meta-Learning Tutorials
============================================================

Test 1: Importaciones
  ✓ PASS torch
  ✓ PASS torchvision
  ✓ PASS numpy
  ✓ PASS matplotlib
  ✓ PASS tqdm

Test 2: OmniglotNWayKShot Sampler (lógica)
  ✓ PASS _organize_by_class crea 10 clases
  ✓ PASS Cada clase tiene 20 ejemplos
  ✓ PASS Support shape correcto (5x1)
  ✓ PASS Query shape correcto (5x15)
  ✓ PASS 5 clases únicas en support
  ✓ PASS 5 clases únicas en query
  ✓ PASS Labels van de 0 a 4

Test 3: Matching Networks (cosine similarity)
  ✓ PASS Cosine similarity shape (10x5)
  ✓ PASS Valores entre -1 y 1
  ✓ PASS Attention shape (10x5)
  ✓ PASS Attention suma a 1

Test 4: OOD Transformations
  ✓ PASS Rotate preserva shape
  ✓ PASS Noise preserva shape
  ✓ PASS Noise valores válidos
  ✓ PASS Invert preserva shape

============================================================
  Resumen
============================================================
  ✓ Importaciones
  ✓ OmniglotNWayKShot
  ✓ Matching Networks
  ✓ OOD Transformations

  Total: 4/4 tests pasados

✓ ¡Todas las verificaciones pasaron!
```

### ⚠️ Si algún test falla:
- Copia el traceback completo
- Verifica que torch y torchvision estén instalados correctamente
- Revisa `VERIFICATION_STATUS.md` para errores conocidos

---

## 📊 Opción 2: Test Completo con Omniglot (45 minutos)

### Paso 1: Instalar todas las dependencias

```bash
pip install -r requirements.txt
```

### Paso 2: Ejecutar Tutorial 03b (Datasets Reales)

```bash
jupyter notebook 02_algoritmos_clave/03b_datasets_reales.ipynb
```

**Celdas críticas a ejecutar**:

1. **Celda de imports** (debe ejecutar sin errores)
2. **Celda de descarga de Omniglot** (descarga ~5MB):
   ```python
   omniglot_train = torchvision.datasets.Omniglot(...)
   ```
   ✅ Esperado: "Omniglot descargado! Train: 19280 imágenes"

3. **Celda de OmniglotNWayKShot** (copiar solución de SOLUTIONS.md)
4. **Celda de test del sampler**:
   ```python
   sampler = OmniglotNWayKShot(omniglot_train, n_way=5, k_shot=1, q_query=15)
   support_x, support_y, query_x, query_y = sampler.sample_episode()
   ```
   ✅ Esperado: "✅ Sampler funciona correctamente!"

### Paso 3: Ejecutar Tutorial 03c (Matching Networks)

```bash
jupyter notebook 02_algoritmos_clave/03c_matching_networks.ipynb
```

**Celdas críticas**:

1. **Test de cosine_similarity**:
   ```python
   run_test(test_cosine, "Test de Similitud Coseno")
   ```
   ✅ Esperado: "✅ Similitud coseno correcta!"

2. **Test de atención**:
   ```python
   run_test(test_attention, "Test de Atención")
   ```
   ✅ Esperado: "✅ Atención calculada correctamente!"

3. **Training de 500 episodios** (~5 minutos):
   ```python
   losses, accs = train_matching_net(matching_net, n_episodes=500)
   ```
   ✅ Esperado:
   - Loss disminuye de ~1.6 a ~0.3
   - Accuracy aumenta de ~0.2 a ~0.9+

### Paso 4: Verificar Tutorial 09 (Proyecto Final)

```bash
jupyter notebook 09_proyecto_final.ipynb
```

**Flujo de verificación**:

1. Copiar soluciones de `SOLUTIONS.md` a las celdas TODO
2. Ejecutar entrenamiento de 2000 episodios (~10 minutos)
3. Evaluar en 5-way 1-shot

✅ **Métricas esperadas** (Omniglot):
- **5-way 1-shot**: 95-98% accuracy
- **5-way 5-shot**: 98-99% accuracy
- **20-way 1-shot**: 85-95% accuracy

---

## 🔍 Checklist de Verificación

### Tests Unitarios
- [ ] `test_solutions.py` ejecuta sin errores
- [ ] Todos los tests pasan (4/4)

### Tutorial 03b (Datasets)
- [ ] Omniglot descarga correctamente
- [ ] OmniglotNWayKShot crea episodios válidos
- [ ] Shapes son correctos: support=(5, 1, 28, 28), query=(75, 1, 28, 28)
- [ ] Visualización muestra 40 caracteres diferentes

### Tutorial 03c (Matching Networks)
- [ ] cosine_similarity retorna valores entre -1 y 1
- [ ] compute_attention suma a 1 por query
- [ ] MatchingNetwork forward funciona
- [ ] Training converge (loss baja, accuracy sube)

### Tutorial 09 (Proyecto Final)
- [ ] Sampler funciona con Omniglot real
- [ ] PrototypicalNetwork se instancia sin errores
- [ ] Training loop ejecuta 2000 episodios
- [ ] Loss converge < 0.5
- [ ] Accuracy en test > 90% (5-way 1-shot)

---

## 🐛 Problemas Comunes y Soluciones

### Problema 1: "No module named 'torch'"

**Solución**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Problema 2: "RuntimeError: Expected all tensors to be on the same device"

**Causa**: Tensores en CPU cuando modelo está en GPU (o viceversa).

**Solución**: Agregar `.to(device)` a todos los tensores:
```python
support_x = support_x.to(device)
support_y = support_y.to(device)
```

### Problema 3: "RuntimeError: expected scalar type Float but found Byte"

**Causa**: Imágenes de Omniglot son uint8, no float32.

**Solución**: Ya está en los transforms:
```python
transforms.ToTensor()  # Convierte a float32 automáticamente
```

### Problema 4: "IndexError: list index out of range" en sampler

**Causa**: Clase con menos ejemplos que `k_shot + q_query`.

**Solución**: Verificar número de ejemplos:
```python
for class_id, indices in sampler.class_to_indices.items():
    if len(indices) < sampler.k_shot + sampler.q_query:
        print(f"⚠️ Clase {class_id} tiene solo {len(indices)} ejemplos")
```

### Problema 5: "CUDA out of memory"

**Soluciones**:
1. Usar CPU: `device = torch.device('cpu')`
2. Reducir batch size (implícito en n_way * k_shot)
3. Usar modelo más pequeño (embedding_dim=32 en lugar de 64)

### Problema 6: Loss no baja / Accuracy estancada en 0.2

**Posibles causas**:
- Learning rate muy alto → Prueba `lr=0.0001`
- Modelo no se entrena → Verificar `model.train()`
- Labels incorrectos → Verificar que support_y esté en range(0, n_way)

**Debug**:
```python
print(f"Model training: {model.training}")
print(f"Support labels: {support_y.unique()}")
print(f"Query labels: {query_y.unique()}")
print(f"Logits range: {logits.min():.2f} to {logits.max():.2f}")
```

---

## 📊 Métricas de Referencia

### Omniglot (desde papers)

| Escenario | Prototypical | Matching | MAML |
|-----------|-------------|----------|------|
| 5-way 1-shot | 98.8% | 98.1% | 98.7% |
| 5-way 5-shot | 99.7% | 98.9% | 99.9% |
| 20-way 1-shot | 96.0% | 95.8% | 95.8% |

### Mini-ImageNet (desde papers)

| Escenario | Prototypical | Matching | MAML |
|-----------|-------------|----------|------|
| 5-way 1-shot | 49.4% | 43.6% | 48.7% |
| 5-way 5-shot | 68.2% | 55.3% | 63.1% |

**Si tus resultados están dentro de ±5%**, ¡excelente trabajo! 🎉

---

## ✅ Criterios de Aprobación

Tu implementación es **correcta** si:

1. ✅ `test_solutions.py` pasa todos los tests
2. ✅ Tutorial 03b descarga Omniglot y crea episodios válidos
3. ✅ Tutorial 03c converge en training (accuracy final > 80%)
4. ✅ Tutorial 09 logra > 90% en Omniglot 5-way 1-shot

Tu implementación es **excelente** si:

1. ✅ Logra > 95% en Omniglot 5-way 1-shot
2. ✅ Logra > 98% en Omniglot 5-way 5-shot
3. ✅ Training converge en < 1000 episodios
4. ✅ Código funciona tanto en CPU como GPU

---

## 🚀 Siguiente Paso Después de Verificar

Una vez que todo funciona:

1. **Experimenta**: Cambia arquitecturas, hiperparámetros
2. **Benchmark**: Prueba con Mini-ImageNet
3. **Mejora**: Implementa técnicas de papers recientes
4. **Aplica**: Usa en tu dominio (medical, robótica, etc.)

---

## 📞 Si Necesitas Ayuda

Si después de seguir esta guía encuentras problemas:

1. Revisa `VERIFICATION_STATUS.md` para errores conocidos
2. Consulta `TROUBLESHOOTING.md` para debugging
3. Verifica que tu entorno cumple los requisitos:
   - Python 3.8+
   - PyTorch 2.0+
   - 8GB RAM (16GB recomendado)
   - GPU opcional (pero acelera 10x)

**Información útil para reportar bugs**:
```python
import torch
import sys

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```
