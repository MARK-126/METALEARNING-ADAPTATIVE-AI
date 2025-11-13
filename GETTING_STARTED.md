# 🚀 Getting Started - Primeros Pasos

Esta guía te llevará paso a paso en tus primeros 30 minutos con el repositorio.

---

## ✅ Paso 1: Verificar Requisitos (5 min)

### Python Version:
```bash
python --version
# Debe mostrar: Python 3.8 o superior
```

Si no tienes Python 3.8+, descárgalo de: https://www.python.org/downloads/

### Git (opcional pero recomendado):
```bash
git --version
```

---

## ⬇️ Paso 2: Clonar/Descargar el Repositorio (2 min)

### Opción A: Con Git
```bash
git clone <URL_DEL_REPO>
cd METALEARNING-ADAPTATIVE-AI
```

### Opción B: Descarga Manual
1. Descarga el ZIP del repositorio
2. Extráelo en una carpeta
3. Abre terminal/cmd en esa carpeta

---

## 🔧 Paso 3: Crear Entorno Virtual (3 min)

**⚡ Altamente Recomendado** para evitar conflictos de paquetes.

### En Linux/Mac:
```bash
python -m venv venv
source venv/bin/activate
```

### En Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

Deberías ver `(venv)` al inicio de tu terminal.

---

## 📦 Paso 4: Instalar Dependencias (5-10 min)

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Nota**: La primera instalación puede tardar. PyTorch es ~1-2 GB.

### Verificar Instalación:
```bash
python -c "import torch; print(f'PyTorch {torch.__version__} instalado correctamente!')"
```

Deberías ver: `PyTorch 2.x.x instalado correctamente!`

---

## 🎓 Paso 5: Iniciar Jupyter (1 min)

```bash
jupyter notebook
```

Esto abrirá tu navegador con la interfaz de Jupyter.

**Alternativa** (si prefieres JupyterLab):
```bash
pip install jupyterlab
jupyter lab
```

---

## 📖 Paso 6: Tu Primer Tutorial (10 min)

### En Jupyter:

1. Navega a: `01_fundamentos/`
2. Abre: `01_introduccion_meta_learning.ipynb`
3. **Lee la sección de teoría** (primeras celdas)
4. **Ejecuta las celdas** una por una (Shift + Enter)

### Cuando llegues a un TODO:

```python
# TODO: Tu código aquí
pass
```

**Opciones**:
- ✏️ Intenta implementarlo tú mismo primero
- 💡 Si te atoras, ejecuta la celda de pistas
- 📝 Si sigues bloqueado, consulta `solutions/SOLUTIONS.md`

### Ejecutar los Tests:

Después de completar un TODO, ejecuta la celda de test:
```python
# ✅ TEST
run_test(test_function, "Test Name")
```

Si ves: **✅ Test pasado!** → ¡Perfecto, continúa!

---

## 🎯 Paso 7: Entender la Estructura (5 min)

Mientras trabajas, es útil entender la organización:

```
METALEARNING-ADAPTATIVE-AI/
│
├── 01_fundamentos/              👈 EMPIEZA AQUÍ
│   ├── 01_introduccion_meta_learning.ipynb
│   └── 02_curva_aprendizaje_comparacion.ipynb
│
├── 02_algoritmos_clave/         👈 DESPUÉS AQUÍ
│   ├── 03_prototypical_networks.ipynb
│   ├── 03b_datasets_reales.ipynb
│   ├── 03c_matching_networks.ipynb
│   ├── 04_maml.ipynb
│   └── 05_meta_learning_memoria.ipynb
│
├── 03_aplicaciones_mundo_real/  👈 LUEGO AQUÍ
│   ├── 06_meta_rl.ipynb
│   ├── 07_skill_discovery.ipynb
│   └── 08_generalizacion_ood.ipynb
│
├── 09_proyecto_final.ipynb      👈 CULMINACIÓN
│
├── utils/                       📚 Funciones helper
├── solutions/                   📝 Soluciones completas
└── datasets/                    💾 Se crea automáticamente
```

---

## 📊 Paso 8: Workflow Típico

### Para Cada Tutorial:

1. **📖 Lee la Teoría** (5-10 min)
   - Entiende los conceptos antes de codificar

2. **✏️ Completa los TODOs** (20-30 min)
   - Intenta implementar sin ver soluciones
   - Usa las pistas si te atoras

3. **✅ Ejecuta los Tests** (2-5 min)
   - Verifica que tu código funciona
   - Debuggea si algo falla

4. **🎨 Ejecuta Visualizaciones** (5 min)
   - Las gráficas te ayudan a entender

5. **💭 Reflexiona** (5 min)
   - ¿Qué aprendiste?
   - ¿Qué te sorprendió?

**Tiempo total por tutorial**: 45-60 minutos

---

## 💡 Consejos para Principiantes

### Si eres Nuevo en PyTorch:

1. **No te preocupes** si no entiendes todo al inicio
2. **Ejecuta cada celda** para ver qué hace
3. **Modifica valores** y re-ejecuta para experimentar
4. **Consulta la doc**: https://pytorch.org/docs/

### Si eres Nuevo en Meta-Learning:

1. **Los primeros 2 tutoriales son cruciales** - no los saltes
2. **Toma notas** de los conceptos clave
3. **Compara** con ML tradicional que ya conoces
4. **No tengas prisa** - es un tema denso

### Si tienes Experiencia:

1. Puedes **saltar las secciones de teoría** si ya conoces el concepto
2. **Intenta resolver TODOs sin pistas**
3. **Experimenta** modificando arquitecturas
4. **Compara** con papers originales

---

## 🆘 ¿Problemas?

### Error Común #1: "Module not found"
```bash
# Activa el entorno virtual:
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Reinstala:
pip install -r requirements.txt
```

### Error Común #2: "Jupyter no encuentra el kernel"
```bash
# Instala el kernel:
pip install ipykernel
python -m ipykernel install --user --name=venv
```

### Error Común #3: "CUDA out of memory"
```python
# En el notebook, cambia a CPU:
device = torch.device('cpu')  # En lugar de 'cuda'
```

### Más Problemas:
Consulta: `TROUBLESHOOTING.md`

---

## 📈 Plan de Estudio Sugerido

### 🟢 Semana 1: Fundamentos (5-8 horas)
- ✅ Tutorial 01: Introducción (1-2h)
- ✅ Tutorial 02: Curvas de Aprendizaje (2-3h)
- ✅ Repaso y experimentación (2-3h)

### 🟡 Semanas 2-3: Algoritmos Clave (12-15 horas)
- ✅ Tutorial 03: Prototypical Networks (3-4h)
- ✅ Tutorial 03b: Datasets Reales (2-3h) ⭐ IMPORTANTE
- ✅ Tutorial 03c: Matching Networks (2-3h)
- ✅ Tutorial 04: MAML (4-5h) ⭐ IMPORTANTE
- ✅ Tutorial 05: Meta-Learning con Memoria (2-3h)

### 🔴 Semanas 4-5: Aplicaciones (10-12 horas)
- ✅ Tutorial 06: Meta-RL (3-4h)
- ✅ Tutorial 07: Skill Discovery (2-3h)
- ✅ Tutorial 08: Generalización OOD (3-4h)

### 🏆 Semana 6: Proyecto Final (8-10 horas)
- ✅ Tutorial 09: Proyecto Integrador ⭐ CULMINACIÓN

**Total: 40-50 horas** de estudio activo

---

## 🎯 Objetivos de Aprendizaje

Al completar todos los tutoriales, serás capaz de:

- ✅ **Explicar** qué es Meta-Learning y por qué es importante
- ✅ **Implementar** Prototypical Networks, Matching Networks y MAML
- ✅ **Trabajar** con datasets reales (Omniglot)
- ✅ **Evaluar** modelos con protocolo estándar de Few-Shot Learning
- ✅ **Aplicar** Meta-Learning a problemas de RL
- ✅ **Entender** robustez OOD y deployment
- ✅ **Construir** un sistema end-to-end completo

---

## 🌟 Recursos Complementarios

### Mientras Estudias:

**Papers Fundamentales** (leer después de cada tutorial):
- Prototypical Networks: https://arxiv.org/abs/1703.05175
- Matching Networks: https://arxiv.org/abs/1606.04080
- MAML: https://arxiv.org/abs/1703.03400

**Videos** (opcional):
- Chelsea Finn's talk on MAML: YouTube
- Stanford CS330: Multi-Task and Meta-Learning

**Comunidades**:
- r/MachineLearning en Reddit
- Papers with Code: Meta-Learning section

---

## ✅ Checklist de Primer Día

Marca lo que completaste:

- [ ] Python 3.8+ instalado
- [ ] Repositorio descargado
- [ ] Entorno virtual creado
- [ ] Dependencias instaladas
- [ ] PyTorch funciona correctamente
- [ ] Jupyter abierto
- [ ] Tutorial 01 iniciado
- [ ] Primer TODO completado
- [ ] Primer test pasado ✅
- [ ] Primera visualización ejecutada

**Si marcaste todo: ¡Excelente! Estás listo para continuar.** 🎉

**Si te falta algo**: Vuelve a los pasos anteriores o consulta `TROUBLESHOOTING.md`

---

## 🚀 ¡Adelante!

Ya tienes todo listo. El aprendizaje de Meta-Learning es un viaje emocionante.

**Recuerda**:
- 🐌 Toma tu tiempo - no hay prisa
- 🤔 Entiende antes de avanzar
- 🔬 Experimenta y juega con el código
- ❓ Pregunta cuando tengas dudas
- 🎯 Disfruta el proceso

**¡Feliz aprendizaje! 🧠✨**

---

**Siguiente paso**: Abre `01_fundamentos/01_introduccion_meta_learning.ipynb`
