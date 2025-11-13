# 🧠 Meta-Learning: Tutoriales Interactivos de "Aprender a Aprender"

![Meta-Learning](https://img.shields.io/badge/Meta--Learning-Adaptive%20AI-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)

## 📚 Descripción

Bienvenido a **METALEARNING-ADAPTATIVE-AI**, una colección completa de tutoriales interactivos diseñados para enseñar **Meta-Learning** (Aprender a Aprender) desde los fundamentos hasta aplicaciones avanzadas en IA adaptativa.

### ¿Qué es Meta-Learning?

Meta-Learning es un paradigma de Inteligencia Artificial que permite a los modelos **aprender cómo aprender**. En lugar de simplemente optimizar parámetros para una tarea específica, el Meta-Learning optimiza el proceso de aprendizaje mismo, permitiendo a los modelos:

- 🚀 Adaptarse rápidamente a nuevas tareas con pocos ejemplos (Few-Shot Learning)
- 🔄 Transferir conocimiento de manera eficiente entre dominios
- 🎯 Generalizar mejor en entornos dinámicos y cambiantes
- 🤖 Servir como base para IA verdaderamente adaptativa

## 🎯 Metodología de Aprendizaje

Estos tutoriales están diseñados con una metodología **práctica e interactiva**:

1. **📖 Teoría**: Cada concepto se explica con claridad y ejemplos visuales
2. **✏️ Práctica**: Secciones de código marcadas como `# TODO` para que las completes
3. **✅ Tests**: Celdas de prueba automáticas para validar tu implementación
4. **💡 Pistas**: Sistema de pistas progresivas para ayudarte cuando te atores
5. **🎨 Visualización**: Gráficas y visualizaciones para entender mejor los conceptos

## 📋 Contenido del Curso (🌟 12 Tutoriales Completos!)

### 🧠 Módulo 1: Fundamentos
Establece las bases conceptuales del Meta-Learning y su importancia.

- **01_introduccion_meta_learning.ipynb**: Conceptos básicos, diferencias con ML tradicional y Transfer Learning
- **02_curva_aprendizaje_comparacion.ipynb**: Comparación cuantitativa de velocidades de adaptación

### 🛠️ Módulo 2: Algoritmos Clave
Implementación práctica de los algoritmos más importantes de Meta-Learning.

- **03_prototypical_networks.ipynb**: Redes prototípicas para clasificación Few-Shot
- **03b_datasets_reales.ipynb**: ⭐ **NUEVO** - Trabajando con Omniglot y datasets reales, samplers N-way K-shot
- **03c_matching_networks.ipynb**: ⭐ **NUEVO** - Matching Networks con mecanismos de atención
- **04_maml.ipynb**: Model-Agnostic Meta-Learning - El algoritmo estrella con gradientes de segundo orden
- **05_meta_learning_memoria.ipynb**: Meta-Learning con memoria usando RNNs/LSTMs como optimizadores

### 🌐 Módulo 3: Aplicaciones al Mundo Real
Puente hacia la IA adaptativa en entornos dinámicos.

- **06_meta_rl.ipynb**: Meta Reinforcement Learning para adaptación rápida en entornos
- **07_skill_discovery.ipynb**: ⭐ **NUEVO** - Aprendizaje de habilidades reutilizables y composición
- **08_generalizacion_ood.ipynb**: ⭐ **NUEVO** - Generalización Out-of-Distribution y robustez en deployment

### 🎓 Proyecto Final
Integración completa de todos los conceptos aprendidos.

- **09_proyecto_final.ipynb**: ⭐ **NUEVO** - Sistema end-to-end con Omniglot, evaluación estándar y benchmarking

## 🚀 Instalación y Configuración

### Requisitos Previos
- Python 3.8 o superior
- pip o conda
- (Opcional) GPU con CUDA para acelerar el entrenamiento

### Pasos de Instalación

1. **Clonar el repositorio**:
```bash
git clone <URL_DEL_REPO>
cd METALEARNING-ADAPTATIVE-AI
```

2. **Crear un entorno virtual** (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

4. **Iniciar Jupyter**:
```bash
jupyter notebook
```

5. **Abrir el primer tutorial** en `01_fundamentos/01_introduccion_meta_learning.ipynb`

## 📖 Cómo Usar Este Curso

1. **Sigue el orden**: Los tutoriales están diseñados para construirse uno sobre otro
2. **Lee la teoría**: Cada notebook comienza con explicaciones conceptuales
3. **Completa los TODOs**: Busca las secciones marcadas como `# TODO: Tu código aquí`
4. **Usa las pistas**: Si te atoras, revisa las celdas de pistas (¡intenta resolverlo primero!)
5. **Ejecuta los tests**: Después de completar cada ejercicio, ejecuta las celdas de test
6. **Experimenta**: Una vez que pasen los tests, modifica parámetros y experimenta

## 🧪 Ejemplo de Estructura de un Tutorial

```python
# 📖 TEORÍA
"""
Explicación del concepto...
"""

# ✏️ EJERCICIO
def mi_funcion(x):
    # TODO: Implementa esta función
    # Pista: La función debe...
    pass

# ✅ TEST
assert mi_funcion(5) == 10, "Test fallido: revisa tu implementación"
print("✅ ¡Test pasado! Excelente trabajo.")
```

## 🗂️ Estructura del Repositorio

```
METALEARNING-ADAPTATIVE-AI/
├── 01_fundamentos/           # Tutoriales de conceptos básicos
├── 02_algoritmos_clave/      # Implementaciones de algoritmos principales
├── 03_aplicaciones_mundo_real/  # Aplicaciones avanzadas
├── datasets/                 # Datasets para los ejercicios
├── utils/                    # Funciones auxiliares reutilizables
├── tests/                    # Tests unitarios adicionales
├── solutions/                # Soluciones completas (¡úsalas solo al final!)
├── requirements.txt          # Dependencias del proyecto
└── README.md                 # Este archivo
```

## 🛠️ Tecnologías Utilizadas

- **PyTorch**: Framework principal de Deep Learning
- **NumPy**: Computación numérica
- **Matplotlib/Seaborn**: Visualización
- **Gymnasium**: Entornos de Reinforcement Learning
- **tqdm**: Barras de progreso
- **pytest**: Testing

## 📈 Progreso Recomendado

- **Semana 1**: Módulo 1 (Fundamentos) - 2 tutoriales
- **Semana 2-4**: Módulo 2 (Algoritmos Clave) - 5 tutoriales incluyendo datasets reales
- **Semana 5-6**: Módulo 3 (Aplicaciones al Mundo Real) - 3 tutoriales
- **Semana 7**: Proyecto Final - Integración completa

**Tiempo estimado total**: 40-50 horas de estudio activo

**Ruta rápida (intensiva)**: Puedes completar en 2-3 semanas dedicando 15-20 horas/semana

## 🤝 Contribuciones

Las contribuciones son bienvenidas! Si encuentras errores, tienes ideas para nuevos tutoriales o mejoras:

1. Abre un issue describiendo el problema/sugerencia
2. Haz un fork del repositorio
3. Crea una rama para tu feature (`git checkout -b feature/nueva-caracteristica`)
4. Commit tus cambios (`git commit -m 'Agrega nueva característica'`)
5. Push a la rama (`git push origin feature/nueva-caracteristica`)
6. Abre un Pull Request

## 📚 Recursos Adicionales

- **Papers Fundamentales**:
  - [MAML](https://arxiv.org/abs/1703.03400) - Finn et al., 2017
  - [Prototypical Networks](https://arxiv.org/abs/1703.05175) - Snell et al., 2017
  - [Meta-RL](https://arxiv.org/abs/1611.05763) - Duan et al., 2016

- **Lecturas Complementarias**:
  - [Meta-Learning: A Survey](https://arxiv.org/abs/1810.03548)
  - [Learning to Learn](https://www.cs.toronto.edu/~rgrosse/courses/csc421_2019/slides/lec20.pdf)

## 📝 Licencia

Este proyecto está bajo la licencia MIT. Ver el archivo `LICENSE` para más detalles.

## ✉️ Contacto

¿Preguntas? ¿Sugerencias? Abre un issue o contacta al mantenedor del repositorio.

---

**¡Feliz Aprendizaje! 🚀🧠**

*"The capacity to learn is a gift; the ability to learn is a skill; the willingness to learn is a choice."* - Brian Herbert
