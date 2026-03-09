# RNN from Scratch con NumPy

Implementación de una **Red Neuronal Recurrente (RNN) multicapa** construida **desde cero con NumPy**, sin frameworks de deep learning.

Este proyecto está pensado como ejercicio de comprensión profunda de las RNN: cómo se propaga la información en el tiempo, cómo se conectan múltiples capas ocultas y cómo funciona el **backpropagation through time (BPTT)** a nivel de gradientes.

---

## Qué hace este proyecto

La clase `RNN` implementa manualmente:

- **Forward pass** sobre secuencias con forma `(T, B, D)`
  - `T`: longitud temporal de la secuencia
  - `B`: batch size
  - `D`: dimensión de entrada
- **Múltiples capas recurrentes**
- **Conexiones recurrentes temporales** dentro de cada capa
- **Conexiones verticales** entre capas consecutivas
- **Cálculo manual del backward**
- **Actualización de pesos por descenso por gradiente**

En otras palabras: aquí no hay magia negra de framework; aquí se ven las tripas del bicho.

---

## Estructura del modelo

La red mantiene cuatro grupos principales de parámetros:

- `pesos_entrada`: transforma la entrada externa hacia la primera capa oculta
- `pesos_hidden`: pesos recurrentes de cada capa para conectar `h_{t-1}` con `h_t`
- `pesos_vertical`: pesos entre capas ocultas dentro del mismo instante temporal
- `pesos_salida`: proyecta la última capa oculta al espacio de salida

Además:

- `bias_hidden`: sesgo compartido para capas ocultas
- `bias_salida`: sesgo de salida

La activación usada en el estado oculto es `tanh`.

---

## Forward pass

Para cada instante temporal `t`:

1. Se toma la entrada `x_t`
2. Se actualiza cada capa recurrente
3. La primera capa recibe:
   - la entrada externa `x_t`
   - el estado oculto anterior `h_{t-1}`
4. Las capas superiores reciben:
   - la salida de la capa inferior en el mismo tiempo `t`
   - su propio estado oculto anterior `h_{t-1}`
5. La salida final se calcula usando la última capa oculta

Esto permite modelar dependencias temporales y profundidad en representación.

---

## Backward pass

La implementación realiza manualmente **BPTT**:

- propaga el error desde la salida a la última capa oculta
- acumula gradientes de salida
- propaga gradientes hacia atrás en el tiempo
- propaga gradientes hacia abajo entre capas
- acumula gradientes de:
  - entrada
  - recurrencia temporal
  - conexiones verticales
  - biases

Finalmente actualiza todos los pesos usando una tasa de aprendizaje dada.

---

## Archivo principal

```bash
rnn.ipynb
```

El notebook contiene la implementación completa de la clase `RNN`.

---

## Requisitos

Solo necesitas:

```bash
pip install numpy
```

---

## Uso básico

```python
import numpy as np
from rnn import RNN

T, B, D = 5, 2, 3
H = 4
O = 2
L = 2

x = np.random.randn(T, B, D)
error = np.random.randn(T, B, O)

modelo = RNN(
    input_size=D,
    hidden_size=H,
    output_size=O,
    layers=L,
    activation="tanh",
    batch_size=B
)

salida, h_final = modelo.forward(x)
gradientes = modelo.backward(learning_rate=0.001, error=error)
```

> Nota: en el notebook actual la clase está definida directamente ahí, así que este ejemplo asume que más adelante se extrae a un archivo `.py` reutilizable.

---

## Objetivo del proyecto

Este repositorio no parece orientado a producción, sino a **aprendizaje profundo de fundamentos**. Y eso es precisamente lo valioso.

Construir una RNN así ayuda a entender:

- qué tensores intervienen realmente
- cómo se almacenan estados intermedios
- cómo se encadenan los gradientes en tiempo y profundidad
- por qué entrenar RNN puede ser delicado
- dónde aparecen problemas como exploding/vanishing gradients

---

## Limitaciones actuales

La implementación es muy útil para estudiar, pero todavía tiene margen para evolucionar:

- No hay **función de pérdida integrada**
- No hay **bucle de entrenamiento completo**
- No hay **dataset de ejemplo**
- No hay **gradient clipping**
- No hay soporte real para varias activaciones, aunque el constructor reciba `activation`
- No hay separación en módulos (`model.py`, `train.py`, etc.)
- No hay tests numéricos para validar gradientes
- No hay comparación con una implementación de referencia como PyTorch

---

## Mejoras recomendadas

### 1. Convertir el notebook en un proyecto más limpio

Una estructura muy razonable sería:

```bash
RNN/
├── README.md
├── rnn.py
├── train_example.py
├── utils.py
├── tests/
│   └── test_gradients.py
└── notebooks/
    └── rnn.ipynb
```

### 2. Añadir una demo entrenable

Por ejemplo:

- predicción del siguiente valor en una secuencia
- clasificación de secuencias simples
- toy problem de caracteres

### 3. Validar gradientes

Hacer **gradient checking** comparando:

- gradiente analítico
- gradiente numérico por diferencias finitas

Eso separa un proyecto “parece que funciona” de uno “sé que funciona”.

### 4. Añadir estabilidad numérica

Sería muy buena idea incluir:

- gradient clipping
- inicialización más controlada
- opción de truncar BPTT

### 5. Documentar formas tensoriales

Tu implementación ya piensa en formas correctamente; conviene dejarlo explicitado en una tabla.

| Símbolo | Significado |
|---|---|
| `T` | longitud de secuencia |
| `B` | tamaño de batch |
| `D` | dimensión de entrada |
| `H` | dimensión oculta |
| `O` | dimensión de salida |
| `L` | número de capas |

---

## Valor del repositorio

Este proyecto tiene valor porque muestra que entiendes varias cosas que mucha gente usa sin comprender del todo:

- diseño tensorial
- flujo temporal en RNN
- almacenamiento de activaciones intermedias
- backpropagation manual
- arquitectura multicapa recurrente

Eso da muy buena señal de base matemática e ingenieril.

---

## Posibles siguientes pasos

Después de esta implementación, una progresión muy potente sería:

1. añadir entrenamiento completo con una loss real
2. implementar **gradient checking**
3. crear una **LSTM desde cero**
4. crear una **GRU desde cero**
5. comparar rendimiento y estabilidad con PyTorch
6. vectorizar y perfilar el código

---

## Autor

Repositorio desarrollado por **David** como proyecto de aprendizaje de redes neuronales recurrentes desde primeros principios.

---
