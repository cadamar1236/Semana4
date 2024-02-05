import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generar datos
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X, y)

# Realizar predicciones para 1830 equipos afectados
equipos_afectados = np.array([[1830]])
costo_predicho = model.predict(equipos_afectados)

# Calcular el costo real para comparar
costo_real = 4 + 3 * equipos_afectados

# Calcular el costo real para el conjunto de entrenamiento
y_pred = model.predict(X)
costo_entrenamiento = mean_squared_error(y, y_pred)

print("Costo predicho para 1830 equipos afectados:", costo_predicho[0][0])
print("Costo real para 1830 equipos afectados:", costo_real[0][0])
print("Costo en el conjunto de entrenamiento:", costo_entrenamiento)
