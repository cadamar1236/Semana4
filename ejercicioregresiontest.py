import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Datos proporcionados
altura = np.array([120, 122, 124, 125, 127, 129, 130, 133, 134, 136, 138, 139, 141, 142, 143, 150, 151, 153, 156, 157, 159, 160, 162, 164, 165, 167, 170, 172, 174, 175, 176, 177, 178, 180, 181, 183, 184])
peso = np.array([45, 45, 47, 46, 47, 48, 50, 52, 51, 54, 55, 56, 58, 59, 58, 60, 61, 62, 64, 65, 65, 67, 68, 69, 69, 69, 70, 72, 73, 73, 75, 76, 77, 76, 77, 78, 79])

# Reshape para que tengan la forma correcta para scikit-learn
altura = altura.reshape(-1, 1)
peso = peso.reshape(-1, 1)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(altura, peso, test_size=0.2, random_state=42)

# Entrenamos el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Hacemos predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Calculamos R² y RSME
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Visualizamos los resultados en el conjunto de prueba
plt.scatter(X_test, y_test, label='Datos reales')
plt.plot(X_test, y_pred, color='red', label='Regresión lineal')
plt.xlabel('Altura')
plt.ylabel('Peso')
plt.title(f'R²: {r2:.4f}, RMSE: {rmse:.4f}')
plt.legend()
plt.show()

# Imprimimos R² y RMSE
print(f"Coeficiente de determinación (R²): {r2:.4f}")
print(f"Raíz del error cuadrático medio (RMSE): {rmse:.4f}")
