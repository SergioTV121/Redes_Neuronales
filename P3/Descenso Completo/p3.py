
# importamos los paquete necesarios
import numpy as np

# cargamos datos de ejemplo
from data_prep import features, targets, features_test, targets_test


n_records, n_features = features.shape
last_loss = None

# En este ejercicio por propósitos de analizar las salidas utilizaremos la misma semilla para los números aleatorios.
np.random.seed(42)

# Definimos algunas funciones necesarias
def sigmoid(x):

    return 1 / (1 + np.exp(-x))

# Initialize weights. 
weights = np.random.normal(scale=1 / n_features**.5, size=n_features)

# Probemos la precisión de la red antes de entrenarla
tes_out = sigmoid(np.dot(features_test, weights))
predictions = tes_out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))

# La precisión debe ser mala seguramente.

# número de épocas
epochs = 1000
# tasa de aprendizaje
learnrate = 0.5

for e in range(epochs):
    del_w = np.zeros(weights.shape)
    # Para todos los renglones de ejemplo, asignar a x la entrada, y a y la salida deseada
    for x, y in zip(features.values, targets):

        #calcula la predicción de la red
        # Tip: NumPy ya tiene una función que calcula el producto punto. Recuerda que también los logits tienen que pasar por la función de activación.
        output = sigmoid(np.dot(x,weights))

        #calcula el error
        error = y-output

        #termino de error
        t_error=error*output*(1-output)

        #calcula el incremento
        del_w+=learnrate*t_error*x

    #Actualiza los pesos
    weights += del_w * (1/len(features.values))
    # Ahora calculemos el error en el conjunto de datos de entrenamiento
    if e % (epochs / 10) == 0:
        out = sigmoid(np.dot(features, weights))
        loss = np.mean((out - targets) ** 2)
        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss



# Cálculo de la precisión

tes_out = sigmoid(np.dot(features_test, weights))
predictions = tes_out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.5f}".format(accuracy))       