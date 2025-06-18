#importamos paquetes y datos
import numpy as np
from data_prep import features, targets, features_test, targets_test

# Definiciones útiles
np.random.seed(21)

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

# Hyperparámetros
n_hidden = 2  # number of hidden units
epochs = 200
learnrate = 0.005

# Obtenemos el número de entradas (features) asi como el número de ejemplos (n_records)
n_records, n_features = features.shape
last_loss = None

# Creamos las matrices de los pesos.
weights_input_hidden = np.random.normal(scale=1 / n_features ** .5,
                                        size=(n_features, n_hidden))
weights_hidden_output = np.random.normal(scale=1 / n_features ** .5,
                                         size=n_hidden)

for e in range(epochs):
    del_w_input_hidden = np.zeros(weights_input_hidden.shape)
    del_w_hidden_output = np.zeros(weights_hidden_output.shape)
    for x, y in zip(features.values, targets):
        ## Forward pass ##
        #Calculate the output
        
        hidden_input = np.dot(x,weights_input_hidden)
        hidden_output = sigmoid(hidden_input)

        output_in=np.dot(hidden_output,weights_hidden_output)
        output = sigmoid(output_in)

        ## Backward pass ##
        # TODO: Calculate the error
        error = y-output

        # TODO: Calculate error gradient in output unit
        output_error = error*sigmoid_prime(output_in)

        # TODO: propagate errors to hidden layer
        hidden_error = output_error*np.multiply(weights_hidden_output,sigmoid_prime(hidden_input))

        # TODO: Update the change in weights
        del_w_hidden_output += learnrate*output_error*hidden_output
        del_w_input_hidden += learnrate*(hidden_error*x[:,None])    #Producto matricial 

    # TODO: Update weights
    weights_input_hidden += del_w_input_hidden
    weights_hidden_output += del_w_hidden_output

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        hidden_output = sigmoid(np.dot(x, weights_input_hidden))
        out = sigmoid(np.dot(hidden_output,
                             weights_hidden_output))
        loss = np.mean((out - targets) ** 2)

        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

# Calculate accuracy on test data
hidden = sigmoid(np.dot(features_test, weights_input_hidden))
out = sigmoid(np.dot(hidden, weights_hidden_output))
predictions = out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.4f}".format(accuracy))