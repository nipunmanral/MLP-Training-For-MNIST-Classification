import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

weights = []
bias = []
layers_config = [784, 512, 10]
average_weights = []
average_bias = []
eta = 0.01
epoch_accuracy = []


def ReLU(x):
    return np.maximum(0, x)

def ReLU_derivative(values):
    result = [1 if x > 0 else 0 for x in values]
    return result

def tanh_activation(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - (np.tanh(x) ** 2)

def Softmax(x):
    return np.exp(x - np.max(x))/(np.sum(np.exp(x - np.max(x))))

def feedforward(input_image):
    a = []
    a.append(input_image.reshape(len(input_image),1))
    for i in range(1, len(layers_config)-1):
        a.append(ReLU((weights[i] @ a[i-1]).reshape(len(weights[i]), 1) + bias[i]))
    y_hat = Softmax((weights[-1] @ a[-1]) + bias[-1]).reshape(len(weights[-1]),)
    a.append(y_hat)
    return a

def backprop(a, ground_output_y):
    delta_error = list(np.empty_like(a))
    index_count = len(layers_config) - 1
    delta_error[index_count] = (a[index_count] - ground_output_y).reshape(len(a[index_count]), 1)
    average_bias[index_count] = average_bias[index_count] + delta_error[index_count] # Output Layer
    average_weights[index_count] = average_weights[index_count] + (delta_error[index_count] @ a[index_count - 1].T) # Output Layer
    for i in range(index_count - 1, 0, -1):
        h_derivative = np.array(ReLU_derivative(a[i])).reshape(1, len(a[i])) * np.eye(len(a[i]))
        delta_error[i] = h_derivative.T @ weights[i+1].T @ delta_error[i+1]
        average_bias[i] = average_bias[i] + delta_error[i]
        average_weights[i] = average_weights[i] + (delta_error[i] @ a[i-1].T)

# He Normalization
def initialize_weights():
    if len(layers_config) < 3:
        print("Incorrect network structure. Check the neural network layer configuration")
    else:
        layer_count = len(layers_config)
        weights.append([])
        bias.append([])
        average_weights.append([])
        average_bias.append([])
        for i in range(1, layer_count):
            neurons_previous = layers_config[i-1]
            neurons_current = layers_config[i]
            single_layer_weights = np.random.normal(0, np.sqrt(2/neurons_previous), (neurons_current, neurons_previous))
            single_layer_bias = np.random.normal(0, np.sqrt(2/neurons_previous), (neurons_current, 1))
            weights.append(single_layer_weights)
            bias.append(single_layer_bias)
            average_weights.append(single_layer_weights)
            average_bias.append(single_layer_bias)


initialize_weights()

data_path = "mnist_traindata.hdf5"
with h5py.File(data_path, 'r') as hf:
    xdata = hf['xdata'][:]
    ydata = hf['ydata'][:]

# split into training and validation data
X_train, X_val, y_train, y_val = train_test_split(xdata, ydata, test_size=10000, random_state=1)

epoch_size = 50
batch_size = 50
batch_numbers = int(len(X_train)/batch_size)

#Single Sample Updates
shuffle_order = np.random.permutation(len(X_train))
for i in range(1, epoch_size + 1):
    print("Running Epoch {}".format(i))
    if i == 2:
        eta = eta/2
    if i == 3:
        eta = eta/2
    shuffle_order = np.random.permutation(len(X_train))
    for j in range(batch_numbers):
        print("Running batch number {} in epoch {}".format(j, i))
        for k in range(batch_size):
            shuffle_index = j * batch_size + k
            sample_x = X_train[shuffle_index,:]
            sample_y = y_train[shuffle_index, :]
            a_values = feedforward(sample_x)
            backprop(a_values, sample_y)
        for a in range(len(weights)):
            value_weight = weights[a]
            value_average_weight = np.multiply(average_weights[a], (eta/batch_size))
            weights[a] = np.subtract(value_weight, value_average_weight)
            value_bias = bias[a]
            value_average_bias = np.multiply(average_bias[a], (eta/batch_size))
            bias[a] = np.subtract(value_bias, value_average_bias)
    print("Running feedforward on validation data for epoch {}".format(i))
    y_output = np.array([feedforward(X_val[m, :])[len(layers_config) - 1] for m in range(len(X_val))])
    class_output = np.argmax(y_output, axis=1)
    label_class = np.argmax(y_val, axis=1)
    number_correct_classification = np.sum(class_output == label_class)
    accuracy_val = number_correct_classification / len(X_val)
    epoch_accuracy.append(accuracy_val)
    print("Accuracy on Validation Set for epoch {} is {}".format(epoch_size, accuracy_val))

plt.plot(range(1, epoch_size + 1), epoch_accuracy)
plt.title('Model accuracy after each epoch')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.axvline(x=1)
plt.show()


with h5py.File('weights_bias.hdf5', 'w') as hf:
    for i in range(1, len(weights)):
        label_w = "w{}".format(i)
        label_b = "b{}".format(i)
        hf.create_dataset(label_w, data=weights[i])
        hf.create_dataset(label_b, data=bias[i])
        hf.attrs['act'] = np.string_("relu")


# Training Set Accuracy
y_output = np.array([feedforward(X_train[i, :])[len(layers_config) - 1] for i in range(len(X_train))])
class_output = np.argmax(y_output, axis=1)
label_class = np.argmax(y_train, axis = 1)
number_correct_classification = np.sum(class_output == label_class)
accuracy_train = number_correct_classification/len(X_train)
print("Accuracy on Training Set is {}".format(accuracy_train))


# Validation Set Accuracy
y_output = np.array([feedforward(X_val[i, :])[len(layers_config) - 1] for i in range(len(X_val))])
class_output = np.argmax(y_output, axis=1)
label_class = np.argmax(y_val, axis = 1)
number_correct_classification = np.sum(class_output == label_class)
accuracy_val = number_correct_classification/len(X_val)
print("Accuracy on Test Set is {}".format(accuracy_val))
