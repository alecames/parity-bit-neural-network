import os, sys, random, csv
import numpy as np

# Alec Ames
# 6843577
# COSC 3P71 - Assign 3 - Neural Networks

# This program will allow the user to create and train a neural network to act as a parity
# checker with a subset of 4bit data permutations over a user specified amount of iterations,
# and test the trained network with the remainder of data.

IN_SIZE = 4 # input size can be changed but it wont fit with anything apart from 1
OUT_SIZE = 1 # either 1 or 0
print_interval = 200 # amount of epochs to wait for printing to output
training_size = 10 # amount of data to be used for training the network (must be less than 16)

# load data from dataset file
data = np.zeros(IN_SIZE)
with open('dataset.txt', 'r') as file:
    # skips line 1 with data titles
    next(file)
    for line in csv.reader(file, dialect="excel-tab"):
        data = np.vstack([data, line])
    data = np.delete(data, 0, 0)

testing_size = len(data) - training_size # subset of training data to be used for testing

# converts half-bit string to vertical matrix
def input_vector(data_line):
    result = []
    for i in range(IN_SIZE):
        result.append(int(data[data_line][0][i]))
    result = np.reshape(result, (IN_SIZE, 1))
    return result

# sigmoid activation function
def sigmoid(x):
    return 1/(1 + np.power((np.e), (-x)))

# derivative for sigmoid function
def derivative(x):
    return np.multiply(x, (1 - x))

# derivative sigmoid function for array
def arr_sigmoid_prime(x):
    for n in range(0,len(x)):
        x[n] = sigmoid(x[n]) * (1 - sigmoid(x[n]))
    return x

# neural network class
class neuralnetwork:
    def __init__(self, sizes, learning_rate):
        self.sizes = sizes
        self.learning_rate = learning_rate
        self.layercount = len(sizes)
        self.weights = [np.random.uniform(-1,1,size) for size in [(sizes[1],sizes[0]),(sizes[2],sizes[1])]]
        self.gradients = [np.ones(size) for size in [(sizes[1],sizes[0]),(sizes[2],sizes[1])]]
        self.x1, self.x2, self.x3 = [],[],0
        self.a1, self.a2, self.a3 = [],[],0

    # function to pass values forward
    def fwdpass(self, input):
        self.a2, self.x2 = [],[]
        self.x1 = input
        self.a1 = sigmoid(input)
        for i in range(hidden_layer_size):
            self.x2.append(np.matmul(self.weights[0][i], self.a1)[0])
            self.a2.append(sigmoid(np.matmul(self.weights[0][i], self.a1)[0]))
        self.x3 = np.matmul(self.weights[1][0], self.a2)
        self.a3 = sigmoid(np.matmul(self.weights[1][0], self.a2))
        return self.a3 # returns output
    
    # backpropagation algorithm
    def backprop(self, exp_out, n):
        self.gradients[1] = 2*(self.a3 - exp_out[n]) * np.matmul((np.vstack(self.a2)), [derivative(sigmoid(self.x3))])
        self.gradients[0] = np.matmul(np.transpose(2*(self.a3 - exp_out[n]) * derivative(sigmoid(self.x3)) * self.weights[1] * np.transpose(arr_sigmoid_prime(self.x2))), np.transpose(self.a1))
        for q in range(0, len(self.sizes) - 1):
            self.weights[q] = np.subtract(self.weights[q], np.multiply(self.learning_rate, self.gradients[q]))

# mean squared error function, y_exp and y_pred are lists/arrays 
def mean_sqr_err(y_exp, y_pred):
    return ((np.subtract(y_exp, y_pred))**2).mean()

# method to train the network
def train(iterations, network, data, train_amt):
    # sub-list of train_amt amount of training examples
    rand_data_indices_train = rand_data_indices[:train_amt]

    for q in range(0, iterations + 1):
        exp_out, pred_out = [],[]
        correct = 0
        total = len(rand_data_indices_train)

        for i in range(0,len(rand_data_indices_train)):
            n = rand_data_indices_train[i]

            prediction = network.fwdpass(input_vector(n))
            if int(data[n][2][0]) == int(np.round(prediction)):
                correct += 1 # +1 for each correct prediction

            exp_out.append(int(data[n][2]))
            pred_out.append(prediction)
            if q: # skips backprop for epoch 0 to show error of initial weights
                network.backprop(exp_out, i)

        # prints to console every ~200 epochs
        if not q % print_interval:
            print (
            "Epoch:", int(-(np.log10(q + 10)) + 6)*(" ") + str(q),
            " | MSE: ", "{:.8f}".format(mean_sqr_err(exp_out, pred_out)),
            " | Success: ", str(correct), "/", total, sep=""
            )

    print("\n————————————————————— Training Results —————————————————————\n",
        "Hidden Nodes: ", hidden_layer_size, " | Learning Rate: ", learning_rate,
        " | Epochs: ", iterations, sep="")
    for i in range(0,len(rand_data_indices_train)):
        n = rand_data_indices_train[i]
        print(
            "Input:", data[n][0],
            "| Expected Output:", exp_out[i],
            "| Prediction:", "{:.8f}".format(pred_out[i]),
            int(np.round(pred_out[i])))
    print(
        "Final MSE: ", "{:.8f}".format(mean_sqr_err(exp_out, pred_out)), 
        " | Final Success: ", str(correct), "/", total, sep="")

# function to test the trained network with the rest of the untouched training examples
def test(network, data, train_amt):
    rand_data_indices_test = rand_data_indices[train_amt:]

    exp_out_test, pred_out_test = [],[]
    correct_test = 0
    total_test = len(rand_data_indices_test)

    print("\n—————————————————————— Testing Results —————————————————————")
    for i in range(0,len(rand_data_indices_test)):
        n = rand_data_indices_test[i]

        prediction = network.fwdpass(input_vector(n))
        prediction = np.absolute(prediction - 1)
        if int(data[n][2][0]) == int(np.round(prediction)):
            correct_test += 1 # +1 for each correct prediction

        exp_out_test.append(int(data[n][2]))
        pred_out_test.append(prediction)

        print(
            "Input:", data[n][0],
            "| Expected Output:", exp_out_test[i],
            "| Prediction:", "{:.8f}".format(pred_out_test[i]),
            int(np.round(pred_out_test[i])))
    print(
        "Testing MSE: ", "{:.8f}".format(mean_sqr_err(exp_out_test, pred_out_test)),
        " | Testing Success: ", str(correct_test), "/", total_test, sep="")

# function to prompt once training or testing complete
def restart_prompt():
    while True:
        try: r = str(input("Restart training? [y/n]: "))
        except ValueError:
            print("Invalid response. Try again.")
        else:
            if r == 'y' or r == 'yes':
                print("\nRestarting...\n")
                os.system(f'python "{sys.argv[0]}"')
                sys.exit(0)
            elif r == 'n' or r == 'no':
                print("Goodbye!\n")
                sys.exit(0)

# small function to get filtered/valid user inputs and 
# fill with default values if no parameters specified
def getParam(type, prompt, minmsg, min, maxmsg, max, default):
    while True:
        userinput = input(prompt)
        if not userinput:
            return default
        try: type(userinput)
        except ValueError:
            print("Invalid Entry.")
        else: 
            userinput = type(userinput)
            if userinput <= max:
                if userinput >= min:
                    return userinput
                else: print(minmsg)
            else: print(maxmsg)

# runs getparam for each parameter with appropriate prompts
print("————————————— Training Parameters —————————————")
hidden_layer_size = getParam(int, "Enter size of hidden layer: ", 
    "Network must have at least one node.", 1,
    "Maximum number of nodes is 256. Try something lower.", 256, 8)

learning_rate = getParam(float, "Enter desired learning rate (0-1): ", 
    "Learning rate cannot be negative.", 0,
    "Learning rate must be less than or equal to one. Try something lower.", 1, 0.15)

iterations = getParam(int, "Enter maximum number of iterations: ", 
    "Iteration count cannot be negative.", 0,
    "That might take too long... Maximum is ~1m iterations.", (2**20), 10000)

print("————————————————— Training... —————————————————")

# initialize network with given parameters
neural_net = neuralnetwork((IN_SIZE, hidden_layer_size, OUT_SIZE), learning_rate)
# create list of randomly ordered data from datafile
rand_data_indices = random.sample(range(len(data)), len(data))
# train network
train(iterations, neural_net, data, training_size)

while True:
        try: r = str(input("\nTest trained network? [y/n]: "))
        except ValueError:
            print("Invalid response. Try again.")
        else:
            if r == 'y' or r == 'yes':
                test(neural_net, data, training_size)
                print()
                restart_prompt()
            elif r == 'n' or r == 'no':
                restart_prompt()

