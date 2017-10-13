import numpy
import scipy.special

class NeuralNetwork:

    # Initialize the neural network
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        # create link weight matrices by sampling weights from a normal probability distribution centred around zero and
        # with a standard deviation that is related to num incoming links into a node (1/sqrt(num incoming links))
        # weight between input and hidden layers
        self.weight_input_hidden = numpy.random.normal(0.0, pow(hidden_nodes,-0.5), (hidden_nodes, input_nodes))
        # weights between hidden and output layers
        self.weight_hidden_output = numpy.random.normal(0.0, pow(output_nodes,-0.5), (output_nodes, hidden_nodes))

        # expit() is a special sigmoid function
        # lambda x: spicpy.special.expit(x) is the same as this in javascript
        # var activation_function = function(x) { return scipy.special.expit(x) }
        self.activation_function = lambda x: scipy.special.expit(x)



    def train(self):
        pass

    # Query the neural network
    # i.e. take input to the neural network and returns the network's output
    # essentially two steps for each layer:
    #   1. Take incoming input matrix and convert to weighted/moderated input matrix; X=WI
    #   2. Convert moderated input matrix to output matrix; O=activate(X)
    def query(self, inputs_list):

        # convert input list to 2d array (matrix)
        inputs = numpy.array(inputs_list, ndmin=2).T

        # compute signals going into hidden layer (CONVERT TO WEIGHTED/MODERATED INPUTS X=WI)
        # X_hidden = Weight_input_hidden * I
        # where:
        #   Weight_input_hidden is the matrix of weights between input and hidden layer
        #   I is the input into the input layer
        #   X_hidden is the combined moderated input (matrix) into the hidden layer
        #   X_hidden matrix will likely result in larger values than the I matrix because each node in the
        #   hidden layer is connected to every node in the input layer!
        hidden_inputs = numpy.dot(self.weight_input_hidden, inputs)

        # compute signals emerging from hidden layer (CONVERT MODERATED INPUTS TO OUTPUTS)
        # O_hidden = sigmoid ( X_hidden )
        # where:
        #   X_hidden is the matrix of inputs into the hidden layer
        #   O_hidden is the matrix of outputs coming out of the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # compute signals going into final output layer (CONVERT TO WEIGHTED/MODERATED INPUTS X=WI)
        final_inputs = numpy.dot(self.weight_hidden_output, hidden_outputs)
        #compute signals emerging from final output layer (CONVERT MODERATED INPUTS TO OUTPUTS)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs




