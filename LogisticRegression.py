from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from scipy.special import expit, logit
import numpy as np


def load_dataset():
    '''
            Loading the Breast Cancer Dataset
                 x_train is of shape: (30, 381)
                 y_train is of shape: (1, 381)
                 x_test is of shape:  (30, 188)
                 y_test is of shape:  (1, 188)
    '''
    
    cancer_data = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(cancer_data.data, cancer_data.target, test_size=0.33)
    x_train = x_train.T
    x_test = x_test.T
    y_train = y_train.reshape(1, (len(y_train)))
    y_test = y_test.reshape(1, (len(y_test)))
    m = x_train.shape[1]
    return x_train, x_test, y_train, y_test, m


class Neural_Network():
    def __init__(self):
        #Assign random weights to a 3 x 1 matrix and bias
        np.random.seed(100)
        self.weights = np.random.randn(30,1)*0.01
        self.bias = np.zeros(shape=(1, 1))

    #The Sigmoid function
    def ___sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    #Train the neural network
    def train(self, x_train, y_train, iterations, m, learning_rate=0.5):

        for i in range(iterations):
            #Pass the training set through our neural network
            z = np.dot(self.weights.T, x_train) + self.bias
            a = self.___sigmoid(z)
            
            # alculate the cost 
            cost = (-1 / m) * np.sum(y_train * np.log(a) + (1 - y_train) * np.log(1 - a))

            if (i % 500 == 0):
                print("Cost after iteration %i: %f" % (i, cost))
                
            #Adjusting the synaptic weights 
            dw = (1 / m) * np.dot(x_train, (a - y_train).T)
            db = (1 / m) * np.sum(a - y_train)

            self.weights = self.weights - learning_rate * dw
            self.bias = self.bias - learning_rate * db
            
    #predict function of neural network
    def predict(self, inputs):
        m = inputs.shape[1]
        y_predicted = np.zeros((1, m))
        z = np.dot(self.weights.T, inputs) + self.bias
        a = self.___sigmoid(z)
        for i in range(a.shape[1]):
            y_predicted[0, i] = 1 if a[0, i] > 0.5 else 0
        return y_predicted




if __name__ == "__main__":
    '''
    step-1 : Loading data set
                 x_train is of shape: (30, 381)
                 y_train is of shape: (1, 381)
                 x_test is of shape:  (30, 188)
                 y_test is of shape:  (1, 188)
    '''
    
    x_train, x_test, y_train, y_test, m = load_dataset()

    '''
    step-2 : Normalize inputs
                 By using sklearn's MinMaxScaler we normalize our inputs
    '''

    scaler = MinMaxScaler()
    x_train_normalized = scaler.fit_transform(x_train.T).T
    x_test_normalized = scaler.transform(x_test.T).T

    '''
    step-3 : Train our network          
    '''

    neuralNet = Neural_Network()
    neuralNet.train(x_train_normalized, y_train, 10000, m)

    '''
    step-4 : Test our network          
    '''
    
    y_predicted = neuralNet.predict(x_test_normalized)

    '''
    step-5 :Calculate the Accuracy of our network          
    '''

    print("Accuracy on given test data: ")
    print(accuracy_score(y_test[0], y_predicted[0])*100)






