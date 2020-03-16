import sys
from myutil import *
import numpy as np
import numpy.random as rand
import numpy.linalg as LA
import random
import matplotlib.pyplot as plt


class generate_data:
    def __init__(self, n, dim, list_of_std, std_y=None):
        self.n = n
        self.dim = dim
        self.list_of_std = list_of_std
        self.std_y = std_y

    def generate_X(self, start, end):
        self.X = rand.uniform(start, end, (self.n, self.dim))

    def white_Gauss(self, std=0.5):
        return rand.normal(0, std, self.n)

    def sigmoid(self, x):
        return 1 / float(1 + np.exp(-x))

    def generate_Y_sigmoid(self):
        self.Y = np.array([self.sigmoid(x.sum() / float(x.shape[0])) for x in self.X])

    def generate_Y_Gauss(self):
        def gauss(x):
            divide_wt = np.sqrt(2 * np.pi) * std
            return np.exp(-(x * x) / float(2 * std * std)) / divide_wt

        std = self.std_y
        x_vec = np.array([x.sum() / float(x.shape[0]) for x in self.X])
        self.Y = np.array(map(gauss, x_vec))

    def generate_Y_Mix_of_Gauss(self, no_Gauss, prob_Gauss):
        self.Y = np.zeros(self.n)
        for itr, p in zip(range(no_Gauss), prob_Gauss):
            w = rand.uniform(0, 1, self.dim)
            self.Y += p * self.X.dot(w)

    def generate_variable_human_prediction(self):
        self.c = {}
        for std in self.list_of_std:
            self.c[str(std)] = self.variable_std_Gauss_inc(np.min(np.array(self.list_of_std)), std,
                                                           self.X.flatten()) ** 2

    def variable_std_Gauss_inc(self, low, high, x):
        m = (high - low) / np.max(x)
        return np.array([rand.normal(0, m * np.absolute(x_i) + low, 1)[0] for x_i in x])

    def generate_human_prediction(self):

        self.human_pred = {}
        for std in self.list_of_std:
            self.human_pred[str(std)] = self.Y + self.white_Gauss(std=std)

    def append_X(self):
        self.X = np.concatenate((self.X, np.ones((self.n, 1))), axis=1)

    def split_data(self, frac):
        indices = np.arange(self.n)
        random.shuffle(indices)
        num_train = int(frac * self.n)
        indices_train = indices[:num_train]
        indices_test = indices[num_train:]
        self.Xtest = self.X[indices_test]
        self.Xtrain = self.X[indices_train]
        self.Ytrain = self.Y[indices_train]
        self.Ytest = self.Y[indices_test]
        self.human_pred_train = {}
        self.human_pred_test = {}

        for std in self.list_of_std:
            self.human_pred_train[str(std)] = self.human_pred[str(std)][indices_train]
            self.human_pred_test[str(std)] = self.human_pred[str(std)][indices_test]

        n_test = self.Xtest.shape[0]
        n_train = self.Xtrain.shape[0]
        self.dist_mat = np.zeros((n_test, n_train))

        for te in range(n_test):
            for tr in range(n_train):
                self.dist_mat[te, tr] = LA.norm(self.Xtest[te] - self.Xtrain[tr])

    def visualize_data(self):
        x = self.X[:, 0].flatten()
        y = self.Y
        plt.scatter(x, y)
        plt.show()


def convert(input_data, output_data):
    def get_err(label, pred):
        return (label - pred) ** 2

    data = load_data(input_data, 'ifexists')
    list_of_std_str = data.human_pred_train.keys()
    test = {'X': data.Xtest, 'Y': data.Ytest, 'c': {}}
    data_dict = {'test': test, 'X': data.Xtrain, 'Y': data.Ytrain, 'c': {}, 'dist_mat': data.dist_mat}

    for std in list_of_std_str:
        data_dict['c'][std] = get_err(data_dict['Y'], data.human_pred_train[std])
        data_dict['test']['c'][std] = get_err(data_dict['test']['Y'], data.human_pred_test[std])

    save(data_dict, output_data)


def main():
    n = 500
    dim = 5
    frac = 0.8
    list_of_options = ['gauss', 'sigmoid', 'Usigmoid', 'Ugauss', 'Wgauss', 'Wsigmoid']
    options = sys.argv[1:]

    if not os.path.exists('data'):
        os.mkdir('data')

    for option in options:
        assert option in list_of_options

        input_data_file = 'data/' + option
        if option in ['Wgauss', 'Wsigmoid']:
            input_data_file = 'data/data_dict_' + option


        if option == 'sigmoid':
            list_of_std = np.array([0.001, 0.005, .01, .05])
            obj = generate_data(n, dim, list_of_std)
            obj.generate_X(-7, 7)
            obj.generate_Y_sigmoid()
            obj.generate_human_prediction()
            obj.append_X()
            obj.split_data(frac)
            # obj.visualize_data()
            save(obj, input_data_file)
            del obj

        if option == 'gauss':
            std_y = 2
            list_of_std = np.array([0.001, .005, 0.01, 0.05])
            obj = generate_data(n, dim, list_of_std, std_y)
            obj.generate_X(-7, 7)
            obj.generate_Y_Gauss()
            obj.generate_human_prediction()
            obj.append_X()
            obj.split_data(frac)
            save(obj, input_data_file)
            del obj

        if option == 'Usigmoid':
            list_of_std = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
            obj = generate_data(n, dim, list_of_std)
            obj.generate_X(-7, 7)
            obj.generate_Y_sigmoid()
            obj.generate_human_prediction()
            obj.append_X()
            obj.split_data(frac)
            # obj.visualize_data()
            save(obj, input_data_file)
            del obj

        if option == 'Ugauss':
            std_y = 2
            list_of_std = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
            obj = generate_data(n, dim, list_of_std, std_y)
            obj.generate_X(-7, 7)
            obj.generate_Y_Gauss()
            obj.generate_human_prediction()
            obj.append_X()
            obj.split_data(frac)
            save(obj, input_data_file)
            del obj

        if option == 'Wsigmoid':
            list_of_std = [0.001]
            obj = generate_data(n=240, dim=1, list_of_std=list_of_std)
            obj.generate_X(-7, 7)
            obj.generate_Y_sigmoid()
            obj.generate_variable_human_prediction()
            obj.append_X()
            full_data = {}
            for std in list_of_std:
                full_data[str(std)] = {'X': obj.X, 'Y': obj.Y, 'c': obj.c[str(std)]}
            save(full_data, input_data_file)

        if option == 'Wgauss':
            list_of_std = [0.001]
            obj = generate_data(n=240, dim=1, list_of_std=list_of_std, std_y=2)
            obj.generate_X(-1, 1)
            obj.generate_Y_Gauss()
            obj.generate_variable_human_prediction()
            obj.append_X()
            full_data = {}
            for std in list_of_std:
                full_data[str(std)] = {'X': obj.X, 'Y': obj.Y, 'c': obj.c[str(std)]}
            save(full_data, input_data_file)

        if option not in ['Wgauss', 'Wsigmoid']:
            if os.path.exists('data/data_dict_' + option + '.pkl'):
                os.remove('data/data_dict_' + option + '.pkl')
            output_data_file = 'data/data_dict_' + option
            print 'converting'
            convert(input_data_file, output_data_file)


if __name__ == "__main__":
    main()
