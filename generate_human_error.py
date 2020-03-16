import numpy.random as rand
from process_image_data import *
import matplotlib.pyplot as plt


def plot_range_of_lambda(data_file):
    lamb = float(sys.argv[1])

    data = load_data(data_file)
    gamma_lower_bound = np.array([data['c']['0.5'][i] / float(data['Y'][i] ** 2) for i in range(data['X'].shape[0])])
    gamma_upper_bound = lamb / (lamb + np.max(LA.norm(data['X'], axis=1).flatten()) ** 2)

    plt.plot(gamma_lower_bound, label='gamma lower bound')
    plt.plot(gamma_upper_bound * np.ones(data['X'].shape[0]), label='gamma upper bound')
    print np.max(LA.norm(data['X'], axis=1).flatten()) ** 2
    plt.legend()
    plt.show()


class Generate_human_error:

    def __init__(self, data_file):
        # print data_file
        self.data = load_data(data_file)
        if 'c' in self.data:
            del self.data['c']
            del self.data['test']['c']
        self.n, self.dim = self.data['X'].shape

    def normalize_features(self, delta=1):
        n, dim = self.data['X'].shape
        for feature in range(dim):
            self.data['X'][:, feature] = np.true_divide(self.data['X'][:, feature],
                                                        100 * LA.norm(self.data['X'][:, feature].flatten()))
            self.data['test']['X'][:, feature] = np.true_divide(self.data['test']['X'][:, feature], 100 * LA.norm(
                self.data['test']['X'][:, feature].flatten()))

        print np.max([LA.norm(x.flatten()) for x in self.data['X']])

    def white_Gauss(self, std=1, n=1, upper_bound=False, y_vec=None):
        init_noise = rand.normal(0, std, n)
        if upper_bound:
            return np.array([noise if noise / y < 0.3 else 0.1 * y for noise, y in zip(init_noise, y_vec)])
        else:
            return init_noise

    def data_independent_noise(self, list_of_std, upper_bound=False):
        self.data['c'] = {}
        self.data['test']['c'] = {}
        for std in list_of_std:
            self.data['c'][str(std)] = self.white_Gauss(std, self.data['Y'].shape[0], upper_bound, self.data['Y']) ** 2
            self.data['test']['c'][str(std)] = self.white_Gauss(std, self.data['test']['Y'].shape[0], upper_bound,
                                                                self.data['test']['Y']) ** 2

    def variable_std_Gauss(self, std_const, x):
        x_norm = LA.norm(x, axis=1).flatten()
        std_vector = std_const * np.reciprocal(x_norm)
        tmp = np.array([rand.normal(0, s, 1)[0] for s in std_vector])
        return tmp

    def data_dependent_noise(self, list_of_std):
        self.data['c'] = {}
        self.data['test']['c'] = {}
        for std in list_of_std:
            self.data['c'][str(std)] = self.variable_std_Gauss(std, self.data['X']) ** 2
            self.data['test']['c'][str(std)] = self.variable_std_Gauss(std, self.data['test']['X']) ** 2

    def modify_y_values(self):
        def get_num_category(y, y_t):
            y = np.concatenate((y.flatten(), y_t.flatten()), axis=0)
            return np.unique(y).shape[0]

        def map_range(v, l, h, l_new, h_new):
            return float(v - l) * ((h_new - l_new) / float(h - l)) + l_new

        num_cat = get_num_category(self.data['Y'], self.data['test']['Y'])
        print num_cat
        self.data['Y'] = np.array([map_range(i, 0, 1, float(1) / num_cat, 1) for i in self.data['Y']]).flatten()
        self.data['test']['Y'] = np.array(
            [map_range(i, 0, 1, float(1) / num_cat, 1) for i in self.data['test']['Y']]).flatten()

    def get_discrete_noise(self, p, num_cat):
        m = 10
        c = []
        for sample in range(self.n):
            if False:
                sum = 0
                for i in range(m):
                    x = np.random.uniform(0, 1)
                    if x < p:
                        sum += (float(1) / num_cat) ** 2
                c.append(float(sum) / m)
            else:
                c.append(((float(1) / num_cat) ** 2) * p)
        return np.array(c)

    def discrete_noise(self, list_of_p):
        def get_num_category(y, y_t):
            y = np.concatenate((y.flatten(), y_t.flatten()), axis=0)
            return np.unique(y).shape[0]

        num_cat = get_num_category(self.data['Y'], self.data['test']['Y'])
        self.data['c'] = {}
        self.data['test']['c'] = {}

        for p in list_of_p:
            self.data['c'][str(p)] = self.get_discrete_noise(p, num_cat)
            self.data['test']['c'][str(p)] = self.get_discrete_noise(p, num_cat)

    def vary_discrete_old_data_format(self, noise_ratio, list_of_high_err_const):
        def nearest(i):
            return np.argmin(self.data['dist_mat'][i])

        n = self.data['X'].shape[0]
        indices = np.arange(n)
        random.shuffle(indices)
        self.data['low'] = {}
        self.data['c'] = {}
        self.data['test']['c'] = {}

        for high_err_const in list_of_high_err_const:
            num_low = int(high_err_const * n)
            self.data['low'][str(high_err_const)] = indices[:num_low]
            self.data['c'][str(high_err_const)] = np.array(
                [0.0001 if i in self.data['low'][str(high_err_const)] else 0.25 for i in range(n)])
            self.data['test']['c'][str(high_err_const)] = np.array(
                [0.0001 if nearest(i) in self.data['low'][str(high_err_const)] else 0.5 for i in
                 range(self.data['test']['X'].shape[0])])

    def vary_discrete_3(self, list_of_noise_ratio):
        def get_num_category(y, y_t):
            y = np.concatenate((y.flatten(), y_t.flatten()), axis=0)
            return np.unique(y).shape[0]

        def nearest(i):
            return np.argmin(self.data['dist_mat'][i])

        self.normalize_features()
        num_cat = get_num_category(self.data['Y'], self.data['test']['Y'])
        high_err_const = 45
        n = self.data['X'].shape[0]
        indices = np.arange(n)
        random.shuffle(indices)
        err = ((float(1) / num_cat) ** 2) / 50
        self.data['low'] = {}
        self.data['c'] = {}
        self.data['test']['c'] = {}

        # list_of_low_indices = []
        for noise_ratio in list_of_noise_ratio:
            num_low = int(noise_ratio * n)
            # print num_low
            self.data['low'][str(noise_ratio)] = indices[:num_low]
            self.data['c'][str(noise_ratio)] = np.array(
                [err if i in self.data['low'][str(noise_ratio)] else high_err_const * err for i in range(n)])
            print 'c min', np.min(self.data['c'][str(noise_ratio)])
            print 'c max', np.max(self.data['c'][str(noise_ratio)])
            self.data['test']['c'][str(noise_ratio)] = np.array(
                [err if nearest(i) in self.data['low'][str(noise_ratio)] else high_err_const * err for i in
                 range(self.data['test']['X'].shape[0])])

    def vary_discrete_3_v2(self, list_of_class_indices):
        def get_num_category(y, y_t):
            y = np.concatenate((y.flatten(), y_t.flatten()), axis=0)
            return np.unique(y).shape[0]

        def nearest(i):
            return np.argmin(self.data['dist_mat'][i])

        self.normalize_features()
        num_cat = get_num_category(self.data['Y'], self.data['test']['Y'])
        err = ((float(1) / num_cat) ** 2) / 50
        self.data['c'] = {}
        self.data['test']['c'] = {}
        list_of_high_indices = []
        c = np.ones(self.data['X'].shape[0]) * err

        for class_index in list_of_class_indices:
            class_label = float(class_index + 1) / num_cat
            new_indices = np.where(self.data['Y'] == class_label)[0]
            list_of_high_indices.extend(list(new_indices))
            err_upper = class_label ** 2
            c[new_indices] = err_upper
            self.data['c'][str(class_index)] = np.copy(c)
            print 'c min', np.min(self.data['c'][str(class_index)])
            print 'c max', np.max(self.data['c'][str(class_index)])
            plt.plot(self.data['c'][str(class_index)])
            plt.show()
            plt.close()
            self.data['test']['c'][str(class_index)] = np.ones(self.data['test']['X'].shape[0]) * err

    def vary_discrete(self, list_of_frac, file_name):

        def nearest(i):
            return np.argmin(self.data['dist_mat'][i])

        self.normalize_features()
        n = self.data['X'].shape[0]
        indices = np.arange(n)
        random.shuffle(indices)

        self.data['low'] = {}
        self.data['c'] = {}
        self.data['test']['c'] = {}

        for frac in list_of_frac:
            num_low = int(frac * n)
            self.data['low'][str(frac)] = indices[:num_low]

            if file_name == 'Umessidor':
                self.data['c'][str(frac)] = np.array(
                    [0.0001 if i in self.data['low'][str(frac)] else 0.4 for i in range(n)])
                self.data['test']['c'][str(frac)] = np.array(
                    [0.0001 if nearest(i) in self.data['low'][str(frac)] else 0.4 for i in
                     range(self.data['test']['X'].shape[0])])

            if file_name == 'Ustare11':
                self.data['c'][str(frac)] = np.array(
                    [0.0001 if i in self.data['low'][str(frac)] else 0.1 for i in range(n)])
                self.data['test']['c'][str(frac)] = np.array(
                    [0.0001 if nearest(i) in self.data['low'][str(frac)] else 0.25 for i in
                     range(self.data['test']['X'].shape[0])])

            if file_name == 'Ustare5':
                self.data['c'][str(frac)] = np.array(
                    [0.0001 if i in self.data['low'][str(frac)] else 0.1 for i in range(n)])
                self.data['test']['c'][str(frac)] = np.array(
                    [0.0001 if nearest(i) in self.data['low'][str(frac)] else 0.1 for i in
                     range(self.data['test']['X'].shape[0])])

    def variable_std_dirichlet(self, x_tr, y_tr, std, file_name):

        def get_num_category(y, y_t):
            y = np.concatenate((y.flatten(), y_t.flatten()), axis=0)
            return np.unique(y)

        cats = get_num_category(self.data['Y'], self.data['test']['Y'])

        if file_name == 'messidor':
            prob = [[6, 3, 1], [2, 6, 2], [1, 3, 6]]

        if file_name == 'stare5':
            prob = [[6, 3, 2, 2, 1, 1], [3, 6, 2, 2, 1, 1], [2, 3, 6, 2, 1, 1], [1, 2, 3, 6, 2, 1], [1, 1, 2, 3, 6, 2],
                    [1, 1, 2, 2, 3, 6]]

        if file_name == 'stare11':
            prob = [[6, 3, 2, 2, 1], [3, 6, 2, 2, 1], [1, 3, 6, 2, 2], [1, 2, 3, 6, 2],
                    [1, 2, 2, 3, 6]]
        tmp = []
        for idx, data in enumerate(x_tr):
            label = y_tr[idx]
            trueindex = np.argwhere(label == cats)
            assert trueindex.size == 1
            trueindex = trueindex.flatten()[0]
            error = 0
            std_vector = np.random.dirichlet(prob[trueindex])
            for idx, probability in enumerate(std_vector):
                error += (probability * ((cats[idx] - cats[trueindex]) ** 2))
            tmp.append(error)

        tmp = np.array(tmp)
        tmp = tmp.reshape(tmp.shape[0])

        return tmp

    def dirichlet(self, list_of_p, file_name):

        self.data['c'] = {}
        self.data['test']['c'] = {}
        for std in list_of_p:
            self.data['c'][str(std)] = self.variable_std_dirichlet(self.data['X'], self.data['Y'], std, file_name)
            self.data['test']['c'][str(std)] = self.variable_std_dirichlet(self.data['test']['X'],
                                                                           self.data['test']['Y'], std, file_name)

    def save_data(self, data_file):
        save(self.data, data_file)


def generate_human_error(file_name_list):
    list_of_option = ['random_noise', 'vary_std_noise', 'norm_rand_noise', \
                      'discrete', 'vary_discrete', 'vary_discrete_old', 'vary_discrete_3', 'dirichlet']

    datasets = ['stare5', 'stare11', 'messidor', 'Ustare5', 'Ustare11', 'Umessidor']

    for file_name in file_name_list:
        assert file_name in datasets

        if file_name.startswith('U'):
            option = 'vary_discrete'
        else:
            option = 'dirichlet'

        if option == 'discrete' or option == 'dirichlet':
            list_of_std = [0.05, 0.08, 0.1, 0.2]
        else:
            if 'vary_discrete' in option:
                list_of_std = [0.2, 0.4, 0.6, 0.8]
            else:
                list_of_std = [(i + 1) * 0.01 for i in range(9)]

        data_file = 'data/data_dict_' + file_name

        obj = Generate_human_error(data_file)

        if option == 'dirichlet':
            obj.dirichlet(list_of_std, file_name)

        if option == 'random_noise':
            obj.data_independent_noise(list_of_std)

        if option == 'vary_std_noise':
            obj.data_dependent_noise(list_of_std)

        if option == 'norm_rand_noise':
            obj.normalize_features(delta=0.001)
            obj.data_independent_noise(list_of_std, upper_bound=True)

        if option == 'discrete':
            obj.discrete_noise(list_of_std)

        if option == 'vary_discrete':
            obj.vary_discrete(list_of_std, file_name)

        if option == 'vary_discrete_old':
            obj.vary_discrete_old_data_format(0.5, list_of_std)

        if option == 'vary_discrete_3':
            obj.vary_discrete_3_v2(list_of_std)

        obj.save_data(data_file)


def main():
    # generating human error for image datasets.
    file_name_list = sys.argv[1:]
    generate_human_error(file_name_list)


if __name__ == "__main__":
    main()
