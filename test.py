import sys
import matplotlib
from brewer2mpl import brewer2mpl
from myutil import *
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from plot_util import latexify


def parse_command_line_input(dataset):
    list_of_option = ['greedy', 'RLSR_Reg', 'distort_greedy', 'kl_triage', 'diff_submod']
    list_of_real = ['messidor', 'stare5', 'stare11', 'hatespeech',
                    'Ustare5', 'Ustare11', 'Umessidor']
    list_of_synthetic = ['sigmoid', 'gauss', 'Ugauss', 'Usigmoid', 'Wgauss', 'Wsigmoid']

    assert (dataset in list_of_real or dataset in list_of_synthetic)

    if dataset in ['sigmoid', 'gauss']:
        list_of_std = [0.001]
        list_of_lamb = [0.005]

    if dataset == 'messidor':
        list_of_std = [0.1]
        list_of_lamb = [1.0]

    if dataset == 'stare5':
        list_of_std = [0.1]
        list_of_lamb = [0.5]

    if dataset == 'stare11':
        list_of_std = [0.1]
        list_of_lamb = [1.0]

    if dataset == 'hatespeech':
        list_of_std = [0.0]
        list_of_lamb = [0.01]

    if dataset in ['Usigmoid', 'Ugauss']:
        list_of_std = [0.01, 0.02, 0.03, 0.04, 0.05]
        list_of_lamb = [0.005]
        list_of_option = ['greedy']

    if dataset in ['Ustare5', 'Ustare11', 'Umessidor']:
        list_of_std = [.2, .4, .6, .8]
        list_of_lamb = [1]
        list_of_option = ['greedy']

    if dataset in ['Wgauss', 'Wsigmoid']:
        list_of_std = [0.001]
        list_of_lamb = [0.001]
        list_of_option = ['greedy']

    return list_of_option, list_of_std, list_of_lamb


class plot_triage_real:

    def __init__(self, list_of_K, list_of_std, list_of_lamb, list_of_option, list_of_test_option, flag_synthetic=None):
        self.list_of_K = list_of_K
        self.list_of_std = list_of_std
        self.list_of_lamb = list_of_lamb
        self.list_of_option = list_of_option
        self.list_of_test_option = list_of_test_option
        self.flag_synthetic = flag_synthetic

    def plot_subset(self, data_file, res_file, list_of_std, list_of_lamb, path, dataset):
        data = load_data(data_file)
        res = load_data(res_file)
        list_of_K = [0.2, 0.4, 0.6, 0.8]

        for K in list_of_K:
            for lamb in list_of_lamb:
                for std in list_of_std:
                    local_data = data[str(std)]
                    local_res = res[str(std)][str(K)][str(lamb)]['greedy']
                    subset_human = local_res['subset']
                    w = local_res['w']
                    n = local_data['X'].shape[0]
                    subset_machine = np.array([i for i in range(n) if i not in subset_human])
                    fig, ax = plt.subplots()
                    fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
                    bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
                    color_list = bmap.mpl_colors

                    x = local_data['X'][subset_machine, 0].flatten()
                    y = local_data['Y'][subset_machine]
                    plt.scatter(x, y, c=color_list[0], label=r'$\mathcal{V}$\textbackslash $\mathcal{S}^*$')

                    x = local_data['X'][subset_human, 0].flatten()
                    y = local_data['Y'][subset_human]
                    plt.scatter(x, y, c=color_list[1], label=r'$\mathcal{S}^*$')
                    x = local_data['X'][:, 0].flatten()
                    y = local_data['X'].dot(w).flatten()
                    plt.scatter(x, y, c=color_list[2], label='$\hat y = {w^*}^\mathsf{T}(\mathcal{S}^*) \mathbf{x}$ ')

                    if dataset == 'Wsigmoid':
                        xlabel = '$[-7,7]$'
                        ax.set_ylim([-0.5, 1.5])
                        x = -5

                    if dataset == 'Wgauss':
                        xlabel = '$[-1,1]$'
                        ax.set_ylim([0.17, 0.21])
                        x = 3

                    plt.legend(prop={'size': 19}, frameon=False,
                               handlelength=0.2)
                    plt.xlabel(r'\textbf{Features} $x$ $\sim$ \textbf{Unif} ' + xlabel, fontsize=23)
                    ax.set_ylabel(r'\textbf{Response} $(y)$', fontsize=23,
                                  labelpad=x)

                    savepath = path + dataset + '_' + str(int(400 * K))
                    plt.savefig(savepath + '.pdf')
                    plt.savefig(savepath + '.png')
                    plt.close()

    def U_get_avg_error_vary_K(self, res_file, test_method, dataset, path):
        res = load_data(res_file)
        savepath = path + dataset + '_' + test_method

        real = ['Umessidor', 'Ustare5', 'Ustare11']
        synthetic = ['Usigmoid', 'Ugauss']

        assert dataset in real or dataset in synthetic

        if dataset in real:
            multtext = r'$\times 10^{-1}$'
            labeltext = r'$\rho_c = $'
            mult = 10

        if dataset in synthetic:
            multtext = r'$\times 10^{-3}$'
            labeltext = r'$\sigma_2 = $'
            mult = 1000

        fig, ax = plt.subplots()
        fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.93)
        bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
        color_list = bmap.mpl_colors
        plt.figtext(0.115, 0.96, multtext, fontsize=17)

        for idx, std in enumerate(self.list_of_std):
            for lamb in self.list_of_lamb:
                for option, ind in zip(self.list_of_option, range(len(self.list_of_option))):
                    err_K_tr = []
                    err_K_te = []

                    for K in self.list_of_K:
                        err_K_tr.append(res[str(std)][str(K)][str(lamb)][option]['train_res']['error'])
                        err_K_te.append(
                            res[str(std)][str(K)][str(lamb)][option]['test_res'][test_method][test_method]['error'])

                    err_K_te_mult = [mult * err for err in err_K_te]  # for synthetic
                    ax.plot((err_K_te_mult), label=labeltext + str(std), linewidth=3, marker='o',
                            markersize=10, color=color_list[idx])

        ax.legend(prop={'size': 18}, frameon=False, handlelength=0.2, loc='best')
        plt.xlabel(r'$n/ | \mathcal{V} | $', fontsize=25)
        ax.set_ylabel(r'\textbf{MSE}', fontsize=25, labelpad=3)
        plt.xticks(range(len(self.list_of_K)), self.list_of_K)

        plt.savefig(savepath + '.pdf')
        plt.savefig(savepath + '.png')
        plt.close()

    def U_get_avg_error_vary_testmethod(self, res_file, dataset, path, std):

        res = load_data(res_file)
        real = ['Umessidor', 'Ustare5', 'Ustare11']
        synthetic = ['Usigmoid', 'Ugauss']

        assert dataset in real or dataset in synthetic

        savepath = path + dataset + '_' + str(std) + '_vary_testmethod'
        if dataset in real:
            multtext = r'$\times 10^{-1}$'
            mult = 10

        if dataset in synthetic:
            multtext = r'$\times 10^{-3}$'
            mult = 1000

        fig, ax = plt.subplots()
        fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.93)
        bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
        color_list = bmap.mpl_colors
        plt.figtext(0.115, 0.96, multtext, fontsize=17)

        for lamb in self.list_of_lamb:
            for option, ind in zip(self.list_of_option, range(len(self.list_of_option))):
                for idx, test_method in enumerate(self.list_of_test_option):
                    label_map = {'MLP': 'Multilayer perceptron', 'LR': 'Logistic regression', 'NN': 'NN neighbor'}
                    err_K_tr = []
                    err_K_te = []

                    for K in self.list_of_K:
                        err_K_tr.append(res[str(std)][str(K)][str(lamb)][option]['train_res']['error'])
                        err_K_te.append(
                            res[str(std)][str(K)][str(lamb)][option]['test_res'][test_method][test_method]['error'])

                    err_K_te_mult = [mult * err for err in err_K_te]  # for synthetic
                    ax.plot((err_K_te_mult), label=label_map[test_method], linewidth=3, marker='o',
                            markersize=10, color=color_list[idx])

        ax.legend(prop={'size': 18}, frameon=False, handlelength=0.2, loc='best')
        plt.xlabel(r'$n/ | \mathcal{V} | $', fontsize=25)
        ax.set_ylabel(r'\textbf{MSE}', fontsize=25, labelpad=3)
        plt.xticks(range(len(self.list_of_K)), self.list_of_K)

        plt.savefig(savepath + '.pdf')
        plt.savefig(savepath + '.png')
        plt.close()

    def get_avg_error_vary_K(self, res_file, image_path, file_name, test_method):
        res = load_data(res_file)

        for std in self.list_of_std:
            for lamb in self.list_of_lamb:
                plot_obj = {}

                for option in self.list_of_option:
                    err_K_te = []
                    for K in self.list_of_K:
                        err_K_te.append(
                            res[str(std)][str(K)][str(lamb)][option]['test_res'][test_method][test_method]['error'])

                    plot_obj[option] = {'test': err_K_te}

                self.plot_err_vs_K(image_path, plot_obj, file_name, test_method)

    def get_avg_error_vary_testmethod(self, res_file, image_path, file_name, option):
        res = load_data(res_file)

        for std in self.list_of_std:
            for lamb in self.list_of_lamb:
                plot_obj = {}

                for test_method in self.list_of_test_option:
                    err_K_te = []

                    for K in self.list_of_K:
                        err_K_te.append(
                            res[str(std)][str(K)][str(lamb)][option]['test_res'][test_method][test_method]['error'])

                    plot_obj[test_method] = {'test': err_K_te}

                self.plot_err_vs_testmethod(image_path, plot_obj, file_name)

    def plot_err_vs_testmethod(self, image_file, plot_obj, file_name):
        savepath = image_file + file_name + '_vary_testmethod'
        fig, ax = plt.subplots()
        fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.93)
        bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
        color_list = bmap.mpl_colors
        key = 'test'
        synthetic = ['sigmoid', 'gauss']
        mult = 10
        multtext = r'$\times 10^{-1}$'

        if file_name in synthetic:
            mult = 1000
            multtext = r'$\times 10^{-3}$'

        for idx, option in enumerate(plot_obj.keys()):
            err = [x * mult for x in plot_obj[option][key]]
            label_map = {'MLP': 'Multilayer perceptron', 'LR': 'Logistic regression', 'NN': 'NN neighbor'}
            plt.plot(err, label=label_map[option], linewidth=3, marker='o',
                     markersize=10, color=color_list[idx])

        plt.figtext(0.115, 0.95, multtext, fontsize=17)
        plt.legend(prop={'size': 18}, frameon=False, handlelength=0.2)
        plt.xlabel(r'$n/ | \mathcal{V} | $', fontsize=25)
        ax.set_ylabel(r'\textbf{MSE}', fontsize=25, labelpad=2)

        plt.xticks(range(len(self.list_of_K)), self.list_of_K)

        plt.savefig(savepath + '.pdf')
        plt.savefig(savepath + '.png')
        # save(plot_obj, image_file)
        plt.close()

    def plot_err_vs_K(self, image_file, plot_obj, file_name, test_method):
        savepath = image_file + file_name + '_' + test_method

        fig, ax = plt.subplots()
        fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.93)
        bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
        color_list = bmap.mpl_colors
        synthetic = ['sigmoid', 'gauss']
        mult = 10
        multtext = r'$\times 10^{-1}$'

        if file_name in synthetic:
            mult = 1000
            multtext = r'$\times 10^{-3}$'

        key = 'test'
        for idx, option in enumerate(plot_obj.keys()):
            err = [x * mult for x in plot_obj[option][key]]
            label_map = {'kl_triage': 'Triage', 'distort_greedy': 'Distorted greedy',
                         'greedy': 'Greedy', 'diff_submod': 'DS', 'RLSR_Reg': 'CRR'}

            ax.plot(err, label=label_map[option], linewidth=3, marker='o',
                    markersize=10, color=color_list[idx])

        plt.figtext(0.115, 0.95, multtext, fontsize=17)
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [4, 0, 1, 2, 3]
        ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], prop={'size': 17}, frameon=False,
                  handlelength=0.2)

        plt.xlabel(r'$n/ | \mathcal{V} | $', fontsize=25)
        ax.set_ylabel(r'\textbf{MSE}', fontsize=25, labelpad=2)
        plt.xticks(range(len(self.list_of_K)), self.list_of_K)

        plt.savefig(savepath + '.pdf')
        plt.savefig(savepath + '.png')
        # save(plot_obj, image_file)
        plt.close()

    def get_NN_human(self, dist, tr_human_ind):
        n_tr = dist.shape[0]
        human_dist = float('inf')
        machine_dist = float('inf')

        for d, tr_ind in zip(dist, range(n_tr)):
            if tr_ind in tr_human_ind:
                if d < human_dist:
                    human_dist = d

            else:
                if d < machine_dist:
                    machine_dist = d

        return human_dist - machine_dist

    def classification_get_test_error(self, res_obj, dist_mat, test_method, X_tr, x, y, y_h=None, c=None):
        w = res_obj['w']
        subset = res_obj['subset']
        n, tr_n = dist_mat.shape
        y_m = x.dot(w)
        err_m = (y - y_m) ** 2

        if y_h == None:
            err_h = c
        else:
            err_h = (y - y_h) ** 2

        from sklearn.neural_network import MLPClassifier
        from sklearn.linear_model import LogisticRegression

        y_tr = np.zeros(tr_n, dtype='uint')
        y_tr[subset] = 1  # human label = 1

        if test_method == 'MLP':
            model = MLPClassifier(max_iter=500)
        if test_method == 'LR':
            model = LogisticRegression(solver='liblinear')

        model.fit(X_tr, y_tr)
        y_pred = model.predict(x)
        subset_te_r = []
        subset_machine_r = []

        for idx, label in enumerate(y_pred):
            if label == 1:
                subset_te_r.append(idx)
            else:
                subset_machine_r.append(idx)

        subset_machine_r = np.array(subset_machine_r)
        subset_te_r = np.array(subset_te_r)

        if subset_te_r.size == 0:
            error_r = err_m.sum() / float(n)
        else:
            error_r = (err_h[subset_te_r].sum() + err_m.sum() - err_m[subset_te_r].sum()) / float(n)

        subset_te_n = np.array([int(i) for i in range(len(y_pred)) if y_pred[i] == 1])
        subset_machine_n = np.array([int(i) for i in range(len(y_pred)) if i not in subset_te_n])
        # print 'sample to human--> ' , str(subset_te_n.shape[0]), ', sample to machine--> ', str( subset_machine_n.shape[0])

        if subset_te_n.size == 0:
            error_n = err_m.sum() / float(n)
        else:
            error_n = (err_h[subset_te_n].sum() + err_m.sum() - err_m[subset_te_n].sum()) / float(n)

        error_n = {'error': error_n, 'human_ind': subset_te_n, 'machine_ind': subset_machine_n}
        error_r = {'error': error_r, 'human_ind': subset_te_r, 'machine_ind': subset_machine_r}

        return error_n, error_r

    def get_test_error(self, res_obj, dist_mat, x, y, y_h=None, c=None):
        w = res_obj['w']
        subset = res_obj['subset']
        n, tr_n = dist_mat.shape
        no_human = int((subset.shape[0] * n) / float(tr_n))
        y_m = x.dot(w)
        err_m = (y - y_m) ** 2

        if y_h == None:
            err_h = c
        else:
            err_h = (y - y_h) ** 2

        diff_arr = [self.get_NN_human(dist, subset) for dist in dist_mat]
        indices = np.argsort(np.array(diff_arr))
        subset_te_r = indices[:no_human]
        subset_machine_r = indices[no_human:]

        if subset_te_r.size == 0:
            error_r = err_m.sum() / float(n)
        else:
            error_r = (err_h[subset_te_r].sum() + err_m.sum() - err_m[subset_te_r].sum()) / float(n)

        subset_te_n = np.array([int(i) for i in range(len(diff_arr)) if diff_arr[i] < 0])
        subset_machine_n = np.array([int(i) for i in range(len(diff_arr)) if i not in subset_te_n])
        # print 'sample to human--> ' , str(subset_te_n.shape[0]), ', sample to machine--> ', str( subset_machine_n.shape[0])

        if subset_te_n.size == 0:
            error_n = err_m.sum() / float(n)
        else:
            error_n = (err_h[subset_te_n].sum() + err_m.sum() - err_m[subset_te_n].sum()) / float(n)

        error_n = {'error': error_n, 'human_ind': subset_te_n, 'machine_ind': subset_machine_n}
        error_r = {'error': error_r, 'human_ind': subset_te_r, 'machine_ind': subset_machine_r}

        return error_n, error_r

    def plot_test_allocation(self, train_obj, test_obj, plot_file_path):
        x = train_obj['human']['x']
        y = train_obj['human']['y']
        plt.scatter(x, y, c='blue', label='train human')

        x = train_obj['machine']['x']
        y = train_obj['machine']['y']
        plt.scatter(x, y, c='green', label='train machine')

        x = test_obj['machine']['x'][:, 0].flatten()
        y = test_obj['machine']['y']
        plt.scatter(x, y, c='yellow', label='test machine')

        x = test_obj['human']['x'][:, 0].flatten()
        y = test_obj['human']['y']
        plt.scatter(x, y, c='red', label='test human')

        plt.legend()
        plt.grid()
        plt.xlabel('<-----------x------------->')
        plt.ylabel('<-----------y------------->')
        plt.savefig(plot_file_path, dpi=600, bbox_inches='tight')
        plt.close()

    def get_train_error(self, plt_obj, x, y, y_h=None, c=None):
        subset = plt_obj['subset']
        w = plt_obj['w']
        n = y.shape[0]

        if y_h == None:
            err_h = c

        else:
            err_h = (y_h - y) ** 2

        y_m = x.dot(w)
        err_m = (y_m - y) ** 2
        error = (err_h[subset].sum() + err_m.sum() - err_m[subset].sum()) / float(n)

        return {'error': error}

    def compute_result(self, res_file, data_file, option, test_method):
        data = load_data(data_file)
        res = load_data(res_file)

        for std, i0 in zip(self.list_of_std, range(len(self.list_of_std))):
            for K, i1 in zip(self.list_of_K, range(len(self.list_of_K))):
                for lamb, i2 in zip(self.list_of_lamb, range(len(self.list_of_lamb))):
                    if option in res[str(std)][str(K)][str(lamb)]:
                        res_obj = res[str(std)][str(K)][str(lamb)][option]
                        train_res = self.get_train_error(res_obj, data['X'], data['Y'], y_h=None, c=data['c'][str(std)])

                        if test_method == 'NN':
                            test_res_n, test_res_r = self.get_test_error(res_obj, data['dist_mat'], data['test']['X'],
                                                                         data['test']['Y'], y_h=None,
                                                                         c=data['test']['c'][str(std)])
                        else:
                            test_res_n, test_res_r = self.classification_get_test_error(res_obj, data['dist_mat'],
                                                                                        test_method,
                                                                                        data['X'], data['test']['X'],
                                                                                        data['test']['Y'], y_h=None,
                                                                                        c=data['test']['c'][str(std)])

                        if 'test_res' not in res[str(std)][str(K)][str(lamb)][option]:
                            res[str(std)][str(K)][str(lamb)][option]['test_res'] = {}

                        res[str(std)][str(K)][str(lamb)][option]['test_res'][test_method] = {'ranking': test_res_r,
                                                                                             test_method: test_res_n}
                        res[str(std)][str(K)][str(lamb)][option]['train_res'] = train_res

        save(res, res_file)

    def split_res_over_K(self, data_file, res_file, unified_K, option):
        res = load_data(res_file)
        for std in self.list_of_std:
            if str(std) not in res:
                res[str(std)] = {}

            for K in self.list_of_K:
                if str(K) not in res[str(std)]:
                    res[str(std)][str(K)] = {}

                for lamb in self.list_of_lamb:
                    if str(lamb) not in res[str(std)][str(K)]:
                        res[str(std)][str(K)][str(lamb)] = {}

                    if option not in res[str(std)][str(K)][str(lamb)]:
                        res[str(std)][str(K)][str(lamb)][option] = {}

                        if K != unified_K:
                            res_dict = res[str(std)][str(unified_K)][str(lamb)][option]
                            if res_dict:
                                res[str(std)][str(K)][str(lamb)][option] = self.get_res_for_subset(data_file, res_dict,
                                                                                                   lamb, K)
        save(res, res_file)

    def get_optimal_pred(self, data, subset, lamb):
        n, dim = data['X'].shape
        subset_c = np.array([int(i) for i in range(n) if i not in subset])
        X_sub = data['X'][subset_c].T
        Y_sub = data['Y'][subset_c]
        subset_c_l = n - subset.shape[0]
        return LA.inv(lamb * subset_c_l * np.eye(dim) + X_sub.dot(X_sub.T)).dot(X_sub.dot(Y_sub))

    def get_res_for_subset(self, data_file, res_dict, lamb, K):
        data = load_data(data_file)
        curr_n = int(data['X'].shape[0] * K)
        subset_tr = res_dict['subset'][:curr_n]
        w = self.get_optimal_pred(data, subset_tr, lamb)
        return {'w': w, 'subset': subset_tr}

    def set_n(self, n):
        self.n = n


def main():
    latexify()
    list_of_test_option = ['MLP', 'LR', 'NN']
    list_of_file_names = sys.argv[1:]

    image_path = 'plots/'
    if not os.path.exists(image_path):
        os.mkdir(image_path)

    for file_name in list_of_file_names:
        print 'plotting ' + file_name
        list_of_option, list_of_std, list_of_lamb = parse_command_line_input(file_name)
        list_of_K = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        data_file = 'data/data_dict_' + file_name
        res_file = 'Results/' + file_name + '_res'

        obj = plot_triage_real(list_of_K, list_of_std, list_of_lamb, list_of_option, list_of_test_option)
        if file_name in ['Wgauss', 'Wsigmoid']:
            obj.plot_subset(data_file, res_file, list_of_std, list_of_lamb, image_path, file_name)

        else:
            obj.set_n(load_data(data_file)['X'].shape[0])
            for idx, test_method in enumerate(list_of_test_option):
                for option in list_of_option:
                    if option not in ['diff_submod', 'RLSR', 'RLSR_Reg']:
                        unified_K = 0.99
                        obj.split_res_over_K(data_file, res_file, unified_K, option)
                    obj.compute_result(res_file, data_file, option, test_method)

                if file_name.startswith('U'):
                    obj.U_get_avg_error_vary_K(res_file, test_method, file_name, image_path)
                else:
                    obj.get_avg_error_vary_K(res_file, image_path,
                                             file_name, test_method)
            if not file_name.startswith('U'):
                obj.get_avg_error_vary_testmethod(res_file, image_path, file_name, 'greedy')


if __name__ == "__main__":
    main()
