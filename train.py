import sys
from myutil import *
from algorithms import triage_human_machine


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


class eval_triage:
    def __init__(self, data_file, real_flag=None, real_wt_std=None):
        self.data = load_data(data_file)
        self.real = real_flag
        self.real_wt_std = real_wt_std

    def eval_loop(self, param, res_file, option):
        print option
        res = load_data(res_file, 'ifexists')
        for std in param['std']:
            if self.real:
                data_dict = self.data
                triage_obj = triage_human_machine(data_dict, self.real)

            else:
                if self.real_wt_std:
                    data_dict = {'X': self.data['X'], 'Y': self.data['Y'], 'c': self.data['c'][str(std)]}
                    triage_obj = triage_human_machine(data_dict, self.real_wt_std)

                else:
                    test = {'X': self.data.Xtest, 'Y': self.data.Ytest,
                            'human_pred': self.data.human_pred_test[str(std)]}
                    data_dict = {'test': test, 'dist_mat': self.data.dist_mat, 'X': self.data.Xtrain,
                                 'Y': self.data.Ytrain, 'human_pred': self.data.human_pred_train[str(std)]}
                    triage_obj = triage_human_machine(data_dict, False)

            if str(std) not in res:
                res[str(std)] = {}

            for K in param['K']:
                if str(K) not in res[str(std)]:
                    res[str(std)][str(K)] = {}

                for lamb in param['lamb']:
                    if str(lamb) not in res[str(std)][str(K)]:
                        res[str(std)][str(K)][str(lamb)] = {}

                    print 'std-->', std, 'K--> ', K, ' Lamb--> ', lamb
                    res_dict = triage_obj.algorithmic_triage({'K': K, 'lamb': lamb, 'DG_T': param['DG_T']},
                                                             optim=option)
                    res[str(std)][str(K)][str(lamb)][option] = res_dict
                    save(res, res_file)

    def check_w(self, res_file, list_of_std, list_of_lamb):
        res = load_data(res_file, 'ifexists')
        list_of_K = [0.2, 0.4, 0.6, 0.8]
        for std in list_of_std:
            for K in list_of_K:
                for lamb, ind in zip(list_of_lamb, range(len(list_of_lamb))):
                    if str(std) not in res:
                        res[str(std)] = {}
                    if str(K) not in res[str(std)]:
                        res[str(std)][str(K)] = {}
                    if str(lamb) not in res[str(std)][str(K)]:
                        res[str(std)][str(K)][str(lamb)] = {}

                    local_data = self.data[str(std)]
                    triage_obj = triage_human_machine(local_data, True)
                    res_dict = triage_obj.algorithmic_triage({'K': K, 'lamb': lamb}, optim='greedy')
                    res[str(std)][str(K)][str(lamb)]['greedy'] = res_dict
                    save(res, res_file)


def main():
    list_of_file_names = sys.argv[1:]

    for file_name in list_of_file_names:
        print 'training ' + file_name
        list_of_option, list_of_std, list_of_lamb = parse_command_line_input(file_name)
        data_file = 'data/data_dict_' + file_name

        if not os.path.exists('Results'):
            os.mkdir('Results')
        res_file = 'Results/' + file_name + '_res'

        if file_name in ['Wgauss', 'Wsigmoid']:
            obj = eval_triage(data_file)
            obj.check_w(res_file, list_of_std, list_of_lamb)
        else:
            DG_T = 5
            if file_name == 'gauss':
                DG_T = 10
            if file_name == 'sigmoid':
                DG_T = 20

            for option in list_of_option:
                if option in ['diff_submod', 'RLSR', 'RLSR_Reg']:
                    list_of_K = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

                else:
                    list_of_K = [0.99]

                param = {'std': list_of_std, 'K': list_of_K, 'lamb': list_of_lamb, 'DG_T': DG_T}
                obj = eval_triage(data_file, real_wt_std=True)
                obj.eval_loop(param, res_file, option)


if __name__ == "__main__":
    main()
