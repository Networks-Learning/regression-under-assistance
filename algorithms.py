import math
from baseline_classes import *
import numpy as np
import numpy.linalg as LA
import random
import time


class triage_human_machine:
    def __init__(self, data_dict, real=None):
        self.X = data_dict['X']
        self.Y = data_dict['Y']

        if real:
            self.c = data_dict['c']

        else:
            self.c = np.square(data_dict['human_pred'] - data_dict['Y'])

        self.dim = self.X.shape[1]
        self.n = self.X.shape[0]
        self.V = np.arange(self.n)
        self.epsilon = float(1)
        self.BIG_VALUE = 100000
        self.real = real

    def get_c(self, subset):
        return np.array([int(i) for i in self.V if i not in subset])

    def get_minus(self, subset, elm):
        return np.array([i for i in subset if i != elm])

    def get_added(self, subset, elm):
        return np.concatenate((subset, np.array([int(elm)])), axis=0)

    def check_delete(self, g_m, subset, approx):
        if subset.size == 0:
            # print 'subset empty'
            return False, subset

        g_m_subset = g_m.eval(subset)
        g_m_subset_vector = g_m.eval_vector(subset)

        if np.max(g_m_subset_vector) >= g_m_subset * approx:
            item_to_del = subset[np.argmax(g_m_subset_vector)]
            subset_left = self.get_minus(subset, item_to_del)
            # print '----Following item is deleted---',item_to_del
            # print 'now subset---> ', subset_left

            return True, subset_left
        # print 'Nothing deleted, subset ---> ', subset
        # print 'No deletion'
        # print 'curr function --> ',g_m_subset
        # print 'after deletion best --> ', np.max(g_m_subset_vector)
        return False, subset

    def check_exchange_greedy(self, g_m, subset, ground_set, approx, K):

        g_m_subset = g_m.eval(subset)
        g_m_exchange, subset_with_null, subset_c_gr = g_m.eval_exch_or_add(subset, ground_set, K)

        if np.max(g_m_exchange) > g_m_subset * approx:
            r, c = np.unravel_index(np.argmax(g_m_exchange, axis=None), g_m_exchange.shape)
            # print 'index of max element ',r,c
            e = subset_with_null[r]
            d = subset_c_gr[c]
            # print e,' is exchanged with ',d
            if e == -1:
                subset_with_null[r] = d
                return True, subset_with_null
            else:
                ind_e = np.where(subset == e)[0]
                subset[ind_e] = d
                return True, subset
        # print 'No Exchange'
        # print 'curr function --> ',g_m_subset
        # print 'after deletion best --> ', np.max(g_m_exchange)
        return False, subset

    def approx_local_search(self, g_m, K, ground_set):
        # max_A (g-m)(A) given |A|<=k  	implementing local search by J.Lee 2009 STOC
        approx = 1 + self.epsilon / float(self.n ** 4)
        curr_subset = np.array([g_m.find_max_elm(ground_set)])
        while True:
            # print ' ---   Delete ----- '
            flag_delete, curr_subset = self.check_delete(g_m, curr_subset, approx)
            if flag_delete:
                # print 'deleted'
                pass
            else:
                # print ' --- Exchange ---- '
                flag_exchange, curr_subset = self.check_exchange_greedy(g_m, curr_subset, ground_set, approx, K)
                # time.sleep(100000)
                if flag_exchange:
                    pass  # print 'exchanged'
                else:
                    break
        return curr_subset

    def constr_submod_max_greedy(self, g_m, K):
        # print 'constr submod max greedy'
        curr_set = np.array([]).astype(int)

        for itr in range(K):
            vector, subset_left = g_m.get_inc_arr(curr_set)
            if np.max(vector) <= 0:
                break

            idx_to_add = subset_left[np.argmax(vector)]
            curr_set = self.get_added(curr_set, idx_to_add)
        return curr_set

    def constr_submod_max(self, g_m, K):
        ground_set = self.V
        # print '----- local search 1 '
        start = time.time()
        subset_1 = self.approx_local_search(g_m, K, ground_set)
        ground_set = self.get_c(subset_1)
        # print '----- local search 2 '
        subset_2 = self.approx_local_search(g_m, K, ground_set)
        finish = time.time()
        print 'Time -- > ', (finish - start)

        if g_m.eval(subset_1) > g_m.eval(subset_2):
            return subset_1
        else:
            return subset_2

    def sel_subset_diff_submod_greedy(self):
        # solve difference of submodular functions
        subset_old = np.array([])
        g_f = G({'X': self.X, 'Y': self.Y, 'c': self.c, 'lamb': self.lamb})
        val_old = g_f.eval(subset_old)
        itr = 0

        while True:
            # print 'modular upper bound '
            f = F({'X': self.X, 'Y': self.Y, 'c': self.c, 'lamb': self.lamb})
            m_f = f.modular_upper_bound(subset_old)
            g_m = SubMod({'X': self.X, 'lamb': self.lamb, 'm': m_f})
            subset = self.constr_submod_max_greedy(g_m, self.K)

            # check whether g-f really improve
            val_curr = g_f.eval(subset)
            if val_curr <= val_old:
                return subset_old

            if set(subset) == set(subset_old):
                return subset
            else:
                subset_old = subset
                val_old = val_curr

            itr += 1

    def sel_subset_diff_submod(self):
        # solve difference of submodular functions
        subset_old = np.array([])
        itr = 0

        while True:
            f = F({'X': self.X, 'Y': self.Y, 'c': self.c, 'lamb': self.lamb})
            m_f = f.modular_upper_bound(subset_old)
            g_m = SubMod({'X': self.X, 'lamb': self.lamb, 'm': m_f})
            subset = self.constr_submod_max(g_m, self.K)
            print 'subset length', subset.shape

            if set(subset) == set(subset_old):
                return subset
            else:
                subset_old = subset

            itr += 1

    def set_param(self, lamb, K):
        self.lamb = lamb
        self.K = K

    def get_optimal_pred(self, subset):
        subset_c = self.get_c(subset)
        X_sub = self.X[subset_c].T
        Y_sub = self.Y[subset_c]
        subset_c_l = self.n - subset.shape[0]
        return LA.inv(self.lamb * subset_c_l * np.eye(self.dim) + X_sub.dot(X_sub.T)).dot(X_sub.dot(Y_sub))

    def plot_subset(self, w, subset):
        plt_obj = {}

        x = self.X[subset, 0].flatten()
        y = self.Y[subset]
        plt_obj['human'] = {'x': x, 'y': y}

        c_subset = self.get_c(subset)
        x = self.X[c_subset, 0].flatten()
        y = self.Y[c_subset]
        plt_obj['machine'] = {'x': x, 'y': y}

        x = self.X[:, 0].flatten()
        y = self.X.dot(w).flatten()
        plt_obj['prediction'] = {'x': x, 'y': y, 'w': w}

        return plt_obj

    def distort_greedy(self, g, K, gamma):
        c_mod = modular_distort_greedy({'X': self.X, 'Y': self.Y, 'c': self.c, 'lamb': self.lamb})
        subset = np.array([]).astype(int)
        g.reset()
        for itr in range(K):
            frac = (1 - gamma / float(K)) ** (K - itr - 1)
            subset_c = self.get_c(subset)
            c_mod_inc = c_mod.get_inc_arr(subset).flatten()
            g_inc_arr, subset_c_ret = g.get_inc_arr(subset)
            g_pos_inc = g_inc_arr.flatten() + c_mod_inc
            inc_vec = frac * g_pos_inc - c_mod_inc

            if np.max(inc_vec) <= 0:
                print 'no increment'
                return subset

            sel_ind = np.argmax(inc_vec)
            elm = subset_c[sel_ind]
            subset = self.get_added(subset, elm)
            g.update_data_str(elm)
        return subset

    def stochastic_distort_greedy(self, g, K, gamma, epsilon):
        c_mod = modular_distort_greedy({'X': self.X, 'Y': self.Y, 'c': self.c, 'lamb': self.lamb})
        subset = np.array([]).astype(int)
        g.reset()
        s = int(math.ceil(self.n * np.log(float(1) / epsilon) / float(K)))
        print 'subset_size', s, 'K-->', K, ', n --> ', self.n

        for itr in range(K):
            frac = (1 - gamma / float(K)) ** (K - itr - 1)
            subset_c = self.get_c(subset)

            if s < subset_c.shape[0]:
                subset_choosen = np.array(random.sample(subset_c, s))
            else:
                subset_choosen = subset_c

            c_mod_inc = c_mod.get_inc_arr(subset, rest_flag=True, subset_rest=subset_choosen)
            g_inc_arr, subset_c_ret = g.get_inc_arr(subset, rest_flag=True, subset_rest=subset_choosen)
            g_pos_inc = g_inc_arr + c_mod_inc
            inc_vec = frac * g_pos_inc - c_mod_inc

            if np.max(inc_vec) <= 0:
                return subset

            sel_ind = np.argmax(inc_vec)
            elm = subset_choosen[sel_ind]
            subset = self.get_added(subset, elm)
            g.update_data_str(elm)

        return subset

    def gamma_sweep_distort_greedy(self, delta=0.01, T=5, flag_stochastic=None):
        g = G({'X': self.X, 'Y': self.Y, 'c': self.c, 'lamb': self.lamb})
        Submod_ratio = 0.7
        if T != 5:
            delta = 0.05
        subset = {}
        G_subset = []
        gamma = 1.0

        for r in range(T + 1):
            if flag_stochastic:
                subset_sel = self.stochastic_distort_greedy(g, self.K, gamma, delta)
            else:
                subset_sel = self.distort_greedy(g, self.K, gamma)
            subset[str(r)] = subset_sel
            G_subset.append(g.eval(subset_sel))
            gamma = gamma * (1 - delta)
        empty_set = np.array([]).astype(int)
        subset[str(T + 1)] = empty_set
        G_subset.append(g.eval(empty_set))
        max_set_ind = np.argmax(np.array(G_subset))

        return subset[str(max_set_ind)]

    def max_submod_greedy(self):
        curr_set = np.array([]).astype(int)
        g = G({'X': self.X, 'Y': self.Y, 'c': self.c, 'lamb': self.lamb})
        # print 'Need to select ', self.K , ' items'
        for itr in range(self.K):
            vector, subset_left = g.get_inc_arr(curr_set)
            idx_to_add = subset_left[np.argmax(vector)]
            curr_set = self.get_added(curr_set, idx_to_add)
            g.update_data_str(idx_to_add)

        return curr_set

    def kl_triage_subset(self):
        kl_obj = kl_triage({'X': self.X, 'Y': self.Y, 'c': self.c, 'lamb': self.lamb})
        return kl_obj.get_subset(self.K)

    def hard_threshold(self, v, k):
        vsorted = np.argsort(v)
        vsorteddec = vsorted[::-1]

        for i in range(k, v.shape[0]):
            v[vsorteddec[i]] = 0

        return v

    def CRR_subset(self):
        X = self.X
        X = np.swapaxes(X, 1, 0)
        y = self.Y
        y = y.reshape(y.shape[0], 1)

        tolerance = .001
        bprev = np.random.uniform(10, 20, y.shape)
        bcur = np.zeros(y.shape)
        Xtr = X.T
        XXtr = X.dot(Xtr)
        XXtrInv = LA.inv(XXtr)
        P_X = (Xtr.dot(XXtrInv)).dot(X)

        while LA.norm(bcur - bprev) > tolerance:
            tmp = np.copy(bcur)
            bcur = self.hard_threshold((P_X.dot(bcur) + (np.eye(X.shape[1]) - P_X).dot(y)).reshape(X.shape[1]), self.K)
            bcur = bcur.reshape(X.shape[1], 1)
            bprev = np.copy(tmp)

        subset = [i for i in range(len(bcur)) if bcur[i] != 0]
        subset = np.array(subset)
        w = (XXtrInv.dot(X)).dot(y - bcur)
        w = w.reshape(w.shape[0])
        return subset

    def CRR_Reg(self):
        X = self.X
        X = np.swapaxes(X, 1, 0)
        y = self.Y
        y = y.reshape(y.shape[0], 1)

        tolerance = .001
        bprev = np.random.uniform(10, 20, y.shape)
        bcur = np.zeros(y.shape)
        Xtr = X.T
        XXtr = X.dot(Xtr)
        XXtr = XXtr + self.lamb * np.eye(self.dim)  # regularized
        XXtrInv = LA.inv(XXtr)
        P_X = (Xtr.dot(XXtrInv)).dot(X)

        while LA.norm(bcur - bprev) > tolerance:
            tmp = np.copy(bcur)
            bcur = self.hard_threshold((P_X.dot(bcur) + (np.eye(X.shape[1]) - P_X).dot(y)).reshape(X.shape[1]), self.K)
            bcur = bcur.reshape(X.shape[1], 1)
            bprev = np.copy(tmp)

        subset = [i for i in range(len(bcur)) if bcur[i] != 0]
        subset = np.array(subset)
        w = (XXtrInv.dot(X)).dot(y - bcur)
        w = w.reshape(w.shape[0])
        return subset

    def algorithmic_triage(self, param, optim):
        # start=time.time()
        self.set_param(param['lamb'], int(param['K'] * self.n))

        if optim == 'RLSR':
            subset = self.CRR_subset()

        if optim == 'RLSR_Reg':
            subset = self.CRR_Reg()

        if optim == 'diff_submod':
            subset = self.sel_subset_diff_submod()

        if optim == 'greedy':
            subset = self.max_submod_greedy()

        if optim == 'diff_submod_greedy':
            subset = self.sel_subset_diff_submod_greedy()

        if optim == 'distort_greedy':
            subset = self.gamma_sweep_distort_greedy(T=param['DG_T'], flag_stochastic=False)

        if optim == 'kl_triage':
            subset = self.kl_triage_subset()

        if optim == 'stochastic_distort_greedy':
            subset = self.gamma_sweep_distort_greedy(flag_stochastic=True)

        if subset.shape[0] == self.n:
            w_m = 0
        else:
            w_m = self.get_optimal_pred(subset)
        # print w_m

        plt_obj = {'w': w_m, 'subset': subset}
        return plt_obj
