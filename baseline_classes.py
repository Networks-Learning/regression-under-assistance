import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt


class kl_triage:
    def __init__(self, data):
        self.X = data['X']
        self.Y = data['Y']
        self.c = data['c']
        self.lamb = data['lamb']
        self.n, self.dim = self.X.shape
        self.V = np.arange(self.n)
        self.training()

    def training(self):
        self.train_machine_error()
        self.train_human_error()

    def get_subset(self, K):
        machine_err = self.X.dot(self.w_machine_error)
        human_err = self.X.dot(self.w_human_error)
        err = np.sqrt(self.c) - np.absolute(self.machine_err)
        indices = np.argsort(err)
        return indices[:K]

    def train_machine_error(self):
        self.w_machine_pred, self.machine_err = self.fit_LR(self.X, self.Y)
        self.w_machine_error, tmp = self.fit_LR(self.X, self.machine_err)

    def train_human_error(self):
        self.w_human_error, tmp = self.fit_LR(self.X, np.sqrt(self.c))

    def fit_LR(self, X, Y):
        w = LA.solve(X.T.dot(X) + self.lamb * np.eye(X.shape[1]), X.T.dot(Y))
        err = np.absolute((X.dot(w) - Y))
        return w, err


class Submodularity_ratio:
    def __init__(self, data):
        self.X = self.normalize_feature(data['X'])
        self.Y = data['Y']
        self.c = data['c']
        self.lamb = data['lamb']
        self.n, self.dim = self.X.shape
        self.V = np.arange(self.n)
        self.g = G({'X': self.X, 'Y': self.Y, 'c': self.c, 'lamb': self.lamb})
        self.k_l = self.get_k_l()

    def normalize_feature(self, X):
        n, dim = X.shape
        for i in range(dim):
            feature = X[:, i]
            X[:, i] = np.true_divide(feature, LA.norm(feature.flatten()))
        return X

    def get_c(self, subset):
        return np.array([int(i) for i in self.V if i not in subset])

    def get_k_l(self):
        empty_set = np.array([]).astype(int)
        l_Null = self.l(empty_set)
        l_V = self.l(self.V)

        print 'l_V', l_V
        l_V_k = [(self.l(self.get_c(np.array([k]))) - l_V) for k in self.V]
        plt.plot(np.array(l_V_k).flatten())
        plt.show()
        print min(l_V_k)

        return (np.log(l_Null - min(l_V_k)) / np.log(l_Null))

    def l(self, subset):
        return self.g.eval_l(subset)

    def get_L(self):
        F_V = - self.g.eval(self.V)
        F_V_k = np.array([- self.g.eval(self.get_c(np.array([k]))) for k in range(self.n)])
        diff_f = np.zeros((self.n, self.n))

        for k1 in range(self.n):
            for k2 in range(self.n):
                if k2 != k1:
                    diff_f[k1, k2] = self.get_f([k1, k2]) - self.get_f([k1])

        term1 = F_V / np.amax(diff_f)  # (1-self.k_l)*
        term2 = F_V / np.max(F_V_k - F_V)

        print 'FV', F_V
        print 'FVK-FV', np.max(F_V_k - F_V)
        print 'diff f ', np.amax(diff_f)
        print term1
        print term2

        return min(term1, term2)

    def get_f(self, list_of_elm):
        subset_c = np.array(list_of_elm)
        y = self.Y[subset_c].reshape(subset_c.shape[0], 1)
        x = self.X[subset_c].T
        yTy = y.T.dot(y)
        xy = x.dot(y)
        xxT = x.dot(x.T)
        c_S = self.c.sum() - self.c[subset_c].sum()
        B = self.lamb * (subset_c.shape[0]) * np.eye(self.dim)
        a11 = yTy + c_S
        a12 = xy.T
        a21 = xy
        a22 = B + xxT

        A = np.vstack((np.hstack((a11, a12)), np.hstack((a21, a22))))

        return np.log(LA.det(A))


class modular_distort_greedy:
    def __init__(self, data):
        self.X = data['X']
        self.Y = data['Y']
        self.c = data['c']
        self.lamb = data['lamb']
        self.g = G({'X': self.X, 'Y': self.Y, 'c': self.c, 'lamb': self.lamb})
        self.n, self.dim = self.X.shape
        self.V = np.arange(self.n)
        self.initialize()

    def initialize(self):
        G_null = self.g.eval(np.array([]).astype(int))
        self.null_val = max(0.0, - self.g.eval(np.array([]).astype(int)))

        G_ascend = np.array([self.g.eval(np.arange(i + 1)) - self.g.eval(np.arange(i)) for i in self.V])
        self.w = np.array([np.max(np.array([0.0, G_ascend[i]])) for i in self.V]).flatten()

    def eval(self, subset):
        return self.null_val + self.w[subset].sum()  # ( )

    def get_c(self, subset):
        return np.array([int(i) for i in range(self.n) if i not in subset])

    def get_inc_arr(self, subset, rest_flag=False, subset_rest=None):
        if rest_flag:
            subset_c = subset_rest
        else:
            subset_c = self.get_c(subset)

        l = np.array([self.w[i] for i in subset_c]).flatten()

        return l


class modular:
    def __init__(self, constant, vec):
        self.constant = constant
        self.vec = vec

    def get_m(self, subset):
        if subset.size == 0:
            return self.constant
        # print '***',subset.astype(int),'****'
        return self.constant + self.vec[subset.astype(int)].sum()

    def get_m_singleton(self, ground_set):
        tmp = np.zeros(ground_set.shape[0])
        for i in range(ground_set.shape[0]):
            tmp[i] = self.get_m(np.array([int(ground_set[i])]))
        return tmp


class G:
    def __init__(self, input):
        self.X = input['X']
        self.Y = input['Y']
        self.lamb = input['lamb']
        self.c = input['c']
        self.dim = self.X.shape[1]
        self.n = self.X.shape[0]
        self.V = np.arange(self.n)
        self.init_data_str()

    def reset(self):
        self.init_data_str()

    def get_c(self, subset):
        return np.array([int(i) for i in self.V if i not in subset])

    def init_data_str(self):
        self.yTy = self.Y.dot(self.Y)
        self.xxT = self.X.T.dot(self.X)
        self.xy = self.X.T.dot(self.Y.reshape(self.n, 1))
        self.c_S = 0
        self.curr_set_len = 0

    def update_data_str(self, elm):
        y = self.Y[elm]
        x = self.X[elm].reshape(self.dim, 1)
        self.yTy -= y * y
        self.xxT -= x.dot(x.T)
        self.xy -= y * x
        self.c_S += self.c[elm]
        self.curr_set_len += 1

    def give_inc(self, elm):
        y = self.Y[elm]
        x = self.X[elm].reshape(self.dim, 1)
        yTy = self.yTy - y * y
        xxT = self.xxT - x.dot(x.T)
        xy = self.xy - y * x
        c_S = self.c_S + self.c[elm]
        B = self.lamb * (self.n - self.curr_set_len - 1) * np.eye(self.dim)

        return -np.log(yTy - xy.T.dot(LA.inv(B + xxT).dot(xy)) + c_S)

    def eval_curr(self):
        B = self.lamb * (self.n - self.curr_set_len) * np.eye(self.dim)
        tmp = -np.log(self.yTy - self.xy.T.dot(LA.inv(B + self.xxT).dot(self.xy)) + self.c_S)
        return tmp

    def get_inc_arr(self, subset, rest_flag=False, subset_rest=None):
        if rest_flag:
            subset_c = subset_rest
        else:
            subset_c = self.get_c(subset)

        vec = []
        G_S = self.eval_curr()[0][0]

        for i in subset_c:
            vec.append(self.give_inc(i)[0][0] - G_S)

        return np.array(vec), subset_c

    def eval(self, subset=None):
        if subset.shape[0] == self.n:
            return -np.log(self.c.sum())

        subset_c = self.get_c(subset)

        if subset.size == 0:
            c_S = 0
        else:
            c_S = self.c[subset].sum()

        y = self.Y[subset_c].reshape(subset_c.shape[0], 1)
        x = self.X[subset_c].T
        yTy = y.T.dot(y)
        xy = x.dot(y)
        xxT = x.dot(x.T)
        B = self.lamb * (self.n - subset.shape[0]) * np.eye(self.dim)

        return -np.log(yTy - xy.T.dot(LA.inv(B + xxT).dot(xy)) + c_S)

    def eval_l(self, subset=None):
        if subset.shape[0] == self.n:
            return (self.c.sum())

        subset_c = self.get_c(subset)

        if subset.size == 0:
            c_S = 0
        else:
            c_S = self.c[subset].sum()

        y = self.Y[subset_c].reshape(subset_c.shape[0], 1)
        x = self.X[subset_c].T
        yTy = y.T.dot(y)
        xy = x.dot(y)
        xxT = x.dot(x.T)
        B = self.lamb * (self.n - subset.shape[0]) * np.eye(self.dim)

        return (yTy - xy.T.dot(LA.inv(B + xxT).dot(xy)) + c_S)


class F:
    def __init__(self, input):
        self.X = input['X']
        self.Y = input['Y']
        self.c = input['c']
        self.lamb = input['lamb']
        self.dim = self.X.shape[1]
        self.n = self.X.shape[0]
        self.V = np.arange(self.n)
        self.BIG_VALUE = 100000

    def get_c(self, subset):
        return np.array([int(i) for i in self.V if i not in subset])

    def get_minus(self, subset, elm):
        return np.array([i for i in subset if i != elm])

    def get_added(self, subset, elm):
        return np.concatenate((subset, np.array([int(elm)])), axis=0)

    def elm_mat(self, elm):
        v = np.hstack((np.array([self.Y[elm]]), self.X[elm]))
        return v.reshape(self.dim + 1, 1).dot(v.reshape(1, self.dim + 1))

    def addend(self, l, subset=None):
        arr = np.eye(self.dim + 1)
        for i in range(self.dim + 1):
            if i == 0:
                if l == 0:
                    arr[0, 0] = 0
                else:
                    arr[0, 0] = self.c[subset].sum()
            else:
                arr[i, i] = self.lamb * (self.n - l)
        return arr

    def modular_upper_bound(self, subset):
        l_subset = subset.shape[0]
        Y_X = np.concatenate((self.Y.reshape(1, self.n), self.X.T), axis=0)
        subset_c = self.get_c(subset)
        Y_X_sub = Y_X[:, subset_c]
        A = Y_X_sub.dot(Y_X_sub.T)
        B = self.addend(l_subset, subset)

        f_subset = np.log(LA.det(A + B))
        f_inc = np.zeros(self.n)
        for elm in subset:
            buffer_new = A + self.elm_mat(elm) + self.addend(l_subset - 1, self.get_minus(subset, elm))
            f_inc[elm] = f_subset - np.log(LA.det(buffer_new))

        A = Y_X.dot(Y_X.T)
        B = self.addend(0, np.array([]))
        f_null = np.log(LA.det(A + B))

        for elm in subset_c:
            buffer_new = A - self.elm_mat(elm) + self.addend(1, np.array([elm]))
            f_inc[elm] = np.log(LA.det(buffer_new)) - f_null

        if subset.size == 0:
            m_f = modular(f_subset, f_inc)
        else:
            m_f = modular(f_subset - f_inc[subset].sum(), f_inc)

        return m_f


class SubMod:
    def __init__(self, input):
        self.X = input['X']
        self.lamb = input['lamb']
        self.dim = self.X.shape[1]
        self.n = self.X.shape[0]
        self.V = np.arange(self.n)
        self.BIG_VALUE = 100000
        self.m = input['m']

    def elm_mat(self, elm):
        return self.X[elm].reshape(self.dim, 1).dot(self.X[elm].reshape(1, self.dim))

    def addend(self, l):
        return self.lamb * (self.n - l) * np.eye(self.dim)

    def get_c(self, subset):
        return np.array([int(i) for i in self.V if i not in subset])

    def get_minus(self, subset, elm):
        return np.array([i for i in subset if i != elm])

    def get_added(self, subset, elm):
        return np.concatenate((subset, np.array([int(elm)])), axis=0)

    def find_max_elm(self, ground_set):
        g = np.zeros(ground_set.shape[0])
        A = self.X.T.dot(self.X)
        B = self.addend(1)

        for elm, idx in zip(ground_set, range(ground_set.shape[0])):
            g[idx] = np.log(LA.det(B + A - self.elm_mat(elm)))

        ind = np.argmax(g - self.m.get_m_singleton(ground_set))

        return ground_set[ind]

    def eval(self, subset):
        l_subset = subset.shape[0]
        subset_c = self.get_c(subset)
        X_sub = self.X[subset_c].T
        A = X_sub.dot(X_sub.T)
        B = self.addend(l_subset)
        g_subset = np.log(LA.det(A + B))

        return (g_subset - self.m.get_m(subset))

    def eval_vector(self, subset):
        l_subset = subset.shape[0]
        subset_c = self.get_c(subset)
        X_sub = self.X[subset_c].T
        A = X_sub.dot(X_sub.T)
        B = self.addend(l_subset - 1)
        g_S = []

        for elm in subset:
            g_S.append(np.log(LA.det(A + self.elm_mat(elm) + B)))
        g_S_m = np.zeros(subset.shape[0])

        for i in range(subset.shape[0]):
            g_S_m[i] = g_S[i] - self.m.get_m(self.get_minus(subset, subset[i]))  # -(m[subset].sum() - m[subset] )

        return g_S_m

    def eval_exch_or_add(self, subset, ground_set, K):

        l_subset = subset.shape[0]
        subset_c = self.get_c(subset)
        subset_c_gr = np.array([i for i in ground_set if i not in subset])
        X_sub = self.X[subset_c].T
        A = X_sub.dot(X_sub.T)
        B = self.addend(l_subset)
        subset_with_null = self.get_added(subset, -1)
        g_m_exchange = np.zeros((subset_with_null.shape[0], subset_c_gr.shape[0]))
        flag_no_add = False

        if subset.shape[0] == K:
            flag_no_add = True
            g_m_exchange[-1] = -1 * self.BIG_VALUE * np.ones(subset_c_gr.shape[0])

        # declare
        for e, row_ind in zip(subset_with_null, range(subset_with_null.shape[0])):
            for d, col_ind in zip(subset_c_gr, range(subset_c_gr.shape[0])):
                if e == -1:
                    if not flag_no_add:
                        g_part = np.log(LA.det(A - self.elm_mat(d) + self.addend(l_subset + 1)))
                        m_part = self.m.get_m(self.get_added(subset, int(d)))
                        g_m_exchange[row_ind][col_ind] = g_part - m_part

                else:
                    g_part = np.log(LA.det(A + self.elm_mat(e) - self.elm_mat(d) + B))
                    m_part = self.m.get_m(self.get_added(self.get_minus(subset, e), d))
                    g_m_exchange[row_ind][col_ind] = g_part - m_part

        return g_m_exchange, subset_with_null, subset_c_gr

    def get_inc_arr(self, subset):

        l_subset = subset.shape[0]
        subset_c = self.get_c(subset)
        X_sub = self.X[subset_c].T
        A = X_sub.dot(X_sub.T)
        B = self.addend(l_subset)
        g_m_subset = np.log(LA.det(A + B)) - self.m.get_m(subset)
        g_inc = np.zeros(subset_c.shape[0])

        for elm, elm_idx in zip(subset_c, range(subset_c.shape[0])):
            g_part = np.log(LA.det(A - self.elm_mat(elm) + self.addend(l_subset + 1)))
            m_part = self.m.get_m(self.get_added(subset, int(elm)))
            g_inc[elm_idx] = g_part - m_part - g_m_subset

        return g_inc, subset_c
