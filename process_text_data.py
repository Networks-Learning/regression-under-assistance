import numpy.random as rand
import codecs
import csv
import random
from myutil import *
import numpy as np
import numpy.linalg as LA


class preprocess_triage_real_data:
    def __init__(self):
        pass

    def process_hate_speech_data(self, src_file, dest_file):

        with open(src_file, 'r') as f:
            f.readline()
            dict_tweet = {}
            response_list = []
            human_annotation_list = []
            while True:
                line_full = f.readline()
                print '&&', line_full, '&&'
                if not line_full:
                    save({'tweets': dict_tweet, 'y': response_list, 'y_h': human_annotation_list}, dest_file)
                    return
                # return dict_tweet,response_list,human_annotation_list
                else:
                    if line_full.isspace():
                        print 'empty'
                    else:
                        line = line_full.split(',', 7)
                        if len(line) == 7:
                            tid = line[0]
                            tweet = line[-1]
                            dict_tweet[tid] = tweet
                            y, y_h = self.get_annotations(line[1:-1])
                            response_list.append(y)
                            human_annotation_list.append(y_h)

    def get_annotations(self, list_of_arg):
        human_response = []
        for i in [1, 2, 3]:
            if int(list_of_arg[i]) > 0:
                human_response.extend([i - 1] * int(list_of_arg[i]))
        response = int(list_of_arg[-1])
        return response, human_response

    def dict_to_txt(self, tweet_dict, file_w):
        with open(file_w, 'w') as f:
            for tweet in tweet_dict.values():
                f.write(tweet)

    def map_range(self, v, l, h, l_new, h_new):
        return float(v - l) * ((h_new - l_new) / float(h - l)) + l_new

    def convert_tweet_to_vector(self, file_dict, file_vec, file_tweet):

        data_dict = load_data(file_dict)
        data_vec = {}
        n_data = len(data_dict['y'])
        data_vec['y'] = np.array([self.map_range(i, 0, 2, 0, 1) for i in data_dict['y']])
        data_vec['c'] = np.zeros(n_data)

        for ind, human_pred, response in zip(range(n_data), data_dict['y_h'], data_vec['y']):
            human_pred_scaled = [self.map_range(i, 0, 2, 0, 1) for i in human_pred]
            data_vec['c'][ind] = (np.mean(np.array(human_pred_scaled)) - float(response)) ** 2
        # self.dict_to_txt(data_dict['tweets'],tweet_file)
        model = fasttext.train_unsupervised(file_tweet, model='skipgram')
        x = []
        for tid in data_dict['tweets'].keys():
            tweet = data_dict['tweets'][tid].replace('\n', ' ')
            x.append(model.get_sentence_vector(tweet).flatten())
        data_vec['x'] = np.array(x)
        # or, cbow model :
        # model = fasttext.train_unsupervised('data.txt', model='cbow')
        save(data_vec, file_vec)

    def truncate_data(self, data_file, data_file_tr):
        data = load_data(data_file)
        n = data['y'].shape[0]
        n_tr = int(n / 4)
        data['x'] = data['x'][:n_tr]
        data['y'] = data['y'][:n_tr]
        data['c'] = data['c'][:n_tr]
        save(data, data_file_tr)

    def split_data(self, frac, file_data, file_data_split):

        data = load_data(file_data)

        print 'x', data['x'].shape
        print 'y', data['y'].shape
        print 'c', data['c'].shape
        # return
        num_data = data['y'].shape[0]
        num_train = int(frac * num_data)
        num_test = num_data - num_train
        indices = np.arange(num_data)
        random.shuffle(indices)
        indices_train = indices[:num_train]
        indices_test = indices[num_train:]
        data_split = {}
        data_split['X'] = data['x'][indices_train]
        data_split['Y'] = data['y'][indices_train]
        data_split['c'] = data['c'][indices_train]
        test = {}
        test['X'] = data['x'][indices_test]
        test['Y'] = data['y'][indices_test]
        test['c'] = data['c'][indices_test]
        data_split['test'] = test
        data_split['dist_mat'] = np.zeros((num_test, num_train))
        for te in range(num_test):
            for tr in range(num_train):
                data_split['dist_mat'][te, tr] = LA.norm(test['X'][te] - data_split['X'][tr])
        save(data_split, file_data_split)

    def change_format_hatespeech(self, data_file, dest_file):
        data = load_data(data_file)
        c = {'0.0': np.copy(data['c'])}
        test_c = {'0.0': np.copy(data['test']['c'])}
        data['c'] = c
        data['test']['c'] = test_c
        save(data, dest_file)


def main():
    # preprocesses hatespeech data
    path = '../../Real_Data/Hatespeech/Davidson/'
    src_file = path + 'labeled_data.csv'
    obj = preprocess_triage_real_data()
    dest_file = path + 'data'
    tweet_file = path + 'tweets.txt'
    vec_file = path + 'data_vectorized'
    vec_tr_file = path + 'data_vectorized_tr'
    vec_split_file = path + 'input_tr'
    vec_full_split_file = path + 'input_full'

    # obj.process_hate_speech_data(src_file,dest_file)
    # obj.convert_tweet_to_vector(dest_file,vec_file,tweet_file)
    # obj.truncate_data(vec_file, vec_tr_file)
    # obj.split_data(0.8, vec_file , vec_full_split_file)

    dest_file = '../data/hatespeech_full'
    obj.change_format_hatespeech(vec_full_split_file, dest_file)


if __name__ == "__main__":
    main()
