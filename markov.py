import random
from load_data import ner_pad_to_max_len


class HMM(object):
    def __init__(self, degree=2):
        self.n_gram = {}
        self.omit = {}
        self.degree = degree
        self.begin_n_grams = [] # n_gram tuples in the beginning of sents
        self.update('CoNLL2003_train.txt')

    def update(self, filename):
        with open(filename, 'r') as f:
            for line in f:
                ids = [int(item.split('/')[0]) for item in line.split(' ')]
                labels = [int(item.split('/')[1]) for item in line.split(' ')]
                for i in range(len(ids)):
                    if ids[i] in self.omit:
                        self.omit[ids[i]][labels[i]] = self.omit[ids[i]].get(labels[i], 0) + 1
                    else:
                        self.omit[ids[i]] = {labels[i]: 1}
                ids = [-1] + ids + [-2]
                if len(ids) <= self.degree:
                    return
                for i in range(len(ids) - self.degree):
                    n_gram_key = tuple(ids[i + j] for j in range(self.degree))
                    if i == 0:
                        self.begin_n_grams.append(n_gram_key)
                    next_id = ids[i + self.degree]
                    if n_gram_key in self.n_gram:
                        self.n_gram[n_gram_key][next_id] = self.n_gram[n_gram_key].get(next_id, 0) + 1
                    else:
                        self.n_gram[n_gram_key] = {next_id: 1}

    def sample(self, n):
        def make_sample_list(d):
            l = []
            for key in d:
                l += [key] * d[key]
            return l

        max_len = 500
        sample_ids, sample_labels = [], []
        while len(sample_ids) < n:
            # sample init ids
            ids = list(random.choice(self.begin_n_grams))
            for j in range(max_len - self.degree):
                d = self.n_gram[tuple(ids[-self.degree:])]
                next_id = random.choice(make_sample_list(d))
                if next_id == -2:
                    break
                ids.append(next_id)
            # get labels through the omit probability
            labels = []
            for input_id in ids[1:]:
                d = self.omit[input_id]
                next_label = random.choice(make_sample_list(d))
                labels.append(next_label)
            if sum(labels[1:]) > 0 and len(labels) >= 3:
                sample_ids.append(ids[1:])
                sample_labels.append(labels)
                assert len(ids) - 1 == len(labels)
                # print(sample_ids[-1])
                # print(sample_labels[-1])
        batch = ner_pad_to_max_len(sample_ids, sample_labels)
        return batch


if __name__ == '__main__':
    hmm = HMM()

    batch = hmm.sample(1000)

    with open('./data/embed/vocab.txt', 'r') as f:
        vocabs = f.read().strip('\n').split('\n')

    input_ids, labels, real_length = batch

    for i in range(1000):
        ids = input_ids[i, :real_length[i]].numpy().tolist()
        l = labels[i, :real_length[i]].numpy().tolist()
        print(" ".join([vocabs[j] for j in ids]))
        print(l)
