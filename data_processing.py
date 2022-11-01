from newsroom import jsonl
from rouge import Rouge
from nltk.tokenize import sent_tokenize
from load_data import load_pkl, write_pkl
import tqdm


def process_newsroom(mode, rouge):
    domains = ['nytimes', 'washingtonpost', 'foxnews', 'theguardian',
               'nydailynews', 'wsj', 'usatoday', 'cnn', 'time', 'mashable']

    def get_domain(url):
        for domain in domains:
            if domain in url:
                return domain
        return None

    labeled_dataset = {}
    for domain in domains:
        labeled_dataset[domain] = []

    cnt = 0
    chunk_cnt = 0
    s = 0
    with jsonl.open('./data/newsroom/{}.jsonl.gz'.format(mode), gzip=True) as train_file:
        for entry in tqdm.tqdm(train_file):
            url = entry['url'].split('/')[2].split('.')
            domain = get_domain(url)
            if domain is None:
                continue
            text_sents = [sent for sent in sent_tokenize(entry['text']) if len(sent.replace('.', '')) >= 1]
            summary_sents = sent_tokenize(entry['summary'])
            scores = []
            try:
                for i in range(len(text_sents)):
                    scores.append(rouge.get_scores(text_sents[i], entry['summary'])[0]['rouge-l']['r'])
            except Exception as e:
                print(e)
                continue
            idx_and_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:len(summary_sents)]
            gold_labels = sorted([idx for idx, _ in idx_and_scores])
            labeled_dataset[domain].append((text_sents, gold_labels))
            #pred = ''.join([text_sents[i] for i in gold_labels])
            #print(pred)
            #print(entry['summary'])
            #s += rouge.get_scores(pred, entry['summary'])[0]['rouge-l']['r']
            cnt += 1
            #print(s/cnt)
            #print('-'*100)
            #if cnt % 1000 == 0:
                #print(cnt)
            if (cnt + 1) % 100000 == 0:
                write_pkl(labeled_dataset, './data/newsroom/{}_{}.pkl'.format(mode, chunk_cnt))
                print('write_to_pkl, chunk_cnt={}'.format(chunk_cnt))
                chunk_cnt += 1
                for domain in domains:
                    labeled_dataset[domain] = []

    print(cnt)
    write_pkl(labeled_dataset, './data/newsroom/{}_{}.pkl'.format(mode, chunk_cnt))

    for domain in domains:
        dataset = []
        for i in range(chunk_cnt+1):
            labeled_dataset = load_pkl('./data/newsroom/{}_{}.pkl'.format(mode, i))
            dataset += labeled_dataset[domain]
        write_pkl(dataset, './data/newsroom/{}_{}.pkl'.format(domain, mode))
        print(domain, len(dataset))


if __name__ == '__main__':
    rouge = Rouge()
    process_newsroom(mode='train', rouge=rouge)