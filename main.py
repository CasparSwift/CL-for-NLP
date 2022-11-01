from torch.utils.data import DataLoader
from load_data import *
from settings import parse_train_args
from model import Document_Encoder, NER_Encoder
from rouge import Rouge
import numpy as np
from markov import HMM
import tqdm

args = parse_train_args()
tokenizer = RNN_tokenizer(args)


def train_task(task_id, model, args, tokenizer, hmm=None):
    model.train()
    if args.task_type == 'ES':
        dataset = TextDataset(task_id=task_id, mode='train', args=args, tokenizer=tokenizer)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.n_workers, collate_fn=dynamic_collate_fn)
    elif args.task_type == 'NER':
        dataset = NERDataset(task=args.tasks[task_id], mode='train', args=args, tokenizer=tokenizer)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.n_workers, collate_fn=ner_collate_fn)

    optimizer = torch.optim.Adam(model.parameters())

    def update_parameters(loss):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        model.zero_grad()

    labeled_sents = []
    for epo in range(4):
        for batch_id, (input_ids, labels, real_lengths) in enumerate(dataloader):
            # print(input_ids)
            # print(labels)
            # print(real_lengths)
            batch_size = input_ids.shape[0]
            if epo == 0:
                for i in range(batch_size):
                    i_ids = input_ids[i, :real_lengths[i]].cpu().numpy().tolist()
                    i_labels = labels[i, :real_lengths[i]].cpu().numpy().tolist()
                    sent = ' '.join([str(iid)+'/'+str(iil) for iid, iil in zip(i_ids, i_labels)])
                    labeled_sents.append(sent)
            if cuda:
                input_ids, labels = input_ids.cuda(), labels.cuda()
            loss, pred = model(input_ids, labels, real_lengths)
            if batch_id % args.logging_steps == 0:
                print('task_id: {} | epoch: {} | loss: {}'.format(task_id, epo, loss.item()))
            update_parameters(loss)
            # generative replay
            if task_id > 0 and (batch_id + 1) % args.replay_steps == 0:
                for _ in range(20):
                    input_ids, labels, real_lengths = hmm.sample(batch_size)
                    if cuda:
                        input_ids, labels = input_ids.cuda(), labels.cuda()
                    loss, pred = model(input_ids, labels, real_lengths)
                    print('replay | epoch: {} | loss: {}'.format(epo, loss.item()))
                    update_parameters(loss)
    with open(args.tasks[task_id] + '_train.txt', 'w') as f:
        f.write('\n'.join(labeled_sents))


def test_task(task_id, model, args, tokenizer):
    model.eval()
    dataset = TextDataset(task_id=task_id, mode='test', args=args, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.n_workers, collate_fn=_dynamic_collate_fn)

    rouge = Rouge()
    cnt = 0
    scores1 = 0
    scores2 = 0
    scoresl = 0
    for input_ids, doc, summary, real_lengths in dataloader:
        if cuda:
            input_ids, real_lengths = input_ids.cuda(), real_lengths.cuda()
        pred = model(input_ids, None, real_lengths).cpu().numpy().tolist()
        offset = 0
        #print(pred)
        for i in range(input_ids.shape[0]):
            hyps = ''.join([doc[i][j] for j in range(len(doc[i])) if pred[j+offset]])
            offset += len(doc[i])
            #print(hyps)
            #print(summary[i])
            #print('-'*50)
            try:
                score = rouge.get_scores(hyps, summary[i])[0]
                scores1 += score['rouge-1']['f']
                scores2 += score['rouge-2']['f']
                scoresl += score['rouge-l']['f']
            except:
                pass
            cnt += 1
    print('task_id: {} | rouge-1: {} | rouge-2: {} | rouge-l: {}'.format(task_id, scores1/cnt, scores2/cnt, scoresl/cnt))


def test_task_ner(task_id, model, args, tokenizer):
    model.eval()
    dataset = NERDataset(task=args.tasks[task_id], mode='test', args=args, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.n_workers, collate_fn=ner_collate_fn)

    correct_cnt = 0
    pred_tot = 0
    entity_cnt = 0
    labeled_sents = []
    fmatrix = np.zeros((5, 5), dtype=np.int)
    for input_ids, labels, real_lengths in dataloader:
        if cuda:
            input_ids, labels = input_ids.cuda(), labels.cuda()
        pred = model(input_ids, labels, real_lengths)[1].cpu().numpy().tolist()
        batch_size = input_ids.shape[0]
        labels = torch.cat([labels[i, :real_lengths[i]] for i in range(batch_size)]).cpu().numpy().tolist()
        for x, y in zip(pred, labels):
            fmatrix[y, x] += 1
            if x == y and x != 0 and y != 0:
                correct_cnt += 1
            if x != 0:
                pred_tot += 1
            if y != 0:
                entity_cnt += 1
        offset = 0
        for i in range(batch_size):
            ids = input_ids[i, :real_lengths[i]].cpu().numpy().tolist()
            sent = tokenizer.decode(ids, labels[offset: offset+real_lengths[i]],
                                    pred[offset: offset+real_lengths[i]])
            offset += real_lengths[i]
            labeled_sents.append(sent)
    with open(args.tasks[task_id] + '.txt', 'w') as f:
        f.write('\n'.join(labeled_sents))
    pre = correct_cnt / pred_tot
    recall = correct_cnt / entity_cnt
    f1 = (2 * pre * recall)/(pre + recall)
    print('task_id: {} | pre: {:.4f} | recall: {:.4f} | F1: {:.4f}'.format(task_id, pre, recall, f1))
    print(fmatrix)


def main():
    if args.task_type == 'ES':
        model = Document_Encoder(args)
    elif args.task_type == 'NER':
        model = NER_Encoder(args)

    cuda = torch.cuda.is_available()
    if cuda:
        model = model.cuda()

    hmm = None
    for train_task_id in range(len(args.tasks)):
        if train_task_id > 0:
            hmm = HMM()
        train_task(train_task_id, model, args, tokenizer, hmm)
        for test_task_id in range(len(args.tasks)):
            if args.task_type == 'ES':
                test_task(test_task_id, model, args, tokenizer)
            elif args.task_type == 'NER':
                test_task_ner(test_task_id, model, args, tokenizer)


if __name__ == '__main__':
    main()