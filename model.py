import numpy as np
import torch
import torch.nn as nn


class Sent_Encoder(nn.Module):
    def __init__(self, args, bi=False, last=True):
        super(Sent_Encoder, self).__init__()
        self.weight = np.load(args.embed_path)
        self.rnn = torch.nn.LSTM(
            input_size=self.weight.shape[1],
            hidden_size=args.hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=bi,
            dropout=0.5
        )
        self.embedding = torch.nn.Embedding(self.weight.shape[0], self.weight.shape[1])
        self.input_dropout = torch.nn.Dropout(0.4)
        self.embedding.weight.data.copy_(torch.from_numpy(np.array(self.weight)))
        self.embedding.weight.requires_grad = True
        self.last = last

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(self.embedding(x), None)
        if self.last:
            return r_out[:, -1, :]
        else:
            return r_out


class Document_Encoder(nn.Module):
    def __init__(self, args):
        super(Document_Encoder, self).__init__()
        self.input_size = args.hidden_size
        self.hidden_size = self.input_size//2
        self.rnn = torch.nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.sent_encoder = Sent_Encoder(args)
        self.dense = nn.Linear(self.hidden_size*2, 1)

        def weighted_cross_entropy_with_logits(out, targets, pos_weight):
            logits = torch.sigmoid(out)
            #print("logits", logits)
            return - (targets.float() * torch.log(logits) * pos_weight +
                      (1 - targets.float()) * torch.log(1 - logits))
        self.loss_func = weighted_cross_entropy_with_logits

    def forward(self, input_ids, labels, real_lengths):
        batch_size = input_ids.shape[0]
        max_doc_len = input_ids.shape[1]
        max_len = input_ids.shape[2]
        input_ids = input_ids.view(-1, max_len)
        sent_encoding = self.sent_encoder(input_ids).view(batch_size, max_doc_len, self.input_size)
        # r_out, _ = self.rnn(sent_encoding)
        # out = torch.cat([r_out[i, :real_lengths[i], :] for i in range(batch_size)])
        r_out = []
        for i in range(batch_size):
            one_out, _ = self.rnn(torch.unsqueeze(sent_encoding[i, :real_lengths[i], :], dim=0))
            r_out.append(torch.squeeze(one_out, dim=0))
        out = torch.cat(r_out)
        out = self.dense(out)
        out = torch.squeeze(out, dim=-1)
        pred = torch.tensor(out > 0, dtype=torch.int)
        if labels is None:
            return pred
        labels = torch.cat([labels[i, :real_lengths[i]] for i in range(batch_size)])
        t = self.loss_func(out, labels, 13)
        #print(t)
        loss = torch.mean(t)
        #print(pred)
        #print(labels)
        return loss, pred


class NER_Encoder(nn.Module):
    def __init__(self, args):
        super(NER_Encoder, self).__init__()
        self.hidden_size = args.hidden_size
        self.sent_encoder = Sent_Encoder(args, bi=True, last=False)
        self.dense = nn.Linear(self.hidden_size*2, args.n_labels)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels, real_lengths):
        batch_size = input_ids.shape[0]
        sent_encoding = self.sent_encoder(input_ids)
        out = torch.cat([sent_encoding[i, :real_lengths[i], :] for i in range(batch_size)])
        out = self.dense(out)
        labels = torch.cat([labels[i, :real_lengths[i]] for i in range(batch_size)])
        loss = self.loss_func(out, labels)
        pred = torch.argmax(out, dim=-1)
        return loss, pred