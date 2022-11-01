import torch.nn as nn
from torch.autograd import Variable
import torch

'''
cnt = 0
for i in range(10):
    v = torch.tensor([0.7, 0.3])
    g = torch.rand(2)
    g = -torch.log(-torch.log(g))
    vv = (torch.log(v) + g)/0.1
    soft = torch.softmax(vv, dim=0)
    if torch.max(soft, 0)[1].item() == 0:
        cnt += 1
print(cnt)
'''
'''
cnt = 0
a = torch.tensor([-2.0])
print(torch.sigmoid(a))
for i in range(1000):
    u = torch.rand(1)
    g = torch.sigmoid((torch.log(u)-torch.log(1-u)+a)/0.1)
    if g.item() > 0.5:
        cnt += 1
    print(g)
print(cnt)
'''
dim = 10
train_set = []
for i in range(100):
    X = torch.randn(dim, requires_grad=False)
    label = int(torch.rand(1).item() > 0.5)
    if label == 0:
        label = -1
    train_set.append((X, label))
a = Variable(torch.randn(dim), requires_grad=True)
D = Variable(1e-3*torch.randn(dim), requires_grad=True)
optimizer = torch.optim.Adam([D, a])

for _ in range(100):
    for X, labels in train_set:
        XX = torch.sigmoid(D)*X
        u = torch.sigmoid(torch.dot(a, XX))
        loss = 0.5*torch.norm(u-labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(torch.sigmoid(D))
        print(loss.item())
exit()


def f(w):
    return 27*w[0]+sum(2*w[i]*w[i+1]+25/16*w[i+1]**2+15*41/16*w[i+1]**2 for i in range(len(w)-1))+512*10+2*(w[0]+49*w[1]+49*w[2])

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.alpha = Variable(1e-3*torch.randn(2, 512//8), requires_grad=True)
        self.index = torch.tensor([i//8 for i in range(512)], requires_grad=False)

    def forward(self):
        a = torch.sigmoid(self.alpha)
        print(a)
        a = torch.index_select(a, dim=-1, index=self.index, out=None)
        w_hat = torch.sum(torch.relu(2*a - 1), dim=-1)
        print(w_hat)
        ff = f([32, w_hat[0], w_hat[1]])/1e6
        print(ff)
        return (ff - 2)**2


m = model()
optimizer = torch.optim.Adam([m.alpha])
for _ in range(1000):
    loss = m()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #print(m.alpha.detach().numpy())
    print(loss.item())
