import numpy as np
import random
import torch
import torch.nn.functional as F
'''
x = torch.rand(2, 32, 5, 5)
print(x)
weight = torch.randn(6, 4)
weight = torch.sigmoid(weight)
index = torch.tensor([0] * 8 + [1] *8 + [2]*8+[3]*8, requires_grad=False)
weight = torch.index_select(weight, dim=-1, index=index, out=None)
print(weight)
#weight = torch.unsqueeze(weight, dim=0)
print(weight.size())
x = torch.transpose(x, dim0=1, dim1=-1)
print(x.size())
x = x * weight
print(x.size())
x = torch.transpose(x, dim0=-1, dim1=1)
print(x.size())
print(x)
'''

K = np.array([[1, 0.3], [0.3, 1]])
print(K)
B = np.diag([0.3, 0.2])
print(B)
A = np.matmul(B, K)
#A = np.matmul(B, A)
k_m = min(np.linalg.eigvals(K))
print(k_m)
A_m = min(np.linalg.eigvals(A))
print(A_m)
#print(A_m / k_m)
#print(np.linalg.norm(B))
exit()


N = 300
dim = 5
alpha = 0.001
X = []
for i in range(N):
    x = np.random.randn(dim)
    x = x / np.linalg.norm(x)
    X.append(x)

K = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        K[i, j] = np.dot(X[i], X[j])


def get_lambda_min(K):
    ans = 100000000
    ij = (-1, -1)
    for i in range(K.shape[0]):
        for j in range(i+1, K.shape[0]):
            lambda_min = min(np.linalg.eigvals(np.array([[K[i, i], K[i, j]], [K[j, i], K[j, j]]])))
            if lambda_min < ans:
                ans = lambda_min
                ij = (i, j)
    i, j = ij
    array = np.array([[K[i, i], K[i, j]], [K[j, i], K[j, j]]])
    return ans, ij, array


print(get_lambda_min(K))

M = 10
S = []


def f(X, S, i):
    for j in S:
        if abs(np.dot(X[i], X[j])) >= 0.7:
            return False
    return True


while len(S) < M:
    i = random.choice([x for x in range(N) if x not in S])
    if len(S) == 0:
        S.append(i)
        continue
    if f(X, S, i):
        S.append(i)
    print(S)


S.sort()
print(S)
small_K = np.zeros((M, M))
for i in range(M):
    for j in range(M):
        small_K[i, j] = K[S[i], S[j]]

print(small_K)

print(get_lambda_min(small_K))
