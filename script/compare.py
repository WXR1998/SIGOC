import numpy as np
res0 = []
res4 = []
d0 = {}
d4 = {}

with open('details_00.txt', 'r') as fin:
    for l in fin:
        num = int(l.split()[0])
        t = float(l.split()[-1])
        if t >= 0 and t <= 1:
            res0.append(t)
            d0[num] = t

with open('details_04.txt', 'r') as fin:
    for l in fin:
        num = int(l.split()[0])
        t = float(l.split()[-1])
        if t >= 0 and t <= 1:
            res4.append(t)
            d4[num] = t

print(np.mean(res0), np.mean(res4))

assert(len(res0) == len(res4))
l = len(res0)

res = []
ores = []

with open('subset.txt', 'w') as fout:
    for i in d0:
        if d0[i] <= d4[i]:
            res.append(d4[i])
            ores.append(d0[i])
            fout.write('%d\n' % i)

print(np.mean(ores), np.mean(res))