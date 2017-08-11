import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

iter = 10000
wt = [0.45, 0.55]
wc = [0.2, 0.8]
wo = [0.7, 0.3]
wh = [0.31, 0.69]
we = [0.34125, 0.65875]
ETCO = dict()
ETCO[0] = [[[0.85, 0.8], [0.4, 0.3]], [[0.6, 0.5], [0.15, 0.1]]]
ETCO[1] = [[[0.15, 0.2], [0.6, 0.7]], [[0.4, 0.5], [0.85, 0.9]]]
HC = dict()
HC[0] = [0.75, 0.2]
HC[1] = [0.25, 0.8]
GEH = dict()
GEH["A"] = [[0.05, 0.1], [0.1, 0.6]]
GEH["B"] = [[0.05, 0.3], [0.3, 0.2]]
GEH["C"] = [[0.1, 0.3], [0.3, 0.1]]
GEH["D"] = [[0.2, 0.2], [0.2, 0.05]]
GEH["F"] = [[0.6, 0.1], [0.1, 0.05]]

def gibbs():
    global iter
    global wc, wt, we, wh
    res = []
    num = 0
    cnt = 0
    T = np.random.choice([0, 1], p=wt)
    C = np.random.choice([0, 1], p=wc)
    E = np.random.choice([0, 1], p=we)
    H = np.random.choice([0, 1], p=wh)
    O = 1
    G = "B"
    for i in range(0, iter):
        weight_T = findWhole(T, C, E, H, O, G)/findT(C, E, H, O, G)
        pT=[0, 0]
        pT[T] = weight_T
        pT[1-T] = 1 - weight_T
        T_next = np.random.choice([0, 1], p=pT)
        weight_C = findWhole(T_next, C, E, H, O, G)/findC(T_next, E, H, O, G)
        pC = [0, 0]
        pC[C] = weight_C
        pC[1-C] = 1 - weight_C
        C_next = np.random.choice([0, 1], p=pC)
        weight_E = findWhole(T_next, C_next, E, H, O, G)/findE(T_next, C_next, H, O, G)
        pE = [0, 0]
        pE[E] = weight_E
        pE[1-E] = 1 - weight_E
        E_next = np.random.choice([0, 1], p=pE)
        weight_H = findWhole(T_next, C_next, E_next, H, O, G)/findH(C_next, E_next, T_next, O, G)
        pH = [0, 0]
        pH[H] = weight_H
        pH[1-H] = 1 - weight_H
        H_next = np.random.choice([0, 1], p=pH)
        T = T_next
        C = C_next
        E = E_next
        H = H_next
        if O == 1 and G == "B":
            num += 1
            if C == 1:
                cnt += 1
        if num == 0:
            p = 0
        else:
            p = float(cnt)/num
        res.append(p)
    return res

def findWhole(t, c, e, h, o, g):
    global wt, wc, wo, HC, ETCO, GEH
    res = wt[t]*wc[c]*wo[o]*HC[h][c]*ETCO[e][t][c][o]*GEH[g][e][h]
    return res

def findT(c, e, h, o, g):
    global wt, wc, wo, HC, ETCO, GEH
    res = findWhole(1, c, e, h, o, g) + findWhole(0, c, e, h, o, g)
    return res

def findC(t, e, h, o, g):
    global wt, wc, wo, HC, ETCO, GEH
    res = findWhole(t, 1, e, h, o, g) + findWhole(t, 0, e, h, o, g)
    return res

def findE(t, c, h, o, g):
    global wt, wc, wo, HC, ETCO, GEH
    res = findWhole(t, c, 1, h, o, g) + findWhole(t, c, 0, h, o, g)
    return res

def findH(c, e, t, o ,g):
    global wt, wc, wo, HC, ETCO, GEH
    res = findWhole(t, c, e, 1, o, g) + findWhole(t, c, e, 0, o, g)
    return res


def calculate(c, o, g):
    global wt, wc, wo, HC, ETCO, GEH
    res = 0
    for t in range(0, 2):
        for e in range(0, 2):
            for h in range(0, 2):
                res += wt[t]*ETCO[e][t][c][o]*HC[h][c]*GEH[g][e][h]
    return res

def bf():
    global wt, wc, wo, HC, ETCO, GEH
    i = 0
    numOfC1 = 0
    num = 0
    iter = []
    p = []
    while i < 10000:
        pT0 = 0.45
        pT1 = 0.55
        # generate T here
        T = np.random.choice(2, 1, p=[pT0, pT1])[0]

        pC0 = 0.2
        pC1 = 0.8
        # generate C here
        C = np.random.choice(2, 1, p=[pC0, pC1])[0]

        pO0 = 0.7
        pO1 = 0.3
        # generate O here
        O = np.random.choice(2, 1, p=[pO0, pO1])[0]

        if C == 0:
            pH0 = 0.75
            pH1 = 0.25
        elif C == 1:
            pH0 = 0.2
            pH1 = 0.8
        else:
            exit("C unexpected value")
        # generate H here
        H = np.random.choice(2, 1, p=[pH0, pH1])[0]

        if T == 0 and C == 0 and O == 0:
            pE0 = 0.85
            pE1 = 0.15
        elif T == 0 and C == 0 and O == 1:
            pE0 = 0.8
            pE1 = 0.2
        elif T == 0 and C == 1 and O == 0:
            pE0 = 0.4
            pE1 = 0.6
        elif T == 0 and C == 1 and O == 1:
            pE0 = 0.3
            pE1 = 0.7
        elif T == 1 and C == 0 and O == 0:
            pE0 = 0.6
            pE1 = 0.4
        elif T == 1 and C == 0 and O == 1:
            pE0 = 0.5
            pE1 = 0.5
        elif T == 1 and C == 1 and O == 0:
            pE0 = 0.15
            pE1 = 0.85
        elif T == 1 and C == 1 and O == 1:
            pE0 = 0.1
            pE1 = 0.9
        else:
            exit("T C O unexpected value")
        # generate E here
        E = np.random.choice(2, 1, p=[pE0, pE1])[0]

        if E == 0 and H == 0:
            pGA = 0.05
            pGB = 0.05
            pGC = 0.1
            pGD = 0.2
            pGF = 0.6
        elif E == 0 and H == 1:
            pGA = 0.1
            pGB = 0.3
            pGC = 0.3
            pGD = 0.2
            pGF = 0.1
        elif E == 1 and H == 0:
            pGA = 0.1
            pGB = 0.3
            pGC = 0.3
            pGD = 0.2
            pGF = 0.1
        elif E == 1 and H == 1:
            pGA = 0.6
            pGB = 0.2
            pGC = 0.1
            pGD = 0.05
            pGF = 0.05
        else:
            exit("E H values unexpected")
        # generate G here
        G = np.random.choice(5, 1, p=[pGA, pGB, pGC, pGD, pGF])[0]
        if G != 1 or O != 1:
            i += 1
        else:
            num += 1
            i += 1
            if C == 1:
                numOfC1 += 1
        iter.append(i)
        if num == 0:
            p.append(0)
        else:
            p.append(float(numOfC1) / float(num))
    return iter, p


iter_, bf_ = bf()
gibbs_ = gibbs()
line1 = plt.plot(iter_, bf_, label="p(C1|O1, GB) - bf")
line2 = plt.plot(iter_, gibbs_, label="p(C1|O1, GB) - gibbs")
# plt.title('brute force')
plt.legend()
plt.show()

