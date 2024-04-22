###
### Logical Arithmetic Sigma-Pi Neuron
###

from .utils import is_all_nge, generate_significant_submindices, generate_significant_submindex
import numpy as np
        
def save_all_mindices(fname, all_mis):
    h = open(fname, "wt")
    for mis in all_mis:
        h.write(repr(mis) + "\n")
    h.close()
    
def read_all_mindices(fname):
    h = open(fname, "rt")
    all_mis = []
    while 1:
        line = h.readline()
        line = line.strip()
        if not line:
            break
        mis - eval(line)
        all_mis.append(mis)
    h.close()
    return all_mis

def first(X):
    return next(iter(X))

def _extend_indices(mindex_k, XA, k, I):
    for j in range(k+1,N):
        xj = XA[j]
        is_nge = False
        for i in mindex_k:
            if not xj[i]:
                is_nge = True
                break
        if is_nge:
            I.append(j)

def generate_total_significant_mindices(X, verbose=True, normality=False):

    N = len(X)
    n = len(X[0])
    XA = np.zeros((N,n), 'd')
    for k in range(N):
        for i in range(n):
            XA[k,i] = X[k][i]
    
    mindices = []
    N = len(X)
    n = len(X[0])
    for k in range(N):
        xk = XA[k]
        mindex_k = tuple(i for i in range(n) if xk[i])
        I = list(range(k))
        if normality:
            _extend_inidices(mindex_k, XA, k, I)
        Xk = XA[I]
                
        mindices_k, mindices_temp = set(), {mindex_k,}
        while mindices_temp:
            finded = set()
            seen = set()
            for mi in mindices_temp:
                submis = {mi for mi in generate_significant_submindices(mi, xk, Xk, seen)}
                if submis:
                    finded |= submis
                else:
                    mindices_k.add(mi)
            mindices_temp = list(finded)
        mindices_k = list(mindices_k)
        mindices_k.sort(key=len)
        if verbose:
            print(k, len(mindices_k), mindices_k[:3])
        yield mindices_k

def generate_significant_mindices(X, verbose=True, normality=False):

    N = len(X)
    n = len(X[0])
    XA = np.zeros((N,n), 'd')
    for k in range(N):
        for i in range(n):
            XA[k,i] = X[k][i]
    
    N = len(X)
    n = len(X[0])
    for k in range(N):
        xk = XA[k]
        mindex_k = tuple(i for i in range(n) if xk[i])
        I = list(range(k))
        if normality:
            _extend_inidices(mindex_k, XA, k, I)
        Xk = XA[I]
                
        while True:
                mi = generate_significant_submindex(mindex_k, xk, Xk)
                if mi is None:
                    break
                mindex_k = mi
        if verbose:
            print(k, len(mindex_k))
        yield [mindex_k]
        
def sort_dataset(X, Y):
    KXY = [(sum(x), x, y) for x,y in zip(X, Y)]
    KXY.sort()
    _, X, Y = zip(*KXY)
    return list(X), list(Y)

class LogicalArithmeticSPN:
    
    __slots__ = 'mindices', 'weights', 'total', 'is_sorted'
    
    def __init__(self, weights=None, mindices=None, total=False):
        self.mindices = mindices if mindices else []
        self.weights = weights if weights else []
        self.total = total
        
    def copy(self):
        spn = LogicalArithmeticSPN(self.weights, self.mindices, self.total)
        return spn

    def evaluate_sum(self, x):
        s = 0
        N = len(self.weights)
        for k in range(N):
            mindex = self.mindices[k]
            p = 1
            for i in mindex:
                if not x[i]:
                    p = 0
                    break
            if p:
                s += self.weights[k]
                
        return s
        
    def evaluate(self, x):
        s = self.evaluate_sum(x)
        if s >= 0:
            return 1
        else:
            return 0
    
    def evaluate_all(self, X):
        return [self.evaluate(x) for x in X]
    
    __call__ = evaluate_all

    def estimate_weight(self, X, Y, k):
        s = self.evaluate_sum(X[k])
        y_k = Y[k]
        if s >= 0 and y_k == 0:
            return -1-s
        elif s < 0 and y_k == 1:
            return -s
        else:
            return None

    def fit_step(self, X, Y, k, mindex):
        w_k = self.estimate_weight(X, Y, k)
        if w_k is None:
            return
        self.mindices.append(mindex)
        self.weights.append(w_k)
    
    def fit(self, X, Y, mindices=None, is_sorted=False):
        self.mindices = []
        self.weights = []
        if not is_sorted:
            X, Y = sort_dataset(X, Y)
#         print(len(X), len(Y))
        total = self.total
        N = len(X)
        n = len(X[0])
        if mindices is None:
            if total:
                mindices = generate_total_significant_mindices(X, False)
            else:
                mindices = generate_significant_mindices(X, False)
#         print(mindices)
        self.mindices = []
        self.weights = []
        for k, mindices_k in enumerate(mindices):
            w_k = self.estimate_weight(X, Y, k)
#             print("k=", k, w_k)
            if w_k is None:
                continue
            if total:
                n = len(mindices_k)
                w_k /= n
                for mindex in mindices_k:
                    self.mindices.append(mindex)
                    self.weights.append(w_k)
            else:
                mindex = first(mindices_k)
                self.mindices.append(mindex)
                self.weights.append(w_k)
    #
    def __str__(self):
        l = []
        for w, mi in zip(self.weights, self.mindices):
            xs = ''.join('x'+str(i) for i in mi)
            if w == 1:
                ws = ''
            elif w == -1:
                ws = '-'
            else:
                ws = str(w)
            l.append(ws+xs)
        text = '+'.join(l)
        text = text.replace('+-', '-')
        text = "H(" + text + ")"
        return text
    #
    def latex(self):
        l = []
        for w, mi in zip(self.weights, self.mindices):
            xs = ''.join('x_{'+str(i)+'}' for i in mi)
            if w == 1:
                ws = ''
            elif w == -1:
                ws = '-'
            else:
                ws = ("%.3g" % w)
            l.append(ws+xs)
        text = '+'.join(l)
        text = text.replace('+-', '-')
        text = "$H(" + text + ")$"
        return text
            
