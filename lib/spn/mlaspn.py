

def product(x, mindex):
    for i in mindex:
        if not x[i]:
            return 0
    return 1

class MLASPN:
    
    __slots__ = 'p_mindices', 's_mindices', 's_weights', 's_offset'
    
    def evaluate(self, x):
        result = []
        product_x = partial(product, x)
        P = map(product_x, self.p_mindices)
        m = len(self.s_indices)
        for j in range(m):
            mindex = self.s_mindices[j]
            weight = self.s_weights[j]
            s = self.s_offset[j]
            n = len(weight)
            for i in range(n):
                k = mindex[i]
                s += weight[i] * P[k]
            result.append(s)
        return tuple(result)
