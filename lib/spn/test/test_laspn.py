import unittest
import sys
import gc
import weakref
import pickle, copy

import numpy as np
import spn.laspn as laspn

class laspnTest(unittest.TestCase):
    
    def test_evaluate(self):
        # XOR
        X = [(0,0), (0,1), (1,0), (1,1)]
        I = [(), (1,), (0,), (0,1)]
        W = [-1, 1, 1, -2]
        S = [-1, 0, 0, -1]
        Y = [0, 1, 1, 0]
        spn = laspn.LogicalArithmeticSPN(W, I)
        self.assertEqual(Y, spn.evaluate_all(X))
        for s, x in zip(S, X):
            self.assertEqual(s, spn.evaluate_sum(x))
            
    def test_is_nge1(self):
        X = np.array([[0,0], [0,1], [1,0], [1,1]], 'd')
        I = [(), (1,), (0,), (0,1)]
        for k in range(4):
#             print(I[k], X[k], laspn.is_all_nge(I[k], X[k], X[:k]))
            self.assertTrue(laspn.is_all_nge(I[k], X[k], X[:k]))
            
    def test_generate_all_significant_submindices(self):
        X = np.array([
            [0,0,0], 
            [1,0,0], 
            [0,1,0], 
            [0,0,1], 
            [1,1,1]], 'd')
        mi = (0,1,2)
        submindices = list(laspn.generate_significant_submindices(mi, X[-1], X[:-1], set()))
        self.assertIn((1,2), submindices)
        self.assertIn((0,1), submindices)
        self.assertIn((0,2), submindices)

    def test_generate_total_significant_mindices(self):
        X = [(1,1,0,0), 
             (0,0,1,1), 
             (1,0,0,1), 
             (1,1,1,0), 
             (0,1,1,1), 
             (1,0,1,1), 
             (0,1,1,1), 
             (1,1,1,1)]
        MI = list(laspn.generate_total_significant_mindices(X, False))
        print(MI)
#         print()
#         for x,mi in zip(X, MI):
#             print(x,mi)
        MI_true = [{(1,), (0,)}, 
                   {(3,), (2,)}, 
                   {(0,3)}, 
                   {(1,2), (0,2)}, 
                   {(1,3)}, 
                   {(0,2,3)}, 
                   {(1,2,3)}, 
                   {(0,1,3)}]
        self.assertEqual(MI_true, [set(mis) for mis in MI])

    def test_fit(self):
        # XOR
        X = [(0,0), (0,1), (1,0), (1,1)]
        I = [(), (1,), (0,), (0,1)]
        W = [-1, 1, 1, -2]
        Y = [0, 1, 1, 0]
        spn = laspn.LogicalArithmeticSPN()
        spn.fit(X, Y, is_sorted=True)
        self.assertEqual(I, spn.mindices)
        self.assertEqual(W, spn.weights)
        self.assertEqual(spn.evaluate_all(X), Y)
#         print(spn.weights)
#         print(spn.mindices)
        
def main():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(laspnTest))
    return suite
    
