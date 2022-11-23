from weighted_sim import *
from experts import *
import numpy as np
import pytest

def test_hyp_combinations():
    combs = hyp_combinations({'AAA':[2,3,4], 'BBB':[lambda x:x, lambda x:1]})
    print(combs)

def test_weighted_alg():
    X1 = np.random.rand(20,10)
    Y1 = np.random.rand(20,1)

    X2 = np.random.rand(20,10)
    Y2 = np.random.rand(20,1)

    H = HistoryLog()
    H.add_data(X1,Y1)

    experts = [SingleFactorOLS(i) for i in range(10)]

    # attach chosen history to experts
    we = WeightedExpert(experts, H)
    H.add_data(X2,Y2)
    
    out = weighted_alg(we, 20, 40)
    print(out['rot'])
    print(out['wot'])

def test_weighted_simulation():
    X1 = np.random.rand(20,10)
    Y1 = np.random.rand(20,1)

    X2 = np.random.rand(20,10)
    Y2 = np.random.rand(20,1)

    X3 = np.random.rand(20,10)
    Y3 = np.random.rand(20,1)

    experts = [SingleFactorOLS(i) for i in range(10)]
    out = weighted_simulation(experts, X1, X2, X3, Y1, Y2, Y3, np.mean(Y1.reshape(-1)))
    print('vip', out['vip'])
    print('hypp', out['hypp'])
    print('r2_is', out['r2_is'])
    print('r2_oos', out['r2_oos'])
    print("wis_ar", out["wis_ar"])
    print("ris_ar", out["ris_ar"])

def test_weighted_alg2():
    X1 = np.array([[1,3], [2,2], [3,1], [4,4]])
    Y1 = np.array([[1],[2],[3],[4]])

    H = HistoryLog()
    H.add_data(X1,Y1)

    experts = [SingleFactorOLS(i) for i in range(2)]

    # attach chosen history to experts
    we = WeightedExpert(experts, H, cost_f=avg_squared_arctan)
    we.update()
    H.add_data(X1,Y1)
    H.add_data(X1,Y1)
    
    results = weighted_alg(we, 4, 8, update_period=2, mtrain=np.mean(Y1))
    print('preds', results['preds'])
    print('wot', results['wot'])
    print('rot', results['rot'])
    
