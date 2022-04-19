from experts import *
import numpy as np
import pytest

def test_history_log():
    H = HistoryLog()
    assert H.get_X() is None
    assert H.get_y() is None

    H.add_data(np.random.rand(10,5), np.random.rand(10,1))
    assert H.get_X().shape == (10,5)
    assert H.get_y().shape == (10,1)
    H.add_data(np.random.rand(5,5), np.random.rand(5, 1))
    assert H.get_X().shape == (15,5)
    assert H.get_y().shape == (15,1)

    H.set_end(8)
    assert H.get_X().shape == (8,5)
    assert H.get_y().shape == (8,1)

    
    assert H.get_X(3,11).shape == (8,5)
    assert H.get_y(3,11).shape == (8,1)

def test_WeightedExpert():
    X1 = np.random.rand(20,10)
    Y1 = np.random.rand(20,1)

    X2 = np.random.rand(20,10)
    Y2 = np.random.rand(20,1)

    H = HistoryLog()
    H.add_data(X1,Y1)

    experts = [SingleFactorOLS(i) for i in range(10)]

    # attach chosen history to experts
    we = WeightedExpert(experts, H)
    we.update()
    assert we.predict(X2, Y2).shape == (20,1)

    H.add_data(X2,Y2)
    H.set_end = 20
    we.update()
    

    assert H.get_X(20,25).shape == (5,10)
    preds = we.predict_from_history(20, 25)
    assert preds.shape == (5,1)

def test_WeightedExpert2():
    X1 = np.array([[1,3], [2,3], [3,1]])
    Y1 = np.array([[1],[2],[3]])

    X2 = X1
    Y2 = Y1

    H = HistoryLog()
    H.add_data(X1,Y1)

    experts = [SingleFactorOLS(i) for i in range(2)]

    # attach chosen history to experts
    we = WeightedExpert(experts, H)
    we.update()


    H.add_data(X1,Y1)
    H.add_data(X1,Y1)
    H.set_end(3)
    we.update()
    
    print(we.weights)
    preds = we.predict_from_history(3,6)
    print(preds)
    print(we.weights)
    preds = we.predict_from_history(6,9)
    print(preds)
    print(we.weights)


if __name__ == '__main__':
    test_WeightedExpert2()