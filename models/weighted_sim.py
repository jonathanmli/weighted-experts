from models.experts import *
import numpy as np
import itertools

def weighted_alg(we : WeightedExpert, start: int, end: int, mtrain=0.0, update_period = 5, update_experts = True, **kwargs):
    '''
    output: r2 over time, r2 aggregate, final weights, weights over time, predictions
    '''
    T = end-start
    out = {}
    # print("ps", T // update_period)
    out['wot'] = np.zeros((T // update_period, len(we.weights)))
    out['rot'] = np.zeros(T // update_period)
    out['preds'] =  np.zeros(T)
    r_chosen_sum = 0

    # training
    for i in range(T // update_period):
        j = i*update_period+start
        we.history.set_end(j)
        preds = we.predict_from_history(j, j+update_period, update_experts=update_experts)
        out['preds'][i * update_period:(i+1) * update_period] = preds.reshape(-1)
        out['wot'][i, :] = we.weights
        ys = we.history.get_y(j, j+update_period)
        r_chosen = np.sum(np.power(preds - ys, 2))
        # print("preds", preds)
        # print("ys", ys)
        # print("r_chosen", r_chosen)
        # print("null", np.sum(np.power(ys - mtrain, 2)))
        out['rot'][i] = 1 - r_chosen / np.sum(np.power(ys - mtrain, 2))
        r_chosen_sum += r_chosen
    
    out['weights'] = we.weights
    out['r2'] = 1 - r_chosen_sum / np.sum(np.power(we.history.get_y(start, end) - mtrain, 2))
    return out

def hyp_combinations(hyps : dict):
    '''
    input: dictionary of hyp names to list of values
    output: list of dictionaries of hype names to value for each possible combination
    '''
    
    comb = list(itertools.product(*list(hyps.values())))
    out = [dict(zip(hyps.keys(), _)) for _ in comb]
    return out

def weighted_simulation(experts, xtrain, xtest, xoos, ytrain, ytest, yoos,
    mtrain=0.0, 
    update_periods = [4000, 8000, 16000], 
    etas=[0.01, 0.1, 0.5], 
    weighted_average=True, 
    exponential_updates=[True, False], 
    cost_fs = [pearson_correlation, avg_squared_arctan], 
    weight_fs = [lambda x: x, soft_max], 
    **kwargs):
    '''
    Used the weighted alg on training, test, and oos data
    experts:
    N: number of cs tickers, used to separate the data into the N assets
    mtrain: mean of training data used to calculate r squared
    xtest/ytest: data that we tune the hyperparameters on
    xtrain/ytrain: data that we train on
    xoos/yoos: data that we test on
    
    the learning rate eta, whether to use weighted averages vs probability,
    and whether to use exponential vs multiplicative weights
    returns is_ar, oos_ar, and w_ar, the arrays of is r2, oos r2, w r2, and array of max hyperparameters respectively each period
    hyper parameters are [max update period, max eta, max scale] currently
    
    '''

    his = HistoryLog()
    his.add_data(xtrain, ytrain)
    his.add_data(xtest, ytest)
    his.add_data(xoos, yoos)

    # Training  
    his.current_end = len(xtrain)
    we = WeightedExpert(experts, his, eta=etas[0], cost_f = cost_fs[0], weight_f=weight_fs[0])
    we.update()
    training_results = weighted_alg(we, 0, len(xtrain), update_period = update_periods[0],update_experts=False, mtrain= mtrain)

    # redo weightings with tuned hyperparameters with weights and experts from before on validation set
    hyps = {'eta': etas, 'update_period' : update_periods, 'exponential_update' :exponential_updates, 'weight_f': weight_fs, 'cost_f':cost_fs}
    combs = hyp_combinations(hyps)
    df_hyps = pd.DataFrame(combs)
    rs = np.zeros(len(combs))
    
    maxi = 0
    maxr = 0
    test_results = None
    for i in range(len(combs)):
        # hyp tuning
        his.current_end = len(xtrain) + len(xtest)
        we = WeightedExpert(experts, his, weights = training_results['weights'].copy(), **combs[i])
        results = weighted_alg(we, len(xtrain), len(xtrain) + len(xtest), mtrain= mtrain, **combs[i])
        if results['r2'] > maxr:
            maxr = results['r2']
            maxi = i
            test_results = results
        rs[i] = results['r2']
 
    print('r2 by hyps')
    df_hyps['r2'] = rs
    df_hyps.sort_values(by='r2', inplace=True, ascending=False)
    print(df_hyps)
    # print(combs)
    # print(rs)

    # # use tuned hyperparameters on test
    # his.current_end = len(xtrain) + len(xtest)
    # we = WeightedExpert(experts, his, weights = training_results['weights'], **combs[maxi])
    # test_results = weighted_alg(we,len(xtrain), len(xtrain) + len(xtest),mtrain=mtrain, **combs[maxi])
    # print('r2is', test_results['r2'])

    # use tuned hyperparameters on OOS
    his.current_end = len(xtrain) + len(xtest) + len(xoos)
    we = WeightedExpert(experts, his, weights = test_results['weights'].copy(), **combs[maxi])
    oos_results = weighted_alg(we, len(xtrain) + len(xtest),len(xtrain) + len(xtest) + len(xoos), mtrain=mtrain, **combs[maxi])
    

    print('r2oos', oos_results['r2'])

    out = {}
    out["wtrain_ar"] = training_results['wot']
    out["rtrain_ar"] = training_results['rot']
    out["ris_ar"] = test_results['rot']
    out["wis_ar"] = test_results['wot']
    out["roos_ar"] = oos_results['rot']
    out["woos_ar"] = oos_results['wot']
    out["hypp"] = list(combs[maxi].values())
    out["preds"] = np.concatenate([training_results['preds'], test_results['preds'], oos_results['preds']])

    # common items
    out["r2_is"] = test_results['r2']
    out["r2_oos"] = oos_results['r2']
    out["model"] = 'weighted experts'
    out["vip"] = oos_results['weights']
    out['hyps'] = df_hyps
    return out