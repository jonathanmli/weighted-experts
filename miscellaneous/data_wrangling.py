import numpy as np
import pandas as pd

def summarize_results(results):
    '''
    results: list of results (dicts)

    1 row for each model, 3+X columns for ris, roos, and vips
    '''
    out = {}
    n_explanatory = len(results[0]['vip'])
    npout = np.zeros((len(results), 2 + len(results[0]['vip'])))

    we_results = None

    for i in range(len(results)):
        if results[i]['model'] == 'weighted experts':
            we_results = results[i]

        # print(results[i]['r2_is'].shape)
        # print(results[i]['r2_is'])
        # print(np.array(results[i]['r2_is']).shape)
        # print(results[i]['r2_oos'].shape)
        # print(results[i]['vip'].shape)

        npout[i] = np.concatenate((np.array(results[i]['r2_is']).reshape(1), np.array(results[i]['r2_oos']).reshape(1), results[i]['vip']), axis = 0)

    pdout = pd.DataFrame(npout, columns=["r2_is", "r2_oos"] + ["vip"]*n_explanatory)
    pdout['model_name'] = [_['model'] for _ in results]

    # additional information to be printed for algorithm
    if we_results is not None:
        we_weights = np.concatenate((we_results['wtrain_ar'], we_results['wis_ar'], we_results['woos_ar']), axis=0)
        we_r2 = np.concatenate((we_results['rtrain_ar'], we_results['ris_ar'], we_results['roos_ar']), axis=0)
        we_hyps = we_results["hyps"]
    
        out['we_hyps'] = we_hyps
        out['we_weights'] = we_weights
        out['we_r2'] = we_r2
        out['we_preds'] = we_results['preds']
    
    
    out['summary'] = pdout
    return out


def data_wrangle(X, y):
    '''
    splits the data into train, test, and oos
    '''
    out = {}

    # split into threes
    total_len = len(X)
    xtrain = X[:total_len//3]
    xtest = X[total_len//3:(total_len*2)//3]
    xoos = X[(total_len*2)//3:]
    total_len = len(y)
    ytrain = y[:total_len//3].reshape(-1, 1)
    ytest = y[total_len//3:(total_len*2)//3].reshape(-1, 1)
    yoos = y[(total_len*2)//3:].reshape(-1, 1)
    

    # calculate statistics
    mu  = 0.2 # *np.sqrt(hh)
    tol = 1e-10
    N       = 200      ### Number of CS tickers
    # m       = nump*2   ### Number of Characteristics
    T       = 180      ### Number of Time Periods

    per     = np.tile(np.arange(N)+1,T)
    # time    = np.repeat(np.arange(T)+1,N)
    # stdv    = 0.05
    # theta_w = 0.005


    ### Demean Returns ### 
    ytrain_demean= ytrain-np.mean(ytrain)
    # ytest_demean = ytest-np.mean(ytest)
    mtrain       = np.mean(ytrain)
    # mtest        = np.mean(ytest)


    ### Calculate Sufficient Stats ###
    # normalize based on sd
    sd           = np.zeros(xtrain.shape[1])
    for i in range(xtrain.shape[1]):
        s = np.std(xtrain[:,i])
        if s > 0:
            xtrain[:,i] = xtrain[:,i] /s
            xtest[:,i]  = xtest[:,i]  /s
            xoos[:,i]   = xoos[:,i]   /s
            sd[i]       = s

    XX           = xtrain.T.dot(xtrain)
    b            = np.linalg.svd(XX)
    L            = b[1][0]
    print('Lasso L=',L)
    Y            = np.matrix(ytrain_demean)
    XY           = xtrain.T.dot(Y)

    

    # demean

    

    # return dictionary of calculated statistics
    out["xtrain"] = xtrain
    out["xtest"] = xtest
    out["xoos"] = xoos
    out["ytrain"] = ytrain
    out["ytest"] = ytest
    out["yoos"] = yoos
    out['XX'] = XX
    out['XY'] = XY
    out['mtrain'] = mtrain
    out['mu'] = mu
    out['tol'] = tol
    out['L'] = L
    out['ytrain_demean'] = ytrain_demean
    return out

    N = 200  ### Number of CS tickers
    m = nump * 2  ### Number of Characteristics
    T = 180  ### Number of Time Periods

    per = np.tile(np.arange(N) + 1, T)
    time = np.repeat(np.arange(T) + 1, N)
    stdv = 0.05
    theta_w = 0.005

    c = pd.read_csv(dirstock + 'c%d.csv' % M, delimiter=',').values
    r1 = pd.read_csv(dirstock + 'r%d_%d_%d.csv' % (mo, M, hh),
                        delimiter=',').iloc[:, 0].values

    ### Add Some Elements ###
    daylen = np.repeat(N, T / 3)
    daylen_test = daylen
    ind = range(0, (N * T // 3))
    xtrain = c[ind, :]
    ytrain = r1[ind]
    ytrain = ytrain.reshape(len(ytrain), 1)
    trainper = per[ind]
    ind = range((N * T // 3), (N * (T * 2 // 3 - hh + 1)))
    xtest = c[ind, :]
    ytest = r1[ind]
    ytest = ytest.reshape(len(ytest), 1)
    testper = per[ind]

    l1 = c.shape[0]
    l2 = len(r1)
    l3 = l2 - np.sum(np.isnan(r1))
    print(l1, l2, l3)
    ind = range((N * T * 2 // 3), min(l1, l2, l3))
    xoos = c[ind, :]
    yoos = r1[ind]
    yoos = yoos.reshape(len(yoos), 1)

    del c
    del r1

    ### Demean Returns ###
    ytrain_demean = ytrain - np.mean(ytrain)
    ytest_demean = ytest - np.mean(ytest)
    mtrain = np.mean(ytrain)
    mtest = np.mean(ytest)

    ### Calculate Sufficient Stats ###
    sd = np.zeros(xtrain.shape[1])
    for i in range(xtrain.shape[1]):
        s = np.std(xtrain[:, i])
        if s > 0:
            xtrain[:, i] = xtrain[:, i] / s
            xtest[:, i] = xtest[:, i] / s
            xoos[:, i] = xoos[:, i] / s
            sd[i] = s

    XX = xtrain.T.dot(xtrain)
    b = np.linalg.svd(XX)
    L = b[1][0]
    print('Lasso L=', L)
    Y = np.matrix(ytrain_demean)
    XY = xtrain.T.dot(Y)