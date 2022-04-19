# import libraries
import pickle
import pandas as pd
import numpy as np
from miscellaneous.data_wrangling import data_wrangle, summarize_results
from models.weighted_sim import weighted_simulation
from models.experts import SingleFactorOLS, soft_max
import models.auxiliary_func
import models.benchmark_models
from miscellaneous.file_management import file_manager

# parameters
filename = "./Simu/Data_real/data.p"
predicted_name = "Y"



if __name__ == '__main__':

    # open file
    fm = file_manager("./Simu/", "100", "test_holistic")

    # N = 200

    # data wrangling
    X = fm.get_data_file("c1.csv", iscsv=True).values
    y = fm.get_data_file("r1_1_1.csv", iscsv=True).iloc[:, 0].values
 

    # process into training, testing, and oos and obtain relevant stats
    wrangled = data_wrangle(X,y) 
    wrangled['N'] = 200

    # run simulation

    # weighted_simulation
    experts = [SingleFactorOLS(i) for i in range(X.shape[1])]
    we_results = weighted_simulation(experts, **wrangled, update_periods = [2000], etas=[0.1])

    # benchmarks

    ### ridge ###
    ### Tuning parameter: the L2 penalty lambda
    lamv            = models.auxiliary_func.sq(0,6,0.1)
    alpha           = 1

    ridge_results = models.benchmark_models.ridge(**wrangled,alpha=alpha,lamv=lamv)

    ### Lasso ###
    ### Tuning parameter: the L1 penalty lambda

    lamv           = models.auxiliary_func.sq(-2,4,0.1)
    alpha          = 1

    lasso_results = models.benchmark_models.Lasso(**wrangled,alpha=alpha,lamv=lamv)

    

    # save results
    out = summarize_results([we_results, lasso_results, ridge_results])

    fm.df_to_csv(pd.DataFrame(out['we_weights']) ,"weights","hol")
    fm.df_to_csv(pd.DataFrame(out['we_hyps']) ,"HYP","hol")
    fm.df_to_csv(pd.DataFrame(out['we_r2']) ,"R2","hol")
    fm.df_to_csv(pd.DataFrame(out['summary']) ,"summary","hol")
    fm.df_to_csv(pd.DataFrame(out['we_preds']) ,"we_preds","hol")
 
        