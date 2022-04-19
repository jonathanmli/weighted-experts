from file_management import *
import numpy as np

def test_all():
    fm = file_manager('/Users/doctorduality/github-repos/research/Weighted Experts/Simu/', 'Test1', 'Test2')
    
    df = pd.DataFrame(np.random.randn(10,20))
    fm.df_to_csv(df, 'test_folder', 'test_results.csv')

