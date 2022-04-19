import os
import pandas as pd


class file_manager:
    '''
    retrieves data from correct folders/files
    outputs results to correct folders/files
    '''
    def __init__(self, root_path, data_index, results_index) -> None:
        self.root_path = root_path
        self.data_path = root_path + 'Data_' + data_index+ '/'
        self.results_path = root_path + 'Data_' + data_index + 'Results_' + results_index+ '/'
        self.write_dir(self.data_path)
        self.write_dir(self.results_path)

    def write_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
    
    def df_to_csv(self, df, folder, name):
        path = self.results_path + folder + '/'
        self.write_dir(path)
        df.to_csv(path + name + '.csv', header=True, index=False)

    def get_results_file(self, folder, name, iscsv = False, **kwargs):
        if iscsv:
            return pd.read_csv(self.results_path + folder +'/'+ name, delimiter=',', **kwargs)
        else:
            raise NotImplementedError()

    def get_data_file(self, name, iscsv = False):
        if iscsv:
            return pd.read_csv(self.data_path + name, delimiter=',')
        else:
            raise NotImplementedError()
    
    def summarize_csv_results(self, folder):
        path = self.results_path + folder + '/'
        # iterate over files in
        # that directory
        n = 0
        aggregate_df = None
        for root, dirs, files in os.walk(path):
            for filename in files:
                n += 1
                pathy = os.path.join(root, filename)
                print(pathy)
                if aggregate_df is None:
                    aggregate_df = pd.read_csv(pathy, header=None)
                else:
                    aggregate_df += pd.read_csv(os.path.join(root, filename), header=None)
                    print(aggregate_df)

        aggregate_df /= n
        return aggregate_df
           

        

