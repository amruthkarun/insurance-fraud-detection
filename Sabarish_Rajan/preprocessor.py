addfrom sklearn.preprocessing import OneHotEncoder
import pandas as pd

class _DataPreprocessor_:

    def __init__(self):
        self.ohe = OneHotEncoder(sparse_output = False, drop = 'first', handle_unknown = 'ignore') 

    def fit_transform(self, df:pd.DataFrame):
        cat_cols = df.select_dtypes(include=['object']).columns
        encoded_df = pd.DataFrame(
            self.ohe.fit_transform(df[cat_cols]), 
            columns = self.ohe.get_feature_names_out(),
            index = df.index
        )
        df = df.drop(cat_cols, axis = 1)
        encoded_df = pd.concat([df, encoded_df], axis = 1)

        return encoded_df, encoded_df.columns


    def preprocess_incident_data(self, input_df:pd.DataFrame):

        
        cat_cols = input_df.select_dtypes(include=['object']).columns
        num_cols = input_df.select_dtypes(include=['int64','float64']).columns

        encoded_cols = pd.DataFrame(
            self.ohe.transform(input_df[cat_cols]),
            columns = self.ohe.get_feature_names_out(),
            index = input_df.index
        )

        processed_df = pd.concat([input_df, encoded_cols], axis = 1)

        processed_df = processed_df.drop(cat_cols, axis = 1)

        return processed_df
        