from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class KeywordTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, keyword_file_name, column_name='keyword'):
        self.column_name = column_name
        self.keyword_file_name = keyword_file_name
        self._df = pd.read_csv(keyword_file_name, index_col=column_name)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if len(X.shape) > 1 and X.shape[1] != 1:
            raise Exception('Can transform only one column')
        X_transformed = pd.merge(X, self._df, on=self.column_name, how='left')
        X_transformed.fillna({'positive_factor': 0.5}, inplace=True)
        return X_transformed

    def get_feature_names_out(self, input_features=None):
        return [self.column_name, 'positive_factor']


def test1():
    t = KeywordTransformer('./keywords_stats.csv')
    df = pd.read_csv('./test.csv')
    print('Shape before', df.shape)
    print(df.sample(n=5))
    df_out = t.fit_transform(df['keyword'])
    print('Shape after', df_out.shape)
    print(df_out.sample(n=5))
    print(t.get_params())
    print(df_out.isnull().sum())
    print(t.get_feature_names_out())


if __name__ == '__main__':
    test1()