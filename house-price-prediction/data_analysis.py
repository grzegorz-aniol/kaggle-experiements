import math
import matplotlib as plt

class DataAnalyzer():
    def __init__(self, force_categorical=None, null_threshold=0.1):
        self.__x = None
        self.__y = None
        self.numerical_features = []
        self.categorical_features = []
        self.result_features = []
        self.df_target_corr = None
        self.df_feature_correlation = None
        self.__null_threshold = null_threshold
        self.__force_categorical = force_categorical
        self.__null_skiped = []
        self.__null_fixed = []

    def fit(self, x_set, y_set):
        self.__x = x_set
        self.__y = y_set
        all_cnt = len(x_set)
        
        self.numerical_features = list(x_set.select_dtypes(include=np.number).columns)
        self.categorical_features = list(x_set.select_dtypes(exclude=np.number).columns)
        
        if self.__force_categorical:
            for fc in self.__force_categorical:
                if fc in self.numerical_features:
                    self.numerical_features.remove(fc)
                    self.categorical_features.append(fc)                    
        
        for col in x_set.columns:
            ds = x_set[col]
            unique = ds.unique()
            unique_values = [v for v in unique if v!=np.nan]
            n_unique = len(unique)
            n_unique_values = len(unique_values)
            null_cnt = ds.isnull().sum()
            null_prc = null_cnt/all_cnt
            print(col)
            print(f' => dtype: {ds.dtype}')
            if null_cnt > 0:
                print(f' => nulls {null_cnt} ({100*null_prc:.0f}%)')
            print(f' => unique ({n_unique})', '...' if n_unique > 20 else unique)
            if n_unique_values == len(x_set):
                # skip probably ID columns
                print(f'ID found {col}')
                if col in self.numerical_features:
                    self.numerical_features.remove(col)
                if col in self.categorical_features:
                    self.categorical_features.remove(col)
            elif n_unique_values == 1:
                # skip column with one value
                print(f'Skip column with const value {col}')
                if col in self.numerical_features:
                    self.numerical_features.remove(col)
                if col in self.categorical_features:
                    self.categorical_features.remove(col)
            elif null_prc > self.__null_threshold:
                if col in self.numerical_features:
                    self.numerical_features.remove(col)
                elif col in self.categorical_features:
                    self.categorical_features.remove(col)
                self.__null_skiped.append(col)
            elif null_prc > 0.0:
                self.__null_fixed.append(col)
        
        if not y_set is None:
            x_numerical = x_set[self.numerical_features]
            self.df_target_corr = pd.DataFrame(data=x_numerical.corrwith(y_set), columns=['y']) 
            self.df_feature_correlation = x_numerical.corr()


    def transform(self, x_set):
        self.__x = x_set
        series = []
        for col in self.__x.columns:
            # skip any column not identified either as numerical or categorical
            if (col not in self.numerical_features) and (col not in self.categorical_features):
                continue
            # skip columns with too many NA/null values
            if col in self.__null_skiped:
                continue
            ds = None
            if col in self.__null_fixed:                
                if col in self.numerical_features:
                    # for numerical features, fix NaN with mean
                    fill_value = self.__x[col].mean()
                else:
                    # for categorical features, fix nulls with const 'NA'
                    fill_value = 'NA'
                ds = self.__x[col].fillna(value=fill_value)
            else:
                ds = self.__x[col]
            if self.__force_categorical and col in self.__force_categorical:
                ds = ds.astype(str)
            if col in self.categorical_features:
                # encode categorical features into dummy columns
                dummies = pd.get_dummies(ds, prefix=col, dtype=float)
                for c in dummies.columns:
                    col_s = dummies[c]
                    n_unique = len(col_s.unique())
                    if n_unique > 1:
                        series.append(dummies[c])
                    else:
                        print(f'Skip encoded {c} - one value')
            else:
                series.append(ds)        
        return pd.concat(series, axis=1)


    def get_target_correlated(self, min_threshold=0.3):
        threshold_cond = abs(self.df_target_corr.iloc[:,0]) > min_threshold
        selected_num_features = list(self.df_target_corr[threshold_cond].index)
        return selected_num_features
    
    def get_correlated_features(self, min_threshold=0.5):
        correlated_pairs = []
        columns = list(self.df_feature_correlation.columns)
        for i,col in enumerate(columns):
            for j in range(i+1, len(columns)):
                c = self.df_feature_correlation.iloc[i,j]
                if abs(c) > min_threshold:
                    correlated_pairs.append((columns[i], columns[j]))
        return correlated_pairs
    
    def plot_input_pdfs(self, h_cols=4, fig_size_x=15):
        self.__plot_pdfs(self.__x[self.numerical_features], h_cols, fig_size_x)

    def plot_output_pdfs(self, h_cols=1, fig_size_x=5):
        self.__plot_pdfs(self.__y, h_cols, fig_size_x)

    def plot_feature_target_xy(self, h_cols=4):
        self.__plot_feature_target_xy(self.__x[self.numerical_features], self.__y, h_cols)

    def plot_input_cat(self, h_cols=4):
        self.__plot_cat(self.__x[self.categorical_features], self.__y, h_cols)

    def __plot_pdfs(self, df, h_cols, fig_size_x):
        h_rows = math.ceil(len(df.columns) / h_cols)
        fig, axes = plt.subplots(h_rows, h_cols, figsize=(fig_size_x, 3*h_rows))
        axes = axes.flatten() if type(axes) is np.ndarray else [axes]
        for i, col in enumerate(df.columns):
            ax = axes[i]
            df[col].hist(ax=ax, bins=25)
            ax.set_title(col)
        plt.tight_layout()
        plt.show()    

    def __plot_feature_target_xy(self, df_inputs, df_target, h_cols):
        h_rows = math.ceil(len(df_inputs.columns) / h_cols)
        fix2, axes2 = plt.subplots(h_rows, h_cols, figsize=(15,4 * h_rows), sharey=True)
        axes2 = axes2.flatten()
        for i, col in enumerate(df_inputs.columns):
            ax = axes2[i]
            ax.scatter(df_inputs[col], df_target[df_target.columns[0]])
            ax.set_title(col)
        plt.show()

    def __plot_cat(self, df_inputs, df_target, h_cols):
        h_rows = math.ceil(len(df_inputs.columns) / h_cols)
        fix2, axes2 = plt.subplots(h_rows, h_cols, figsize=(15,3 * h_rows), sharey=True)
        axes2 = axes2.flatten()
        for i, col in enumerate(df_inputs.columns):
            ax = axes2[i]
            sns.boxplot(ax=ax, x=df_inputs[col], y=df_target.iloc[:,0])
        plt.show()

