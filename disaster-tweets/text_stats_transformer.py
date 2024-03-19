from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import string
import nltk
import validators as vld
import re


class TextStatsTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, url_stats_file='url_stats.csv'):
        self.url_stats_file=url_stats_file
        self.__PUNCTUATION = set(string.punctuation)
        self.__STOP_WORDS = set(nltk.corpus.stopwords.words('english'))
        self.__LEMM = nltk.WordNetLemmatizer()
        self.__features_out = []
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):        
        v_clean_text = list()
        v_text_length = list()
        v_upper_text_factor = list()
        v_tags_count = list()
        v_punc_factor = list()
        v_ann_count= list()
        v_urls_count = list()
        v_tokens_count = list()
        v_stop_words_factor = list()
        v_clean_tokens_factor = list()
        v_url_domains = list()
        v_url_redirects = list()

        # load URL statistics
        df_url_stats = pd.read_csv(self.url_stats_file, index_col='url')

        for _, row in X.iterrows():
            text = row['text']
            clean_text = self.clean_text(text)
            tokens = re.split('\\s+', text)
            text_length = len(text) - text.count(' ')
            upper_text_length = sum(1 for c in text if c.isupper())
            upper_text_factor = upper_text_length / text_length
            tags_count = sum(1 for c in text if c=='#')
            punct_factor = sum(1 for c in text if c in self.__PUNCTUATION) / text_length
            ann_count = sum(1 for c in text if c=='@')
            all_urls = list(filter(vld.url, tokens))
            urls_count = len(all_urls)
            tokens_count = len(tokens)
            stop_words_count = sum(1 for token in tokens if token in self.__STOP_WORDS)
            stop_words_factor = stop_words_count / tokens_count
            clean_tokens_count = tokens_count - stop_words_count - urls_count
            clean_tokens_factor = clean_tokens_count / tokens_count

            domains = []
            url_redirects_count = 0            
            for url in all_urls:
                if url in df_url_stats.index:
                    url_info_row = df_url_stats.loc[url]
                    domain = url_info_row['final_domain']
                    redirects = int(url_info_row['url_redirects'])
                    domains.append(domain)
                    url_redirects_count += redirects
            domains_text = ' '.join(domains)            

            v_clean_text.append(clean_text)
            v_text_length.append(text_length)
            v_upper_text_factor.append(upper_text_factor)
            v_tags_count.append(tags_count)
            v_punc_factor.append(punct_factor)
            v_ann_count.append(ann_count)
            v_urls_count.append(urls_count)
            v_tokens_count.append(tokens_count)
            v_stop_words_factor.append(stop_words_factor)
            v_clean_tokens_factor.append(clean_tokens_factor)
            v_url_domains.append(domains_text)
            v_url_redirects.append(url_redirects_count)

        X_transformed = X.copy()
        X_transformed['clean_text'] = v_clean_text
        X_transformed['text_length'] = v_text_length
        X_transformed['upper_text_factor'] = v_upper_text_factor
        X_transformed['tags_count'] = v_tags_count
        X_transformed['punct_factor'] = v_punc_factor
        X_transformed['ann_count'] = v_ann_count
        X_transformed['urls_count'] = v_urls_count
        X_transformed['tokens_count'] = v_tokens_count
        X_transformed['stop_words_factor'] = v_stop_words_factor
        X_transformed['clean_tokens_factor'] = v_clean_tokens_factor
        X_transformed['url_domains'] = v_url_domains
        X_transformed['url_redirects_count'] = v_url_redirects

        self.__features_out = X_transformed.columns

        return X_transformed

    def get_feature_names_out(self, input_features=None):
        return self.__features_out

    def is_not_stopword(self, word):
        return word not in self.__STOP_WORDS

    def is_not_number(self, token):
        return not token.isdigit()

    def is_not_url(self, token):
        return not vld.url(token)

    def remove_punctuation(self, input):
        return ''.join([c for c in input if c not in self.__PUNCTUATION])

    def tokenize(self, input):
        return nltk.word_tokenize(input.lower())

    def clean_text(self, input):
        tokens = re.split('\\s+', input.lower())
        tokens = filter(self.is_not_url, tokens)
        tokens = filter(self.is_not_stopword, tokens)
        tokens = map(self.remove_punctuation, tokens)
        tokens = filter(self.is_not_number, tokens)
        tokens = map(self.__LEMM.lemmatize, tokens)
        return ' '.join(tokens)


def test1():
    df = pd.read_csv('./disaster-tweets/train.csv', index_col='id').sample(n=20)
    print(df.head())
    t = TextStatsTransformer(url_stats_file='./disaster-tweets/url_stats.csv')
    print('Input shape', df.shape)
    df_out = t.fit_transform(df)
    print('Output shape', df_out.shape)
    print(len(t.get_feature_names_out()), t.get_feature_names_out())
    print(df_out[['text', 'urls_count', 'url_domains', 'url_redirects_count']])


if __name__ == '__main__':
    test1()