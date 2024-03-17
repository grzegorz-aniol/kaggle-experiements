import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import re
import spacy
import geonamescache as geo


class LocationTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, column_name='location'):
        self.column_name = column_name

        self.__nlp = spacy.load("en_core_web_sm")
        self.__gc = geo.GeonamesCache()
        self.us_states = dict()
        self.can_states = dict()
        self.countries = None

        self.__build_countries_cache()
        self.__build_states_cache()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if len(X.shape) > 1 and X.shape[1] != 1:
            raise Exception('Can transform only one column')
        source = X if isinstance(X, pd.Series) else X[self.column_name]
        output = source.apply(
            lambda row: self.__build_new_cols(row))
        X_transformed = pd.concat((source, output), axis=1)
        return X_transformed

    def get_feature_names_out(self, input_features=None):
        return [self.column_name, 'country', 'state', 'city', 'missing_location']

    def __build_new_cols(self, location):
        country = city = state = ''
        if not location or not isinstance(location, str) or len(location) == 0:
            missing_location = 1
        else:
            country, state, city = self.__analyze_location(location)        
            missing_location = 0
        if not country:
            country = ''
        if not state:
            state = ''
        if not city:
            city = ''
        return pd.Series({'country': country, 'state': state, 'city': city, 'missing_location': missing_location})

    def __build_countries_cache(self):
        countries_dict = self.__gc.get_countries_by_names()
        countries = {}
        for key, item in countries_dict.items():
            name = str(key)
            countries[key] = name
            countries[item['iso']] = name
        countries['USA'] = countries['United States']
        countries['US'] = countries['United States']
        self.countries = countries

    def __build_states_cache(self):
        can_states = {}
        us_states = {}

        canadian_states = {
            'AB': 'Alberta',
            'BC': 'British Columbia',
            'MB': 'Manitoba',
            'NB': 'New Brunswick',
            'NL': 'Newfoundland and Labrador',
            'NT': 'Northwest Territories',
            'NS': 'Nova Scotia',
            'NU': 'Nunavut',
            'ON': 'Ontario',
            'PE': 'Prince Edward Island',
            'QC': 'Quebec',
            'SK': 'Saskatchewan',
            'YT': 'Yukon'
        }

        for key, value in canadian_states.items():
            can_states[value] = value
            can_states[key] = value

        states_dict = self.__gc.get_us_states_by_names()
        for key, value in states_dict.items():
            us_states[key] = key
            us_states[value['code']] = key

        self.us_states = us_states
        self.can_states = can_states

    def __standardize_country(self, country):
        if country in ['USA', 'US']:
            return 'United States'
        return country

    def __analyze_location(self, txt):
        if txt is np.nan or txt is None or len(txt) == 0:
            return None, None, None

        if txt in self.countries:
            return self.__standardize_country(txt), None, None

        doc = self.__nlp(txt)
        tokens = [ent.text for ent in doc.ents if ent.label_ == 'GPE']
        country = city = state = None

        # Safeguard: detect state in US & Canada
        words = re.split(',', txt)
        if len(words) > 1:
            state_candidate = words[-1].strip()
            if state_candidate in self.us_states:
                state = self.us_states[state_candidate]
                country = 'United States'
            elif state_candidate in self.can_states:
                state = self.can_states[state_candidate]
                country = 'Canada'

        # Search for city first
        known_tokens = []
        for loc in tokens:
            name = str(loc)
            cities = sorted(self.__gc.search_cities(name, attribute='name', case_sensitive=False),
                            key=lambda x: int(x['population']), reverse=True)
            cities = list(filter(lambda x: x['name'].lower() == name.lower(), cities))
            if len(cities) > 0:
                known_tokens.append(name)
                city = cities[0]['name']
                country_code = cities[0]['countrycode']
                country_candidate = self.countries[country_code]
                if (country and country == country_candidate) or not country:
                    country = country_candidate
                    if country_code == 'US':
                        state_code = cities[0]['admin1code']
                        if state_code:
                            state = self.us_states[state_code]
                break

        for loc in tokens:
            name = str(loc)
            if name in known_tokens:
                continue
            if not country and name in self.countries:
                country = name
            elif not state and name in self.us_states:
                state = self.us_states[name]

        if country or state or city:
            return self.__standardize_country(country), state, city
        return None, None, None


def test1():
    t = LocationTransformer()
    df = pd.read_csv('./disaster-tweets/test.csv', index_col='id')
    print('Shape before', df.shape)
    print(df.sample(n=5))
    df_out = t.fit_transform(df['location'])
    print('Shape after', df_out.shape)
    print(df_out.sample(n=5))
    print(t.get_params())
    print(df_out.isnull().sum())
    print(t.get_feature_names_out())


if __name__ == '__main__':
    test1()
