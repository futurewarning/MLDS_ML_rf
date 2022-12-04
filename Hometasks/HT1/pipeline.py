import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle


class PrerocessingPipeline():
    def __init__(self):
        self.NUMERIC_COLUMNS = ['year', 'km_driven', 'mileage', 'engine',
                                'max_power', 'seats']
        self.CATEGORICAL_FEATURES = ['fuel', 'seller_type', 'transmission', 'owner']

        self.scaler = None
        self.encoder = None
        self.na_value = None

    def process_mileage(self, x):
        if x is not None:
            x = str(x)
            if 'kmpl' in x:
                return float(x[:x.find(' kmpl')])
            elif 'km/kg' in x:
                return float(x[:x.find(' km/kg')])
        else:
            return x

    def process_engine(self, x):
        if x is not None:
            x = str(x)
            if 'CC' in x:
                return int(x[:x.find(' CC')])
        else:
            return x

    def process_max_power(self, x):
        if x is not None:
            x = str(x)
            if 'bhp' in x:
                # One item seems to be ' bhp' so we treat it as None
                try:
                    return float(x[:x.find(' bhp')])
                except:
                    return None
        else:
            return x

    def _cast(self, X: pd.DataFrame):
        x = X.copy()
        x.engine = x.engine.astype('int64')
        x.seats = x.seats.astype('int64')
        x.engine = x.engine.astype('int64')
        x.seats = x.seats.astype('int64')
        return x

    def _pre_transform(self, X: pd.DataFrame):
        x = X.copy()
        x.mileage = x.mileage.apply(self.process_mileage)
        x.engine = x.engine.apply(self.process_engine)
        x.max_power = x.max_power.apply(self.process_max_power)
        x.drop('torque', inplace=True, axis=1)
        x.drop('name', axis=1)
        return x

    def fit(self, X: pd.DataFrame):
        x = X.copy()
        x = self._pre_transform(x)

        self.na_value = x.median()
        x.fillna(self.na_value, inplace=True)

        x = self._cast(x)

        self.scaler = StandardScaler()
        self.scaler.fit(x[self.NUMERIC_COLUMNS])

        self.encoder = OneHotEncoder(drop='first', sparse=False)
        self.encoder.fit(x[self.CATEGORICAL_FEATURES])
        return self

    def transform(self, X: pd.DataFrame):
        x = X.copy()
        x = self._pre_transform(x)
        x.fillna(self.na_value, inplace=True)
        x = self._cast(x)
        encoded = pd.DataFrame(self.encoder.transform(x[self.CATEGORICAL_FEATURES]))
        x = pd.DataFrame(self.scaler.transform(x[self.NUMERIC_COLUMNS]),
                         columns=self.NUMERIC_COLUMNS)
        x = pd.concat((x, encoded), axis=1)
        return x

def main():
    df = pd.read_csv('Data/cars_train.csv')
    df = df[~df.drop("selling_price", axis=1).duplicated()]
    df.reset_index(drop=True)

    test = PrerocessingPipeline()
    test.fit(df)

    pickle.dump(test, open('models/process_pipeline.pkl', 'wb'))


if __name__ == '__main__':
    main()