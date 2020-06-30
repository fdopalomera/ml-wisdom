""" TODO:
    - Samples dictionary
"""

import pandas as pd
import numpy as np
from unidecode import unidecode
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from time import time
import random


class PrepML:
    """
    Clase para realizar el preproces de los datos requerido para los modelos de Machine Learning
    """
    def __init__(self, df):
        self.df = df.dropna().reset_index(drop=True)
        self.columns = list(df.columns)
        self.transformers = []
        self.df_ct = self._clean_categories()

    def _clean_categories(self):
        """
        Convierte los caracterés no alfanúmericos en guiones de todas las columnas del DataFrame de tipo 'object'
        :return: DataFrame con columnas 'object' con valores convertidos
        """
        
        df_ct = self.df
        for var in df_ct.select_dtypes('object').columns:
            df_ct[var] = df_ct[var].map(lambda x: "".join(c if c.isalnum() else "_" for c in str(x)))

        return df_ct

    def onehot_encoder(self, columns, drop_first=True):
        """
        Recodifica las columnas seleccionadas (variables categóricas), creando k o k-1 nuevas columnas
            por cada clase que posea la columna original, imputando con valores 1 y 0 según si en el registro
            se presenta o no la categoría.

        :param columns: [list] lista de columnas del df que se desean procesar por el encoder
        :param drop_first: [bool]
        """

        aux = {'drop': {True: 'first', False: None},
               'unknown': {True: 'error', False: 'ignore'},
               'names': {True: 1, False: 0}
               }

        df_oh = self.df[columns]
        # Convertir valores con caracteres no alfanuméricos
        for var in columns:
            df_oh[var] = df_oh[var].map(lambda x: "".join(c if c.isalnum() else "_" for c in unidecode(str(x))))
        # Categorías para one-hot
        categories_dict = {var: list(df_oh[var]
                                     .value_counts()
                                     .sort_values(ascending=False)
                                     .index)
                           for var in df_oh}
        # Nombre de columnas dummy
        ix = aux['names'][drop_first]
        dummy_names = [f'{var}_{cat}' for var, cat_list in categories_dict.items() for cat in cat_list[ix:]]

        # Instanciamos  objeto de preproceso
        oh_enc = OneHotEncoder(list(categories_dict.values()),
                               sparse=True,
                               drop=aux['drop'][drop_first],
                               handle_unknown=aux['unknown'][drop_first])
        # Entrenamos y transformamos columnas con el encoder
        dummy_data = oh_enc.fit_transform(df_oh)
        prep_df = pd.DataFrame.sparse.from_spmatrix(data=dummy_data,
                                                    columns=dummy_names)
        # Actualizamos la base
        self.df = pd.concat(objs=[self.df.drop(columns=columns),
                                  prep_df],
                            axis=1)
        # Actualizamos atributos
        self.columns = list(self.df.columns)
        self.transformers += [('onehot', oh_enc, columns)]

    def standard_scaler(self, columns):
        """
        Recodifica las columnas seleccionadas (variables continuas) escalando sus valores a través
            de la transformación: (x - mean(X)) / std(X)
        :param columns: columnas seleccionadas
        """

        # Instanciamos y entrenamos/transformamos con objeto de preproceso
        std_enc = StandardScaler()
        std_data = std_enc.fit_transform(self.df[columns])
        prep_df = pd.DataFrame(data=std_data,
                               columns=columns)
        # Actualizamos la base
        self.df = pd.concat(objs=[self.df.drop(columns=columns),
                                  prep_df],
                            axis=1)
        # Actualizamos atributos
        self.columns = list(self.df.columns)
        self.transformers += [('std_scaler', std_enc, columns)]

    def transform_columns(self, transformer_instance, transformer_name, columns):
        """
        Realiza en el DataFrame la instancia de la transformación asignada
        :param transformer_instance: intancia de la clase que genera la transformación
        :param transformer_name: nombre a asignar al transformador
        :param columns: columnas a aplicar la transformación
        """

        # Instanciamos y entrenamos/transformamos con objeto de preproceso
        data = transformer_instance.fit_transform(self.df[columns])
        prep_df = pd.DataFrame(data=data,
                               columns=columns)
        # Actualizamos la base
        self.df = pd.concat(objs=[self.df.drop(columns=columns),
                                  prep_df],
                            axis=1)
        # Actualizamos atributos
        self.columns = list(self.df.columns)
        self.transformers += [(transformer_name, transformer_instance, columns)]

    def remove_outliers(self, columns, iqr_multiplier=1.5, print_diff=False):
        """
        Remove los outliers de las columnas númericas según el critero de los boxplot de John Tucky
        :param columns: columnas para buscar outliers
        :param iqr_multiplier: multiplicador del rango intercuartil
        :param print_diff: imprimir diferencias entre la muestra de entrenamiento de antes y después
                            de la eliminación de outliers.
        """
        before = self.df.shape[0]
        q1 = self.df[columns].quantile(0.25)
        q3 = self.df[columns].quantile(0.75)
        iqr = q3 - q1
        self.df = self.df[~((self.df < (q1 - iqr_multiplier * iqr)) |
                            (self.df > (q3 + iqr_multiplier * iqr))
                            ).any(axis=1)].reset_index(drop=True).copy()
        self.df_ct = self.df

        if print_diff:
            #  Cálculo de diferencia en el tamaño de la muestra de entrenamiento
            after = self.df.shape[0]
            print(f'Cantidad de datos antes de eliminación de outliers: {before}')
            print(f'Cantidad de datos después eliminación de outliers: {after}')
            print(f'Proporción de datos eliminados: {1 - round(after / before, 3)}')

    def log_transformer(self, column):
        """
        Realiza una transformación logística a las columnas seleccionadas

        :param column:  columnas seleccionadas a transformar
        """

        self.df[column] = self.df[column].map(lambda x: np.log(x))

    def to_ml_samples(self, sample_col, target, val_size=.3, random_state=42):
        """
        Divide la base en 6 muestras: de entrenamiento, de prueba y de validación.

        :param sample_col: variable que clasifica el tipo de muestra
        :param target: variable objetivo
        :param val_size: dimensiones de la muestra de prueba
        :param random_state: semilla pseudo-aleatoria
        :return: X_train, y_train, X_test, y_test, X_val, y_val
        """

        start = time()
        random.seed(random_state)

        df_test = self.df[self.df[sample_col] == 'test'].reset_index(drop=True)

        ix = self.df[self.df['sample'] == 'train'].index
        val_number = int(np.floor(len(ix) * val_size))
        val_ix = random.choices(ix, k=val_number)
        train_ix = list(set(ix) - set(val_ix))
        df_val = self.df.loc[val_ix].reset_index(drop=True)
        df_train = self.df.loc[train_ix].reset_index(drop=True)

        X_train = df_train.drop(columns=[target, sample_col])
        y_train = df_train.pop(target)
        X_val = df_val.drop(columns=[target, sample_col])
        y_val = df_val.pop(target)
        X_test = df_test.drop(columns=[target, sample_col])
        y_test = df_test.pop(target)

        length = round(time() - start, 0)
        print(f'Realizado en {length}s')

        return [X_train, y_train, X_test, y_test, X_val, y_val]