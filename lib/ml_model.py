""" TODO:
    - Ojo con categorías no observadas: cambiar a drop_first a 'False'
    - Atributo mejores parámetros
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor, XGBClassifier
from time import time
import pickle


class MLModel:
    """
    Clase que engloba las principales tareas del flujo de trabajo de Machine Learning para entrenar, evaluar y exportar modelos.
    """

    def __init__(self, model):
        self.model = model
        self.best_model = None
        self.target = None
        self.features = None

    @classmethod
    def from_pickle(cls, filepath):
        """
        Constructor alternativo de la clase, para instanciarla a partir de un modelo en pickle
        """
        best_model = pickle.load(open(filepath, 'rb'))

        obj = cls.__new__(cls)
        super(MLModel, obj).__init__()
        obj.model = None
        obj.best_model = best_model

        return obj

    def fit(self, x_train, y_train):
        """
        Entrena el modelo.

        :param x_train: Atributos de entrenamiento
        :param y_train: Variable Objetivo de entrenamiento
        """

        start = time()
        self.best_model = self.model.fit(x_train, y_train)
        self.target = y_train.name
        self.features = x_train.columns

        length = round(time() - start, 0)
        print(f'Realizado en {length}s')

    def grid_search(self, x_train, y_train, param_grid, cv=5, n_jobs=-2):
        """
        Realiza la optimización de hiperparámetros a partir de la grilla definida.

        :param x_train: Atributos de entrenamiento
        :param y_train: Variable objetivo de entrenamiento
        :param param_grid: Grilla de hiperparámetros
        :param cv: Número de validaciones cruzadas a realizar
        :param n_jobs: número de trabajos a realizar en paralelo
        """
        start = time()
        grid = GridSearchCV(estimator=self.model,
                            param_grid=param_grid,
                            n_jobs=n_jobs,
                            cv=cv)
        if isinstance(self.model, XGBRegressor) | isinstance(self.model, XGBClassifier):
            grid.fit(x_train.values, y_train)
        else:
            grid.fit(x_train, y_train)

        print(f'Mejores parámetros:\n{grid.best_params_}\n')
        length = round(time() - start, 0)
        print(f'Realizado en {length}s')

        # Actualzación de atributos
        self.best_model = grid.best_estimator_
        self.target = y_train.name
        self.features = x_train.columns

    def metrics(self, x_test, y_test, ml_problem='reg', print_results=False):
        """
        Devuelve las principales métricas utilizadas para un modelo de regeresión

        :param x_test: Atributos de muestra de prueba
        :param y_test: Variable objetivo de muestra de prueba
        :param print_results: si desea que los resultados sean impresos
        :param ml_problem: tipo de problema de Machine Learning 'reg' o 'clf'
        :return: diccionario con las métricas (RSME, MAE y R2)
        """

        if isinstance(self.best_model, XGBRegressor) | isinstance(self.best_model, XGBClassifier):
            y_hat = self.best_model.predict(x_test.values)
        else:
            y_hat = self.best_model.predict(x_test)
        # Cálculo de métricas según problema de ML: 'reg' o 'clf'
        if ml_problem == 'reg':
            metrics = {'rsme': round(np.sqrt(mean_squared_error(y_true=y_test,
                                                                y_pred=y_hat)), 1),
                       'mae': round(mean_absolute_error(y_true=y_test,
                                                        y_pred=y_hat), 1),
                       'r2': round(r2_score(y_true=y_test,
                                            y_pred=y_hat), 3)}
        elif ml_problem == 'clf':
            metrics = {'roc_score': round(roc_auc_score(y_test, y_hat), 3),
                       'confusion_matrix': confusion_matrix(y_test, y_hat).round(3),
                       'classification_report': classification_report(y_test, y_hat)}
        else:
            raise ValueError("Problema de Machine Learning no valido, escoga 'reg' o 'clf'")

        # Retorno: Imprimir resultados o devlover diccionario
        if print_results:
            aux = {'reg': '', 'clf': '\n'}
            for key, value in metrics.items():
                print(f'{key}:{aux[ml_problem]} {value}')
        else:
            return metrics

    def train_val_metrics(self, X_train, y_train, X_val, y_val):
        """
        Devuelve las principales métricas utilizadas para evaluar modelos de regeresión para las muestra de entrenamiento y de validación.

        :param X_train: Atributos de la muestra de entrenamiento
        :param y_train: Variable objetivo de la muestra de entrenamiento
        :param X_val: Atributos de la muestra de prueba
        :param y_val: Variable objetivo de la muestra de prueba
        :return: DataFrame con las métricas
        """

        train_met = self.metrics(X_train, y_train, print_results=False)
        test_met = self.metrics(X_val, y_val, print_results=False)
        data = [[val for key, val in train_met.items()],
                [val for key, val in test_met.items()]
                ]
        cols = ['RSME', 'MAE', 'R2']
        ix = ['Train', 'Val']

        return pd.DataFrame(data=data, columns=cols, index=ix)

    def feature_importances(self, columns_name):
        """
        Crea un objeto Series con los atributos más relevantes.

        :param columns_name: Nombre de las columnas de la muestra de entrenamiento
        :return: Series con Atributos más relevantes
        """

        if hasattr(self.best_model, 'feature_importances_'):
            return pd.Series(data=self.best_model.feature_importances_,
                             index=columns_name).sort_values(ascending=False)
        else:
            raise ValueError("El algoritmo no tiene el atributo feature_importances")

    def to_pickle(self, car_category):
        """
        Serializa el mejor modelo y guarda el archivo en el sistema.

        :param car_category: categoría del vehículo del modelo
        """

        model_name = self.best_model.__class__.__name__.lower()
        pickle.dump(self.best_model, open(f'best_models/{car_category}_{model_name}.sav', 'wb'))

    def to_pipeline(self, transformers, X_ct):
        """
        Crea un pipeline con el mejor modelo y la lista transformadores asignado.

        :param transformers: lista con tuplas de los trasnformadores
        :param X_ct: Atributos para entrenar el objeto ColumnTransformer
        :return: pipeline instanciado
        """

        col_tf = ColumnTransformer(transformers).fit(X_ct)
        pipeline = Pipeline([
            ('preprocessor', col_tf),
            ('model', self.best_model)
        ])

        return pipeline
