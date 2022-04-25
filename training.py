import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib
from timeit import default_timer

csv_path = 'D:/TCC/CSV/1995_2000_2005_2010_extra_agua.csv'

df = pd.read_csv(csv_path, sep=",")

data_class = df['Classe'].values
#Para Landsat 5
data_all = df[['SAMPLE_1','SAMPLE_2','SAMPLE_3','SAMPLE_4','SAMPLE_5','SAMPLE_6','SAMPLE_7']].values
#Para Landsat 8
#data_all = df[['SAMPLE_1','SAMPLE_2','SAMPLE_3','SAMPLE_4','SAMPLE_5','SAMPLE_6','SAMPLE_7','SAMPLE_8','SAMPLE_9','SAMPLE_10']].values

start = default_timer()

param_grid = {
                "criterion": ['entropy', 'gini'],
                "n_estimators": [20,30,40],#[10, 20, 30, 40, 50, 100],# [20,30,40]
                "bootstrap": [False, True],
                "max_depth": [10,20,30],#[10, 20, 30, 40, 50],# [10,20,30]
                "max_features": ['auto','sqrt'], #['auto',0.1,0.2,0.3]
                "min_samples_leaf": [2, 4, 6], #[2,4,6]
                "min_samples_split": [2, 5, 10], #[2,5,10]
            }

rf = RandomForestClassifier()

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', cv=15)
grid_search.fit(data_all, data_class)

best_estimator = grid_search.best_estimator_
maior_pont = grid_search.best_score_

print('Maior pontuação {}'.format(maior_pont))
end = default_timer()
print('Duração ' + str(int((end - start) / 60)) + ' min')
modelo = 'D:/TCC/tcc_model_16bits_extra.pkl'
joblib.dump(best_estimator, modelo)