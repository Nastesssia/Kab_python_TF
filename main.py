import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# Шаг 1: Загрузка данных
data = pd.read_csv('creditcard.csv')

# Шаг 2: Исследование данных
print(data.head())
print(data.info())
print(data.isnull().sum())
print(data.describe())

# Шаг 3: Предварительная обработка данных
data['NormalizedAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
data = data.drop(['Time', 'Amount'], axis=1)

# Разделение данных на тренировочные и тестовые
X = data.drop('Class', axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Создание модели
model = RandomForestClassifier(random_state=42)

# Обучение модели
model.fit(X_train, y_train)

# Оценка модели на тестовых данных
y_pred = model.predict(X_test)
print("F1 Score:", f1_score(y_test, y_pred))

# Настройка гиперпараметров модели
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring=make_scorer(f1_score))
grid_search.fit(X_train, y_train)

# Лучшие параметры и F1-мера
print("Лучшие параметры:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# Оценка лучшей модели на тестовых данных
y_pred_best = best_model.predict(X_test)
print("F1 Score лучшей модели:", f1_score(y_test, y_pred_best))
# Шаг 6: Разделение данных на тренировочные и тестовые наборы
X = data.drop('Class', axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Шаг 4 & 5: Создание и обучение модели
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Предсказание на тестовом наборе
y_pred = classifier.predict(X_test)

# Шаг 7: Оценка качества модели
print(classification_report(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Дополнительно: настройка гиперпараметров модели

parameters = {
    'n_estimators': [10, 50, 100],  # Количество "деревьев" в случайном "лесу"
    'max_depth': [None, 10, 20, 30],  # Максимальная глубина "дерева"
    'min_samples_split': [2, 5, 10],  # Минимальное количество образцов, необходимое для разделения узла
    'min_samples_leaf': [1, 2, 4]  # Минимальное количество образцов, необходимое в "листе"
}

cv = GridSearchCV(classifier, parameters, cv=5, scoring='f1')
cv.fit(X_train, y_train)

print("Лучшие параметры:", cv.best_params_)
best_model = cv.best_estimator_

# Предсказание с использованием лучшей модели
y_pred_best = best_model.predict(X_test)

# Оценка лучшей модели
print(classification_report(y_test, y_pred_best))
print("F1 Score лучшей модели:", f1_score(y_test, y_pred_best))
print("Accuracy Score лучшей модели:", accuracy_score(y_test, y_pred_best))
