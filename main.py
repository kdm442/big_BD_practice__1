import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer

#Загрузка данных
file_path = '/Users/mihaililicev/Downloads/iot_telemetry_data.csv'
data = pd.read_csv(file_path)

print("Размер набора данных:", data.shape)
print("Первые строки:\n", data.head())
print("Проверка наличия пропусков: ", data.isnull().sum()) # Проверка наличия пропусков
print("Просмотр статистики", data.describe())  # Просмотр статистики

# Разведочный анализ даннных(EDA)
if 'device' in data.columns:
    top_devices = data['device'].value_counts().head(10)
    plt.figure(figsize=(10,5))
    sns.barplot(x=top_devices.index, y=top_devices.values)
    plt.title("Топ-10 устройств по числу записей")
    plt.xticks(rotation=45)
    plt.show()

# распределение категориальных признаков light и motion
for col in ['light', 'motion']:
    if col in data.columns:
        plt.figure(figsize=(5,4))
        sns.countplot(x=col, data=data)
        plt.title(f"Распределение {col}")
        plt.show()

# распределение числовых признаков
num_cols = ['co', 'humidity', 'lpg', 'smoke', 'temp']
data[num_cols].hist(bins=30, figsize=(12,8))
plt.suptitle("Гистограммы сенсорных показателей")
plt.show()

# корреляционная матрица
plt.figure(figsize=(8,6))
sns.heatmap(data[num_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Корреляция сенсоров")
plt.show()

# Предобработка данных

# Заполнение пропусков
imputer = SimpleImputer(strategy="median")
data[num_cols] = imputer.fit_transform(data[num_cols])

# Кодирование категориальных признаков
for col in ['device', 'light', 'motion']:
    if col in data.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))

# Масштабирование числовых данных
scaler = StandardScaler()
data[num_cols] = scaler.fit_transform(data[num_cols])

# Конструирование и отбор признаков

data['gas_sum'] = data['co'] + data['lpg'] + data['smoke']
data['gas_ratio'] = data['co'] / (data['smoke'] + 1e-5)
data['temp_hum_ratio'] = data['temp'] / (data['humidity'] + 1e-5)

data['high_temp'] = (data['temp'] > data['temp'].mean()).astype(int)
data['high_smoke'] = (data['smoke'] > data['smoke'].mean()).astype(int)

# Считаем аномалией превышение CO или дыма
if 'anomaly' not in data.columns:
    data['anomaly'] = ((data['co'] > 1.0) | (data['smoke'] > 1.0)).astype(int)

print("Класс-аннотации (anomaly):")
print(data['anomaly'].value_counts())

# Отбор признаков 
X = data.drop(columns=['anomaly', 'ts'])  
y = data['anomaly']

selector = SelectKBest(score_func=f_classif, k=5)
X_new = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support()]
print("Топ-5 признаков:", selected_features)