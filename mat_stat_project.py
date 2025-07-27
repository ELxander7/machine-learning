# Импорт необходимых библиотек
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import shapiro
from scipy.stats import stats
from scipy.stats import norm
# Игнорирование предупреждений
import warnings
warnings.filterwarnings('ignore')

#1-Загрузка данных
df = pd.read_csv('d:/учёба/2 курс/4 семестр/матстат/практика/проект/social_media_vs_productivity.csv')

# Основная информация
print("\n1.Основная информация о данных.")
print("Количество наблюдений:", len(df))
print("Количество переменных:", len(df.columns))


print("\nТипы данных:")
print(df.dtypes)
print("\nКоличество пропущенных значений:")
print(df.isnull().sum())

print("\nКоличество пропущенных значений, после заполнения:")
df['daily_social_media_time'].fillna(df['daily_social_media_time'].mean(), inplace=True)
df['perceived_productivity_score'].fillna(df['perceived_productivity_score'].mean(), inplace=True)
df['stress_level'].fillna(df['stress_level'].mean(), inplace=True)
df['sleep_hours'].fillna(df['sleep_hours'].mean(), inplace=True)
df['actual_productivity_score'].fillna(df['actual_productivity_score'].mean(), inplace=True)
df['screen_time_before_sleep'].fillna(df['screen_time_before_sleep'].mean(), inplace=True)
df['job_satisfaction_score'].fillna(df['job_satisfaction_score'].mean(), inplace=True)
print(df.isna().sum())

#2-Основные статистические характеристики
print("\n2.Основные характеристики данных.")
print("\nСтатистика:")
#numeric_cols = ['daily_social_media_time']
numeric_cols = ['age','daily_social_media_time', 'number_of_notifications', 'work_hours_per_day', 'actual_productivity_score', 'stress_level','days_feeling_burnout_per_month','job_satisfaction_score']

statistics_df = df[numeric_cols].describe().T
with pd.option_context('display.max_columns', 8):
    print(statistics_df)

# Дисперсия
print("\nДисперсия:")
print(df[numeric_cols].var())

# Корреляция
print("\nКорреляция:")
with pd.option_context('display.max_columns', 8):
    print(df[numeric_cols].corr())

# 3. Визуализация данных (оптимизированная)

# Цикл для создания графиков для каждого числового столбца
for column in numeric_cols:
    # Создание нового графика с 2 подграфиками
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

   # Гистограмма c графиком функции распределения
    sns.histplot(x=df[column], bins=30, kde=True, ax=axes[0], color='royalblue')
    axes[0].set_title('Гистограмма с графиком функции распределения', fontsize=16)

    # Визуализация Box plot (ящик с усами)
    sns.boxplot(data=df, x="gender", y=df[column], color='lightcoral')
    axes[1].set_title('Боксплот', fontsize=16)

    plt.suptitle(f'{column}')
    plt.tight_layout(pad=2.0, w_pad=1.0, h_pad=4.0)

    plt.show()

# Диаграммы рассеивания
for column in numeric_cols:
   if column != 'actual_productivity_score':
       plt.figure(figsize=(10, 8))
       sns.scatterplot(x=df[column], y=df['actual_productivity_score'])

       plt.title(f'Диаграмма рассеивания. Соотношение: {column} vs actual_productivity_score')
       plt.show()

#4-Нахождение выбросов через IQR
def find_outliers_iqr(column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

print(f"\n4.Количество выбросов:")
df_outliers = {}
for column in numeric_cols:
    df_outliers[column] = find_outliers_iqr(column)
    print(f"{column}:", len(df_outliers[column]))


print("\n5. Проверка гипотез.")
# Сравнение средних между Фактическим показателем продуктивности и Oценки удовлетворения_работы (t-test):
print("\nt-тест сравнения средних:")
for column in numeric_cols:
    if column == 'actual_productivity_score':  # Пропускаем столбец
        continue
    t_stat, p_val = stats.ttest_ind(df[column], df['actual_productivity_score'])
    print(f"\nt-тест: {column} и actual_productivity_score")
    print(f"t= {t_stat:.4f}, p = {p_val:.4f}")
    if p_val < 0.05:
        print("Средние значения статистически различаются")
    else:
        print("Средние значения не различаются")

#Проверка гипотезы о нормальности распределения (Shapiro-Wilk test):
print("\nПроверка гипотезы о нормальности распределения (Shapiro-Wilk test):")
for column in numeric_cols:
    stat, p = shapiro(df[column].dropna())
    print(f"{column}: W-статистика = {stat:.8f}, p-value = {p:.8f}")
    if p > 0.05:
        print("Распределение нормальное (не отвергаем H0)")
    else:
        print("Распределение не нормальное (отвергаем H0)")
    print()


for col in numeric_cols:
    plt.figure(figsize=(10, 4))
    sns.histplot(df[col], kde=True, stat="density", bins=100, label='Empirical')
    plt.title(f'Распределение: {col}')
    if not df[col].isnull().any():
        x = np.linspace(df[col].min(), df[col].max(), 100)
        normal_curve = norm.pdf(x, loc=df[col].mean(), scale=df[col].std())
        plt.plot(x, normal_curve, 'm:', label='Нормальное распределение', color='red')

    plt.legend()
    plt.show()

#6-Линейная регрессия
# Выбор признака и целевой переменной
X_subset = df[['job_satisfaction_score']]  # выбрали 'признак' = job_satisfaction_score
y_subset = df['actual_productivity_score']  # выбрали 'целевая_переменная' = job_satisfaction_score
X = X_subset[:1000]
y = y_subset[:1000]

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# Прогнозирование значений
y_pred = model.predict(X_test)

# Создание графика
plt.scatter(X_test, y_test, color='blue', label='Исходные данные')
plt.plot(X_test, y_pred, color='red', label='Линейная регрессия')
plt.xlabel('job_satisfaction_score')
plt.ylabel('actual_productivity_score')
plt.title('График линейной регрессии')
plt.legend()
plt.show()

# Коэффициенты модели
print(f"Коэффициент регересии: {model.coef_[0]:.4f}")
print(f"Свободный член: {model.intercept_:.4f}")
print(f"R²: {model.score(X, y):.4f}")
print(f"Уравнение регрессии: [actual_productivity_score] = {model.intercept_:.4f} + {model.coef_[0]:.4f} * [job_satisfaction_score] ")

# Оценка модели
mse = mean_squared_error(y_test, y_pred)
print(f"Среднеквадратичная ошибка: {mse:.4f}")