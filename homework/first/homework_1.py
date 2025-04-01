import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import os
from PIL import Image


def find_optimal_k(X, k_range=range(2, 11)):
    print("Определение оптимального количества кластеров (для демонстрации)...")
    inertia = []
    silhouette_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))

    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f'Оптимальное количество кластеров (для демонстрации): {optimal_k}')

    # Визуализация методов
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertia, 'bo-')
    plt.xlabel('Количество кластеров')
    plt.ylabel('Инерция')
    plt.title('Метод локтя')

    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_scores, 'bo-')
    plt.xlabel('Количество кластеров')
    plt.ylabel('Силуэтный коэффициент')
    plt.title('Метод силуэта')

    plt.tight_layout()
    plt.show()

    return optimal_k


def manual_kmeans(X, K=3, max_iter=10, tol=1e-6):
    print("\nЗапуск ручной реализации K-means (3 кластера)...")

    # Стандартизация данных
    X_scaled = StandardScaler().fit_transform(X)
    n_samples, n_features = X_scaled.shape

    # Цвета для кластеров (фиксированные 3 цвета)
    colors = ['red', 'blue', 'green']

    # Инициализация центроидов
    np.random.seed(42)
    centroids = X_scaled[np.random.choice(n_samples, K, replace=False)]

    # Создание папки для изображений
    if not os.path.exists('kmeans_iterations'):
        os.makedirs('kmeans_iterations')

    # Все возможные пары признаков для визуализации
    feature_pairs = list(combinations(range(n_features), 2))

    def plot_clusters(X, labels, centroids, iteration, feature_pairs):
        plt.figure(figsize=(15, 10))
        for i, (f1, f2) in enumerate(feature_pairs, 1):
            plt.subplot(2, 3, i)
            for k in range(K):
                cluster_points = X[labels == k]
                plt.scatter(cluster_points[:, f1], cluster_points[:, f2],
                            label=f'Кластер {k+1}', color=colors[k], alpha=0.7)
                plt.scatter(centroids[k, f1], centroids[k, f2],
                            marker='x', s=200, c='black', linewidths=2)
            plt.xlabel(iris.feature_names[f1])
            plt.ylabel(iris.feature_names[f2])
            plt.title(f'{iris.feature_names[f1]} vs {iris.feature_names[f2]}')
            plt.legend()
        plt.suptitle(f'Итерация {iteration}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'kmeans_iterations/iteration_{iteration}.png')
        plt.close()

    # Основной цикл K-means
    for iteration in range(max_iter):
        # Назначение точек кластерам
        distances = np.sqrt(((X_scaled - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=0)

        # Визуализация текущего состояния кластеров
        plot_clusters(X_scaled, labels, centroids, iteration, feature_pairs)

        # Обновление центроидов
        new_centroids = np.array([X_scaled[labels == k].mean(axis=0) for k in range(K)])

        # Проверка сходимости
        if np.allclose(centroids, new_centroids, atol=tol):
            print(f'Сходимость достигнута на итерации {iteration}')
            break

        centroids = new_centroids

    # Создание GIF-анимации
    print("Создание анимации...")
    images = []
    for i in range(iteration + 1):
        img = Image.open(f'kmeans_iterations/iteration_{i}.png')
        images.append(img)

    images[0].save('kmeans_animation.gif', save_all=True, append_images=images[1:],
                   optimize=False, duration=1000, loop=0)

    return labels, centroids


def final_visualization(X, labels, centroids, K=3):
    print("\nФинальная визуализация (3 кластера)...")

    # Стандартизация данных
    X_scaled = StandardScaler().fit_transform(X)
    n_samples, n_features = X_scaled.shape

    # Все возможные пары признаков для визуализации
    feature_pairs = list(combinations(range(n_features), 2))

    # Цвета для кластеров (фиксированные 3 цвета)
    colors = ['red', 'blue', 'green']

    # Финальная визуализация всех проекций
    print("Создание финальной визуализации...")
    plt.figure(figsize=(15, 10))
    for i, (f1, f2) in enumerate(feature_pairs, 1):
        plt.subplot(2, 3, i)
        for k in range(K):
            cluster_points = X_scaled[labels == k]
            plt.scatter(cluster_points[:, f1], cluster_points[:, f2],
                        label=f'Кластер {k+1}', color=colors[k], alpha=0.7)
            plt.scatter(centroids[k, f1], centroids[k, f2],
                    marker='x', s=200, c='black', linewidths=2)
        plt.xlabel(iris.feature_names[f1])
        plt.ylabel(iris.feature_names[f2])
        plt.title(f'{iris.feature_names[f1]} vs {iris.feature_names[f2]}')
        plt.legend()

    plt.suptitle('Финальный результат кластеризации - все проекции (3 кластера)', fontsize=16)
    plt.tight_layout()
    plt.savefig('final_clusters.png')
    plt.show()

if __name__ == "__main__":
    # Загрузка данных
    iris = load_iris()
    X = iris.data

    # Определение оптимального K
    optimal_k = find_optimal_k(X)

    # Ручной K-means (3 кластера)
    cluster_labels, cluster_centroids = manual_kmeans(X)

    # Финальная визуализация (3 кластера)
    final_visualization(X, cluster_labels, cluster_centroids)

    print("\nГотово! Результаты сохранены в файлы:")
    print("- kmeans_animation.gif (анимация процесса кластеризации)")
    print("- final_clusters.png (финальные кластеры)")
