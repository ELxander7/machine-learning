import random
import numpy as np
import folium
from geopy.distance import geodesic
from typing import List, Tuple

class Point:
    def __init__(self, id: int, name: str, lat: float, lon: float, priority: float):
        self.id = id
        self.name = name
        self.lat = lat
        self.lon = lon
        self.priority = priority
    def __repr__(self): #определяет строковое представление объекта
        return f"{self.name} (prio: {self.priority})"

def load_points_from_data() -> List[Point]:
    points = [
        Point(1, "Красная площадь", 55.753960, 37.620393, 8),
        Point(2, "Белорусский вокзал", 55.774779, 37.588144, 5),
        Point(3, "Парк Сокольники", 55.796312, 37.739975, 10),
        Point(4, "Новоспасский мост", 55.728202, 37.599849, 7),
        Point(5, "Театр на Таганке", 55.759745, 37.604854, 6),
        Point(6, "Гастрономический музей", 55.747778, 37.626672, 9),
    ]
    return points

def calculate_distance(point1: Point, point2: Point) -> float:
    return geodesic((point1.lat, point1.lon), (point2.lat, point2.lon)).km

def total_priority_and_time(route: List[Point], speed_kmh: float) -> Tuple[float, float]:
    total_priority = sum(point.priority for point in route)
    total_time = 0.0

    for i in range(len(route) - 1):
        distance = calculate_distance(route[i], route[i + 1])
        total_time += distance / speed_kmh

    return total_priority, total_time

class Ant:
    def __init__(self, start_node: int):
        self.visited = [start_node]
        self.total_time = 0
        self.total_priority = 0

    def move(self, pheromone: np.ndarray, priorities: List[float], distances: np.ndarray, alpha: float, beta: float, max_time: float, speed_kmh: float) -> bool:
        current = self.visited[-1]
        unvisited = [i for i in range(len(priorities)) if i not in self.visited]

        if not unvisited:
            return False

        probabilities = []
        for next_node in unvisited:
            time_to_next = distances[current][next_node] / speed_kmh
            if self.total_time + time_to_next > max_time:
                continue
            prob = (pheromone[current][next_node] ** alpha) * ((priorities[next_node]) * beta) / (time_to_next + 1e-5)
            probabilities.append((next_node, prob))

        if not probabilities:
            return False

        total_prob = sum(p for _, p in probabilities)
        probabilities = [(n, p / total_prob) for n, p in probabilities]
        next_node = random.choices([n for n, _ in probabilities], weights = [p for _, p in probabilities])[0]

        self.visited.append(next_node)
        self.total_time += distances[current][next_node] / speed_kmh
        self.total_priority += priorities[next_node]
        return True

def ant_colony_optimization(points: List[Point], max_time_hours: float, speed_kmh: float, ants_count: int, iterations: int, alpha: float, beta: float, evaporation_rate: float, q: float) -> List[Point]:
    num_points = len(points)
    distances = np.zeros((num_points, num_points))
    priorities = [point.priority for point in points]

    for i in range(num_points):
        for j in range(num_points):
            if i != j:
                distances[i][j] = calculate_distance(points[i], points[j])

    pheromone = np.ones((num_points, num_points)) * 0.1
    best_route = None
    best_score = 0

    for iteration in range(iterations):
        ant_routes = []
        for _ in range(ants_count):
            start_node = random.randint(0, num_points - 1)
            ant = Ant(start_node)
            while ant.move(pheromone, priorities, distances, alpha, beta, max_time_hours, speed_kmh):
                pass
            ant_routes.append((ant.total_priority, ant.visited))

        pheromone *= evaporation_rate
        for score, route in ant_routes:
            for i in range(len(route) - 1):
                pheromone[route[i]][route[i + 1]] += q * score
            if score > best_score:
                best_score = score
                best_route = route

        print(f"Ieration {iteration + 1}, Best Priority: {best_score}")

    return [points[i] for i in best_route]

def plot_route_with_info(route: List[Point], filename: str = "route.html", speed_kmh: float = 5):
    if not route:
        print("Empty route, nothing to plot")
        return

    m = folium.Map(location=[route[0].lat, route[0].lon], zoom_start=12)

    for i, point in enumerate(route):
        folium.Marker(
            location=[point.lat, point.lon],
            popup=f"{i + 1}. {point.name} (prio: {point.priority})",
            icon=folium.Icon(color='green' if i == 0 else 'blue' if i == len(route) - 1 else 'red')
        ).add_to(m)

    route_coords = [(point.lat, point.lon) for point in route]
    folium.PolyLine(route_coords, color = "blue", weight = 2.5, opacity=1).add_to(m)

    total_priority, total_time = total_priority_and_time(route, speed_kmh)
    num_points = len(route)

    title_html = f"""
        <h3 align="center" style="font-size:16px"><b>Оптимальный маршрут</b></h3>
        <p align="center">Всего точек: {num_points}<br>
        Общий приоритет: {total_priority:.2f}<br>
        Общее время: {total_time:.2f} часов</p>
        """
    m.get_root().html.add_child(folium.Element(title_html))

    m.save(filename)
    print(f"Route map saved to {filename}")

def main():
    points = load_points_from_data()

    max_time_hours = 4
    speed_kmh = 5
    ants_count = 20
    iterations = 100
    alpha = 1 #влияние феромона
    beta = 2 #влияние приоритета
    evaporation_rate = 0.5 #скорость испарения феромонов
    q = 100 #константа обновления феромонов

    print("Running Ant Colony Optimization...")
    best_route = ant_colony_optimization(points, max_time_hours, speed_kmh, ants_count, iterations, alpha, beta, evaporation_rate, q)

    print("\nBest route found:")
    for i, point in enumerate(best_route):
        print(f"{i + 1}. {point}")

    total_priority, total_time = total_priority_and_time(best_route, speed_kmh)
    print(f"\nTotal priority: {total_priority: .2f}")
    print(f"Total time: {total_time:.2f} hours")
    print(f"Time limit: {max_time_hours} hours")

    plot_route_with_info(best_route, filename="best_route.html", speed_kmh=speed_kmh)

if __name__ == "__main__":
    main()