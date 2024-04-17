import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from random import choice, random
import random
import warnings
from typing import List, Tuple, Set

warnings.filterwarnings("ignore")

class fullSearch:
    def __init__(self,m):
        self.matrix = GraphMatrix(m)
        self.full_search()
        print(self.solution)
        show_solution(self.matrix.matrix, self.solution)
        self.matrix.print_matrix()

    def is_cover(self) -> bool:
        covers = True
        m = len(self.matrix.matrix)

        for i in range(m):
            for j in range(i + 1, m, 1):
                if self.matrix.matrix[i][j]:
                    if (i in self.solution and j in self.solution):
                        covers = False
                        break
            if not covers:
                break

        return covers


    def full_search(self):
        m = len(self.matrix.matrix)
        visited = [False] * m
        self.solution = []
        def search(cover: List[int], n: int) -> bool:
            if n == 0:
                return self.is_cover()
            else:
                for i in range(m):
                    if not visited[i]:
                        visited[i] = True
                        cover.append(i)
                        if search(cover, n - 1):
                            return True
                        else:
                            visited[i] = False
                            cover.pop()
            return False

        for n in range(m-1, 1,-1):
            if search(self.solution, n):
                break

        return self.solution


class simpleGreed:
    def __init__(self,m):
        self.matrix = GraphMatrix(m)
        self.simple_greedy()
        print(self.solution)
        show_solution(self.matrix.matrix, self.solution)
        self.matrix.print_matrix()
    def simple_greedy(self):
        self.vertexes = [i for i in range(len(self.matrix.matrix))]
        self.solution = []
        def remove_vertexes(vert):
            self.vertexes = [ v for v in self.vertexes if not self.matrix.matrix[v][vert]]
        while len(self.vertexes) != 0:
            #vert = choice(self.vertexes)
            vert = max(self.vertexes,key=lambda x:sum([i for i in self.matrix.matrix[x] if i not in self.solution]))
            self.vertexes.remove(vert)
            self.solution.append(vert)
            remove_vertexes(vert)

        return self.solution


class GraphMatrix:
    def __init__(self,m):
        self.matrix = self.generate_matrix(m)
    def generate_matrix(self,m):
        """Генерация случайной матрицы смежности графа."""
        self.matrix = [[False] * m for _ in range(m)]

        for i in range(m):
            for j in range(i + 1, m):
                if choice((True, False)):
                    self.matrix[i][j] = True
                    self.matrix[j][i] = True

        return self.matrix
    def edges_to_ndarray(edges: List[Tuple[int, int]]):
        num_edges = len(edges)

        res = np.ndarray(shape=(num_edges, 2), dtype=int)
        for i, (v1, v2) in enumerate(edges):
            res[i, 0] = v1
            res[i, 1] = v2

        return res
    def print_matrix(self):
        m = len(self.matrix)
        width = len(str(m)) + 1

        text = [f'{"":^{width}}{"".join(f"{i:^{width}}" for i in range(m))}']
        text.append('-' * len(text[0]))
        for i, row in enumerate(self.matrix):
            text.append(f'{f"{i}":^{width}}|{"".join(f"{1 if self.matrix[i][j] else 0:^{width}}" for j in range(m))}')

        print('\n'.join(text))


class AdvancedVertexSupportAlgorithm:
    def __init__(self,m):
        self.matrix = GraphMatrix(m)
        self.avsa()
        print(self.solution)
        show_solution(self.matrix.matrix, self.solution)
        self.matrix.print_matrix()
    def avsa(self) -> List[int]:
        matrix = self.matrix.matrix
        m = len(matrix)
        edges = matrix_to_edges(matrix)
        self.solution = []
        self.verticles = [i for i in range(m)]


        def degree(edges: List[Tuple[int, int]], v: int) -> int:
            degree = 0
            for v1, v2 in edges:
                if v1 == v or v2 == v:
                    degree += 1
            return degree

        def support(edges: List[Tuple[int, int]], degrees: List[int], v: int) -> int:
            res = 0
            for (v1, v2) in edges:
                if v1 == v:
                    res += degrees[v1]
                elif v2 == v:
                    res += degrees[v2]

            return res

        def min_elements(support_values: List[int]) -> List:
            """Нахождение вершинын наименьшим значением поддержки."""
            res = [None] * len(support_values)
            l = [value for value in support_values if value is not None]
            if l:
                m = min([value for value in support_values if value is not None])
                for i, value in enumerate(support_values):
                    if value is not None:
                        if value == m:
                            res[i] = value

            return res

        def find_max_neighbor(min_support: List[int], edges: List[Tuple[int, int]]) -> int:
            """Нахождение соседа с максимальным значением поддержки"""
            max_neighbor = None
            max_support = None

            for v1 in self.verticles:
                sup1 = 0
                for v2, sup2 in enumerate(min_support):
                    if sup2 is not None:
                        if (v1, v2) in edges or (v2, v1) in edges:
                            sup1 += sup2

                if max_neighbor is None:
                    max_neighbor = v1
                    max_support = sup1
                elif max_support < sup1:
                    max_neighbor = v1
                    max_support = sup1

            return max_neighbor

        while len(self.verticles) != 0:
            degrees = [degree(edges, v) for v in range(m)]
            support_values = [support(edges, degrees, v) if degrees[v] != 0 else None for v in range(m)]
            min_support = min_elements(support_values)
            neighbor = find_max_neighbor(min_support, edges)
            self.solution.append(neighbor)
            self.verticles = [v for v in self.verticles if (neighbor,v) not in edges and (v,neighbor) not in edges]
            self.verticles.remove(neighbor)


def edges_to_ndarray(edges: List[Tuple[int, int]]):
    num_edges = len(edges)

    res = np.ndarray(shape=(num_edges, 2), dtype=int)
    for i, (v1, v2) in enumerate(edges):
        res[i, 0] = v1
        res[i, 1] = v2

    return res

def genetic_alg(m):
    matrix = GraphMatrix(m)
    m = len(matrix.matrix)
    edges = matrix_to_edges(matrix.matrix)
    num_edges = len(edges)
    edges = edges_to_ndarray(edges)

    class Agent:
        def __init__(self, generation, dna, mutation_rate):
            self.generation = generation
            self.dna = dna
            self.mutation_rate = mutation_rate
            self.fitness = 0

        def mutate(self):
            """Оператор мутации"""
            self.dna = self.dna.copy()
            mutation_mask = np.random.rand(m) < self.mutation_rate / m
            self.dna[mutation_mask] = 1 - self.dna[mutation_mask]  # перестановка
            return self

        def crossover(self, other_agent, offspring: int):
            """Скрещивание особей"""
            selection = np.random.choice([0, 1], size=(m)).astype(np.bool_)
            dna = np.choose(selection, [self.dna, other_agent.dna])
            return [Agent(self.generation + 1, dna, self.mutation_rate)
                    for i in range(offspring)]

        def update_fitness(self):
            """Приспособленность особи = сумма чисел - число непокрытых ребер * число ребер"""
            vert_cover_fitness = np.full([num_edges], -num_edges)
            mask = (self.dna[edges[:, 0]] | self.dna[edges[:, 1]]).astype(bool)
            vert_cover_fitness[mask] = 1.0
            self.fitness = np.sum(vert_cover_fitness) - np.sum(self.dna)
            return self.fitness

    class GA:
        def solve(self, population_size=30, mutation_rate=1):
            self.population_fitness = []
            self.populatin_size = population_size

            initial_dna = np.ones(m, dtype=np.int_)
            population = [Agent(1, initial_dna, mutation_rate) for _ in range(self.populatin_size)]
            population_fitness = [agent.update_fitness() for agent in population]
            self.population_fitness.append(np.max(population_fitness))

            result = self.run(population)

            result.sort(key=lambda ag: ag.fitness, reverse=True)
            return result[0]

        def selection(self, population):
            """
            Выбор 20% лучших по значению приспособленности
            и выбор 5% случайных особей
            """
            top_n = round(self.populatin_size * 0.2)
            lucky_losers = round(self.populatin_size * 0.05)
            winners = sorted(population, key=lambda ag: ag.fitness, reverse=True)
            return winners[:top_n] + random.sample(winners[top_n:], lucky_losers)

        def crossover(self, parents):
            """Скрещивание особей"""
            offspring = round(self.populatin_size / (len(parents) / 2))
            random.shuffle(parents)
            children = []
            for i in range(0, len(parents) // 2):
                children += parents[i * 2].crossover(parents[i * 2 + 1], offspring)
            return children

        def run(self, population):
            generation = 0
            delta_fitness = 2

            while delta_fitness > 1 and generation < 100:
                parents = self.selection(population)  # селекция особей
                next_gen = self.crossover(parents)  # скрещивание
                mut_gen = [agent.mutate() for agent in next_gen]  # мутирование нового поколения
                population_fitness = [agent.update_fitness() for agent in mut_gen]  # вычисление приспособленности

                if len(self.population_fitness) >= 50:
                    delta_fitness = abs(np.sum(np.diff(self.population_fitness[-50:])))
                population = mut_gen

                generation += 1
                self.population_fitness.append(np.max(population_fitness))
            return population

    best_agent = GA().solve()
    result = [i for i, gene in enumerate(best_agent.dna) if gene == 1]
    result = [i for i in range(len(matrix.matrix)) if i not in result]
    print(result)
    show_solution(matrix.matrix,result)
    print(matrix.print_matrix(),sep='\n')


def show_solution(graph: List[List[bool]], cover: List[int]):
        edges = matrix_to_edges(graph)
        m = len(graph)

        g = nx.Graph()
        g.add_nodes_from(range(m))
        g.add_edges_from(edges)

        pos = nx.spring_layout(g)
        nx.draw(g,
                pos=pos,
                with_labels=True)
        nx.draw_networkx_nodes(g,
                               pos=pos,
                               nodelist=[v for v in range(m) if v in cover],
                               node_color='g')
        plt.show()

def matrix_to_edges(matrix: List[List[bool]]) -> List[Tuple[int, int]]:
        """Конвертация матрицы смежности вершин в список ребер"""
        m = len(matrix)
        edges = []

        for i in range(m):
            for j in range(i + 1, m, 1):
                if matrix[i][j]:
                    edges.append((i, j))

        return edges

# print("Точный алгоритм")
# f = fullSearch(5)
print("Жадный алгоритм")
f = simpleGreed(10)
# print("AVSA")
# f = AdvancedVertexSupportAlgorithm(7)
# print("Генетический алгоритм")
# print(genetic_alg(7))