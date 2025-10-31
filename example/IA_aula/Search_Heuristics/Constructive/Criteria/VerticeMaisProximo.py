from heapq import heappush, heappushpop
from Search_Heuristics.Constructive.Criteria.CriterioGuloso import CriterioGuloso
import math
import random


class VerticeMaisProximo(CriterioGuloso):
    def __init__(self, distancias):
        super().__init__("vmp", distancias)

    def melhor_elemento(self, nao_visitados: set[int], ultimo: int) -> int:
        melhor = math.inf
        melhor_elemento = 0
        for i in nao_visitados:
            if self.distancias[i][ultimo] < melhor:
                melhor = self.distancias[i][ultimo]
                melhor_elemento = i
        return melhor_elemento

    def melhor_alfa(self, nao_visitados: set[int], ultimo: int, alfa: float) -> int:
        melhores_alfa = math.ceil(len(nao_visitados) * alfa)
        if melhores_alfa <= 1:
            return self.melhor_elemento(nao_visitados, ultimo)
        heap_max = []
        for i in nao_visitados:
            if len(heap_max) < melhores_alfa:
                # Multiplica por -1 para transformar heap min (padrão) em heap max
                heappush(heap_max, (self.distancias[i][ultimo] * -1, i))
            elif heap_max[0][0] < self.distancias[i][ultimo] * -1:
                heappushpop(heap_max, (self.distancias[i][ultimo] * -1, i))
        return heap_max[random.randint(0, melhores_alfa - 1)][1]
