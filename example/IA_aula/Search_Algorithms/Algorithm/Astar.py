import heapq
import time

from Search_Algorithms.Algorithm.Algorithm import Algorithm
from Search_Algorithms.Heuristics.Heuristica import Heuristic
from Search_Algorithms.Problem.Jogo8.Estado import Estado
from Search_Algorithms.Tree.Arvore import Arvore


class Astar(Algorithm):
    def __init__(self, estado_inicial: Estado, objetivo: Estado, heuristica: Heuristic):
        super().__init__("A* - " + heuristica.nome, Arvore(estado_inicial, 0, heuristica), objetivo)

    def encontrar_solucao(self):
        inicio = time.time()
        analisar = [self.arvore_busca]
        analisados = set()
        while True:
            # Retira menor elemento, simulando uma fila de prioridades
            arvore = heapq.heappop(analisar)
            analisados.add(arvore.estado.tabuleiro)
            self.estados_analisados += 1
            # Verifica se estado analisado é objetivo
            if not self.eh_objetivo(arvore.estado):
                for vertice in self.expandir(arvore, analisados):
                    heapq.heappush(analisar, vertice)
            else:
                self.solucao = arvore.retornar_acoes()
                break
        fim = time.time()
        self.tempo_execucao = fim - inicio


