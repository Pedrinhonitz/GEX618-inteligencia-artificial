import time
from collections import deque

from Search_Algorithms.Algorithm.Algorithm import Algorithm
from Search_Algorithms.Problem.Jogo8.Estado import Estado
from Search_Algorithms.Tree.Arvore import Arvore


class BFS(Algorithm):
    def __init__(self, estado_inicial: Estado, objetivo: Estado):
        super().__init__("BFS", Arvore(estado_inicial, 0), objetivo)

    def encontrar_solucao(self):
        inicio = time.time()
        analisar = deque([self.arvore_busca])
        analisados = set()
        while True:
            # Retira primeiro elemento, simulando uma fila
            arvore = analisar.popleft()
            analisados.add(arvore.estado.tabuleiro)
            self.estados_analisados += 1
            # Verifica se estado analisado é objetivo
            if not self.eh_objetivo(arvore.estado):
                analisar.extend(self.expandir(arvore, analisados))
            else:
                self.solucao = arvore.retornar_acoes()
                break
        fim = time.time()
        self.tempo_execucao = fim - inicio
