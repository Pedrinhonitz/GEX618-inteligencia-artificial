import math
from Search_Heuristics.AlgoritmoBusca import AlgoritmoBusca
from Search_Heuristics.Constructive.Criteria.CriterioGuloso import CriterioGuloso
from Search_Heuristics.Solucao import Solucao
import time


class BuscaConstrutivaGulosoAlfa(AlgoritmoBusca):
    def __init__(self, criterio_guloso: CriterioGuloso, solucao_otima: int, alfa: float):
        super().__init__("BCGα"+str(alfa)+criterio_guloso.nome, criterio_guloso.distancias, solucao_otima)
        self.criterio_guloso: CriterioGuloso = criterio_guloso  # Critério guloso usado para selecionar próximo elemento
        self.alfa: float = alfa  # Configura a quantidade % dentre os primeiros elementos que podem ser selecionados

    def buscar_solucao(self) -> list[Solucao]:
        melhor_qualidade = math.inf
        solucao_list = []
        while time.time() < self.tempo_limite:
            ultimo = 0
            solucao = Solucao(0, [])
            nao_visitados = set(range(1, self.tamanho))
            for _ in range(1, self.tamanho):
                elemento = self.criterio_guloso.melhor_alfa(nao_visitados, ultimo, self.alfa)
                solucao.ciclo.append(elemento)
                solucao.qualidade += self.distancias[ultimo][elemento]
                nao_visitados.remove(elemento)
                ultimo = elemento
            solucao.qualidade += self.distancias[ultimo][0]
            if solucao.qualidade < melhor_qualidade:
                melhor_qualidade = solucao.qualidade
                solucao.tempo = time.time() - self.tempo_limite
                solucao_list.append(solucao)
                if melhor_qualidade == self.solucao_otima:
                    return solucao_list
        return solucao_list
