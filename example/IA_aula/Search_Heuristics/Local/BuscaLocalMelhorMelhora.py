import time
from Search_Heuristics.AlgoritmoBusca import AlgoritmoBusca
from Search_Heuristics.Local.Neighbourhood.Vizinhanca import Vizinhanca
from Search_Heuristics.Solucao import Solucao


class BuscaLocalMelhorMelhora(AlgoritmoBusca):
    def __init__(self, vizinhanca: Vizinhanca, solucao_otima: int, solucao: Solucao = None):
        super().__init__("BLMM"+vizinhanca.nome, vizinhanca.distancias, solucao_otima)
        self.vizinhanca: Vizinhanca = vizinhanca
        if solucao is None:
            self.solucao = self.gerar_solucao_inicial_aleatoria()
        else:
            self.solucao = solucao

    def buscar_solucao(self) -> list[Solucao]:
        solucao_list = [self.solucao]
        melhor_qualidade = self.solucao.qualidade
        while time.time() < self.tempo_limite:
            self.solucao = self.vizinhanca.melhor_vizinho(self.solucao, set())
            if self.solucao.qualidade < melhor_qualidade:
                melhor_qualidade = self.solucao.qualidade
                self.solucao.tempo = time.time() - self.tempo_limite
                solucao_list.append(self.solucao)
                if melhor_qualidade == self.solucao_otima:
                    return solucao_list
            else:  # não houve melhoras
                break
        return solucao_list
