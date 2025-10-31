import time
from Search_Heuristics.AlgoritmoBusca import AlgoritmoBusca
from Search_Heuristics.Local.Neighbourhood.Vizinhanca import Vizinhanca
from Search_Heuristics.Solucao import Solucao


class BuscaTabu(AlgoritmoBusca):
    def __init__(self, vizinhanca: Vizinhanca, solucao_otima: int, parametro_mandato: int, solucao: Solucao = None):
        super().__init__("BT"+vizinhanca.nome+"-m"+str(parametro_mandato), vizinhanca.distancias, solucao_otima)
        self.parametro_mandato: int = parametro_mandato
        self.vizinhanca: Vizinhanca = vizinhanca
        if solucao is None:
            self.solucao = self.gerar_solucao_inicial_aleatoria()
        else:
            self.solucao = solucao

    def buscar_solucao(self) -> list[Solucao]:
        solucao_list = [self.solucao]
        melhor_qualidade = self.solucao.qualidade
        tabu = set()
        tabu_mandato = list()
        mandato = round(self.tamanho/self.parametro_mandato)
        qtd_trocas = self.vizinhanca.qtd_trocas
        while time.time() < self.tempo_limite:
            self.solucao = self.vizinhanca.melhor_vizinho(self.solucao, tabu)
            tabu.add(self.solucao.i_movimento)
            tabu_mandato.append(self.solucao.i_movimento)
            if qtd_trocas == 2:
                tabu.add(self.solucao.j_movimento)
                tabu_mandato.append(self.solucao.j_movimento)
            if self.solucao.qualidade < melhor_qualidade:
                melhor_qualidade = self.solucao.qualidade
                self.solucao.tempo = time.time() - self.tempo_limite
                solucao_list.append(self.solucao)
                if melhor_qualidade == self.solucao_otima:
                    return solucao_list
            if mandato == 0:
                tabu.remove(tabu_mandato.pop(0))
                if qtd_trocas == 2:
                    tabu.remove(tabu_mandato.pop(0))
            else:
                mandato -= 1
        return solucao_list
