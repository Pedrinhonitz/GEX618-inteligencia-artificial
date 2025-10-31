from Search_Algorithms.Algorithm.Astar import Astar
from Search_Algorithms.Algorithm.BFS import BFS
from Search_Algorithms.Heuristics.Heuristica import Heuristic1, Heuristic2
from Search_Algorithms.Problem.Jogo8.Complexidade import Complexidade
from Search_Algorithms.Problem.Jogo8.Jogo8 import Jogo8


c = Complexidade.Facil
i = 3
for _ in range(1):
    instancia = Jogo8(i, c)
    instancia.estado_inicial.tabuleiro = (6, 4, 7, 8, 5, 0, 3, 2, 1)
    instancia.estado_inicial.vazio = 5

    #instancia.estado_inicial.tabuleiro = (8, 6, 7, 2, 5, 4, 3, 0, 1)
    #instancia.estado_inicial.vazio = 7

    print(instancia.estado_inicial)
    print(c.name)

    algoritmos = [Astar(instancia.estado_inicial, instancia.estado_final, Heuristic2(instancia.estado_final)),
                  Astar(instancia.estado_inicial, instancia.estado_final, Heuristic1(instancia.estado_final)),
                  BFS(instancia.estado_inicial, instancia.estado_final)]

    print(algoritmos[0].arvore_busca.valor_h)

    for alg in algoritmos:
        alg.encontrar_solucao()
        print(f"\n{alg.nome}")
        print(f"Solução Ótima: {alg.solucao}")
        print(f"Custo da S*: {len(alg.solucao)}")
        print(f"Tempo de execução: {alg.tempo_execucao:.3f}s")
        print(f"Fator de Ramificação Médio: {(alg.ramificacao/alg.estados_analisados):.2f}")
        print(f"Quantidade de estados analisados: {alg.estados_analisados}")
