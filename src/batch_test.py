import os
import sys
import numpy as np
import pickle
import re

sys.path.append(os.getcwd())

from rl_game import MazeEnv, QLearningAgent
from config import Config

MODEL_FOLDER = "./models"
RESULTS_FILE = "output/results_qlearning.csv"

def test_model(model_path, runs=1000):
    # Para não abrir janelas (garantir para não renderizar)
    Config.RENDERS = False
    
    env = MazeEnv()
    agent = QLearningAgent(env.action_space, Config.ALPHA, Config.GAMMA, 0)

    try: 
        agent.load_model(model_path)
    except:
        print(f"❌ Erro ao carregar {model_path}")
        return 0, 0

    total_reward = 0
    total_success = 0

    for _ in range(runs):
        state, _ = env.reset(isnumpy=False)
        done = False
        episode_reward = 0

        while not done:
            action = agent.get_action(state)
            state, reward, done, _, _ = env.step(action, isnumpy=False)
            episode_reward += reward

            if reward == Config.GOAL_REWARD:
                total_success += 1

        total_reward += episode_reward

    avg_reward = total_reward / runs
    success_rate = total_success / runs

    return avg_reward, success_rate

# Ordena por número os arquivos para ficar legal no CSV :)
def extract_number_file(file):
    number = re.search(r'\d+', file)
    return int(number.group()) if number else 0

if __name__ == "__main__":
    
    # Vai criar pasta de output se não existir
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    
    files = [f for f in os.listdir(MODEL_FOLDER) if f.endswith(".pkl")]
    models = sorted(files, key=extract_number_file)
    

    if not models:
        print("❌ Nenhum modelo encontrado em ./models/")
        print("   -> Rode o batch_train.py primeiro!")
        exit()

    print(f"📌 Testando {len(models)} modelos...")
    print(f"   -> Resultados serão salvos em: {RESULTS_FILE}")

    with open(RESULTS_FILE, "w") as f:
        f.write("id_modelo,avg_reward,success_rate\n")

        for model in models:
            model_path = os.path.join(MODEL_FOLDER, model)
            print(f"▶ Testando {model} ...")

            avg_reward, success = test_model(model_path)
            
            f.write(f"{model},{avg_reward},{success}\n")
            print(f"R: {avg_reward:.2f} | S: {success:.2%}")

    print("\n✅ Testes concluídos!")
    print(f"📄 Resultados salvos em: {RESULTS_FILE}")
