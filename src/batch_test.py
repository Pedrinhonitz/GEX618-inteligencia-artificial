import os
import numpy as np
import pickle
from rl_game import MazeEnv, QLearningAgent
from config import Config

MODEL_FOLDER = "./models"
RESULTS_FILE = "output/results_qlearning.csv"

def test_model(model_path, runs=1000):
    env = MazeEnv()
    agent = QLearningAgent(env.action_space, Config.ALPHA, Config.GAMMA, 0)

    agent.load_model(model_path)

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


if __name__ == "__main__":
    models = sorted([f for f in os.listdir(MODEL_FOLDER) if f.endswith(".pkl")])

    if not models:
        print("❌ Nenhum modelo encontrado em ./models/")
        exit()

    print("📌 Testando modelos...")

    with open(RESULTS_FILE, "w") as f:
        f.write("id_modelo,avg_reward,success_rate\n")

        for model in models:
            model_path = os.path.join(MODEL_FOLDER, model)
            print(f"▶ Testando {model} ...")

            avg_reward, success = test_model(model_path)

            f.write(f"{model},{avg_reward},{success}\n")

    print("\n✅ Testes concluídos!")
    print(f"📄 Resultados salvos em: {RESULTS_FILE}")
