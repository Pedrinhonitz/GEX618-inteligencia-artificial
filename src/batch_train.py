import os
import csv
import shutil
from config import Config

PARAM_FOLDER = "./seeds"

RAW_MODEL = "q_learning_model.pkl"

MODEL_FOLDER = "./models"
os.makedirs(MODEL_FOLDER, exist_ok=True)


def load_csv_into_config(csv_path):
    print(f"\n==============================")
    print(f"🔧 Carregando parâmetros: {csv_path}")
    print("==============================")

    with open(csv_path, "r") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) != 2:
                continue

            key, value = row[0], row[1]

            try:
                value = float(value)
                if value.is_integer():
                    value = int(value)
            except:
                pass

            if hasattr(Config, key):
                setattr(Config, key, value)
                print(f"  - {key} = {value}")
            else:
                print(f"⚠️ Aviso: parâmetro desconhecido no CSV -> {key}")


def run_training(csv_name):
    csv_path = os.path.join(PARAM_FOLDER, csv_name)

    load_csv_into_config(csv_path)

    setattr(Config, "TRAIN", True)
    setattr(Config, "RENDERS", False)

    print("\n🚀 Iniciando treinamento:", csv_name)
    print("==============================================")

    os.system("python3 src/rl_game.py")

    if not os.path.exists(RAW_MODEL):
        print(f"❌ ERRO: {RAW_MODEL} não foi encontrado! Treino falhou.")
        return

    new_model_name = f"model_{csv_name.replace('.csv', '')}.pkl"
    new_model_path = os.path.join(MODEL_FOLDER, new_model_name)

    shutil.move(RAW_MODEL, new_model_path)

    print(f"✅ Modelo salvo como {new_model_path}\n")


if __name__ == "__main__":
    csv_files = sorted([f for f in os.listdir(PARAM_FOLDER) if f.endswith(".csv")])

    if not csv_files:
        print("❌ Nenhum CSV encontrado em ./seeds/")
        exit()

    print(f"🔎 {len(csv_files)} CSVs encontrados:\n", csv_files)

    for csv_file in csv_files:
        run_training(csv_file)

    print("\n🎉 Finalizado! Todos os modelos foram treinados e movidos para ./models/")
