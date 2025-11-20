create-env:
	python3 -m venv venv

start-env:
	source venv/bin/activate

requirements-install:
	pip3 install -r requirements.txt

run-rl:
	python3 src/rl_game.py

run-all-train:
	python3 src/batch_train.py

run-all-test:
	python3 src/batch_test.py