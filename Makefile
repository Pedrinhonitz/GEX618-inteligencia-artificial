create-env:
	python3 -m venv venv

start-env:
	source venv/bin/activate

requirements-install:
	pip3 install -r requirements.txt

run:
	python3 src/RL-game.py