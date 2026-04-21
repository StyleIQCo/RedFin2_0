.PHONY: install data train test serve docker clean drift-demo

install:
	pip install -r requirements.txt

data:
	python -m data.generate

train:
	python -m src.avm.train
	python -m src.recommender.train

test:
	pytest tests/ -v

serve:
	uvicorn src.serving.app:app --reload --host 0.0.0.0 --port 8000

docker:
	docker build -t redfin-ml-mvp:latest .

docker-up:
	docker-compose up --build

drift-demo:
	python -m scripts.simulate_drift

clean:
	rm -rf __pycache__ .pytest_cache data/*.parquet artifacts/ registry/models/
	find . -name "__pycache__" -type d -exec rm -rf {} +

all: install data train test
