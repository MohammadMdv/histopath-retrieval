.PHONY: download-data build-index run eval stop build help

COMPOSE = docker compose
RUN_IN = $(COMPOSE) run --rm retrieval

help:
	@echo "Targets:"
	@echo "  download-data   Download NCT-CRC-HE-100K + CRC-VAL-HE-7K from Zenodo"
	@echo "  build-index     Embed index images and build FAISS index"
	@echo "  run             Start the web app (http://localhost:8000)"
	@echo "  eval            Run offline evaluation with bootstrap CI"
	@echo "  build           (Re-)build the Docker image"
	@echo "  stop            Stop and remove containers"

build:
	$(COMPOSE) build

download-data:
	$(RUN_IN) python scripts/download_data.py \
		--data-dir /data \
		--subsample $${SUBSAMPLE_PER_CLASS:-1000}

build-index:
	$(RUN_IN) python scripts/build_index.py --config /app/config.yaml

run:
	$(COMPOSE) up

eval:
	$(RUN_IN) python scripts/eval.py --config /app/config.yaml

stop:
	$(COMPOSE) down
