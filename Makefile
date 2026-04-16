.PHONY: help start stop restart status logs health build ingest clean

help:
	@echo "Available commands:"
	@echo "  make start        Start all services"
	@echo "  make stop         Stop all services"
	@echo "  make restart      Restart all services"
	@echo "  make status       Show service status"
	@echo "  make logs         Show service logs"
	@echo "  make health       Check all services health"
	@echo "  make build        Build and start all services"
	@echo "  make api          Start the FastAPI server locally"
	@echo "  make clean        Stop and remove all volumes"

start:
	docker compose up -d

stop:
	docker compose down

restart:
	docker compose restart

status:
	docker compose ps

logs:
	docker compose logs --tail=50 -f

build:
	docker compose up --build -d

api:
	uvicorn api:app --reload --port 8000

health:
	@echo "=== API ==="
	@curl -sf http://localhost:8000/health && echo " ✓" || echo " ✗ not running"
	@echo "=== OpenSearch ==="
	@curl -sf http://localhost:9202/_cluster/health | python -m json.tool | grep status || echo " ✗ not running"
	@echo "=== PostgreSQL ==="
	@docker exec rag-postgres pg_isready -U rag_user -d rag_db 2>/dev/null && echo " ✓" || echo " ✗ not running"
	@echo "=== Redis ==="
	@docker exec rag-redis redis-cli ping 2>/dev/null || echo " ✗ not running"
	@echo "=== Ollama ==="
	@curl -sf http://localhost:11434/api/tags | python -m json.tool | grep name || echo " ✗ not running"

clean:
	docker compose down -v
	@echo "All containers and volumes removed."
