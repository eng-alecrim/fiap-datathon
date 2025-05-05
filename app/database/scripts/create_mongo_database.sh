#!/bin/bash

set -e
set -o pipefail
set -u

# Cria .venv, instala dependências e ativa
uv sync
source ./.venv/bin/activate

# Define a pasta onde estão os scripts
SCRIPT_DIR="./data_engineering"

# Lista, ordena e executa cada script .py dentro da pasta
for script in $(ls "$SCRIPT_DIR"/*.py | sort); do
  echo "Executando: $script"
  python "$script"
  echo "Concluído: $script"
  echo "---------------------------"
done

echo "✅ Todos os scripts foram executados com sucesso!"
