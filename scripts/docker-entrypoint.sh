#!/bin/bash
set -e

echo "Neo4j is up - executing command"

if [ "${POPULATE_DATABASE_NIPS2025}" = "true" ]; then
    echo "Importing NeurIPS 2025 papers..."
    bash scripts/import_neurips2025_kg.sh
else
    echo "Skipping NeurIPS 2025 paper import (POPULATE_DATABASE_NIPS2025 is not set to 'true')"
fi

echo "Starting main application..."
exec "$@"