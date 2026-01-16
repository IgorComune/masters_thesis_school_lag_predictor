#!/bin/bash

echo "🔧 Iniciando Prefect..."
prefect server start &

echo "🚀 Prefect UI disponível em http://localhost:4200"
