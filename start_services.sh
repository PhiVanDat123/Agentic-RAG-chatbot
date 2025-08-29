#!/bin/bash
# start_services.sh

# Start FastAPI backend in background
python main.py backend &

# Wait a moment for backend to start
sleep 5

# Start Streamlit frontend
streamlit run main.py --server.port 8501 --server.address 0.0.0.0
