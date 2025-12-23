#!/bin/bash

# Navigate to the directory where this script is located
cd "$(dirname "$0")"

#  Run the streamlit app
echo "Starting BIST100 AI Dashboard..."
python3 -m streamlit run app.py
