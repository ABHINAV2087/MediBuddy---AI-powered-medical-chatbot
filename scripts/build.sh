# #!/bin/bash
# set -o errexit

# echo "Running setup..."
# ./scripts/setup.sh

# echo "Installing Python dependencies..."
# pip install -r requirements.txt

# echo "Checking for vectorstore files..."
# VECTORSTORE_DIR="vectorstore/db_faiss"
# if [ -f "$VECTORSTORE_DIR/index.faiss" ] && [ -f "$VECTORSTORE_DIR/index.pkl" ]; then
#     echo "Vectorstore files exist, skipping creation..."
# else
#     echo "Creating vectorstore..."
    
#     # Check if PDFs exist
#     if [ -z "$(ls -A data/)" ]; then
#         echo "Error: No PDF files found in data/ directory"
#         exit 1
#     fi
    
#     python create_vectorstore.py
    
#     # Verify creation was successful
#     if [ ! -f "$VECTORSTORE_DIR/index.faiss" ] || [ ! -f "$VECTORSTORE_DIR/index.pkl" ]; then
#         echo "Error: Vectorstore creation failed"
#         exit 1
#     fi
# fi

# echo "Build completed successfully."