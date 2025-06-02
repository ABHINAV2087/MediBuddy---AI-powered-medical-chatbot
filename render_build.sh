#!/bin/bash

echo "🚀 Starting Render build process..."

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data
mkdir -p vectorstore
mkdir -p templates

# Build vectorstore if data files exist
echo "🔨 Checking for data files..."
if [ "$(ls -A data/ 2>/dev/null)" ]; then
    echo "📚 Found data files, building vectorstore..."
    python build_vectorstore.py
    if [ $? -eq 0 ]; then
        echo "✅ Vectorstore built successfully"
    else
        echo "❌ Failed to build vectorstore"
        exit 1
    fi
else
    echo "⚠️ No data files found in data/ directory"
    echo "⚠️ Vectorstore will be built on first request"
fi

echo "✅ Build process completed"