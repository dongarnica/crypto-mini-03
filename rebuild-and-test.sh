#!/bin/bash

echo "🔄 Rebuilding Docker image with historical data..."

# Build the image
docker build -t crypto-trading-engine:latest .

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully!"
    
    echo "🚀 Running container to test historical data loading..."
    
    # Run the container (stop after 30 seconds for testing)
    timeout 30s docker run --rm crypto-trading-engine:latest
    
    echo "✅ Test completed! Check the logs above to see if historical data was found."
else
    echo "❌ Docker build failed!"
    exit 1
fi
