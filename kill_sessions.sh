#!/bin/bash

# Kill All Backspace Sessions Script
echo "🔥 Killing all Backspace application sessions..."

# Kill all uvicorn processes
echo "Killing uvicorn processes..."
pkill -f uvicorn 2>/dev/null || echo "No uvicorn processes found"

# Kill all Python processes running the app
echo "Killing Python app processes..."
pkill -f "python.*app" 2>/dev/null || echo "No Python app processes found"

# Kill processes by port (common ports: 8000, 8001)
echo "Killing processes on ports 8000 and 8001..."
if command -v lsof &> /dev/null; then
    lsof -ti:8000 | xargs kill -9 2>/dev/null || echo "No processes on port 8000"
    lsof -ti:8001 | xargs kill -9 2>/dev/null || echo "No processes on port 8001"
else
    # Alternative method using netstat (if available)
    if command -v netstat &> /dev/null; then
        netstat -tulnp 2>/dev/null | grep :8000 | awk '{print $7}' | cut -d/ -f1 | xargs kill -9 2>/dev/null || echo "No processes on port 8000"
        netstat -tulnp 2>/dev/null | grep :8001 | awk '{print $7}' | cut -d/ -f1 | xargs kill -9 2>/dev/null || echo "No processes on port 8001"
    else
        echo "Neither lsof nor netstat available - skipping port-based cleanup"
    fi
fi

# Kill all Docker containers (if any)
echo "Killing Docker containers..."
if command -v docker &> /dev/null; then
    docker kill $(docker ps -q) 2>/dev/null || echo "No Docker containers to kill"
else
    echo "Docker not available - skipping Docker cleanup"
fi

# Kill all processes containing 'backspace' in the name
echo "Killing processes with 'backspace' in name..."
pkill -f -i backspace 2>/dev/null || echo "No backspace processes found"

echo ""
echo "✅ Session cleanup complete!"
echo ""
echo "Remaining processes check:"
ps aux | grep -E "(uvicorn|python.*app|backspace)" | grep -v grep || echo "✅ No related processes found" 