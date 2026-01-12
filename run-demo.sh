#!/bin/bash
# AI Duet Messenger Demo Launcher

echo "🚀 AI Duet Messenger Demo Launcher"
echo "=================================="
echo ""

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: Please run this script from the ai-duet-fastapi root directory"
    exit 1
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found"
    echo "Creating .env from example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✅ Created .env file. Please edit it with your API keys."
        echo ""
    else
        echo "❌ .env.example not found. Please create .env manually."
        exit 1
    fi
fi

# Check Python dependencies
echo "📦 Checking Python dependencies..."
python3 -c "import fastapi, uvicorn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Missing Python dependencies. Installing..."
    pip install -r requirements.txt
fi

# Check if demo exists
if [ ! -d "demo" ]; then
    echo "❌ Demo directory not found"
    exit 1
fi

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

# Check if demo dependencies are installed
if [ ! -d "demo/node_modules" ]; then
    echo "📦 Installing demo dependencies..."
    cd demo
    npm install
    cd ..
fi

# Ask user which mode to run
echo ""
echo "Choose how to run the demo:"
echo "1) Development mode (separate servers, hot reload)"
echo "2) Production mode (integrated, build first)"
echo "3) Just build the demo (no server)"
echo ""
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo "🔧 Starting in development mode..."
        echo "Backend will run on http://localhost:8000"
        echo "Frontend will run on http://localhost:3000"
        echo ""
        echo "Press Ctrl+C to stop both servers"
        echo ""
        
        # Start backend in background
        echo "Starting backend..."
        uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
        BACKEND_PID=$!
        
        # Wait a bit for backend to start
        sleep 3
        
        # Start frontend
        echo "Starting frontend..."
        cd demo
        npm run dev &
        FRONTEND_PID=$!
        cd ..
        
        # Wait for user to stop
        echo ""
        echo "✅ Both servers running!"
        echo "Open http://localhost:3000 in your browser"
        echo ""
        
        # Trap Ctrl+C and kill both processes
        trap "echo ''; echo 'Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT
        wait
        ;;
    2)
        echo ""
        echo "🏗️  Building demo for production..."
        cd demo
        npm run build
        cd ..
        
        echo ""
        echo "🚀 Starting backend server..."
        echo "Demo will be available at http://localhost:8000/demo"
        echo "Main app at http://localhost:8000"
        echo ""
        echo "Press Ctrl+C to stop"
        echo ""
        
        uvicorn main:app --host 0.0.0.0 --port 8000
        ;;
    3)
        echo ""
        echo "🏗️  Building demo..."
        cd demo
        npm run build
        cd ..
        
        echo ""
        echo "✅ Build complete!"
        echo "Built files are in: demo/dist/"
        echo ""
        echo "To serve the demo:"
        echo "  uvicorn main:app --host 0.0.0.0 --port 8000"
        echo "  Then open http://localhost:8000/demo"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac
