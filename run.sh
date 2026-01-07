#!/bin/bash

# AI Duet FastAPI - Startup Script
# This script helps set up and run the AI Duet server

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║           AI Duet FastAPI - Startup Script           ║"
    echo "╚══════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_step() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

check_dependencies() {
    print_step "Checking dependencies..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 is not installed"
        exit 1
    fi
    
    # Check pip
    if ! command -v pip3 &> /dev/null; then
        print_warning "pip3 not found, installing..."
        sudo apt-get update && sudo apt-get install -y python3-pip
    fi
    
    # Check FFmpeg (required for audio processing)
    if ! command -v ffmpeg &> /dev/null; then
        print_warning "FFmpeg not found, installing..."
        sudo apt-get update && sudo apt-get install -y ffmpeg
    fi
    
    print_step "Dependencies check completed"
}

setup_environment() {
    print_step "Setting up environment..."
    
    # Check if .env exists
    if [ ! -f ".env" ]; then
        print_warning ".env file not found"
        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_step "Created .env from example"
            echo -e "${YELLOW}Please edit .env file with your API keys${NC}"
        else
            print_error ".env.example not found"
            exit 1
        fi
    fi
    
    # Check if API keys are set
    if grep -q "your_fireworks_api_key_here" .env || grep -q "your_deepinfra_api_key_here" .env; then
        print_warning "API keys not configured in .env file"
        echo -e "${YELLOW}Please update .env with your actual API keys:${NC}"
        echo "1. FIREWORKS_API_KEY from https://fireworks.ai"
        echo "2. DEEPINFRA_API_KEY from https://deepinfra.com"
        read -p "Continue anyway? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    print_step "Environment setup completed"
}

install_requirements() {
    print_step "Installing Python requirements..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_step "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_step "Requirements installed successfully"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
}

download_models() {
    print_step "Checking for Whisper models..."
    
    # Check if faster-whisper model needs to be downloaded
    # The model will be automatically downloaded on first run
    print_step "Whisper models will be downloaded automatically on first run"
}

start_server() {
    print_step "Starting AI Duet server..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Get host and port from .env or use defaults
    HOST=${HOST:-0.0.0.0}
    PORT=${PORT:-8000}
    
    # Check if .env sets different values
    if [ -f ".env" ]; then
        source .env
    fi
    
    echo -e "${GREEN}"
    echo "┌──────────────────────────────────────────────────────┐"
    echo "│   AI Duet Server Starting...                         │"
    echo "│   Host: ${HOST}                                      │"
    echo "│   Port: ${PORT}                                      │"
    echo "│                                                      │"
    echo "│   WebSocket: ws://${HOST}:${PORT}/ws/{session_id}    │"
    echo "│   Test Client: http://${HOST}:${PORT}/client         │"
    echo "└──────────────────────────────────────────────────────┘"
    echo -e "${NC}"
    
    # Start the server
    uvicorn main:app --host "$HOST" --port "$PORT" --reload
}

run_tests() {
    print_step "Running quick tests..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    if [ -f "test_client.py" ]; then
        echo -e "${YELLOW}Running test client (requires server to be running)${NC}"
        echo "Run this in another terminal:"
        echo "  cd $(pwd) && ./test_client.py"
    fi
    
    print_step "Tests completed"
}

show_help() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  setup     - Install dependencies and set up environment"
    echo "  install   - Install Python packages only"
    echo "  run       - Start the server (default)"
    echo "  test      - Run tests"
    echo "  all       - Full setup and start"
    echo "  help      - Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 setup    # Initial setup"
    echo "  $0 run      # Start server"
    echo "  $0 all      # Do everything and start"
}

main() {
    print_header
    
    case "${1:-run}" in
        "setup")
            check_dependencies
            setup_environment
            install_requirements
            download_models
            ;;
        "install")
            install_requirements
            ;;
        "run")
            setup_environment
            install_requirements
            start_server
            ;;
        "test")
            setup_environment
            install_requirements
            run_tests
            ;;
        "all")
            check_dependencies
            setup_environment
            install_requirements
            download_models
            start_server
            ;;
        "help")
            show_help
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"