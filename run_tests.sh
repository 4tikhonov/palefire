#!/bin/bash
# Test runner script for Pale Fire

set -e

echo "🧪 Pale Fire Test Suite"
echo "======================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${YELLOW}⚠️  pytest not found. Installing test dependencies...${NC}"
    pip install -r requirements.txt
fi

# Parse arguments
MODE=${1:-all}

case $MODE in
    all)
        echo -e "${BLUE}Running all tests...${NC}"
        pytest tests/ -v
        ;;
    
    unit)
        echo -e "${BLUE}Running unit tests...${NC}"
        pytest tests/ -v -m unit
        ;;
    
    integration)
        echo -e "${BLUE}Running integration tests...${NC}"
        pytest tests/ -v -m integration
        ;;
    
    coverage)
        echo -e "${BLUE}Running tests with coverage...${NC}"
        pytest tests/ -v --cov=. --cov-report=html --cov-report=term
        echo ""
        echo -e "${GREEN}✅ Coverage report generated in htmlcov/index.html${NC}"
        ;;
    
    fast)
        echo -e "${BLUE}Running fast tests (excluding slow)...${NC}"
        pytest tests/ -v -m "not slow"
        ;;
    
    config)
        echo -e "${BLUE}Testing configuration module...${NC}"
        pytest tests/test_config.py -v
        ;;
    
    core)
        echo -e "${BLUE}Testing PaleFireCore module...${NC}"
        pytest tests/test_palefire_core.py -v
        ;;
    
    search)
        echo -e "${BLUE}Testing search functions...${NC}"
        pytest tests/test_search_functions.py -v
        ;;
    
    api)
        echo -e "${BLUE}Testing API...${NC}"
        pytest tests/test_api.py -v
        ;;
    
    watch)
        echo -e "${BLUE}Running tests in watch mode...${NC}"
        if command -v pytest-watch &> /dev/null; then
            pytest-watch tests/
        else
            echo -e "${YELLOW}⚠️  pytest-watch not installed. Install with: pip install pytest-watch${NC}"
            exit 1
        fi
        ;;
    
    help|--help|-h)
        echo "Usage: ./run_tests.sh [MODE]"
        echo ""
        echo "Modes:"
        echo "  all         - Run all tests (default)"
        echo "  unit        - Run only unit tests"
        echo "  integration - Run only integration tests"
        echo "  coverage    - Run tests with coverage report"
        echo "  fast        - Run fast tests (exclude slow)"
        echo "  config      - Test config module only"
        echo "  core        - Test PaleFireCore module only"
        echo "  search      - Test search functions only"
        echo "  api         - Test API only"
        echo "  watch       - Run tests in watch mode"
        echo "  help        - Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./run_tests.sh              # Run all tests"
        echo "  ./run_tests.sh coverage     # Run with coverage"
        echo "  ./run_tests.sh config       # Test config only"
        exit 0
        ;;
    
    *)
        echo -e "${YELLOW}Unknown mode: $MODE${NC}"
        echo "Run './run_tests.sh help' for usage information"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}✅ Tests completed!${NC}"

