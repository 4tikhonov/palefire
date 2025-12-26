#!/bin/bash
# Quick test verification script

echo "🔍 Verifying Pale Fire Test Suite"
echo "=================================="
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "❌ pytest not found. Install with: pip install -r requirements.txt"
    exit 1
fi

echo "✅ pytest found"
echo ""

# Count test files
test_files=$(find tests -name "test_*.py" | wc -l | tr -d ' ')
echo "📁 Test files: $test_files"

# Collect tests
echo ""
echo "📊 Collecting tests..."
pytest --collect-only -q tests/ 2>&1 | tail -1

# Run tests
echo ""
echo "🧪 Running tests..."
pytest tests/ -v --tb=no -q

# Get exit code
exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "✅ All tests passed!"
else
    echo "❌ Some tests failed. Run './run_tests.sh' for details."
fi

exit $exit_code

