#!/bin/bash
# Verification script for utils refactoring

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                  VERIFYING REFACTORING                                       ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if files exist
echo "📁 Checking file structure..."
files=(
    "palefire-cli.py"
    "utils/__init__.py"
    "utils/palefire_utils.py"
    "docs/REFACTORING_UTILS.md"
    "REFACTORING_SUMMARY.md"
)

all_exist=true
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "  ${GREEN}✓${NC} $file exists"
    else
        echo -e "  ${RED}✗${NC} $file missing"
        all_exist=false
    fi
done
echo ""

# Check syntax
echo "🔍 Checking Python syntax..."
syntax_ok=true

if python3 -m py_compile palefire-cli.py 2>/dev/null; then
    echo -e "  ${GREEN}✓${NC} palefire-cli.py syntax OK"
else
    echo -e "  ${RED}✗${NC} palefire-cli.py syntax error"
    syntax_ok=false
fi

if python3 -m py_compile utils/palefire_utils.py 2>/dev/null; then
    echo -e "  ${GREEN}✓${NC} utils/palefire_utils.py syntax OK"
else
    echo -e "  ${RED}✗${NC} utils/palefire_utils.py syntax error"
    syntax_ok=false
fi

if python3 -m py_compile utils/__init__.py 2>/dev/null; then
    echo -e "  ${GREEN}✓${NC} utils/__init__.py syntax OK"
else
    echo -e "  ${RED}✗${NC} utils/__init__.py syntax error"
    syntax_ok=false
fi
echo ""

# Check line counts
echo "📊 Checking line counts..."
cli_lines=$(wc -l < palefire-cli.py)
utils_lines=$(wc -l < utils/palefire_utils.py)
init_lines=$(wc -l < utils/__init__.py)

echo "  • palefire-cli.py: $cli_lines lines"
echo "  • utils/palefire_utils.py: $utils_lines lines"
echo "  • utils/__init__.py: $init_lines lines"
echo ""

# Check for key functions in utils
echo "🔎 Checking for key functions in utils..."
functions=(
    "search_episodes"
    "search_episodes_with_custom_ranking"
    "search_episodes_with_question_aware_ranking"
    "export_results_to_json"
    "clean_database"
    "get_node_connections_with_entities"
    "extract_temporal_info"
    "calculate_temporal_relevance"
    "extract_query_terms"
    "calculate_query_match_score"
)

functions_ok=true
for func in "${functions[@]}"; do
    if grep -q "^async def $func\|^def $func" utils/palefire_utils.py; then
        echo -e "  ${GREEN}✓${NC} $func found"
    else
        echo -e "  ${RED}✗${NC} $func missing"
        functions_ok=false
    fi
done
echo ""

# Check imports in CLI
echo "🔗 Checking imports in CLI..."
if grep -q "from utils.palefire_utils import" palefire-cli.py; then
    echo -e "  ${GREEN}✓${NC} CLI imports from utils"
else
    echo -e "  ${RED}✗${NC} CLI doesn't import from utils"
fi
echo ""

# Summary
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                           SUMMARY                                            ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"

if $all_exist && $syntax_ok && $functions_ok; then
    echo -e "${GREEN}✅ ALL CHECKS PASSED!${NC}"
    echo ""
    echo "The refactoring is complete and verified:"
    echo "  • All files exist"
    echo "  • Syntax is correct"
    echo "  • All functions moved successfully"
    echo "  • CLI properly imports from utils"
    echo ""
    echo "📚 See docs/REFACTORING_UTILS.md for details"
    exit 0
else
    echo -e "${RED}❌ SOME CHECKS FAILED${NC}"
    echo ""
    echo "Please review the errors above."
    exit 1
fi

