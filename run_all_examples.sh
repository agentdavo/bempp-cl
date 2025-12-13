#!/bin/bash
#
# Run All bempp_audio Examples
#
# This script runs the three main examples to demonstrate bempp_audio capabilities:
#   1. piston_minimal.py - Ultra-minimal piston on infinite baffle
#   2. waveguide_infinite_baffle.py - Exponential horn on infinite baffle
#   3. waveguide_on_box.py - Waveguide on finite box enclosure (unified mesh)
#   4. compression_driver_network_minimal.py - Lumped compression-driver network (fast, no BEM)
#
# Usage:
#   ./run_all_examples.sh              # Run all three examples sequentially
#   ./run_all_examples.sh --parallel   # Run all three in parallel (faster)
#   ./run_all_examples.sh --help       # Show this help
#

set -e  # Exit on error

# Use numba backend by default (more reliable than OpenCL on many systems)
export BEMPP_DEVICE_INTERFACE="${BEMPP_DEVICE_INTERFACE:-numba}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EXAMPLES_DIR="$SCRIPT_DIR/examples/bempp_audio"

# Check if virtual environment exists
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo -e "${RED}Error: Virtual environment not found at $SCRIPT_DIR/venv${NC}"
    echo "Please create it with: python -m venv venv && source venv/bin/activate && pip install -e ."
    exit 1
fi

# Parse arguments
PARALLEL=0
if [ "$1" = "--parallel" ]; then
    PARALLEL=1
    echo -e "${YELLOW}Running examples in parallel mode${NC}"
elif [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    head -n 15 "$0" | tail -n 13
    exit 0
fi

# Function to run a single example
run_example() {
    local name=$1
    local script=$2
    local duration_est=$3
    
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Running: $name${NC}"
    echo -e "${BLUE}Estimated time: $duration_est${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    
    START_TIME=$(date +%s)
    
    # Activate venv and run
    cd "$SCRIPT_DIR"
    source venv/bin/activate
    python "$EXAMPLES_DIR/$script"
    EXIT_CODE=$?
    
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    MINUTES=$((ELAPSED / 60))
    SECONDS=$((ELAPSED % 60))
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✓ $name completed successfully in ${MINUTES}m ${SECONDS}s${NC}"
        return 0
    else
        echo ""
        echo -e "${RED}✗ $name failed with exit code $EXIT_CODE${NC}"
        return $EXIT_CODE
    fi
}

# Main execution
echo ""
echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}bempp_audio Example Suite${NC}"
echo -e "${GREEN}======================================================================${NC}"
echo ""
echo "Running three examples demonstrating BEM acoustic simulation:"
echo "  1. Piston on infinite baffle (minimal example)"
echo "  2. Exponential waveguide on infinite baffle"
echo "  3. Waveguide on finite box enclosure (unified mesh)"
echo "  4. Compression driver network (lumped, fast)"
echo ""

TOTAL_START=$(date +%s)

if [ $PARALLEL -eq 1 ]; then
    # Run in parallel
    echo -e "${YELLOW}Starting all examples in parallel...${NC}"
    echo ""
    
    # Run in background with output redirection
    (run_example "1. Piston Minimal" "piston_minimal.py" "~2-5 min" > /tmp/bempp_ex1.log 2>&1 && echo -e "${GREEN}✓ Example 1 complete${NC}" || echo -e "${RED}✗ Example 1 failed${NC}") &
    PID1=$!
    
    (run_example "2. Waveguide on Infinite Baffle" "waveguide_infinite_baffle.py" "~10-20 min" > /tmp/bempp_ex2.log 2>&1 && echo -e "${GREEN}✓ Example 2 complete${NC}" || echo -e "${RED}✗ Example 2 failed${NC}") &
    PID2=$!
    
    (run_example "3. Waveguide on Box" "waveguide_on_box.py" "~15-30 min" > /tmp/bempp_ex3.log 2>&1 && echo -e "${GREEN}✓ Example 3 complete${NC}" || echo -e "${RED}✗ Example 3 failed${NC}") &
    PID3=$!

    (run_example "4. Compression Driver Network" "compression_driver_network_minimal.py" "~1-5 sec" > /tmp/bempp_ex4.log 2>&1 && echo -e "${GREEN}✓ Example 4 complete${NC}" || echo -e "${RED}✗ Example 4 failed${NC}") &
    PID4=$!
    
    echo "Waiting for all examples to complete..."
    echo "  PID $PID1: piston_minimal.py (log: /tmp/bempp_ex1.log)"
    echo "  PID $PID2: waveguide_infinite_baffle.py (log: /tmp/bempp_ex2.log)"
    echo "  PID $PID3: waveguide_on_box.py (log: /tmp/bempp_ex3.log)"
    echo "  PID $PID4: compression_driver_network_minimal.py (log: /tmp/bempp_ex4.log)"
    echo ""
    echo "You can monitor progress with:"
    echo "  tail -f /tmp/bempp_ex1.log"
    echo "  tail -f /tmp/bempp_ex2.log"
    echo "  tail -f /tmp/bempp_ex3.log"
    echo "  tail -f /tmp/bempp_ex4.log"
    echo ""
    
    # Wait for all
    wait $PID1
    EXIT1=$?
    wait $PID2
    EXIT2=$?
    wait $PID3
    EXIT3=$?
    wait $PID4
    EXIT4=$?
    
    # Show logs
    echo ""
    echo -e "${BLUE}======== Example 1 Output ========${NC}"
    tail -20 /tmp/bempp_ex1.log
    echo ""
    echo -e "${BLUE}======== Example 2 Output ========${NC}"
    tail -20 /tmp/bempp_ex2.log
    echo ""
    echo -e "${BLUE}======== Example 3 Output ========${NC}"
    tail -20 /tmp/bempp_ex3.log
    echo ""
    echo -e "${BLUE}======== Example 4 Output ========${NC}"
    tail -20 /tmp/bempp_ex4.log
    
    # Check results
    FAILED=0
    [ $EXIT1 -ne 0 ] && FAILED=$((FAILED+1))
    [ $EXIT2 -ne 0 ] && FAILED=$((FAILED+1))
    [ $EXIT3 -ne 0 ] && FAILED=$((FAILED+1))
    [ $EXIT4 -ne 0 ] && FAILED=$((FAILED+1))
    
else
    # Run sequentially
    FAILED=0
    
    run_example "1. Piston Minimal" "piston_minimal.py" "~2-5 min"
    [ $? -ne 0 ] && FAILED=$((FAILED+1))
    
    run_example "2. Waveguide on Infinite Baffle" "waveguide_infinite_baffle.py" "~10-20 min"
    [ $? -ne 0 ] && FAILED=$((FAILED+1))
    
    run_example "3. Waveguide on Box" "waveguide_on_box.py" "~15-30 min"
    [ $? -ne 0 ] && FAILED=$((FAILED+1))

    run_example "4. Compression Driver Network" "compression_driver_network_minimal.py" "~1-5 sec"
    [ $? -ne 0 ] && FAILED=$((FAILED+1))
fi

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))
TOTAL_MINUTES=$((TOTAL_ELAPSED / 60))
TOTAL_SECONDS=$((TOTAL_ELAPSED % 60))

# Summary
echo ""
echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}Summary${NC}"
echo -e "${GREEN}======================================================================${NC}"
echo ""
echo "Total time: ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All examples completed successfully!${NC}"
    echo ""
    echo "Generated output files:"
    echo "  - piston_ultra_minimal_*.png"
    echo "  - waveguide_infinite_baffle_*.png"
    echo "  - waveguide_infinite_baffle_*.html"
    echo "  - waveguide_on_box_*.png"
    echo "  - waveguide_on_box_*.html"
    echo ""
    echo "View the HTML files in a browser for interactive 3D visualizations!"
    exit 0
else
    echo -e "${RED}✗ $FAILED example(s) failed${NC}"
    echo ""
    echo "Check the output above for error messages."
    if [ $PARALLEL -eq 1 ]; then
        echo "Full logs available at:"
        echo "  /tmp/bempp_ex1.log"
        echo "  /tmp/bempp_ex2.log"
        echo "  /tmp/bempp_ex3.log"
    fi
    exit 1
fi
