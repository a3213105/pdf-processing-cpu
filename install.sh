#!/bin/bash

set -e  # Exit immediately on error

echo "========================================"
echo "  PDF Processing Skill - Install Script"
echo "========================================"
echo ""

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SKILL_DIR"

# Check Python version
echo "1. Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Detected Python version: $python_version"

# Check whether Python version is in 3.10-3.13 range
python_major=$(echo $python_version | cut -d. -f1)
python_minor=$(echo $python_version | cut -d. -f2)

if [ "$python_major" -lt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -lt 10 ]); then
    echo "   ❌ Error: Python version is too low. Python 3.10+ is required"
    echo "   Current version: $python_version"
    exit 1
fi

if [ "$python_major" -gt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -gt 13 ]); then
    echo "   ⚠️  Warning: Python version is higher than recommended. Use Python 3.10-3.13"
fi

echo "   ✓ Python version is supported"
echo ""

# Check pip
echo "2. Checking pip..."
if ! command -v pip3 &> /dev/null; then
    echo "   ❌ Error: pip3 not found"
    exit 1
fi
echo "   ✓ pip3 is installed"
echo ""

# Upgrade pip
echo "3. Upgrading pip..."
python3 -m pip install --upgrade pip
echo "   ✓ pip upgraded"
echo ""

# Install
echo "4. Installing dependencies..."
echo "   This may take a few minutes..."
pip install -U -r requirements.txt
echo "   ✓ Dependencies installed"
echo ""

echo "5. Downloading model files..."
echo "   This may take a few minutes..."
modelscope download --model "a3213105/pdf-processing-cpu" --local_dir "$SKILL_DIR/models"
echo "   ✓ Model files downloaded"
echo ""

# Verify installation
echo "6. Verifying installation..."
python script/main.py -v > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✓ pdf-processing-cpu command runs successfully"
else
    echo "   ❌ Warning: pdf-processing-cpu command failed"
fi
echo ""

echo "========================================"
echo "  ✅ Installation completed!"
echo "========================================"
echo ""
echo "PDF Processing Skill is ready to use."
echo ""
echo "Test command:"
echo "  python $SKILL_DIR/script/main.py -i /path/to/document.pdf -o output_dir"
echo ""
echo "For more details, see:"
echo "  $SKILL_DIR/README.md"
echo ""
