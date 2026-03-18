#!/bin/bash

set -e  # 遇到错误立即退出

echo "========================================"
echo "  PDF Processing Skill - 安装脚本"
echo "========================================"
echo ""

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SKILL_DIR"

# 检查 Python 版本
echo "1. 检查 Python 版本..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   检测到 Python 版本: $python_version"

# 检查 Python 版本是否在 3.10-3.13 范围内
python_major=$(echo $python_version | cut -d. -f1)
python_minor=$(echo $python_version | cut -d. -f2)

if [ "$python_major" -lt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -lt 10 ]); then
    echo "   ❌ 错误: Python 版本过低。需要 Python 3.10 或更高版本"
    echo "   当前版本: $python_version"
    exit 1
fi

if [ "$python_major" -gt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -gt 13 ]); then
    echo "   ⚠️  警告: Python 版本过高。推荐使用 Python 3.10-3.13"
fi

echo "   ✓ Python 版本符合要求"
echo ""

# 检查 pip
echo "2. 检查 pip..."
if ! command -v pip3 &> /dev/null; then
    echo "   ❌ 错误: 未找到 pip3"
    exit 1
fi
echo "   ✓ pip3 已安装"
echo ""

# 更新 pip
echo "3. 更新 pip..."
python3 -m pip install --upgrade pip
echo "   ✓ pip 已更新"
echo ""

# 安装
echo "5. 下载依赖模块..."
echo "   这可能需要几分钟时间，请耐心等待..."
pip install -U -r requirements.txt
echo "   ✓ 依赖模块下载完成"
echo ""

echo "5. 下载模型文件..."
echo "   这可能需要几分钟时间，请耐心等待..."
modelscope download --model "a3213105/pdf-processing-cpu" --local_dir "$SKILL_DIR/models"
echo "   ✓ 模型文件下载完成"
echo ""

# 验证安装
echo "6. 验证安装..."
python script/main.py -v > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✓ pdf-processing-cpu 命令运行正常"
else
    echo "   ❌ 警告: pdf-processing-cpu 命令运行异常"
fi
echo ""

echo "========================================"
echo "  ✅ 安装完成！"
echo "========================================"
echo ""
echo "现在可以使用 PDF Processing Skill 了！"
echo ""
echo "测试命令:"
echo "  python $SKILL_DIR/pdf-processing-cpu/script/main.py -i /path/to/document.pdf -o output_dir"
echo ""
echo "更多信息请查看:"
echo "  $SKILL_DIR/pdf-processing-cpu/README.md"
echo ""
