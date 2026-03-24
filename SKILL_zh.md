---
name: pdf-processing-cpu (Intel® Xeon® Optimized)
description: 基于 Intel® Xeon® AMX 指令集优化加速的本地化 MinerU PDF 文档解析工具，支持将 PDF 转换为 Markdown。
---

## 工具列表

### 1. pdf-processing-cpu

将 PDF 文档转换为 Markdown 格式，保留文档结构、公式、表格和图片。

**描述**：使用 MinerU 解析 PDF 文档并输出为 Markdown，支持 OCR、公式识别、表格提取等能力。

**参数**：
- `input` (string, required)：PDF 文件绝对路径
- `output_dir` (string, required)：输出目录绝对路径

**返回值**：
```json
#Success
{
  "success": true,
  "message" : "processing info", 
  "outputs":
  [
    {
      "input_name" : "/path/to/input",
      "output_path": "/path/to/output"
    },
    {
      "input_name" : "/path/to/input",
      "output_path": "/path/to/output"
    }
  ]
}

#Failed
{
  "success": false,
  "message" : "error info",
  "outputs": []
}
```

**示例**：
```bash
# 输出 markdown
python $HOME/.openclaw/workspace/skills/pdf-processing-cpu/script/main.py -i /path/to/document.pdf -o output_dir

#or dir for PDF
python $HOME/.openclaw/workspace/skills/pdf-processing-cpu/script/main.py -i /path/to/documents_dir -o output_dir

```

---

## 安装说明

### 1. 安装依赖
```bash
pip install -r $HOME/.openclaw/workspace/skills/pdf-processing-cpu/requirements.txt
```

### 2. 下载模型
```bash
modelscope download --model a3213105/pdf-processing-cpu --local_dir $HOME/.openclaw/workspace/skills/pdf-processing-cpu/models
```

### 2. 验证安装
```bash
python $HOME/.openclaw/workspace/skills/pdf-processing-cpu/script/main.py -v
```

### 3. 系统要求

- **Python 版本**：3.10-3.13
- **操作系统**：Linux
- **内存**：
  - 最低 2GB，推荐 16GB+
- **磁盘空间**：最低 4GB（推荐 SSD）
- **CPU**：Intel® Xeon® 4代以上CPU，带AMX可以提供更优性能

## 使用场景

1. **学术论文解析**：提取公式、表格、图片等结构化内容  
2. **技术文档转换**：将 PDF 转为 Markdown 便于版本管理和在线发布  
3. **OCR 处理**：处理扫描版 PDF 和乱码 PDF  
4. **多语言文档**：支持 109 种语言 OCR 识别  
5. **批量处理**：批量转换多个 PDF 文档  

## 注意事项

1. **文件路径**：所有路径必须是绝对路径  
2. **输出目录**：不存在的目录会自动创建  
3. **性能**：使用带 AMX 的 XEON 可显著提升解析速度  
5. **内存**：处理大型文档可能消耗更多内存  

## 故障排除

### 常见问题

1. **安装失败**：
   - 确保使用 Python 3.10-3.13
   - Windows 仅支持 Python 3.10-3.12（ray 不支持 3.13）
   - 使用 `uv pip install` 可解决大多数依赖冲突

## 相关资源

- MinerU 官方文档：https://opendatalab.github.io/MinerU/
- MinerU GitHub：https://github.com/opendatalab/MinerU
- 在线体验：https://mineru.net/--
