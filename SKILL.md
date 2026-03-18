---
name: pdf-processing-cpu
description: PDF document parsing tool based on local MinerU, supports converting PDF to Markdown.
---

## Tool List

### 1. pdf-processing-cpu

Convert PDF documents to Markdown format, preserving document structure, formulas, tables, and images.

**Description**: Use MinerU to parse PDF documents and output in Markdown format, supporting OCR, formula recognition, table extraction, and other features.

**Parameters**:
- `input` (string, required): Absolute path to the PDF file
- `output_dir` (string, required): Absolute path to the output directory

**Return Value**:
```json
#Complete processing
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
  "success": False,
  "message" : "error info",
}
```

**Examples**:
```bash
#output markdown
python $HOME/.openclaw/workspace/skills/pdf-processing-cpu/script/main.py -i /path/to/document.pdf -o output_dir

#or dir for PDF
python $HOME/.openclaw/workspace/skills/pdf-processing-cpu/script/main.py -i /path/to/documents_dir -o output_dir

```

---

## Installation Instructions

### 1. install Requirements
```bash
pip install -r $HOME/.openclaw/workspace/skills/pdf-processing-cpu/requirements.txt
```

### 2. download models
```bash
modelscope download --model a3213105/pdf-processing-cpu --local_dir $HOME/.openclaw/workspace/skills/pdf-processing-cpu/models
```

### 3. Verify Installation
```bash
python $HOME/.openclaw/workspace/skills/pdf-processing-cpu/script/main.py -v
```

### 4. System Requirements

- **Python Version**: 3.10-3.13
- **Operating System**: Linux
- **Memory**:
  - minimum 16GB, recommended 32GB+
- **Disk Space**: minimum 20GB (SSD recommended)

## Use Cases

1. **Academic Paper Parsing**: Extract structured content such as formulas, tables, and images
2. **Technical Document Conversion**: Convert PDF documents to Markdown for version control and online publishing
3. **OCR Processing**: Process scanned PDFs and garbled PDFs
4. **Multilingual Documents**: Supports OCR recognition for 109 languages
5. **Batch Processing**: Batch convert multiple PDF documents


## Notes

1. **File Paths**: All paths must be absolute paths
2. **Output Directory**: Non-existent directories will be created automatically
3. **Performance**: Using XEON with AMX can significantly improve parsing speed
5. **Memory**: Processing large documents may consume more memory

## Troubleshooting

### Common Issues

1. **Installation Failure**:
   - Ensure using Python 3.10-3.13
   - Windows only supports Python 3.10-3.12 (ray does not support 3.13)
   - Using `uv pip install` can resolve most dependency conflicts


## Related Resources

- MinerU Official Documentation: https://opendatalab.github.io/MinerU/
- MinerU GitHub: https://github.com/opendatalab/MinerU
- Online Demo: https://mineru.net/
