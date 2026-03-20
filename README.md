# OpenClaw Skill - pdf-processing-cpu

基于 Intel® Xeon® AMX 指令集优化加速的本地化 MinerU PDF 文档解析工具，支持将 PDF 转换为 Markdown

## 功能概览

- 识别PDF文件，生成markdown结构化数据
- 零成本调用： 部署于本地服务器，无需支付按分钟或 Token 计费的 API 费用
- 硬核加速： 深度集成 Intel AMX 矩阵加速技术，针对 BF16/INT8 混合精度推理进行调优，CPU 处理速度可媲美入门级 GPU。
- 极致隐私： 语音数据在本地 Xeon 节点完成处理，满足个人或企业对数据合规性的严苛要求。

## 技术特性
- 算子级优化： 基于 Intel® Xeon® AMX 指令集优化加速

## 典型输入

- pdf文档

## 典型输出

- Markdown文件

See SKILL.md for usage and tuning.
