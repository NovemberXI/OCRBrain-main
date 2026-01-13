# OCRBrain (Project Gemini) - 离线智能 OCR 底座
**Building the Visual Perception Core for Offline AI Agents**

OCRBrain 是一个专为 **离线环境 (Air-gapped)** 设计的高性能光学字符识别引擎。它不仅是文字识别工具，更是为本地化 AI Agent 打造的**“视觉感知核心”**。

---

## 🚀 项目特点与核心优势 (Key Features & Advantages)

*   **完全离线运行**: 所有模型与依赖**均**本地化部署，**无需**外部网络连接，**确保**数据安全与隐私。
*   **企业级工程规范**: 严格遵循 `gemini.md` 定义的 MVC 分层架构与统一命名规范，**保障**代码可维护性、可扩展性，并**为**未来向 C++/Rust 等原生语言迁移**奠定**平滑基础。
*   **统一工具链**: 通过 `LogTool`, `ConfigTool`, `FileTool` 等静态工具类**提供**一致且高效的基础设施服务，**简化**开发与维护。
*   **高性能多模态架构**: **采用** Qwen 系列语言模型与 `vary_b` 视觉编码器结合，**实现**对复杂文档和图像的深度理解和精准文本提取。
*   **硬件兼容性**: 代码**适配** CPU 推理，**同时支持** GPU 加速，**满足**从低配设备到高性能工作站的不同计算需求。

---

## 🧠 技术架构与模型详解 (Technical Architecture & Model Details)

本项目**采用** Python Sidecar 模式，核心技术栈如下：

*   **OCR 核心引擎**: 基于 `transformers` 库，**使用`GOTQwenForCausalLM`模型架构**。此架构**结合**了强大的 Qwen 系列语言模型与先进的视觉编码器，**实现**多模态理解能力。
    *   **推理优化**: 当前模型主要侧重于准确性，性能优化**正在进行中**。
    *   模型权重与配置**存储于** `app/code/data/` 目录。

### 模型架构概述

本项目**采用**的多模态模型架构**包含**以下两个核心组成部分：

1.  **视觉编码器 (Vision Encoder)**:
    *   **功能**: **负责**从输入的图像中提取高维语义特征。
    *   **实现**: 项目**集成了`vary_b.py`模块作为视觉编码器**，该模块**基于 Transformer 架构**（ViT），**将图像切片转换为一系列特征向量**。
    *   **输入**: 原始图像像素数据。
    *   **输出**: 图像的语义特征表示（高维特征向量序列）。

2.  **语言模型 (Language Model - LLM)**:
    *   **模型**: Qwen 系列模型，**是一个强大的通用语言模型**，具备上下文理解与文本生成能力。
    *   **功能**: **接收**视觉编码器提取的图像特征，并**结合**由 `conversation` 模块生成的文本指令 (prompt)，**进而生成 OCR 结果文本**。
    *   **实现**: `ocr_model.py` **封装了`GOTQwenForCausalLM`模型的加载与 OCR 推理逻辑**。
    *   **输入**: 视觉编码器输出的图像特征，以及**结构化文本 prompt**。
    *   **输出**: 识别出的原始文本序列。

---

### 模型处理流程 (Model Processing Flow)

为实现高效精准的OCR功能，本项目设计了一套端到端的模型处理流程，涵盖预处理、模型推理及后处理三个阶段。

1.  **预处理 (Preprocessing)**
    *   **图像加载与转换**: 用户输入的图像文件（PNG, JPG, WEBP等）或PDF页面，首先由 `FileTool` 加载并转换为统一的 PIL.Image 格式。
    *   **视觉特征准备**: 转换后的图像随后由 `blip_process` 模块进行进一步处理。此步骤包括图像尺寸调整、归一化等操作，最终生成符合视觉编码器输入要求的张量形式（`image_tensor`）。

2.  **模型推理输入 (Model Inference Input)**
    *   **多模态融合**: 模型的实际输入是一个融合了图像特征和文本指令的多模态序列。
    *   **图像特征**: 预处理阶段生成的 `image_tensor` 被传递给 `vary_b` 视觉编码器，提取出高维的图像特征向量。
    *   **文本指令 (Prompt)**: `conversation` 模块根据 `ocrtype` 参数（例如 `plain`）构建一个结构化的文本 prompt。这个 prompt 明确指示模型执行何种OCR任务。
    *   **输入序列构建**: 视觉特征与文本 prompt 通过 `tokenizer` 编码后，结合特定的 `<image>` 和 `<imgpad>` 占位符，形成模型最终接受的输入序列 (`input_ids`)。

3.  **模型推理输出 (Model Inference Output)**
    *   **文本生成**: `GOTQwenForCausalLM` 模型接收融合后的输入序列，基于其强大的多模态理解能力，生成原始的文本序列作为识别结果。此过程在 GPU (CUDA) 或 CPU 上执行，并利用 `torch.autocast` 进行类型优化以提升效率。

4.  **后处理 (Postprocessing)**
    *   **结果解码**: 模型生成的原始文本序列由 `tokenizer` 解码为可读的字符串。
    *   **格式清理**: 解码后的文本会进行额外的清理，例如移除特定的停止符 (`stop_str`) 或空白字符。
    *   **JSON封装**: 最终的纯文本识别结果与输入文件信息一起，被封装成 JSON 格式的数据。

---

### 系统级输入输出形式 (System-Level Input/Output Formats)

*   **系统级输入**:
    *   **支持**图像文件：PNG, JPG, JPEG, GIF, BMP, WEBP 格式。
    *   **支持** PDF 文档。
*   **模型层输入**:
    *   图像：经过 `blip_process` 模块预处理的图像张量。
    *   文本：通过 `conversation` 模块构建的结构化文本 prompt。
*   **系统级输出**:
    *   OCR 识别结果**以 JSON 格式精确保存**。JSON 文件名**遵循** `原始文件名.原始扩展名.json` 格式，**彻底避免**同名但不同类型文件的结果覆盖问题。

---

## 🚀 用户指南 (User Guide)

### 1. 环境安装 (Installation)

**推荐**使用 Conda 进行环境隔离：

```bash
# 1. 克隆代码
git clone [YOUR_REPO_URL_HERE] # 请替换为实际的项目仓库URL
cd OCRBrain-main # 确保进入项目根目录

# 2. 创建环境 (Python 3.10)
conda create -n ocr_brain python=3.10
conda activate ocr_brain

# 3. 安装依赖
# 包含 PyTorch, Hugging Face Transformers, Pillow 等核心库
pip install -r app/code/requirements.txt
```

### 2. 模型准备 (Model Setup)

*   **下载模型**: 将您的 OCR 模型权重和相关配置文件（例如 Qwen 模型）**下载并放置**在 `app/code/data/` 目录下。
*   **配置路径**: **修改** `app/config/models.yaml` 文件，**指定** `ocr.modelPath` 为您模型所在的目录：
    ```yaml
    ocr:
      modelPath: "app/code/data" # 替换为您的模型实际路径
    ```

### 3. 数据准备 (Data Preparation)

您可以准备图像文件（如 `.jpg`, `.png`）或 PDF 文件作为 OCR 输入。
*   **存放位置**: **建议**将测试文件放置在 `app/code/input/` 目录**（需自行创建）**。

### 4. 运行 OCR (Usage)

OCRBrain **提供**命令行接口进行 OCR 处理，**支持**图像、PDF 文件或目录作为输入，并将结果保存到指定目录的 JSON 文件。

#### 核心命令行参数

*   `-i` 或 `--input`: **必需**。**指定**输入路径，**支持**单个图像文件、单个 PDF 文件，**或**一个包含图片/PDF 的目录。
*   `-o` 或 `--output_dir`: **可选**。**指定** OCR 结果的输出目录。**默认值为**项目根目录下的 `out` 文件夹。输出的 JSON 文件**采用** `原始文件名.原始扩展名.json` 的格式命名，**彻底防止**同名但不同类型文件的结果被覆盖。
*   `--ocrtype`: **可选**。**指定** OCR 处理的类型。**当前支持 `plain` (默认值，纯文本提取)**。未来**将扩展支持**更多类型（例如 `form`, `table` 等），具体可查看 `app/code/core/OcrService.py` 中的 `performOcr` 函数实现。

#### 场景示例

**场景 A: 对单个图像文件执行 OCR**

```bash
# 运行命令 (-i 指定输入图像文件路径)
python app/code/main.py -i "app/code/input/example.jpg"
# 结果将保存到项目根目录的 out/example.jpg.json

# 指定输出目录
python app/code/main.py -i app/code/input/1.webp -o results
# 结果将保存到 results/1.webp.json
```

**场景 B: 对单个 PDF 文件执行 OCR**

```bash
# 运行命令 (-i 指定输入 PDF 文件路径)
python app/code/main.py -i "app/code/input/document.pdf"
# 结果将保存到项目根目录的 out/document.pdf.json

# 指定输出目录
python app/code/main.py -i app/code/input/1.pdf -o results
# 结果将保存到 results/1.pdf.json
```

**场景 C: 对目录下的多个文件执行 OCR**

```bash
# 运行命令 (-i 指定包含图片/PDF的目录)
python app/code/main.py -i "app/code/input/"
# 目录下所有支持的文件均进行 OCR，结果分别保存到 out/ 目录下

# 指定输出目录
python app/code/main.py -i "app/code/input/" -o batch_results
# 目录下所有支持的文件均进行 OCR，结果分别保存到 batch_results/ 目录下
```

**场景 D: 指定 OCR 类型**

```bash
# 使用默认的 'plain' OCR 类型 (无需明确指定)
python app/code/main.py -i "app/code/input/example.jpg" --ocrtype plain

# 未来支持其他 OCR 类型 (例如 'form')
# python app/code/main.py -i "app/code/input/form_document.jpg" --ocrtype form
```

---

## 🛠️ 质量保障与工程规范 (Quality Assurance & Engineering Standards)

*   **分层架构**: 严格**遵守** MVC 分层设计，**实现**职责分离，**提高**模块内聚与解耦。
*   **代码风格**: **强制采用驼峰命名法** (CamelCase) 贯穿 Python 代码，**为**未来 C++/Rust 跨语言协作**奠定**一致性基础。

---

## 📅 未来计划 (Coming Soon)

1.  **添加测试用例**: **逐步为核心功能和业务逻辑编写单元测试和集成测试**，**提升**代码质量和稳定性。
2.  **Prompt 工程优化**: **深入研究和集成更有效的prompt策略**，**例如**针对不同文档类型**定制化** prompt 模板，以提高模型在特定OCR任务（如表格识别、版面分析）上的表现力。
3.  **模型精细化集成**: 持续探索并**集成最新、性能更优的 OCR 模型**，**例如**考虑引入专门针对手写识别或特定行业文档的微调模型，**或**评估更高效的 Transformer 变体。
4.  **推理性能深度优化**: **采用模型量化、剪枝**等高级优化技术，**使用onnx加速推理引擎**，提升在资源受限环境下的推理速度。
5.  **增强多模态理解**: 除了基础文本提取，**目标是实现结构化信息提取**（如表格数据、表单字段识别），**并结合LLM进行结果的智能分析和交互**。
6.  **用户图形界面开发**: **配合** Tauri 框架**开发直观友好的用户图形界面**，**降低**用户使用门槛，**提升**用户体验。

---

## ⚠️ 当前局限 (Current Limitations)

**注意：本项目目前处于 Phase 1 初期，OCR 功能正在持续完善。**

本项目**采用**大型语言模型 (LLM) 与视觉模型。**因此，对计算资源要求较高**，在 CPU 或性能受限的 GPU 环境下，**OCR 推理速度慢于传统轻量级 OCR 方案**。我们**正持续投入**资源进行模型优化与推理流程改进，以提升整体性能。

---

## 📄 License

MIT License