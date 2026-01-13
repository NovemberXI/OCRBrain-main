# gemini.md - 全栈工程化开发规范 (Project Constitution)

此文档为本项目开发的最高指导原则。所有代码生成、架构设计与重构工作必须严格遵循本规范。

---

## 1. 项目愿景与交付标准 (Vision & Delivery)

### 1.1 项目定义
* **核心功能**：构建本地化智能底座，集成 Agent (LLM) +  **OCR (当前实现)**。
* **演进路线**：
    * **Phase 1 (MVP)**：Tauri (前端) + Python Sidecar (后端微服务)。
    * **Phase 2 (Performance)**：逐步迁移至 C++/Rust 原生实现。
* **部署要求**：
    * **纯离线环境 (Air-gapped)**：严禁依赖外部 API，所有模型与依赖必须本地化。
    * **硬件兼容**：代码需适配 CPU 推理（低配保底）与 GPU 加速（高配增强）。

### 1.2 工程哲学
* **拒绝 Demo 心态**：代码必须具备工业级可维护性与可扩展性。
* **简单至上**：优先使用**高中英语水平**的基础词汇命名，确保多语言（Python/C++/Java）开发者的认知兼容性。

---

## 2. 目录结构规范 (Directory Structure)

遵循 Linux 标准与项目约定，根目录结构如下：

```text
app/
├── bin/                 # [Binary] 编译产物
│   └── OCRBrain.exe     # Phase 1: PyInstaller 打包后的可执行文件
├── code/                # [Source Code] 源代码
│   ├── main.py          # 入口文件
│   ├── core/            # 核心业务 (Service/Controller)
│   │   ├── ocr_model.py
│   │   ├── OcrService.py
│   │   ├── plug/        # 插件/辅助模块 (e.g., blip_process)
│   │   │   └── blip_process.py
│   │   └── vision_encoder/ # 视觉编码器相关 (e.g., vary_b)
│   │       └── vary_b.py
│   ├── dao/             # 数据访问 (DAO)
│   ├── utils/           # 通用工具 (Static Tools)
│   │   ├── AudioTool.py
│   │   ├── ConfigTool.py
│   │   ├── FileTool.py
│   │   ├── LogTool.py
│   │   └── ocr_internal/ # OCR 内部工具 (e.g., conversation, constants, utils)
│   │       ├── arguments.py
│   │       ├── constants.py
│   │       ├── conversation.py
│   │       └── utils.py
│   └── tests/           # 单元测试
├── config/              # [Config] 配置文件 (XML/YAML/JSON)
│   ├── appDev.yaml      # 开发环境
│   ├── appProd.yaml     # 生产环境
│   └── models.yaml      # 模型路径配置
├── logs/                # [Log] 运行时日志
│   ├── app.log          # 实时日志
│   ├── error.log        # 错误日志 (独立)
│   └── archive/         # 历史归档 (按天分割)
├── data/                # [Data] 业务数据
│   └── tempAudio/       # 临时音频切片
└── out/                 # [Output] 业务输出
```

## 3. 命名规范 (Naming Convention)

为了平滑过渡 Phase 2 的 C++ 重构，Python 代码强制打破 PEP8 规范，全面采用 **驼峰命名法 (CamelCase)**。

### 3.1 核心法则
* **词汇选择**：使用简单、兼容性强的高中词汇。
    * ✅ `User`, `Config`, `Tool`, `Data`
    * ❌ `Administrator`, `Configuration`, `Utility`, `Repository`
* **类名 (Class)**：**大驼峰 (UpperCamelCase)**
    * 示例：`LogTool`, `ConfigTool`, `MainService`
* **方法/变量 (Method/Variable)**：**小驼峰 (lowerCamelCase)**
    * 示例：`loadConfig()`, `printLog()`, `userList`, `retryCount`

---

## 4. 基础设施规范 (Infrastructure)

所有工具类必须采用 **静态类 + 静态方法** 模式，禁止实例化，严格模拟 C++/Java 的使用习惯。

### 4.1 日志系统 (LogTool)
* **文件分割**：必须按天切割日志 (Daily Rotation)。
* **错误隔离**：ERROR 级别日志必须双写，一份进入 `app.log`，一份独立写入 `error.log`。
* **调用规范**：
    ```python
    LogTool.info("System init start")
    LogTool.error("Connect failed", e)
    ```

### 4.2 配置中心 (ConfigTool)
* **多环境支持**：支持 `dev` (开发), `test` (测试), `prod` (生产), `uat` (验收)。
* **格式**：优先使用 YAML 或 JSON (易读易解析)。
* **调用规范**：
    ```python
    # 静态方法，命名空间风格
    ConfigTool.load("appDev.yaml")
    port = ConfigTool.get("server.port")
    ```

### 4.3 通用工具箱 (Utils)
* `FileTool`: 文件操作 (e.g., `pdfToImage`) - **已实现**
* `LogTool`: 日志记录 (e.g., `info`, `error`) - **已实现**
* `ConfigTool`: 配置读取 (e.g., `load`, `get`) - **已实现**
* `TimeTool`: 时间处理 (e.g., `getNowStr`) - **待实现**
* `StringTool`: 字符串处理 (e.g., `isEmpty`) - **待实现**
* `DecimalTool`: 数值计算 (e.g., `add`, `sub`) - **待实现**
* `DateTool`: 日期计算 (e.g., `addDays`) - **待实现**
* `JsonTool`: 序列化 (e.g., `toJson`, `fromJson`) - **待实现**

---

## 5. 架构分层规范 (MVC Architecture)

系统严格按照职责分离原则设计。

### 5.1 DAO 层 (Data Access Object)
* **职责**：仅负责数据的 **存 (Save)** 和 **取 (Get)**。
* **资源管理**：必须使用 `try-catch-finally` 结构，确保资源（数据库连接、文件句柄、显存）在 `finally` 中释放。
* **异常处理**：捕获 -> 打印堆栈 -> 向上抛出。
    ```python
    def getUserData(id):
        try:
            # 申请资源
            conn = DB.connect()
            return conn.query(id)
        except Exception as e:
            LogTool.error("DB Error", e)
            raise e  # 抛出给上层
        finally:
            # 释放资源
            conn.close()
    ```

### 5.2 Service 层 (Business Logic)
* **职责**：核心业务逻辑、算法实现。
* **示例**：
    * **逻辑**：余额 = 原有数据 - 2000
    * **流程**：调用 `DAO.get()` -> 计算 -> 调用 `DAO.save()`

### 5.3 Controller/API 层 (Interface)
* **职责**：
    1.  接收参数 (e.g., `id=8888`, `amount=2000`)
    2.  数据合法性校验 (e.g., `amount > 0`)
    3.  格式化输出 (Return JSON)

---

## 6. 数据模型规范 (Data Modeling)

贯彻“程序 = 数据结构 + 算法 + 展示”的理念。

### 6.1 实体映射 (Class & Attribute)
* **类 (Class)**：真实世界在数字世界的映射。
    * 例：`User` (用户), `Meeting` (会议)
* **属性 (Attribute)**：实体的特征。
    * 例：`User.name`, `Meeting.startTime`

### 6.2 关系与结构 (Data Structures)
根据业务场景严格选择数据结构：
* **列表/数组 (List/Array)**：存储同类数据的集合（如：`userList`）。
* **队列 (Queue)**：先进先出，用于 ASR 音频流缓冲（如：`audioQueue`）。
* **树 (Tree)**：层级关系，用于组织架构或文件目录（如：`deptTree`）。

---

## 7. 测试规范 (Testing)

* **覆盖原则**：每编写一个核心功能函数，必须编写对应的测试用例。
* **测试范围**：优先覆盖 **主要业务流程 (Happy Path)**，确保主干通畅。
* **位置**：`app/code/tests/` 目录下。