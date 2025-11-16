# LangChainStudy

一个用于学习和研究LangChain的Python项目模板。

## 项目结构

```
LangChainStudy/
├── src/                    # 源代码目录
│   ├── __init__.py
│   └── langchainstudy/     # 主项目包
│       ├── __init__.py
│       └── example.py      # 示例模块
├── tests/                  # 测试文件目录
├── docs/                   # 文档目录
├── examples/               # 示例代码目录
├── main.py                 # 主程序入口
├── requirements.txt        # 项目依赖
├── .gitignore              # Git忽略文件
└── README.md               # 项目说明文档
```

## 功能特点

- 标准的Python项目结构
- 虚拟环境支持
- 完整的依赖管理
- 示例代码和文档结构
- 开发工具配置

## 安装步骤

### 1. 克隆项目

```bash
git clone <repository-url>
cd LangChainStudy
```

### 2. 激活虚拟环境

#### Windows
```bash
venv\Scripts\activate
```

#### macOS/Linux
```bash
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 运行主程序

```bash
python main.py
```

### 开发新功能

1. 在`src/langchainstudy/`目录下创建新的模块
2. 在`main.py`中导入并使用新模块
3. 运行测试确保功能正常

## 开发指南

### 代码规范

- 使用Black进行代码格式化
- 使用isort进行导入排序
- 使用flake8进行代码检查

```bash
# 格式化代码
black .
isort .

# 代码检查
flake8 .
```

### 测试

使用pytest进行测试：

```bash
pytest
```

## 环境变量

创建`.env`文件并添加必要的环境变量：

```
# OpenAI API密钥
OPENAI_API_KEY=your-api-key-here

# 其他环境变量
```

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

## 联系方式

如有问题或建议，请通过以下方式联系：

- Email: your-email@example.com
- GitHub: [your-github-username](https://github.com/your-github-username)