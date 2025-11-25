# 医学文献检索增强生成（RAG）对话系统开发
![项目状态](https://img.shields.io/badge/status-active-brightgreen.svg) ![技术栈](https://img.shields.io/badge/tech-LangChain%20%7C%20LangGraph%20%7C%20LLM-orange.svg) ![应用场景](https://img.shields.io/badge/scene-MRI%20Brain%20Network%20Analysis-blue.svg)
## 项目概述

> 本项目是一个自用的检索增强生成（RAG）工具，专注于MRI脑网络分析领域常用概念的知识检索。基于LangChain与LangGraph搭建，通过多Agent协同工作，结合本地部署的向量数据库与大语言模型，实现专业概念的精准查询与解答，同时配套精度验证模块保障结果可靠性。

## 技术栈

- 核心框架：LangChain（工具链整合）、LangGraph（多Agent工作流编排）

- 向量存储：Chroma（文档向量存储与检索）

- 大语言模型：阿里通义千问（qwen-max，通过DashScope调用）

- 嵌入模型：DashScope text-embedding-v1（文本向量化处理）

- 评估工具：RAGAS（检索与生成结果评估）

- 辅助工具：dotenv（环境变量管理）、matplotlib（评估结果可视化）、datasets（数据处理）

## 核心模块与流程

1. 多Agent协同模块

	- Super Agent（主管智能体）：接收用户查询，判断处理策略（检索/重写/直接结束），实现任务分发

	- Rewrite Agent（重写智能体）：基于历史对话优化模糊查询，最多重写3次，提升检索精度

	- Grade Agent（评分智能体）：对检索到的文档进行相关性评估，决定是否用于生成回答

	- Generate Agent（生成智能体）：结合检索上下文与LLM，生成专业回答

2. 数据库模块

	- 向量数据库：Chroma（存储路径：./chroma_db），集合名称为health_docs（需自行修改为MRI脑网络相关集合名）

	- 检索配置：默认返回Top 6个相关文档，支持通过search_kwargs调整

3. Prompt管理模块

	- 支持本地Prompt模板自由加载，模板文件为：

	  - prompts_template_generate.json（生成回答用）

	  - prompts_template_agent.json（Super Agent决策用）

	  - prompts_template_grade.json（文档评分用）

	  - prompts_template_rewrite.json（问题重写用）

	- 加载方式：通过LangChain的load_prompt函数读取，支持UTF-8编码

4. 精度验证模块

	- 评估指标：上下文精确率（context_precision）、上下文召回率（context_recall）、回答忠实度（faithfulness）、回答相关性（answer_relevancy）

	- 评估数据：基于QApairs.json中的问答对（问题-标准答案）

	- 结果输出：生成Pandas DataFrame表格与2x2散点图可视化

## 系统流程

> 用户查询 → Super Agent决策 → （检索/重写）
→ 检索模块获取相关文档 → Grade Agent评分 → （合格→生成回答；不合格→重写查询）→ 生成最终回答 → 纳入RAGAS评估

## 项目结构

    项目根目录/
    ├── all.py               # 主程序（含所有模块逻辑、工作流编排）
    ├── prompts_template_*.json  # 各Agent的Prompt模板文件
    ├── chroma_db/           # Chroma向量数据库存储目录
    ├── QApairs.json         # 评估用问答对（问题-标准答案）
    └── .env                 # 环境变量配置（存储DASHSCOPE_API_KEY）

## 环境配置

1. 安装依赖（根据代码导入推断核心依赖）：

	```python
	pip install langchain langgraph langchain-chroma dashscope python-dotenv ragas datasets matplotlib pandas
	```

2. 配置环境变量：在.env文件中添加：

	```python
	DASHSCOPE_API_KEY=你的DashScope API密钥
	```

## 使用方法（自用）

1. 准备MRI脑网络相关文档，导入Chroma向量库（需自行确保数据已正确存入./chroma_db）

2. 调整QApairs.json中的问答对（替换为MRI脑网络领域的测试问题与标准答案）

3. 直接运行all.py：

  - 自动处理预设查询（代码中query列表）

  - 批量处理QApairs.json中的测试问题并完成RAGAS评估

  - 输出评估表格与可视化图表

## 关键参数说明（自用可调）

- 检索TopK：search_kwargs={"k": 6}（默认返回6个相关文档）

- 重写上限：times_rewrite最大为3（超过则终止重写）

- LLM温度系数：temperature=0（保证回答严谨，无随机性）

- 评估指标：默认启用4项核心指标，可在evaluate函数中增减
