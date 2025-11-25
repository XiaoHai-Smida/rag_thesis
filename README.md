# 医学文献检索增强生成（RAG）对话系统开发
![项目状态](https://img.shields.io/badge/status-active-brightgreen.svg) ![技术栈](https://img.shields.io/badge/tech-LangChain%20%7C%20RAG%20%7C%20LLM-orange.svg) ![应用场景](https://img.shields.io/badge/scene-Knowledge%20Retrieval-blue.svg)
## 项目概述

> 本项目是一个自用的检索增强生成（RAG）工具，专注于MRI脑网络分析领域常用概念的知识检索。基于LangChain与LangGraph搭建，通过多Agent协同工作，结合本地部署的向量数据库与大语言模型，实现专业概念的精准查询与解答，同时配套精度验证模块保障结果可靠性。

## 技术栈

- 核心框架：LangChain（工具链整合）、LangGraph（多Agent工作流编排）

- 向量存储：Chroma（文档向量存储与检索）

- LLM：Qwen1.5-7B-Chat （使用 vLLM 本地部署）

- Embedding Model：DashScope text-embedding-v1（文本向量化处理）

## 核心模块与流程

1. 多Agent协同

	- Super Agent：接收用户查询，判断处理策略（检索/重写/直接结束），实现任务分发

	- Rewrite Agent：基于历史对话优化模糊查询，最多重写3次，提升检索精度

	- Grade Agent：对检索到的文档进行相关性评估，决定是否用于生成回答

	- Generate Agent：结合检索上下文与LLM，生成专业回答

2. 数据库构建

	- 向量数据库：Chroma（存储路径：./chroma_db）

	- 检索配置：默认返回Top 6个相关文档，支持通过search_kwargs调整

3. Prompt管理

	- 支持本地Prompt模板自由加载，模板文件为：

	  - prompts_template_generate.json（生成回答用）

	  - prompts_template_agent.json（Super Agent决策用）

	  - prompts_template_grade.json（文档评分用）

	  - prompts_template_rewrite.json（问题重写用）

	- 加载方式：通过LangChain的load_prompt函数读取，支持UTF-8编码

4. 精度验证

	- 评估指标：上下文精确率（context_precision）、上下文召回率（context_recall）、回答忠实度（faithfulness）、回答相关性（answer_relevancy）

	- 评估数据：基于QApairs.json中的问答对

## 系统流程

> 用户查询 → Super Agent决策 → （检索/重写）→ 检索模块获取相关文档
> → Grade Agent评分 → （合格→生成回答；不合格→重写查询）→ 生成最终回答 → 纳入RAGAS评估

## 项目结构

    项目根目录/
    ├── langgraph_mian.ipynb 	 # 主程序（含所有模块逻辑、工作流编排）
	├── langgraph_mian_v1.ipynb	 # 主程序（本地LLM替换为在线模型 qwen-max）
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
	DASHSCOPE_API_KEY=your_DashScope_API_key
	```

## 关键参数说明（自用可调）

- 检索TopK：search_kwargs={"k": 6}（默认返回6个相关文档）

- 重写上限：times_rewrite最大为3（超过则终止重写）

- LLM温度系数：temperature=0（保证回答严谨，无随机性）

- 评估指标：默认启用4项核心指标，可在evaluate函数中增减
