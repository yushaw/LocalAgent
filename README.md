# Agent 实现
1. 本项目探索使用 Agent 从本地文件中处理数据，进而生成复杂内容。例如，输入与特定行业相关的文档，基于此生成行业分析报告。
2. 由于存在 token 数量的限制，我们不能将大量数据直接交由语言模型处理。传统的 embedding 数据库提供的功能相对基础，多半只能处理简单的问答情形。

# 原理
1. 采用了典型的 Tasken Driven 的 Agent 结构；
2. 考虑到 embedding 数据库功能单一，我们确保任务精细化，以便语言模型可以从数据库中找到对应的数据或信息；
3. 需要对 prompt 进行大量修改和调整。例如，如果在 embedding 数据库中未找到内容，可能是因为任务划分不够细致，需进行再次处理（本库未体现）；
4. Prompt 采用标准的 ReAct 形式。

# 待完成
1. 实现 prompt 的 ToT (Tree of Thoughts)，用以处理复杂的数据或逻辑推理问题；
2. 自动生成 prompt。目前，根据任务调整 prompt 的工作量很大。

# 代码实现
1. 基于 OpenAI 和 Langchain，使用了 Chromadb 的本地数据库;
2. 为了快速实现功能，借用了大量来自 aurelio-labs/funkagent 和 embedchain/embedchain 的代码。



