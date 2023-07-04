import agents
import os
import openai
from embedchain import App
from tool.wiki import wikipedia
import json

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']


data_chat_bot = App()
dir_path = "data"

# 遍历文件夹
for root, dirs, files in os.walk(dir_path):
    for file_name in files:
        # os.path.join用于拼接文件或目录路径
        file_path = os.path.join(root, file_name)
        data_chat_bot.add_local("pdf_file",file_path)
    
    print("files added successfully")

# 初始化本地搜索的工具
def localDataSearch(query: str) -> str:
    """
    search the 地热行业的行业报告 for the given query.
    
    :param query: the subject you want to search
    :return: The answer to the query
    """
    return data_chat_bot.query(query)

test = data_chat_bot.query("地热行业市场规模有多大？")
print(test)

# 创建 agent
bot = agents.Agent(openai_api_key=OPENAI_API_KEY, functions = [localDataSearch, wikipedia], thinkDepth=5)

answer = bot.ask("对地热行业做一个简要的市场分析")

content = answer["choices"][0]["message"]["content"]
print(content)
