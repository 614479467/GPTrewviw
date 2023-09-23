
import openai

# 替换 YOUR_API_KEY 为您的实际API密钥
openai.api_key = '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'

# 尝试调用OpenAI API来获取引擎列表
try:
    engines = openai.Engine.list()
    print("Connection successful! Available engines:", engines)
except Exception as e:
    print("Connection failed! Error:", e)