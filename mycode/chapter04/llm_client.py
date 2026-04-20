import os
from typing import List, Dict

from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env 文件中的环境变量
load_dotenv()


class HelloAgentsLLM:
    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = None):
        self.model = model or os.getenv("LLM_MODEL_ID")
        apiKey = apiKey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")

        # 判断参数齐全
        if not all([self.model, apiKey, baseUrl]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")
        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def think(self, messages: List[Dict[str, str]], temperature: float = 0):
        print(f"🧠 正在调用 {self.model} 模型...")

        try:
            response = self.client.chat.completions.create(model=self.model, messages=messages, temperature=temperature,
                                                           stream=True)
            print("✅ 大语言模型响应成功:")
            collected_content = []
            for chunk in response:
                # 有可能最后一个chunk没有数据
                if not chunk.choices:
                    continue
                # 调用模型传n代表一次性预测几个回答，默认1，delta代表增量 因为是流式输出
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print("\n")
            return "".join(collected_content)

        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None


if __name__ == "__main__":
    llm = HelloAgentsLLM()
    content = llm.think(messages=[{"role": "system", "content": "你是一个写python代码有用的助手"},
                                {"role": "user", "content": "写一个快速排序的代码"}], temperature=0.5)
    if content:
        print("\n\n--- 完整模型响应 ---")
        print(content)
