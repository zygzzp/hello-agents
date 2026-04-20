"""
组件：记忆模块
    反思智能体 包含行动->反思->优化
"""
from typing import List, Dict

from mycode.chapter04.llm_client import HelloAgentsLLM


class Memory:
    def __init__(self):
        self.record: List[Dict[str, str]] = []

    def add_record(self, record_type: str, content: str):
        """
        添加记忆
        record_type: 记忆类型，例如 "execution" 或 "reflection"
        content: 记忆内容
        """
        self.record.append({"record_type": record_type, "content": content})
        print(f"📝 记忆已更新，新增一条 '{record_type}' 记录。")

    def get_trajectory(self) -> str:
        """将所有记忆记录格式化为一个连贯的字符串文本"""
        trajectory = ""
        for record in self.record:
            record_type = record.get("record_type")
            content = record.get("content")

            if record_type == "execution":
                trajectory += f"--- 上一轮尝试 (代码) ---\n{content}\n\n"
            elif record_type == "reflection":
                trajectory += f"--- 上一轮尝试 (反思) ---\n{content}\n\n"
        return trajectory

    def get_last_execution(self) -> str:
        for record in reversed(self.record):
            if record["record_type"] == "execution":
                return record["content"]
        return None


# 执行提示词：任务
INITIAL_PROMPT_TEMPLATE = """
你是一位资深的Python程序员。请根据以下要求，编写一个Python函数。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。

要求: {task}

请直接输出代码，不要包含任何额外的解释。
"""

# 反思提示词 : 原始任务 执行结果
REFLECT_PROMPT_TEMPLATE = """
你是一位极其严格的代码评审专家和资深算法工程师，对代码的性能有极致的要求。
你的任务是审查以下Python代码，并专注于找出其在**算法效率**上的主要瓶颈。

# 原始任务:
{task}

# 待审查的代码:
```python
{code}
```

请分析该代码的时间复杂度，并思考是否存在一种**算法上更优**的解决方案来显著提升性能。
如果存在，请清晰地指出当前算法的不足，并提出具体的、可行的改进算法建议（例如，使用筛法替代试除法）。
如果代码在算法层面已经达到最优，才能回答“无需改进”。

请直接输出你的反馈，不要包含任何额外的解释。
"""

# 优化提示词：任务 执行结果 审查结果
REFINE_PROMPT_TEMPLATE = """
你是一位资深的Python程序员。你正在根据一位代码评审专家的反馈来优化你的代码。

# 原始任务:
{task}

# 你上一轮尝试的代码:
{last_code_attempt}

# 评审员的反馈:
{feedback}

请根据评审员的反馈，生成一个优化后的新版本代码。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。
请直接输出优化后的代码，不要包含任何额外的解释。
"""


class ReflectionAgent:

    def __init__(self, llm_client: HelloAgentsLLM, max_iterations=3):
        self.llm_client = llm_client
        self.memory = Memory()
        self.max_iterations = max_iterations

    def run(self, task: str):
        print(f"\n--- 开始处理任务 ---\n任务: {task}")

        # 初始执行
        initial_prompt = INITIAL_PROMPT_TEMPLATE.format(task=task)
        initial_code = self.llm_client.think(messages=[{"role": "user", "content": initial_prompt}], temperature=0.5)
        self.memory.add_record("execution", initial_code)
        print(f"\n 初始执行结果:\n{initial_code}\n")

        for i in range(self.max_iterations):
            print(f"\n--- 迭代 {i + 1} ---")

            last_execution = self.memory.get_last_execution()

            reflect_prompt = REFLECT_PROMPT_TEMPLATE.format(task=task, code=last_execution)
            feedback = self.llm_client.think(messages=[{"role": "user", "content": reflect_prompt}],
                                             temperature=0.5)
            print(f"\n 反思结果:\n{feedback}\n")
            self.memory.add_record("reflection", feedback)

            # b. 检查是否需要停止
            if "无需改进" in feedback or "no need for improvement" in feedback.lower():
                print("\n✅ 反思认为代码已无需改进，任务完成。")
                break

            reflect_prompt = REFINE_PROMPT_TEMPLATE.format(task=task, last_code_attempt=last_execution,
                                                           feedback=feedback)
            refined_code = self.llm_client.think(messages=[{"role": "user", "content": reflect_prompt}],
                                                 temperature=0.5)
            print(f"\n 优化后代码:\n{refined_code}\n")
            self.memory.add_record("execution", refined_code)

if __name__ == '__main__':
    # 1. 初始化LLM客户端 (请确保你的 .env 和 llm_client.py 文件配置正确)
    try:
        llm_client = HelloAgentsLLM()
    except Exception as e:
        print(f"初始化LLM客户端时出错: {e}")
        exit()

    # 2. 初始化 Reflection 智能体，设置最多迭代2轮
    agent = ReflectionAgent(llm_client, max_iterations=2)

    # 3. 定义任务并运行智能体
    task = "编写一个Python函数，找出1到n之间所有的素数 (prime numbers)。"
    agent.run(task)