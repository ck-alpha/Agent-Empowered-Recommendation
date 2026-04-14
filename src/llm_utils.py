import requests
import json
import logging

# 初始化日志记录器，用于在控制台或日志文件中记录运行时的状态和错误信息
logger = logging.getLogger(__name__)

class OllamaLLM:
    """
    Ollama大语言模型接口类。
    主要负责与本地部署的Ollama服务进行网络通信, 封装底层API调用,
    实现向指定的本地大语言模型发送提示词并获取生成结果的功能。
    """
    def __init__(self, model='qwen2.5:14b', temperature=0.1, max_tokens=256):
        # 指定使用的大语言模型名称（需与ollama pull下载的模型名称一致）
        self.model = model
        # 控制模型生成文本的随机性（温度参数）。
        # 推荐系统通常需要较低的温度值（如0.1），以保证模型输出的推荐列表稳定、可控且符合逻辑
        self.temperature = temperature
        # 限制模型单次生成的最大Token数量，防止生成过长无关文本占用资源
        self.max_tokens = max_tokens
        # 本地Ollama服务的默认API请求地址及端口
        self.api_url = "http://127.0.0.1:11434/api/generate"

    def generate(self, prompt, system_prompt=""):
        """
        核心生成方法。接收提示词，调用大模型并返回生成的纯文本结果。
        
        参数:
            prompt (str): 用户输入的主提示词（例如推荐任务的具体上下文、用户历史行为等）。
            system_prompt (str): 系统提示词，用于设定模型的全局角色（例如：探索智能体或利用智能体）。
            
        返回:
            str: 模型生成的文本字符串。如果调用失败, 则返回空JSON字符串 "{}"。
        """
        # 拼接系统提示词和用户提示词，构建完整的输入上下文
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        # 构建符合Ollama API规范的请求负载（Payload）数据字典
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,  # 关闭流式输出，要求模型在后台生成完毕后一次性返回完整结果
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }
        
        try:
            # 向Ollama服务发送HTTP POST请求，包含JSON格式的请求体
            response = requests.post(self.api_url, json=payload)
            # 检查HTTP响应状态码，如果请求失败（如404, 500等）则主动抛出HTTPError异常
            response.raise_for_status()
            # 解析返回的JSON格式数据，提取并返回 "response" 字段中的文本内容
            return response.json().get("response", "")
        except Exception as e:
            # 捕获并记录网络请求失败、连接超时或解析异常等错误信息
            logger.error(f"Ollama 调用失败: {e}")
            # 发生错误时返回默认的空JSON格式字符串，防止外层解析代码因返回None而抛出异常
            return "{}"