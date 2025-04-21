import importlib
from typing import Any
from colorama import Fore, Style, init
import os
import requests
import json

_SUPPORTED_PROVIDERS = {
    "openai",
    "anthropic",
    "azure_openai",
    "cohere",
    "google_vertexai",
    "google_genai",
    "fireworks",
    "ollama",
    "together",
    "mistralai",
    "huggingface",
    "groq",
    "bedrock",
    "dashscope",
    "xai",
    "deepseek",
    "deepseek_hw",
    "litellm",
    "qwen_local",
}


class GenericLLMProvider:

    def __init__(self, llm):
        self.llm = llm

    @classmethod
    def from_provider(cls, provider: str, **kwargs: Any):
        if provider == "openai":
            _check_pkg("langchain_openai")
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(**kwargs)
        elif provider == "anthropic":
            _check_pkg("langchain_anthropic")
            from langchain_anthropic import ChatAnthropic

            llm = ChatAnthropic(**kwargs)
        elif provider == "azure_openai":
            _check_pkg("langchain_openai")
            from langchain_openai import AzureChatOpenAI

            if "model" in kwargs:
                model_name = kwargs.get("model", None)
                kwargs = {"azure_deployment": model_name, **kwargs}

            llm = AzureChatOpenAI(**kwargs)
        elif provider == "cohere":
            _check_pkg("langchain_cohere")
            from langchain_cohere import ChatCohere

            llm = ChatCohere(**kwargs)
        elif provider == "google_vertexai":
            _check_pkg("langchain_google_vertexai")
            from langchain_google_vertexai import ChatVertexAI

            llm = ChatVertexAI(**kwargs)
        elif provider == "google_genai":
            _check_pkg("langchain_google_genai")
            from langchain_google_genai import ChatGoogleGenerativeAI

            llm = ChatGoogleGenerativeAI(**kwargs)
        elif provider == "fireworks":
            _check_pkg("langchain_fireworks")
            from langchain_fireworks import ChatFireworks

            llm = ChatFireworks(**kwargs)
        elif provider == "ollama":
            _check_pkg("langchain_community")
            _check_pkg("langchain_ollama")
            from langchain_ollama import ChatOllama
            
            llm = ChatOllama(base_url=os.environ["OLLAMA_BASE_URL"], **kwargs)
        elif provider == "together":
            _check_pkg("langchain_together")
            from langchain_together import ChatTogether

            llm = ChatTogether(**kwargs)
        elif provider == "mistralai":
            _check_pkg("langchain_mistralai")
            from langchain_mistralai import ChatMistralAI

            llm = ChatMistralAI(**kwargs)
        elif provider == "huggingface":
            _check_pkg("langchain_huggingface")
            from langchain_huggingface import ChatHuggingFace

            if "model" in kwargs or "model_name" in kwargs:
                model_id = kwargs.pop("model", None) or kwargs.pop("model_name", None)
                kwargs = {"model_id": model_id, **kwargs}
            llm = ChatHuggingFace(**kwargs)
        elif provider == "groq":
            _check_pkg("langchain_groq")
            from langchain_groq import ChatGroq

            llm = ChatGroq(**kwargs)
        elif provider == "bedrock":
            _check_pkg("langchain_aws")
            from langchain_aws import ChatBedrock

            if "model" in kwargs or "model_name" in kwargs:
                model_id = kwargs.pop("model", None) or kwargs.pop("model_name", None)
                kwargs = {"model_id": model_id, "model_kwargs": kwargs}
            llm = ChatBedrock(**kwargs)
        elif provider == "dashscope":
            _check_pkg("langchain_dashscope")
            from langchain_dashscope import ChatDashScope

            llm = ChatDashScope(**kwargs)
        elif provider == "xai":
            _check_pkg("langchain_xai")
            from langchain_xai import ChatXAI

            llm = ChatXAI(**kwargs)
        elif provider == "deepseek":
            _check_pkg("langchain_openai")
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(openai_api_base='https://api.deepseek.com',
                     openai_api_key=os.environ["DEEPSEEK_API_KEY"],
                     **kwargs
                )
        elif provider == "deepseek_hw":
            _check_pkg("langchain_openai")
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(openai_api_base='https://infer-modelarts-cn-southwest-2.modelarts-infer.com/v1/infers/952e4f88-ef93-4398-ae8d-af37f63f0d8e/v1/',
                     openai_api_key="cKf5GPVxUVlhnFwQO-2ooeCWmnH0IvZ4DtVTEcSIPcHjt-2cXguFo3j7Sj13Vo29gZ-tYjIrcci8EANOCEO2yg",
                     **kwargs
                )
        elif provider == "litellm":
            _check_pkg("langchain_community")
            from langchain_community.chat_models.litellm import ChatLiteLLM

            llm = ChatLiteLLM(**kwargs)
        elif provider == "gigachat":
            _check_pkg("langchain_gigachat")
            from langchain_gigachat.chat_models import GigaChat

            kwargs.pop("model", None) # Use env GIGACHAT_MODEL=GigaChat-Max
            llm = GigaChat(**kwargs)
        elif provider == "qwen_local":
            _check_pkg("langchain_openai")
            from langchain_openai import ChatOpenAI
            
                
            llm = ChatOpenAI(
                openai_api_base="http://39.105.44.103:8294/v1/", 
                openai_api_key="dummy_key",
                **kwargs
            )
    
        else:
            supported = ", ".join(_SUPPORTED_PROVIDERS)
            raise ValueError(
                f"Unsupported {provider}.\n\nSupported model providers are: {supported}"
            )
        return cls(llm)


    # async def get_chat_response(self, messages, stream, websocket=None):
    #     if not stream:
    #         # Getting output from the model chain using ainvoke for asynchronous invoking
    #         print("message", messages)
    #         output = await self.llm.ainvoke(messages)

    #         return output.content

    #     else:
    #         return await self.stream_response(messages, websocket)

    async def get_chat_response(self, messages, stream, websocket=None):
        if not stream:
            url = "http://39.105.44.103:8294/v1/chat/completions"
            api_key = ""

            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}'
            }
            data = {
                "model": "./qwen2.5-7B",
                "max_tokens": 1024,
                "messages": messages,
                "stream": False,
                "temperature": 1.0
            }
            
            try:
                response = requests.post(url, headers=headers, data=json.dumps(data), verify=False)
                
                if response.status_code != 200:
                    raise Exception(f"请求失败: {response.status_code} - {response.text}")
                
                content = response.json()
                if 'choices' not in content or not content['choices']:
                    raise Exception("响应格式错误: 缺少 choices 字段")
                
                return content['choices'][0]['message']['content']
                
            except Exception as e:
                print(f"[ERROR] 请求失败: {str(e)}")
                raise

        else:
            return await self.stream_response(messages, websocket)

    async def stream_response(self, messages, websocket=None):
        paragraph = ""
        response = ""

        # Streaming the response using the chain astream method from langchain
        async for chunk in self.llm.astream(messages):
            content = chunk.content
            if content is not None:
                response += content
                paragraph += content
                if "\n" in paragraph:
                    await self._send_output(paragraph, websocket)
                    paragraph = ""

        if paragraph:
            await self._send_output(paragraph, websocket)

        return response

    async def _send_output(self, content, websocket=None):
        if websocket is not None:
            await websocket.send_json({"type": "report", "output": content})
        else:
            print(f"{Fore.GREEN}{content}{Style.RESET_ALL}")


def _check_pkg(pkg: str) -> None:
    if not importlib.util.find_spec(pkg):
        pkg_kebab = pkg.replace("_", "-")
        # Import colorama and initialize it
        init(autoreset=True)
        # Use Fore.RED to color the error message
        raise ImportError(
            Fore.RED + f"Unable to import {pkg_kebab}. Please install with "
            f"`pip install -U {pkg_kebab}`"
        )
