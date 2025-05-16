import asyncio
import datetime
import re
import logging
from typing import Dict, List

from fastapi import WebSocket
from langdetect import detect

from backend.report_type import BasicReport, DetailedReport
from backend.chat import ChatAgentWithMemory

from gpt_researcher.utils.enum import ReportType, Tone
from multi_agents.main import run_research_task
from gpt_researcher.actions import stream_output
from backend.server.server_utils import CustomLogsHandler
from gpt_researcher.utils.llm import get_llm

logger = logging.getLogger(__name__)

async def is_english(text: str) -> bool:
    """检测文本是否为英文"""
    try:
        # 快速检查中/日/韩文字符
        if re.search(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7a3]', text):
            return False
        return detect(text) == 'en'
    except Exception as e:
        logger.error(f"Error detecting language: {e}")
        return False

async def translate_to_english(text: str) -> str:
    """使用 Qwen 模型翻译文本"""
    try:
        # 使用 qwen_local 作为翻译模型
        llm = get_llm(
            "qwen_local",
            model="./qwen2.5-7B",
            temperature=1.0,
            max_tokens=1024
        )
        messages = [{"role": "user", "content": f"请将以下内容准确翻译成英文（保留专业术语）：\n\n{text}\n\n只需返回翻译结果，不要添加任何解释。"}]
        response = await llm.get_chat_response(messages, stream=False)
        return response.strip()
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return text  # 失败时返回原文

class WebSocketManager:
    """Manage websockets"""

    def __init__(self):
        """Initialize the WebSocketManager class."""
        self.active_connections: List[WebSocket] = []
        self.sender_tasks: Dict[WebSocket, asyncio.Task] = {}
        self.message_queues: Dict[WebSocket, asyncio.Queue] = {}
        self.chat_agent = None

    async def start_sender(self, websocket: WebSocket):
        """Start the sender task."""
        queue = self.message_queues.get(websocket)
        if not queue:
            return

        while True:
            message = await queue.get()
            if websocket in self.active_connections:
                try:
                    if message == "ping":
                        await websocket.send_text("pong")
                    else:
                        await websocket.send_text(message)
                except:
                    break
            else:
                break

    async def connect(self, websocket: WebSocket):
        """Connect a websocket."""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.message_queues[websocket] = asyncio.Queue()
        self.sender_tasks[websocket] = asyncio.create_task(
            self.start_sender(websocket))

    async def disconnect(self, websocket: WebSocket):
        """Disconnect a websocket."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            self.sender_tasks[websocket].cancel()
            await self.message_queues[websocket].put(None)
            del self.sender_tasks[websocket]
            del self.message_queues[websocket]

    async def start_streaming(self, task, report_type, report_source, source_urls, document_urls, tone, websocket, headers=None, query_domains=[]):
        """Start streaming the output."""
        tone = Tone[tone]
        # add customized JSON config file path here
        config_path = "default"
        report = await run_agent(task, report_type, report_source, source_urls, document_urls, tone, websocket, headers = headers, query_domains = query_domains, config_path = config_path)
        #Create new Chat Agent whenever a new report is written
        self.chat_agent = ChatAgentWithMemory(report, config_path, headers)
        return report

    async def chat(self, message, websocket):
        """Chat with the agent based message diff"""
        if self.chat_agent:
            await self.chat_agent.chat(message, websocket)
        else:
            await websocket.send_json({"type": "chat", "content": "Knowledge empty, please run the research first to obtain knowledge"})

async def run_agent(task, report_type, report_source, source_urls, document_urls, tone: Tone, websocket, headers=None, query_domains=[], config_path=""):
    """Run the agent."""    
    # Create logs handler for this research task
    logs_handler = CustomLogsHandler(websocket, task)
    
    # 在创建 researcher 之前先检查并翻译查询
    if not await is_english(task):
        logger.info(f"Translating non-English query: {task}")
        task = await translate_to_english(task)
        logger.info(f"Translated to: {task}")
        # 更新日志处理器中的查询
        await logs_handler.send_json({
            "query": task,
            "sources": [],
            "context": [],
            "report": ""
        })
    
    # Initialize researcher based on report type
    if report_type == "multi_agents":
        logger.info(f"use multi_agents")
        report = await run_research_task(
            query=task,  # 使用翻译后的查询
            websocket=logs_handler,
            stream_output=stream_output, 
            tone=tone, 
            headers=headers
        )
        report = report.get("report", "")
        
    elif report_type == ReportType.DetailedReport.value:
        logger.info(f"use detail_agents")
        researcher = DetailedReport(
            query=task,  # 使用翻译后的查询
            query_domains=query_domains,
            report_type=report_type,
            report_source=report_source,
            source_urls=source_urls,
            document_urls=document_urls,
            tone=tone,
            config_path=config_path,
            websocket=logs_handler,
            headers=headers
        )
        report = await researcher.run()
        
    else:
        researcher = BasicReport(
            query=task,  # 使用翻译后的查询
            query_domains=query_domains,
            report_type=report_type,
            report_source=report_source,
            source_urls=source_urls,
            document_urls=document_urls,
            tone=tone,
            config_path=config_path,
            websocket=logs_handler,
            headers=headers
        )
        report = await researcher.run()

    return report
