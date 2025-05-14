import asyncio
from typing import List, Dict, Set, Optional, Any
from fastapi import WebSocket
import logging

from gpt_researcher import GPTResearcher

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('detailed_report')

class DetailedReport:
    def __init__(
        self,
        query: str,
        report_type: str,
        report_source: str,
        source_urls: List[str] = [],
        document_urls: List[str] = [],
        query_domains: List[str] = [],
        config_path: str = None,
        tone: Any = "",
        websocket: WebSocket = None,
        subtopics: List[Dict] = [],
        headers: Optional[Dict] = None
    ):
        self.query = query
        self.report_type = report_type
        self.report_source = report_source
        self.source_urls = source_urls
        self.document_urls = document_urls
        self.query_domains = query_domains
        self.config_path = config_path
        self.tone = tone
        self.websocket = websocket
        self.subtopics = subtopics
        self.headers = headers or {}

        self.gpt_researcher = GPTResearcher(
            query=self.query,
            query_domains=self.query_domains,
            report_type="research_report",
            report_source=self.report_source,
            source_urls=self.source_urls,
            document_urls=self.document_urls,
            config_path=self.config_path,
            tone=self.tone,
            websocket=self.websocket,
            headers=self.headers,
        )
        self.existing_headers: List[Dict] = []
        self.global_context: List[str] = []
        self.global_written_sections: List[str] = []
        self.global_urls: Set[str] = set(
            self.source_urls) if self.source_urls else set()

    async def run(self) -> str:
        await self._initial_research()
        subtopics = await self._get_all_subtopics()
        report_introduction = await self.gpt_researcher.write_introduction()
        _, report_body = await self._generate_subtopic_reports(subtopics)
        self.gpt_researcher.visited_urls.update(self.global_urls)
        report = await self._construct_detailed_report(report_introduction, report_body)
        return report

    async def _initial_research(self) -> None:
        await self.gpt_researcher.conduct_research()
        self.global_context = self.gpt_researcher.context
        self.global_urls = self.gpt_researcher.visited_urls
        logger.info("=== _initial_research ===")
        logger.info(f"global_context: {self.global_context }")
        logger.info(f"global_urls: {self.global_urls}")
        logger.info("============================================")

    async def _get_all_subtopics(self) -> List[Dict]:
        subtopics_data = await self.gpt_researcher.get_subtopics()
        
        logger.info("=== Subtopics Data from _get_all_subtopics ===")
        logger.info(f"Raw subtopics_data: {subtopics_data}")
        logger.info(f"Subtopics type: {type(subtopics_data)}")
        if hasattr(subtopics_data, 'subtopics'):
            logger.info(f"Subtopics list: {subtopics_data.subtopics}")
        logger.info("============================================")

        all_subtopics = []
        if subtopics_data and subtopics_data.subtopics:
            for subtopic in subtopics_data.subtopics:
                all_subtopics.append({"task": subtopic.task})
        else:
            logger.warning(f"Unexpected subtopics data format: {subtopics_data}")

        return all_subtopics

    async def _generate_subtopic_reports(self, subtopics: List[Dict]) -> tuple:
        subtopic_reports = []
        subtopics_report_body = ""
        name = ['report_1','report_2','report_3']
        for index,subtopic in enumerate(subtopics):
            result = await self._get_subtopic_report(subtopic,name[index])
            if result["report"]:
                logger.info("=== sub_topic_report ===")
                logger.info(f"report_introduction: {result}")
                logger.info("============================================")
                subtopic_reports.append(result)
                subtopics_report_body += f"\n\n\n{result['report']}"

        return subtopic_reports, subtopics_report_body

    async def _get_subtopic_report(self, subtopic: Dict, name:str) -> Dict[str, str]:
        current_subtopic_task = subtopic.get("task")
        subtopic_assistant = GPTResearcher(
            query=current_subtopic_task,
            query_domains=self.query_domains,
            report_type="subtopic_report",
            report_source=self.report_source,
            websocket=self.websocket,
            headers=self.headers,
            parent_query=self.query,
            subtopics=self.subtopics,
            visited_urls=self.global_urls,
            agent=self.gpt_researcher.agent,
            role=self.gpt_researcher.role,
            tone=self.tone,
        )

        subtopic_assistant.context = list(set(self.global_context))
        await subtopic_assistant.conduct_research()

        draft_section_titles = await subtopic_assistant.get_draft_section_titles(current_subtopic_task)

        if not isinstance(draft_section_titles, str):
            draft_section_titles = str(draft_section_titles)

        parse_draft_section_titles = self.gpt_researcher.extract_headers(draft_section_titles)
        parse_draft_section_titles_text = [header.get(
            "text", "") for header in parse_draft_section_titles]

        relevant_contents = await subtopic_assistant.get_similar_written_contents_by_draft_section_titles(
            current_subtopic_task, parse_draft_section_titles_text, self.global_written_sections
        )

        subtopic_report = await subtopic_assistant.write_report(
            existing_headers=self.existing_headers,
            relevant_written_contents=relevant_contents,
            name=name
        )

        self.global_written_sections.extend(self.gpt_researcher.extract_sections(subtopic_report))
        self.global_context = list(set(subtopic_assistant.context))
        self.global_urls.update(subtopic_assistant.visited_urls)

        self.existing_headers.append({
            "subtopic task": current_subtopic_task,
            "headers": self.gpt_researcher.extract_headers(subtopic_report),
        })

        return {"topic": subtopic, "report": subtopic_report}

    async def _construct_detailed_report(self, introduction: str, report_body: str) -> str:
        toc = self.gpt_researcher.table_of_contents(report_body)
        conclusion = await self.gpt_researcher.write_report_conclusion(report_body)
        conclusion_with_references = self.gpt_researcher.add_references(
            conclusion, self.gpt_researcher.visited_urls)
        report = f"{introduction}\n\n{toc}\n\n{report_body}\n\n{conclusion_with_references}"
        return report
