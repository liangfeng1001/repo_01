import asyncio
import random
import json
import re
from typing import Dict, Optional, List, Any, AsyncGenerator
import logging
import os
from ..actions.utils import stream_output
from ..actions.query_processing import plan_research_outline, get_search_results
from ..document import DocumentLoader, OnlineDocumentLoader, LangChainDocumentLoader
from ..utils.enum import ReportSource, ReportType, Tone
from ..utils.logging_config import get_json_handler, get_research_logger
from pathlib import Path
import markdown
from weasyprint import HTML
import openai
import numpy as np



class ResearchConductor:
    """Manages and coordinates the research process."""

    def __init__(self, researcher):
        self.researcher = researcher
        self.logger = logging.getLogger('research')
        self.json_handler = get_json_handler()


    async def plan_research(self, query, query_domains=None):
        self.logger.info(f"Planning research for query: {query}")
        if query_domains:
            self.logger.info(f"Query domains: {query_domains}")
        
        await stream_output(
            "logs",
            "planning_research",
            f"🌐 Browsing the web to learn more about the task: {query}...",
            self.researcher.websocket,
        )

        search_results = await get_search_results(query, self.researcher.retrievers[0], query_domains)
        self.logger.info(f"Initial search results obtained: {len(search_results)} results")

        await stream_output(
            "logs",
            "planning_research",
            f"🤔 Planning the research strategy and subtasks...",
            self.researcher.websocket,
        )

        outline = await plan_research_outline(
            query=query,
            search_results=search_results,
            agent_role_prompt=self.researcher.role,
            cfg=self.researcher.cfg,
            parent_query=self.researcher.parent_query,
            report_type=self.researcher.report_type,
            cost_callback=self.researcher.add_costs,
        )
        self.logger.info(f"Research outline planned: {outline}")
        return outline

    async def conduct_research(self):
        """Runs the GPT Researcher to conduct research"""
        if self.json_handler:
            self.json_handler.update_content("query", self.researcher.query)
        
        self.logger.info(f"Starting research for query: {self.researcher.query}")
        
        # Reset visited_urls and source_urls at the start of each research task
        self.researcher.visited_urls.clear()
        research_data = []

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "starting_research",
                f"🔍 Starting the research task for '{self.researcher.query}'...",
                self.researcher.websocket,
            )

        if self.researcher.verbose:
            await stream_output("logs", "agent_generated", self.researcher.agent, self.researcher.websocket)

        # Research for relevant sources based on source types below
        if self.researcher.source_urls:
            self.logger.info("Using provided source URLs")
            research_data = await self._get_context_by_urls(self.researcher.source_urls)
            if research_data and len(research_data) == 0 and self.researcher.verbose:
                await stream_output(
                    "logs",
                    "answering_from_memory",
                    f"🧐 I was unable to find relevant context in the provided sources...",
                    self.researcher.websocket,
                )
            if self.researcher.complement_source_urls:
                self.logger.info("Complementing with web search")
                additional_research = await self._get_context_by_web_search(self.researcher.query, [], self.researcher.query_domains)
                research_data += ' '.join(additional_research)

        elif self.researcher.report_source == ReportSource.Web.value:
            self.logger.info("Using web search")
            research_data = await self._get_context_by_web_search(self.researcher.query, [], self.researcher.query_domains)

        # ... rest of the conditions ...
        elif self.researcher.report_source == ReportSource.Local.value:
            self.logger.info("Using local search")
            document_data = await DocumentLoader(self.researcher.cfg.doc_path).load()
            self.logger.info(f"Loaded {len(document_data)} documents")
            if self.researcher.vector_store:
                self.researcher.vector_store.load(document_data)

            research_data = await self._get_context_by_web_search(self.researcher.query, document_data, self.researcher.query_domains)

        # Hybrid search including both local documents and web sources
        elif self.researcher.report_source == ReportSource.Hybrid.value:
            if self.researcher.document_urls:
                document_data = await OnlineDocumentLoader(self.researcher.document_urls).load()
            else:
                document_data = await DocumentLoader(self.researcher.cfg.doc_path).load()
            if self.researcher.vector_store:
                self.researcher.vector_store.load(document_data)
            docs_context = await self._get_context_by_web_search(self.researcher.query, document_data, self.researcher.query_domains)
            web_context = await self._get_context_by_web_search(self.researcher.query, [], self.researcher.query_domains)
            research_data = f"Context from local documents: {docs_context}\n\nContext from web sources: {web_context}"

        elif self.researcher.report_source == ReportSource.Azure.value:
            from ..document.azure_document_loader import AzureDocumentLoader
            azure_loader = AzureDocumentLoader(
                container_name=os.getenv("AZURE_CONTAINER_NAME"),
                connection_string=os.getenv("AZURE_CONNECTION_STRING")
            )
            azure_files = await azure_loader.load()
            document_data = await DocumentLoader(azure_files).load()  # Reuse existing loader
            research_data = await self._get_context_by_web_search(self.researcher.query, document_data)
            
        elif self.researcher.report_source == ReportSource.LangChainDocuments.value:
            langchain_documents_data = await LangChainDocumentLoader(
                self.researcher.documents
            ).load()
            if self.researcher.vector_store:
                self.researcher.vector_store.load(langchain_documents_data)
            research_data = await self._get_context_by_web_search(
                self.researcher.query, langchain_documents_data, self.researcher.query_domains
            )

        elif self.researcher.report_source == ReportSource.LangChainVectorStore.value:
            research_data = await self._get_context_by_vectorstore(self.researcher.query, self.researcher.vector_store_filter)

        # Rank and curate the sources
        self.researcher.context = research_data
        if self.researcher.cfg.curate_sources:
            self.logger.info("Curating sources")
            self.researcher.context = await self.researcher.source_curator.curate_sources(research_data)

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "research_step_finalized",
                f"Finalized research step.\n💸 Total Research Costs: ${self.researcher.get_costs()}",
                self.researcher.websocket,
            )
            if self.json_handler:
                self.json_handler.update_content("costs", self.researcher.get_costs())
                self.json_handler.update_content("context", self.researcher.context)

        self.logger.info(f"Research completed. Context size: {len(str(self.researcher.context))}")
        return self.researcher.context

    async def _get_context_by_urls(self, urls):
        """Scrapes and compresses the context from the given urls"""
        self.logger.info(f"Getting context from URLs: {urls}")
        
        new_search_urls = await self._get_new_urls(urls)
        self.logger.info(f"New URLs to process: {new_search_urls}")

        scraped_content = await self.researcher.scraper_manager.browse_urls(new_search_urls)
        self.logger.info(f"Scraped content from {len(scraped_content)} URLs")

        if self.researcher.vector_store:
            self.logger.info("Loading content into vector store")
            self.researcher.vector_store.load(scraped_content)

        context = await self.researcher.context_manager.get_similar_content_by_query(
            self.researcher.query, scraped_content
        )
        return context

    # Add logging to other methods similarly...

    async def _get_context_by_vectorstore(self, query, filter: Optional[dict] = None):
        """
        Generates the context for the research task by searching the vectorstore
        Returns:
            context: List of context
        """
        self.logger.info(f"Starting vectorstore search for query: {query}")
        context = []
        # Generate Sub-Queries including original query
        sub_queries = await self.plan_research(query)
        # If this is not part of a sub researcher, add original query to research for better results
        if self.researcher.report_type != "subtopic_report":
            sub_queries.append(query)

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "subqueries",
                f"🗂️  I will conduct my research based on the following queries: {sub_queries}...",
                self.researcher.websocket,
                True,
                sub_queries,
            )

        # Using asyncio.gather to process the sub_queries asynchronously
        context = await asyncio.gather(
            *[
                self._process_sub_query_with_vectorstore(sub_query, filter)
                for sub_query in sub_queries
            ]
        )
        return context

    async def _calculate_query_similarity(self, main_query, sub_query):
        """计算主查询和子查询的相似度"""
        main_embedding = await self.researcher.memory.get_embeddings().aembed_query(main_query)
        sub_embedding = await self.researcher.memory.get_embeddings().aembed_query(sub_query)
        similarity = np.dot(main_embedding, sub_embedding) / (
            np.linalg.norm(main_embedding) * np.linalg.norm(sub_embedding)
        )
        return similarity

    def _calculate_context_rank_score(self, total_results, current_rank):
        """计算上下文排序得分"""
        return (total_results - current_rank + 1) / total_results

    def _calculate_source_authority_score(self, source):
        """计算来源权威性得分"""
        source_scores = {
            "pubmed": 2,
            "arxiv": 1,
            "tavily": 1
        }
        return source_scores.get(source.lower(), 0)

    async def _process_and_score_sub_query(self, main_query, sub_query, scraped_data, query_domains):
        """处理子查询并计算得分"""
        # 计算主查询和子查询的相似度
        query_similarity = await self._calculate_query_similarity(main_query, sub_query)
        
        # 获取子查询的上下文
        context = await self._process_sub_query(sub_query, scraped_data, query_domains)
        if not context:
            return []
        
        # 解析上下文内容
        context_items = []
        current_item = {}
        for line in context.split('\n'):
            if line.startswith('Source:'):
                if current_item:
                    context_items.append(current_item)
                current_item = {'source': line.replace('Source:', '').strip()}
            elif line.startswith('Title:'):
                current_item['title'] = line.replace('Title:', '').strip()
            elif line.startswith('Content:'):
                current_item['content'] = line.replace('Content:', '').strip()
        
        if current_item:
            context_items.append(current_item)
        
        # 对每个上下文项进行分类
        for item in context_items:
            source = item['source'].lower()
            if 'arxiv' in source:
                item['source_type'] = 'arxiv'
            elif 'ncbi' in source:
                item['source_type'] = 'pubmed'
            else:
                item['source_type'] = 'tavily'
        
        # 计算每个上下文项的得分
        total_items = len(context_items)
        scored_items = []
        for i, item in enumerate(context_items):
            # 计算上下文排序得分
            context_rank_score = self._calculate_context_rank_score(total_items, i)
            
            # 计算来源权威性得分
            source_score = self._calculate_source_authority_score(item['source_type'])
            
            # 计算总得分 (w1=0.4, w2=0.4, w3=0.2)
            total_score = (
                0.4 * query_similarity +  # 主查询与子查询相关度
                0.4 * context_rank_score +  # 上下文排序得分
                0.2 * source_score  # 来源权威性得分
            )
            
            scored_items.append({
                'content': item['content'],
                'source': item['source'],
                'title': item['title'],
                'source_type': item['source_type'],
                'similarity_score': query_similarity,
                'context_rank_score': context_rank_score,
                'source_authority_score': source_score,
                'score': total_score
            })
        
        return scored_items

    async def _get_context_by_web_search(self, query, scraped_data: list = [], query_domains: list = []):
        """
        Generates the context for the research task by searching the query and scraping the results
        Returns:
            context: List of context
        """
        self.logger.info(f"Starting web search for query: {query}")
        
        # Generate Sub-Queries including original query
        sub_queries = await self.plan_research(query, query_domains)
        self.logger.info(f"Generated sub-queries: {sub_queries}")
        
        # If this is not part of a sub researcher, add original query to research for better results
        if self.researcher.report_type != "subtopic_report":
            sub_queries.append(query)

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "subqueries",
                f"🗂️ I will conduct my research based on the following queries: {sub_queries}...",
                self.researcher.websocket,
                True,
                sub_queries,
            )

        # 获取原始搜索结果
        context = await asyncio.gather(
            *[
                self._process_sub_query(sub_query, scraped_data, query_domains)
                for sub_query in sub_queries
            ]
        )
        
        # 记录原始搜索结果
        self.logger.info("原始搜索结果:")
        for idx, content in enumerate(context):
            if content:
                self.logger.info(f"子查询 {idx + 1} 的结果:")
                self.logger.info(content)
                self.logger.info("-" * 50)
        
        # 处理所有子查询并计算得分
        try:
            all_scored_items = []
            for sub_query in sub_queries:
                scored_items = await self._process_and_score_sub_query(
                    query, sub_query, scraped_data, query_domains
                )
                all_scored_items.extend(scored_items)
            
            # 记录排序前的得分情况
            self.logger.info("\n排序前的得分情况:")
            for idx, item in enumerate(all_scored_items):
                self.logger.info(f"项目 {idx + 1}:")
                self.logger.info(f"  来源: {item['source']}")
                self.logger.info(f"  来源类型: {item['source_type']}")
                self.logger.info(f"  标题: {item['title']}")
                self.logger.info(f"  内容长度: {len(item['content'])}")
                self.logger.info(f"  相似度得分: {item['similarity_score']:.4f}")
                self.logger.info(f"  上下文排名得分: {item['context_rank_score']:.4f}")
                self.logger.info(f"  来源权威性得分: {item['source_authority_score']:.4f}")
                self.logger.info(f"  总分: {item['score']:.4f}")
                self.logger.info("-" * 50)
            
            # 按得分排序
            all_scored_items.sort(key=lambda x: x['score'], reverse=True)
            
            # 记录排序后的结果
            self.logger.info("\n排序后的结果:")
            for idx, item in enumerate(all_scored_items):
                self.logger.info(f"排名 {idx + 1}:")
                self.logger.info(f"  来源: {item['source']}")
                self.logger.info(f"  来源类型: {item['source_type']}")
                self.logger.info(f"  标题: {item['title']}")
                self.logger.info(f"  内容长度: {len(item['content'])}")
                self.logger.info(f"  相似度得分: {item['similarity_score']:.4f}")
                self.logger.info(f"  上下文排名得分: {item['context_rank_score']:.4f}")
                self.logger.info(f"  来源权威性得分: {item['source_authority_score']:.4f}")
                self.logger.info(f"  总分: {item['score']:.4f}")
                self.logger.info("-" * 50)
            
            # 设置阈值（可以根据需要调整）
            threshold = 0.5
            filtered_items = [item for item in all_scored_items if item['score'] >= threshold]
            
            # 记录过滤后的结果
            self.logger.info(f"\n阈值过滤后的结果 (阈值: {threshold}):")
            self.logger.info(f"过滤前项目数: {len(all_scored_items)}")
            self.logger.info(f"过滤后项目数: {len(filtered_items)}")
            
            # 格式化输出
            context = []
            for item in filtered_items:
                context.append(
                    f"Source: {item['source']}\n"
                    f"Title: {item['title']}\n"
                    f"Content: {item['content']}\n"
                )
            
            if context:
                combined_context = " ".join(context)
                self.logger.info(f"最终组合上下文大小: {len(combined_context)}")
                return combined_context
            return []
            
        except Exception as e:
            self.logger.error(f"Error during web search: {e}", exc_info=True)
            return []

    async def _process_sub_query(self, sub_query: str, scraped_data: list = [], query_domains: list = []):
        """Takes in a sub query and scrapes urls based on it and gathers context."""
        if self.json_handler:
            self.json_handler.log_event("sub_query", {
                "query": sub_query,
                "scraped_data_size": len(scraped_data)
            })
        
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "running_subquery_research",
                f"\n🔍 Running research for '{sub_query}'...",
                self.researcher.websocket,
            )

        try:
            if not scraped_data:
                scraped_data = await self._scrape_data_by_urls(sub_query, query_domains)
                self.logger.info(f"Scraped data size: {len(scraped_data)}")

            content = await self.researcher.context_manager.get_similar_content_by_query(sub_query, scraped_data)
            self.logger.info(f"Content found for sub-query: {len(str(content)) if content else 0} chars")
            # content 进行分类
            result = await self.classify_content(content)

            if content and self.researcher.verbose:
                await stream_output(
                    "logs", "subquery_context_window", f"📃 {result}", self.researcher.websocket
                )
            elif self.researcher.verbose:
                await stream_output(
                    "logs",
                    "subquery_context_not_found",
                    f"🤷 No content found for '{sub_query}'...",
                    self.researcher.websocket,
                )
            if content:
                if self.json_handler:
                    self.json_handler.log_event("content_found", {
                        "sub_query": sub_query,
                        "content_size": len(content)
                    })
            return content
        except Exception as e:
            self.logger.error(f"Error processing sub-query {sub_query}: {e}", exc_info=True)
            return ""


    async def _process_sub_query_with_vectorstore(self, sub_query: str, filter: Optional[dict] = None):
        """Takes in a sub query and gathers context from the user provided vector store

        Args:
            sub_query (str): The sub-query generated from the original query

        Returns:
            str: The context gathered from search
        """
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "running_subquery_with_vectorstore_research",
                f"\n🔍 Running research for '{sub_query}'...",
                self.researcher.websocket,
            )

        content = await self.researcher.context_manager.get_similar_content_by_query_with_vectorstore(sub_query, filter)

        if content and self.researcher.verbose:
            await stream_output(
                "logs", "subquery_context_window", f"📃 {content}", self.researcher.websocket
            )
        elif self.researcher.verbose:
            await stream_output(
                "logs",
                "subquery_context_not_found",
                f"🤷 No content found for '{sub_query}'...",
                self.researcher.websocket,
            )
        return content

    async def classify_content(self, content: str) -> Dict[str, List[Dict]]:
        """
        异步分类内容到不同类别（arxiv/tavily）
    
         Args:
            content: 包含多个内容块的原始文本
        
        Returns:
        分类后的字典结构 {
            "arxiv": [包含arxiv的块],
            "tavily": [其他块]
        }
        """
        # 使用正则表达式分割内容块
        blocks = re.split(r'\n(?=Source: https?://)', content.strip())
        
        classified = {"arxiv": [], "pubmed": [], "tavily": []}  # 初始化所有可能的类别
        
        for block in blocks:
            if not block.strip():
                continue
                
            # 解析块内容
            source_match = re.search(r'^Source: (https?://[^\s]+)', block, re.M)
            title_match = re.search(r'Title: (.+)', block)
            content_match = re.search(r'Content: (.+)', block, re.DOTALL)
            
            if not all([source_match, title_match, content_match]):
                continue
                
            parsed_block = {
                "source": source_match.group(1), 
                "title": title_match.group(1).strip().replace('\n', ''),
                "content": content_match.group(1).strip().replace('\n', ' ')
            }

            # 分类逻辑
            if 'arxiv' in parsed_block['source'].lower():
                classified['arxiv'].append(parsed_block)
            elif 'ncbi' in parsed_block['source'].lower():
                classified['pubmed'].append(parsed_block)
            else:
                classified['tavily'].append(parsed_block)

        # 将没有内容的分类进行剔除，不在输出展示
        classified = {key: value for key, value in classified.items() if value}
        # 使用 json.dumps 将字典转换为 JSON 字符串，确保使用双引号
        classified_json = json.dumps(classified, ensure_ascii=False, indent=2)

        return classified_json

    async def _get_new_urls(self, url_set_input):
        """Gets the new urls from the given url set.
        Args: url_set_input (set[str]): The url set to get the new urls from
        Returns: list[str]: The new urls from the given url set
        """

        new_urls = []
        for url in url_set_input:
            if url not in self.researcher.visited_urls:
                self.researcher.visited_urls.add(url)
                new_urls.append(url)
                if self.researcher.verbose:
                    await stream_output(
                        "logs",
                        "added_source_url",
                        f"✅ Added source url to research: {url}\n",
                        self.researcher.websocket,
                        True,
                        url,
                    )

        return new_urls

    async def _search_relevant_source_urls(self, query, query_domains: list = []):
        new_search_urls = []

        # Iterate through all retrievers
        for retriever_class in self.researcher.retrievers:
            # Instantiate the retriever with the sub-query
            retriever = retriever_class(query, query_domains=query_domains)

            # Perform the search using the current retriever
            search_results = await asyncio.to_thread(
                retriever.search, max_results=self.researcher.cfg.max_search_results_per_query
            )

            # Collect new URLs from search results
            search_urls = [url.get("href") for url in search_results]
            new_search_urls.extend(search_urls)

        # Get unique URLs
        new_search_urls = await self._get_new_urls(new_search_urls)
        random.shuffle(new_search_urls)

        return new_search_urls

    async def _scrape_data_by_urls(self, sub_query, query_domains: list = []):
        """
        Runs a sub-query across multiple retrievers and scrapes the resulting URLs.

        Args:
            sub_query (str): The sub-query to search for.

        Returns:
            list: A list of scraped content results.
        """
        new_search_urls = await self._search_relevant_source_urls(sub_query, query_domains)

        # Log the research process if verbose mode is on
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "researching",
                f"🤔 Researching for relevant information across multiple sources...\n",
                self.researcher.websocket,
            )

        # Scrape the new URLs
        scraped_content = await self.researcher.scraper_manager.browse_urls(new_search_urls)

        if self.researcher.vector_store:
            self.researcher.vector_store.load(scraped_content)

        return scraped_content
