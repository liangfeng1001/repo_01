import asyncio
import random
import json
import re
import html
from typing import Dict, Optional, List, Any, AsyncGenerator
import logging
import os
from gpt_researcher.llm_provider.generic.base import GenericLLMProvider
from ..actions.utils import stream_output
from ..actions.query_processing import plan_research_outline, get_search_results, generate_pubmed_sub_queries
from ..document import DocumentLoader, OnlineDocumentLoader, LangChainDocumentLoader
from ..utils.enum import ReportSource, ReportType, Tone
from ..utils.logging_config import get_json_handler, get_research_logger
from pathlib import Path
import markdown
from weasyprint import HTML
import openai
import numpy as np
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import requests
import pandas as pd
from ..retrievers import PubmedDianSearch


authors_t = 1

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
            cost_callback=self.researcher.add_costs
        )
        self.logger.info(f"Research outline planned: {outline}")
        return outline

    async def conduct_research(self):
        """Runs the GPT Researcher to conduct research"""
        if self.json_handler:
            self.json_handler.update_content("query", self.researcher.query)
        
        self.logger.info(f"Starting research for query: {self.researcher.query}")
        self.logger.info(f"Report type: {self.researcher.report_type}")
        
        # 如果是主查询（非子主题报告），重置结果
        if self.researcher.report_type != "subtopic_report":
            self.logger.info(f"Main query detected, resetting accumulated results")

            # self.researcher.accumulated_classified_results = {
            #     "arxiv": [],
            #     "pubmed": [],
            #     "tavily": []
            # }

            #todo 修+（试）
            self.researcher.accumulated_classified_results = {}



        else:
            self.logger.info(f"Subtopic report detected, using existing accumulated results")
            if not hasattr(self.researcher, 'accumulated_classified_results'):
                self.logger.error("No existing accumulated results found for subtopic report")
                raise AttributeError("accumulated_classified_results not found for subtopic report")
        
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
            research_data = await self._get_context_by_web_search(self.researcher.query, [], self.researcher.query_domains, self.researcher.accumulated_classified_results)

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
            "pubmed": 5,
            "arxiv": 3,
            "tavily": 2
        }
        return source_scores.get(source.lower(), 0)

    async def _get_context_by_web_search(self, query, scraped_data: list = [], query_domains: list = [], accumulated_classified_results: dict = None):
        """
        Generates the context for the research task by searching the query and scraping the results
        Returns:
            context: List of context
        """
        self.logger.info(f"Starting web search for query: {query}")
        
        # 载入期刊影响因子Excel数据
        journal_df = None
        excel_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data/journal_impact_factors.xlsx")
        try:
            journal_df = pd.read_excel(excel_path)
            self.logger.info(f"Successfully loaded journal impact factors from {excel_path}")
            self.logger.info(f"Loaded {len(journal_df)} journal entries")
        except Exception as e:
            self.logger.warning(f"Could not load journal impact factors: {e}. Will use default scores.")
        
        # Generate Sub-Queries including original query
        sub_queries = await self.plan_research(query, query_domains)
        self.logger.info(f"Generated sub-queries: {sub_queries}")
        
        # If this is not part of a sub researcher, add original query to research for better results
        if self.researcher.report_type != "subtopic_report":
            mark_query = "\n" + "".join([f"## {q.strip()}\n" for q in sub_queries])
            await stream_output(
                "logs",
                "mark_query",
                f"# {query}{mark_query}",
                self.researcher.websocket,
            )
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

        # 测试用：
        self.logger.info(f"全部的查询query的内容是：{sub_queries}")

        # 获取原始搜索结果
        context = await asyncio.gather(
            *[
                self._process_sub_query(sub_query, scraped_data, query_domains)
                for sub_query in sub_queries
            ]
        )


        # 加+
        min_jif = self.researcher.min_jif_score  # 需添加最低期待影响因子得分的字段

        #todo 修+
        source_type_keys = ["tavily", "arxiv", "pubmed"]  # 定义来源类型键

        # 记录原始搜索结果
        self.logger.info("raw search results:")
        for idx, contents in enumerate(context):
            if contents:
                self.logger.info(f"subquery {idx + 1} - '{sub_queries[idx]}' research results:")
                # 将contents按块分割
                content_blocks = contents.split("\n\n")
                self.logger.info(f"find {len(content_blocks)} content_blocks")
                self.logger.info("-" * 50)
        
        # 处理所有搜索结果并计算得分
        try:
            all_scored_items = []

            # 直接遍历context列表
            for idx, contents in enumerate(context):
                if not contents:
                    continue
                
                # 将内容按块分割
                content_blocks = contents.split("\n\n")

                # 遍历该查询的所有内容块
                for block in content_blocks:
                    if not block.strip():
                        continue

                    # 解析块内容
                    source_match = re.search(r'^Source: (https?://[^\s]+)', block, re.M)
                    title_match = re.search(r'Title: (.+)', block)
                    content_match = re.search(r'Content: (.+)', block, re.DOTALL)
                    
                    if not all([source_match, title_match, content_match]):
                        continue
                        
                    url = source_match.group(1)
                    title = title_match.group(1).strip()
                    content_text = content_match.group(1).strip()



                    # 确定来源默认类型
                    source_type = source_type_keys[0]

                    # 从内容块中获取检索器类型
                    retriever_type_match = re.search(r'RetrieverType: (\w+)', block)
                    if retriever_type_match:
                        source_type = retriever_type_match.group(1)
                        self.logger.info(f"get source_type from content_block: {source_type}")  # 打印检索器类型
                    else:
                        # 从scraped_data中获取检索器类型
                        for item in scraped_data:
                            if item.get('url') == url and 'retriever_type' in item:  # 检查URL和检索器类型是否存在
                                source_type = item['retriever_type']  # 获取检索器类型
                                self.logger.info(f"get source_type from scraped_data: {source_type}")
                                break

                    # 修+
                    # 检查来源类型是否有效
                    if source_type not in source_type_keys:  # 如果来源类型无效，则跳过或者强制修改为默认类型
                        # continue
                        # 强制修改为默认类型
                        source_type = source_type_keys[0]  # 默认类型
                        self.logger.info(
                            f"Invalid source_type: {source_type}. Forcing to default: {source_type_keys[0]}")


                    # 如果来源类型是tavily，根据URL特征进行二次分类
                    if source_type == source_type_keys[0]: # 修+
                        if 'arxiv' in url.lower():  # 如果URL包含arxiv，则将来源类型改为arxiv
                            source_type = "arxiv"
                            self.logger.info("according to url, change source_type from tavily to arxiv")
                        elif 'ncbi' in url.lower() or 'pubmed' in url.lower():  # 如果URL包含ncbi或pubmed，则将来源类型改为pubmed
                            source_type = "pubmed"
                            self.logger.info("according to url, change source_type from tavily to pubmed")

                    # 计算来源权威性得分
                    source_authority_score = self._calculate_source_authority_score(source_type)
                    
                    # 计算上下文排序得分 - 使用当前结果在所有结果中的位置
                    context_rank_score = self._calculate_context_rank_score(len(all_scored_items) + 1, len(all_scored_items))

                    # 计算内容与查询的相似度
                    content_similarity = await self._calculate_query_similarity(query, sub_queries[idx])  

                    impact_factor = 0


                    self.logger.info(f'our min_jif_score is {min_jif}')


                    journal_name = ""
                    if source_type == "pubmed" or source_type == "tavily":
                        # 提取期刊信息
                        journal_info = await self._extract_journal_info_from_url(url)
                        journal_name = journal_info.get("journal_name", "")

                        self.logger.info(f"our journal_name is <<{journal_name}>> in this time")
                        # 查找期刊影响因子
                        if journal_df is not None and journal_name:
                            try:
                                self.logger.info("We are starting to search for journal factors ")
                                # 标准化查询的期刊名称
                                normalized_journal_name = await self._normalize_journal_name(journal_name)

                                # 尝试精确匹配
                                # 首先创建一个标准化的期刊名称列
                                journal_df['NormalizedName'] = journal_df['Name'].apply(
                                    lambda x: str(x).replace(' - ', '-').replace(' And ', ' & ').replace(' and ', ' & ').replace('&amp;', '&').replace('&AMP;', '&') if not pd.isna(x) else ""
                                )
                                journal_match = journal_df[journal_df['NormalizedName'].str.upper() == normalized_journal_name.upper()]



                                # 如果没有精确匹配，尝试部分匹配
                                if journal_match.empty:

                                    self.logger.info("Liang__is trying》》》》 journal_math is also empty")

                                    for _, row in journal_df.iterrows():
                                        normalized_db_name = row['NormalizedName'].upper()
                                        
                                        if normalized_db_name in normalized_journal_name.upper() or normalized_journal_name.upper() in normalized_db_name:
                                            journal_match = pd.DataFrame([row])
                                            break
                                            
                                        # 也检查缩写名
                                        if 'AbbrName' in row and not pd.isna(row['AbbrName']):
                                            abbr_name = str(row['AbbrName']).upper()
                                            normalized_abbr_name = await self._normalize_journal_name(abbr_name)
                                            
                                            if normalized_abbr_name in normalized_journal_name.upper() or normalized_journal_name.upper() in normalized_abbr_name:
                                                journal_match = pd.DataFrame([row])
                                                break
                                
                                # 如果有ISSN匹配
                                if journal_match.empty and 'issn' in journal_info:

                                    self.logger.info("期刊匹配依然没有成功！！")

                                    issn = journal_info['issn']
                                    journal_match = journal_df[(journal_df['ISSN'] == issn) | (journal_df['EISSN'] == issn)]
                                
                                if not journal_match.empty:

                                    impact_factor = float(journal_match.iloc[0]['JIF']) if 'JIF' in journal_match.columns and not pd.isna(journal_match.iloc[0]['JIF']) else 0
                                    self.logger.info(f"期刊 '{journal_name}' 的影响因子 (JIF): {impact_factor}")
                            except Exception as e:
                                self.logger.warning(f"期刊影响因子查询错误: {e}")


                    if impact_factor < min_jif:  # 如果影响因子小于最低期待影响因子，则跳过
                       continue


                    # 获取对应文献的发表年、卷期、页码（若没有，只给年份就好）
                    published_date = ""  # 初始化发表日期
                    authors = ""  # 初始化作者
                    vol = ""  # 初始化卷号
                    pagination = ""  # 初始化页码
                    if source_type == source_type_keys[2]:  # 如果来源类型是pubmed

                        published_date_match = re.search(r'Published Date: (.+)', block)  # 匹配发表日期
                        vol_match = re.search(r'Volume: (.+)', block)  # 匹配卷号
                        pagination_match = re.search(r'Pagination: (.+)', block)  # 匹配页码
                        authors_match = re.search(r'Authors: (.+)', block)  # 匹配作者
                        # 提取信息
                        if authors_match:
                            self.logger.info("作者匹配了！！")
                            authors = authors_match.group(1).strip()  # 获取作者
                        if pagination_match:
                            self.logger.info("页码匹配了！！")
                            pagination = pagination_match.group(1).strip()  # 获取页码
                        if vol_match:
                            self.logger.info("卷号匹配了！！")
                            vol = vol_match.group(1).strip()  # 获取卷号
                        if published_date_match:
                            self.logger.info("日期匹配了！！")
                            published = published_date_match.group(1).strip()  # 获取发表日期
                            # published_date = published.split(" ")[0] # 提取发表日期
                            published_date = published[:4]

                    elif source_type == source_type_keys[1]:  # 如果来源类型是arxiv
                        arxiv_info = self._extract_arxiv_info_from_url(url)  # 提取arxiv信息
                        published_date = arxiv_info.get("published_date", "")  # 获取发表日期
                        # # 检查发表日期是否在2020年之后
                        # if published_date and int(published_date.split(".")[0]) < 2020:  # 检查发表日期是否在2020年之后
                        #     continue  # 如果不是，则跳过
                        authors = arxiv_info.get("arxiv_id", "Unknown ID")  # 获取作者（先用ID代替）


                    if vol:
                        if pagination:  # 如果有卷号和页码
                            vol_pagination = f"Vol.{vol}:{pagination}"  # 拼接卷号和页码
                        else:  # 如果只有卷号
                            vol_pagination = f"Vol.{vol}"  # 拼接卷号
                    else:
                        vol_pagination = ""

                        # 标准化影响因子得分到0-1范围
                    # 假设最高影响因子为100（可根据实际情况调整）
                    max_impact_factor = 503.1
                    normalized_impact_factor = min(impact_factor / max_impact_factor, 1.0)
                    
                    # 计算总得分 (调整权重以反映新的评分方式)
                    total_score = (
                        0.3 * content_similarity +      # 内容与查询的相似度
                        0.2 * context_rank_score +     # 上下文排序得分
                        0.2 * source_authority_score + # 来源权威性得分
                        0.3 * normalized_impact_factor # 期刊影响因子得分
                    )
                    
                    self.logger.info(f"***the content is from: {source_type}***")
                    all_scored_items.append({
                        'content': content_text,
                        'source': url,
                        'title': title,
                        'journal_name': journal_name,
                        'source_type': source_type,
                        'similarity_score': content_similarity,  # 更新为内容相似度
                        'context_rank_score': context_rank_score,
                        'source_authority_score': source_authority_score,
                        'impact_factor': impact_factor,
                        'normalized_impact_factor': normalized_impact_factor,
                        'score': total_score,
                        'published_date': published_date,   # 发表日期
                        'authors': authors,   # 作者
                        'vol_pagination': vol_pagination,   # 卷号+页码

                    })
                
            # 记录排序前的得分情况
            # self.logger.info("\n排序前的得分情况:")
            # for idx, item in enumerate(all_scored_items):
            #     try:
            #         self.logger.info(f"  item {idx + 1}:")
            #         self.logger.info(f"  url_source: {item['source']}")
            #         self.logger.info(f"  journal_name: {item['journal_name']}")
            #         self.logger.info(f"  source_type: {item['source_type']}")
            #         # 处理标题中的特殊字符
            #         title = item['title'].encode('utf-8', errors='ignore').decode('utf-8')
            #         self.logger.info(f"  标题: {title}")
            #         self.logger.info(f"  内容长度: {len(item['content'])}")
            #         self.logger.info(f"  相似度得分: {item['similarity_score']:.4f}")
            #         self.logger.info(f"  上下文排名得分: {item['context_rank_score']:.4f}")
            #         self.logger.info(f"  来源权威性得分: {item['source_authority_score']:.4f}")
            #         self.logger.info(f"  影响因子: {item['impact_factor']}")
            #         self.logger.info(f"  标准化影响因子得分: {item['normalized_impact_factor']:.4f}")
            #         self.logger.info(f"  总分: {item['score']:.4f}")
            #         self.logger.info("-" * 50)
            #     except Exception as e:
            #         self.logger.warning(f"记录项目 {idx + 1} 时出错: {str(e)}")
            #         continue

            # 测试用：
            self.logger.info(f'本次的影响因子最低要求是：{min_jif}')
            self.logger.info(f'总共{len(all_scored_items)}个被收录的文献')



            # 按得分排序
            all_scored_items.sort(key=lambda x: x['score'], reverse=True)

            # self.logger.info(f'展示被收录采纳的文献信息：：')
            # for item in all_scored_items:
            #     try:
            #         self.logger.info(f"  item: {item['title']}")
            #         self.logger.info(f"  source: {item['source']}")
            #         self.logger.info(f"  journal_name: {item['journal_name']}")
            #         self.logger.info(f"  impact_factor: {item['impact_factor']}")
            #         self.logger.info(f"  source_type: {item['source_type']}")
            #         self.logger.info(f"  authors: {item['authors']}")
            #         self.logger.info(f"  published_date: {item['published_date']}")
            #         self.logger.info(f"  vol_pagination: {item['vol_pagination']}")
            #         self.logger.info(f'------------------------')
            #     except Exception as e:
            #         self.logger.warning(f"记录项目时出错: {str(e)}")
            #         continue

            # # 记录排序后的结果
            # self.logger.info("\n排序后的结果:")
            # for idx, item in enumerate(all_scored_items):
            #     self.logger.info(f"排名 {idx + 1}:")
            #     self.logger.info(f"  来源: {item['source']}")
            #     self.logger.info(f"  期刊名称: {item['journal_name']}")
            #     self.logger.info(f"  来源类型: {item['source_type']}")
            #     self.logger.info(f"  标题: {item['title']}")
            #     self.logger.info(f"  内容长度: {len(item['content'])}")
            #     self.logger.info(f"  相似度得分: {item['similarity_score']:.4f}")
            #     self.logger.info(f"  上下文排名得分: {item['context_rank_score']:.4f}")
            #     self.logger.info(f"  来源权威性得分: {item['source_authority_score']:.4f}")
            #     self.logger.info(f"  影响因子: {item['impact_factor']}")
            #     self.logger.info(f"  标准化影响因子得分: {item['normalized_impact_factor']:.4f}")
            #     self.logger.info(f"  总分: {item['score']:.4f}")
            #     self.logger.info("-" * 50)
            
            # 设置阈值或取前N个结果
            threshold = 0.4  # 默认阈值
            max_results = 20  # 最大结果数量
            
            # 应用阈值或取前N个
            filtered_items = [item for item in all_scored_items if item['score'] >= threshold]
            if not filtered_items:  # 如果阈值过滤后没有结果，取排名前N的结果
                filtered_items = all_scored_items[:max_results]
            elif len(filtered_items) > max_results:  # 如果结果过多，限制数量
                filtered_items = filtered_items[:max_results]
            
            # 记录最终选择的结果
            self.logger.info(f"\n最终选择的结果 (阈值: {threshold}, 最大数量: {max_results}):")
            self.logger.info(f"选择的项目数: {len(filtered_items)}")
            
            # 格式化输出，保持与原始格式兼容
            formatted_context = []
            for item in filtered_items:
                formatted_block = (
                    f"Source: {item['source']}\n"
                    f"Title: {item['title']}\n"
                    f"Content: {item['content']}\n"
                )
                formatted_context.append(formatted_block)

            # 将filtered_items转换为分类格式

            #固定初始值格式：列表
            initial_value = []  #

            classified_items = {key: initial_value.copy() for key in source_type_keys}

            for item in filtered_items:
                source = item['source']
                parsed_block = {
                    "source": source,
                    "JournalName": item['journal_name'],
                    "title": item['title'],
                    "content": item['content'],
                    # 加++
                    'impact_factor': item['impact_factor'],
                    'published_date': item.get('published_date', ''),  # 获取发布日期，如果不存在则默认为'',
                    'authors': item.get('authors', ''),
                    'vol_pagination': item.get("vol_pagination")

                }
                
                # if item['source_type'] == "pubmed":
                #     classified_items['pubmed'].append(parsed_block)
                # elif item['source_type'] == "arxiv":
                #     classified_items['arxiv'].append(parsed_block)
                # else:
                #     classified_items['tavily'].append(parsed_block)

                #todo 修+
                #检查来源类型并添加到对应的分类中
                classified_items[item['source_type']].append(parsed_block)
            
            # accumulated_classified_results = accumulated_classified_results
            # # 更新累积的分类结果，保持分类结构
            # self.logger.info("Updating accumulated results")
            # for category in classified_items:
            #     try:
            #         # 检查是否有重复的source
            #         existing_sources = {item['source'] for item in self.researcher.accumulated_classified_results[category]}
            #         # 只添加新的source
            #         new_items = [item for item in classified_items[category] if item['source'] not in existing_sources]
            #         if new_items:
            #             self.logger.info(f"Adding {len(new_items)} new items to {category}")
            #             self.researcher.accumulated_classified_results[category].extend(new_items)
            #     except Exception as e:
            #         self.logger.error(f"Error processing category {category}: {str(e)}")
            #         continue

            #todo 修+(试)
            #更新累积的分类结果，保持分类结构
            self.logger.info("Updating accumulated results")

            for category, context in classified_items.items():
                if not self.researcher.accumulated_classified_results.setdefault(category, None):
                    self.researcher.accumulated_classified_results[category] = []
                le_s = len(self.researcher.accumulated_classified_results[category])
                try:
                    for item_dict in context:
                        if len(self.researcher.accumulated_classified_results[category]) == 0:
                            self.researcher.accumulated_classified_results[category].append(item_dict)
                        else:
                            inf = True
                            for i in range(len(self.researcher.accumulated_classified_results[category])):
                                if self.researcher.accumulated_classified_results[category][i]['source'] == item_dict[
                                    'source']:
                                    self.researcher.accumulated_classified_results[category][i]['content'] += item_dict[
                                        'content']
                                    inf = False
                                    break
                            if inf:
                                self.researcher.accumulated_classified_results[category].append(item_dict)

                    le_e = len(self.researcher.accumulated_classified_results[category])

                    self.logger.info(f"Added {le_e - le_s} new items to {category}")

                except Exception as e:
                    self.logger.error(f"Error processing category {category}: {str(e)}")
                    continue


            # 记录当前累积的结果
            self.logger.info("Current accumulated results:")
            for category in self.researcher.accumulated_classified_results:
                self.logger.info(f"{category}: {len(self.researcher.accumulated_classified_results[category])} items")

            # # 测试用：
            # for category,item_list in self.researcher.accumulated_classified_results.items():
            #     self.logger.info(f"展示搜索引擎{category}: {len(item_list)} items，获取的内容："+"\n\n")
            #     for item in item_list:
            #         self.logger.info(f"{item}"+"\n")


            # 转换为JSON字符串
            self.logger.info("Converting to JSON")
            classified_json = json.dumps(self.researcher.accumulated_classified_results, ensure_ascii=False, indent=2)
            self.logger.info(f"分类结果: {classified_json}") #
            await stream_output(
                    "logs", "subquery_context_window", f"{classified_json}", self.researcher.websocket
                )
            if formatted_context:
                combined_context = " ".join(formatted_context)
                self.logger.info(f"最终组合上下文大小: {len(combined_context)}")
                return combined_context
            return []
            
        except Exception as e:
            self.logger.error(f"Error during web search: {e}", exc_info=True)
            import traceback
            self.logger.error(traceback.format_exc())
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
            
            # 解析内容块并添加检索器类型信息
            if content:
                content_blocks = re.split(r'\n(?=Source: https?://)', content.strip())
                processed_blocks = []
                
                for block in content_blocks:
                    if not block.strip():
                        continue
                        
                    # 解析块内容
                    source_match = re.search(r'^Source: (https?://[^\s]+)', block, re.M)
                    title_match = re.search(r'Title: (.+)', block)
                    content_match = re.search(r'Content: (.+)', block, re.DOTALL)
                    
                    if not all([source_match, title_match, content_match]):
                        continue
                        
                    url = source_match.group(1)
                    title = title_match.group(1).strip()
                    content_text = content_match.group(1).strip()
                    
                    # 从scraped_data中获取检索器类型
                    retriever_type = "tavily"  # 默认类型
                    for item in scraped_data:
                        if item.get('url') == url and 'retriever_type' in item:
                            retriever_type = item['retriever_type']
                            self.logger.info(f"从scraped_data中获取到检索器类型: {retriever_type}")
                            break


                    # 加++
                    # 从scraped_data中获取出版时间信息
                    published_date = ""  # 默认出版时间为空
                    for item in scraped_data:
                        if item.get('url') == url and 'published_date' in item:
                            published_date = item['published_date']
                            self.logger.info(f"从scraped_data中获取到出版时间: {published_date}")
                            break

                    # 从scraped_data中获取作者信息
                    authors = ""  # 默认作者为空
                    for item in scraped_data:
                        if item.get('url') == url and 'authors' in item:
                            authors = item['authors']
                            self.logger.info(f"从scraped_data中获取到作者信息: {authors}")
                            break

                    if authors and isinstance(authors, list):  # 如果作者信息存在,
                        author_names = ""
                        # 提取作者的名字
                        for idx in range(len(authors)):
                            if idx > authors_t - 1:
                                break
                            author_names += authors[idx] + ", "
                        authors = author_names[:-2] + ", et al."
                        self.logger.info(f"最后提取到的作者信息: {authors}")

                    # 从scraped_data中获取卷号信息
                    vol = ""  # 默认卷号为空
                    for item in scraped_data:
                        if item.get('url') == url and 'vol' in item:
                            vol = item['vol']
                            self.logger.info(f"从scraped_data中获取到卷号信息: {vol}")
                            break

                    # 从scraped_data中获取页码信息
                    pagination = ""  # 默认页码为空
                    for item in scraped_data:
                        if item.get('url') == url and 'pagination' in item:
                            pagination = item['pagination']
                            self.logger.info(f"从scraped_data中获取到页码信息: {pagination}")
                            break

                    # 修++
                    # 构建新的内容块，包含检索器类型信息
                    processed_block = (
                        f"Source: {url}\n"
                        f"Title: {title}\n"
                        f"Content: {content_text}\n"
                        f"RetrieverType: {retriever_type}\n"
                        f"Published Date: {published_date}\n"
                        f"Volume: {vol}\n"
                        f"Pagination: {pagination}\n"
                        f"Authors: {authors}\n"

                    )
                    processed_blocks.append(processed_block)
                
                # 重新组合处理后的内容
                content = "\n".join(processed_blocks)

            if content and self.researcher.verbose:
                print(f"Content found 12345")
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
        search_results_with_type = []  # 存储带检索器类型的结果

        # 遍历所有检索器
        for retriever_class in self.researcher.retrievers:
            # 获取当前检索器类型
            current_retriever_type = retriever_class.__name__.lower() 
            # 映射检索器类型
            retriever_type_mapping = {
                'tavilysearch': 'tavily',
                'arxivsearch': 'arxiv',
                'pubmeddiansearch': 'pubmed'  # 使用小写形式
            }
            current_retriever_type = retriever_type_mapping.get(current_retriever_type, 'tavily')
            self.logger.info(f"Current retriever type: {current_retriever_type} (original: {retriever_class.__name__})")
            
            # 根据检索器类型处理查询
            processed_query = query
            if current_retriever_type == "pubmed":
                # 对 PubMed 检索器进行特殊处理
                self.logger.info("Processing query for PubMed retriever")
                # 使用 generate_pubmed_sub_queries 生成规范化的查询
                sub_queries = await generate_pubmed_sub_queries(
                    query=query,
                    cfg=self.researcher.cfg,
                    cost_callback=self.researcher.add_costs
                )
                # 使用第一个子查询作为主要查询
                if sub_queries:
                    processed_query = sub_queries[0]
                    self.logger.info(f"Processed PubMed query: {processed_query}")
                else:
                    self.logger.warning("No PubMed sub-queries generated, skipping PubMed search")
                    continue  # 如果没有生成子查询，跳过当前检索器
            
            # 实例化当前检索器
            retriever = retriever_class(processed_query, query_domains=query_domains)
            self.logger.info(f"*****use retriever: {current_retriever_type} search query: {processed_query}")
            # 执行搜索
            search_results = await asyncio.to_thread(
                retriever.search, max_results=self.researcher.cfg.max_search_results_per_query
            )
            self.logger.info(f"*****use retriever 2: {current_retriever_type}")
            ceshi_c = 0
            # 为搜索结果添加检索器类型标识
            for result in search_results:
                result['retriever_type'] = current_retriever_type  # 添加检索器类型字段
                search_results_with_type.append(result)  # 将结果添加到带检索器类型的结果列表中
                ceshi_c += 1
                self.logger.info(f"测试检索器：<{ceshi_c}>{current_retriever_type} \n 搜索结果：{result['href']} ")
                # self.logger.info(f"测试pubmed搜索的字段是否存在：0：{result.get('href')}")
                # self.logger.info(f"测试pubmed搜索的字段是否存在：1：{result.get('published')}")
                # self.logger.info(f"测试pubmed搜索的字段是否存在：2：{result.get('pagination')}")
                # self.logger.info(f"测试pubmed搜索的字段是否存在：3：{result.get('authors')}")
                # self.logger.info(f"测试pubmed搜索的字段是否存在：4：{result.get('vol')}")
                # self.logger.info(f"测试pubmed搜索的字段是否存在：5：{result.get('title')}")

            # 收集URL
            search_urls = [url.get("href") for url in search_results]
            new_search_urls.extend(search_urls)

        # 获取新的URL并随机打乱
        new_search_urls = await self._get_new_urls(new_search_urls)
        random.shuffle(new_search_urls)

        # 返回URL列表和带检索器类型的结果
        return new_search_urls, search_results_with_type

    async def _scrape_data_by_urls(self, sub_query, query_domains: list = []):
        """
        Runs a sub-query across multiple retrievers and scrapes the resulting URLs.

        Args:
            sub_query (str): The sub-query to search for.

        Returns:
            list: A list of scraped content results.
        """
        # 获取URL列表和带检索器类型的结果
        new_search_urls, search_results_with_type = await self._search_relevant_source_urls(sub_query, query_domains)

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
        
        # 记录抓取结果
        self.logger.info(f"抓取结果数量: {len(scraped_content)}")
        
        # 创建URL到检索器类型的映射
        url_to_type = {}

        # 加++
        # 初始化其他字段的映射
        url_to_published_date = {}
        url_to_authors = {}
        url_to_vol = {}
        url_to_pagination = {}

        for result in search_results_with_type:
            if 'href' in result and 'retriever_type' in result:
                normalized_url = self._normalize_url(result['href'])
                url_to_type[normalized_url] = result['retriever_type']
                self.logger.info(f"添加检索器<-URL映射:{result['retriever_type']} <-- {normalized_url}")

                ## 加++修++
                # if result['retriever_type'] == 'pubmed':  # 如果检索器类型是pubmed
                self.logger.info(f"pubmed检索器类型, 开始处理pubmed字段")
                if 'published' in result:  # 如果结果中包含发布日期字段
                    url_to_published_date[normalized_url] = result['published']  # 添加到映射中
                    self.logger.info(f"添加发布日期映射: {normalized_url} -> {result['published']}")
                if 'authors' in result:  # 如果结果中包含作者字段
                    url_to_authors[normalized_url] = result['authors']  # 添加到映射中
                    self.logger.info(f"添加作者映射: {normalized_url} -> {result['authors']}")
                if 'vol' in result:  # 如果结果中包含卷号字段
                    url_to_vol[normalized_url] = result['vol']  # 添加到映射中
                    self.logger.info(f"添加卷号映射: {normalized_url} -> {result['vol']}")
                if 'pagination' in result:  # 如果结果中包含分页信息字段
                    url_to_pagination[normalized_url] = result['pagination']  # 添加到映射中
                    self.logger.info(f"添加分页信息映射: {normalized_url} -> {result['pagination']}")


        self.logger.info(f"发布时间的映射字典的长度是：{len(url_to_published_date)}")

        # 为每个内容添加检索器类型信息
        for content in scraped_content:
            if 'url' in content:
                content_url = content['url']
                normalized_content_url = self._normalize_url(content_url)
                self.logger.info(f"处理URL: {content_url}")
                self.logger.info(f"标准化后的URL: {normalized_content_url}")
                
                # 首先尝试从映射中获取类型
                if normalized_content_url in url_to_type:
                    content['retriever_type'] = url_to_type[normalized_content_url]
                    self.logger.info(f"从URL映射中获取到检索器类型: {content['retriever_type']}")
                else:
                    # 如果没有找到匹配的检索器类型，使用URL特征判断
                    if 'arxiv' in normalized_content_url:
                        content['retriever_type'] = 'arxiv'
                        self.logger.info(f"根据URL特征判断为arxiv")
                    elif 'ncbi' in normalized_content_url or 'pubmed' in normalized_content_url:
                        content['retriever_type'] = 'pubmed'
                        self.logger.info(f"根据URL特征判断为pubmed")
                    else:
                        content['retriever_type'] = 'tavily'
                        self.logger.info(f"根据URL特征判断为tavily")

                ## 修++
                # 为每个内容添加:1. 出版时间 published_date 2. 期刊卷号 vol  3. 作者信息 authors 4. 分页信息 pagination
                if normalized_content_url in url_to_published_date:  # 如果URL在发布日期映射中
                    content['published_date'] = url_to_published_date[normalized_content_url]  # 添加发布日期字段
                    self.logger.info(f"添加发布日期: {content['published_date']}")
                if normalized_content_url in url_to_authors:  # 如果URL在作者映射中
                    content['authors'] = url_to_authors[normalized_content_url]  # 添加作者字段
                    self.logger.info(f"添加作者: {content['authors']}")
                if normalized_content_url in url_to_vol:  # 如果URL在卷号映射中
                    content['vol'] = url_to_vol[normalized_content_url]  # 添加卷号字段
                    self.logger.info(f"添加卷号: {content['vol']}")
                if normalized_content_url in url_to_pagination:  # 如果URL在分页信息映射中
                    content['pagination'] = url_to_pagination[normalized_content_url]  # 添加分页信息字段
                    self.logger.info(f"添加分页信息: {content['pagination']}")


        if self.researcher.vector_store:
            self.researcher.vector_store.load(scraped_content)

        return scraped_content

    def _normalize_url(self, url):
        """标准化URL以便进行匹配"""
        if not url:
            return ""
        # 移除URL中的协议前缀
        url = re.sub(r'^https?://', '', url)
        # 移除尾部斜杠
        url = url.rstrip('/')
        # 移除URL参数
        url = url.split('?')[0]
        # 移除锚点
        url = url.split('#')[0]
        return url.lower()

    async def _normalize_journal_name(self, name):
        """标准化期刊名称以提高匹配成功率"""
        if not name:
            return name
            
        # 解码HTML实体
        normalized = html.unescape(name)
        
        # 标准化空格
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # 标准化连字符和常见替换
        normalized = normalized.replace(' - ', '-')
        normalized = normalized.replace(' And ', ' & ')
        normalized = normalized.replace(' and ', ' & ')
        normalized = normalized.replace('&amp;', '&')
        normalized = normalized.replace('&AMP;', '&')
        
        # 处理常见的期刊名称变体
        replacements = {
            'J ': 'Journal ',
            'Intl': 'International',
            'Int ': 'International ',
            'Int.': 'International',
            'Rev ': 'Review ',
            'Rev.': 'Review',
            'Sci ': 'Science ',
            'Sci.': 'Science',
            'Adv ': 'Advances ',
            'Adv.': 'Advances',
            'Res ': 'Research ',
            'Res.': 'Research',
            'Chem ': 'Chemistry ',
            'Chem.': 'Chemistry',
            'Biol ': 'Biology ',
            'Biol.': 'Biology',
            'Med ': 'Medicine ',
            'Med.': 'Medicine',
        }
        
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        
        return normalized

    async def _extract_journal_info_from_url(self, url):
        """从URL中提取期刊信息，优先通过DOI识别"""
        journal_info = {"journal_name": None}
        
        try:
            # 首先尝试直接从URL提取DOI
            doi_match = re.search(r'(10\.\d{4,}[\/.][\w\.\-\/]+)', url)
            
            # 如果URL中没有DOI，尝试访问页面获取DOI
            if not doi_match:
                content = await self._fetch_url_content(url)
                if content:
                    # 尝试从HTML中提取DOI
                    soup = BeautifulSoup(content, "html.parser")
                    
                    # 检查各种可能包含DOI的元数据标签
                    meta_doi = soup.find("meta", {"name": "citation_doi"}) or \
                            soup.find("meta", {"name": "dc.identifier"}) or \
                            soup.find("meta", {"name": "DC.Identifier"}) or \
                            soup.find("meta", {"scheme": "doi"})
                    
                    if meta_doi and meta_doi.get("content"):
                        doi = meta_doi.get("content")
                        if doi.startswith("doi:"):
                            doi = doi[4:]
                        doi_match = re.match(r'(10\.\d{4,}[\/.][\w\.\-\/]+)', doi)
                    else:
                        # 尝试从正文中查找DOI
                        doi_regex = r'(?:doi|DOI):\s*(10\.\d{4,}[\/.][\w\.\-\/]+)'
                        content_match = re.search(doi_regex, content)
                        if content_match:
                            doi_match = re.match(r'(10\.\d{4,}[\/.][\w\.\-\/]+)', content_match.group(1))
            
            # 如果找到DOI，通过CrossRef API获取期刊信息
            if doi_match:
                doi = doi_match.group(1)
                journal_info["doi"] = doi
                self.logger.info(f"Found DOI: {doi}")
                
                # 调用CrossRef API
                try:
                    crossref_api_url = f"https://api.crossref.org/works/{doi}"
                    response = await self._fetch_url_content(crossref_api_url)
                    
                    if response:
                        data = json.loads(response)
                        if "message" in data:
                            message = data["message"]
                            
                            # 提取期刊名称
                            if "container-title" in message and message["container-title"]:
                                journal_info["journal_name"] = message["container-title"][0]
                                self.logger.info(f"Found journal name from CrossRef: {journal_info['journal_name']}")
                            
                            # 提取ISSN (可用于进一步匹配期刊影响因子)
                            if "ISSN" in message and message["ISSN"]:
                                journal_info["issn"] = message["ISSN"][0]
                                
                            # 提取出版商信息
                            if "publisher" in message:
                                journal_info["publisher"] = message["publisher"]
                                
                            return journal_info
                except Exception as e:
                    self.logger.warning(f"Error fetching CrossRef metadata: {e}")
            
            # 如果通过DOI无法获取，回退到基于URL的方法
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            path = parsed_url.path
            
            # 处理常见学术网站的URL模式
            # arXiv
            if "arxiv.org" in domain:
                journal_info["journal_name"] = "arXiv"
                return journal_info
                
            # PubMed/PMC
            if "pubmed.ncbi.nlm.nih.gov" in domain or "pmc.ncbi.nlm.nih.gov" in domain:
                if "pubmed.ncbi.nlm.nih.gov" in domain:
                    pmid_match = re.search(r'pubmed\.ncbi\.nlm\.nih\.gov\/(\d+)', url)
                    if pmid_match:
                        pmid = pmid_match.group(1)
                        try:
                            api_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={pmid}&retmode=json"
                            response = await self._fetch_url_content(api_url)
                            if response:
                                data = json.loads(response)
                                if 'result' in data and pmid in data['result']:
                                    result = data['result'][pmid]
                                    if 'fulljournalname' in result:
                                        journal_info["journal_name"] = result['fulljournalname']
                                    elif 'source' in result:
                                        journal_info["journal_name"] = result['source']
                        except Exception as e:
                            self.logger.warning(f"Error fetching PubMed metadata: {e}")
                return journal_info
            
            # Nature
            if "nature.com" in domain:
                if "nature.com/articles/s41586" in url:
                    journal_info["journal_name"] = "Nature"
                elif "nature.com/articles/s41467" in url:
                    journal_info["journal_name"] = "Nature Communications"
                elif re.search(r'nature\.com\/([^\/]+)\/journal', url):
                    journal_match = re.search(r'nature\.com\/([^\/]+)\/journal', url)
                    if journal_match:
                        journal_name = journal_match.group(1).replace('-', ' ').title()
                        journal_info["journal_name"] = f"Nature {journal_name}"
                else:
                    # 尝试从s开头的DOI风格路径中提取期刊标识符
                    s_match = re.search(r'nature\.com\/articles\/(s\d+)', url)
                    if s_match:
                        # Nature期刊DOI前缀映射表
                        nature_prefixes = {
                            's41586': 'Nature',
                            's41467': 'Nature Communications',
                            's41598': 'Scientific Reports',
                            's41587': 'Nature Biotechnology',
                            's41591': 'Nature Medicine',
                            's41593': 'Nature Neuroscience',
                            's41594': 'Nature Structural & Molecular Biology',
                            's41561': 'Nature Geoscience',
                            's41563': 'Nature Materials',
                            's41565': 'Nature Nanotechnology',
                            's41566': 'Nature Photonics',
                            's41567': 'Nature Physics',
                            's41577': 'Nature Reviews Immunology',
                            's41573': 'Nature Reviews Drug Discovery',
                            's41574': 'Nature Reviews Endocrinology',
                            's41575': 'Nature Reviews Gastroenterology & Hepatology',
                            's41576': 'Nature Reviews Genetics',
                            's41577': 'Nature Reviews Immunology',
                            's41578': 'Nature Reviews Materials',
                            's41579': 'Nature Reviews Microbiology',
                            's41580': 'Nature Reviews Molecular Cell Biology',
                            's41581': 'Nature Reviews Nephrology',
                            's41582': 'Nature Reviews Neurology',
                            's41584': 'Nature Reviews Rheumatology',
                            's41571': 'Nature Reviews Clinical Oncology',
                            's42255': 'Nature Metabolism',
                            's42256': 'Nature Machine Intelligence',
                            's41567': 'Nature Physics',
                            's41557': 'Nature Chemistry',
                            's41558': 'Nature Climate Change',
                            's41559': 'Nature Ecology & Evolution',
                            's41560': 'Nature Energy',
                            's41564': 'Nature Microbiology',
                            's41570': 'Nature Reviews Chemistry',
                            's41589': 'Nature Chemical Biology',
                            's41592': 'Nature Methods',
                            's41596': 'Nature Protocols',
                            's41597': 'Scientific Data',
                            's41598': 'Scientific Reports',
                            's41699': 'Communications Biology',
                            's41746': 'npj Digital Medicine',
                            's42003': 'Communications Biology',
                        }
                        
                        s_id = s_match.group(1)
                        for prefix, journal in nature_prefixes.items():
                            if s_id.startswith(prefix):
                                journal_info["journal_name"] = journal
                                break
                
                return journal_info
            
            # BioMedCentral
            if "biomedcentral.com" in domain:
                journal_subdomain = domain.split(".biomedcentral.com")[0]
                bmc_journals = {
                    "ann-clinmicrob": "Annals of Clinical Microbiology and Antimicrobials",
                    "bmcgenomics": "BMC Genomics",
                    "bmcinfectdis": "BMC Infectious Diseases",
                    "bmcmicrobiol": "BMC Microbiology",
                    "genomemedicine": "Genome Medicine",
                    "translationalmedicine": "Journal of Translational Medicine",
                    "bmcbiol": "BMC Biology",
                    "bmccancer": "BMC Cancer",
                    "bmcneurosci": "BMC Neuroscience",
                    "bmcpsychiatry": "BMC Psychiatry",
                    "jnanobiotechnology": "Journal of Nanobiotechnology",
                    "mbio": "mBio",
                    "microbiome": "Microbiome",
                    "parasitesandvectors": "Parasites & Vectors",
                    "retrovirology": "Retrovirology",
                    "virologyj": "Virology Journal",
                }
                
                if journal_subdomain in bmc_journals:
                    journal_info["journal_name"] = bmc_journals[journal_subdomain]
                
                return journal_info
            
            # Lancet
            if "thelancet.com" in domain:
                if "thelancet.com/journals/lancet" in url:
                    journal_info["journal_name"] = "The Lancet"
                else:
                    lancet_match = re.search(r'thelancet\.com\/journals\/([^\/]+)', url)
                    if lancet_match:
                        journal_code = lancet_match.group(1)
                        lancet_journals = {
                            "laninf": "The Lancet Infectious Diseases",
                            "lanmic": "The Lancet Microbe",
                            "lanepe": "The Lancet Regional Health - Europe",
                            "lanpsy": "The Lancet Psychiatry",
                            "landia": "The Lancet Diabetes & Endocrinology",
                            "langas": "The Lancet Gastroenterology & Hepatology",
                            "lanhae": "The Lancet Haematology",
                            "lanhiv": "The Lancet HIV",
                            "laneur": "The Lancet Neurology",
                            "lanplh": "The Lancet Planetary Health",
                            "lanpub": "The Lancet Public Health",
                            "lanres": "The Lancet Respiratory Medicine",
                            "lanrhe": "The Lancet Rheumatology",
                            "lanonc": "The Lancet Oncology",
                            "lancet": "The Lancet",
                            "lanchi": "The Lancet Child & Adolescent Health",
                            "landig": "The Lancet Digital Health",
                            "eclinm": "EClinicalMedicine",
                        }
                        if journal_code in lancet_journals:
                            journal_info["journal_name"] = lancet_journals[journal_code]
                
                return journal_info
            
            # MDPI
            if "mdpi.com" in domain:
                mdpi_match = re.search(r'mdpi\.com\/journal\/([^\/]+)', url)
                if mdpi_match:
                    journal_name = mdpi_match.group(1).replace('-', ' ').title()
                    journal_info["journal_name"] = journal_name
                else:
                    # 尝试从URL提取ISSN
                    issn_match = re.search(r'mdpi\.com\/(\d{4}-\d{3}[\dX])', url)
                    if issn_match:
                        journal_info["issn"] = issn_match.group(1)
                
                return journal_info
            
            # Frontiers
            if "frontiersin.org" in domain:
                frontiers_match = re.search(r'frontiersin\.org\/journals\/([^\/]+)', url)
                if frontiers_match:
                    journal_slug = frontiers_match.group(1).replace('-', ' ')
                    journal_info["journal_name"] = f"Frontiers in {journal_slug.title()}"
                
                return journal_info
            
            # RSC (Royal Society of Chemistry)
            if "rsc.org" in domain or "pubs.rsc.org" in domain:
                rsc_match = re.search(r'\/([^\/]+)\/article', url)
                if rsc_match:
                    journal_code = rsc_match.group(1)
                    rsc_journals = {
                        "c0": "Chemical Communications",
                        "cc": "Chemical Communications",
                        "cs": "Chemical Science",
                        "dt": "Dalton Transactions",
                        "gc": "Green Chemistry",
                        "cp": "Chemical Physics",
                        "sc": "Sustainable Chemistry",
                        "an": "Analyst",
                        "ce": "Chemical Education",
                        "cy": "Catalysis Science & Technology",
                        "ee": "Energy & Environmental Science",
                        "en": "Environmental Science",
                        "fd": "Faraday Discussions",
                        "lc": "Lab on a Chip",
                        "nr": "Nanoscale",
                        "ob": "Organic & Biomolecular Chemistry",
                        "py": "Polymer Chemistry",
                        "ra": "RSC Advances",
                        "sm": "Soft Matter",
                    }
                    if journal_code in rsc_journals:
                        journal_info["journal_name"] = rsc_journals[journal_code]
                
                return journal_info
            
            # 如果仍无法确定期刊，尝试从内容中提取
            if not journal_info["journal_name"] and not content:
                content = await self._fetch_url_content(url)
                
            if content and not journal_info["journal_name"]:
                soup = BeautifulSoup(content, "html.parser")
                
                # 尝试从元数据中提取期刊名
                journal_meta = soup.find("meta", {"name": "citation_journal_title"}) or \
                            soup.find("meta", {"name": "prism.publicationName"}) or \
                            soup.find("meta", {"name": "dc.source"}) or \
                            soup.find("meta", {"property": "og:site_name"})
                            
                if journal_meta and journal_meta.get("content"):
                    journal_info["journal_name"] = journal_meta.get("content")
                    self.logger.info(f"Found journal name from HTML metadata: {journal_info['journal_name']}")
        # 标准化期刊名称
            if journal_info["journal_name"]:
                journal_info["journal_name"] = await self._normalize_journal_name(journal_info["journal_name"])
                self.logger.info(f"Normalized journal name: {journal_info['journal_name']}")
                              
            return journal_info
            
        except Exception as e:
            self.logger.warning(f"Error extracting journal info from URL: {e}")
            return journal_info

    async def _fetch_url_content(self, url, timeout=10):
        """获取URL内容的辅助函数"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = await asyncio.to_thread(
                lambda: requests.get(url, headers=headers, timeout=timeout)
            )
            
            if response.status_code == 200:
                return response.text
            return None
        except Exception as e:
            self.logger.warning(f"Error fetching URL {url}: {e}")
            return None


    #! 加++
    def _extract_arxiv_info_from_url(self, url):
        """从arXiv URL中提取信息"""
        try:
            arxiv_info = {}
            # 从URL中提取arXiv ID
            arxiv_id_match = re.search(r'arxiv\.org/pdf/(\d{4})\.(.*?)$', url)
            if not arxiv_id_match:
                arxiv_id_match = re.search(r'\.([^./]+)$', url)  # 匹配最后一个点后非斜杠内容
                arxiv_id = arxiv_id_match.group(1)
                if not arxiv_id_match:
                    arxiv_id = "Unknown document"  # 无法提取arxiv ID，返回空字符串
            else:
                arxiv_id = arxiv_id_match.group(2)

            arxiv_info["arxiv_id"] = arxiv_id
            arxiv_info["published_date"] = self.extract_and_convert_arxiv_date(url)

            return arxiv_info

        except Exception as e:
            self.logger.warning(f"Error extracting arXiv info from URL: {e}")

    def extract_and_convert_arxiv_date(self,url: str) -> str:
        """
        从arXiv链接中提取日期标记并转换为期待日期格式

        参数:
        url (str): arXiv论文链接，如"http://arxiv.org/pdf/2410.15367v1"

        返回:
        str: 期待日期格式：如"2024.10"
        """
        # 使用正则表达式匹配arxiv ID中的日期部分（如2410或1704）
        match = re.search(r'arxiv\.org/pdf/(\d{2})(\d{2})', url, re.IGNORECASE)
        if not match:
            match = re.search(r'\.(\d{2})(\d{2})', url[::-1])
            if match:
                digits_m = match.group(1)[::-1]
                digits_y = match.group(2)[::-1]
                digits = digits_y + digits_m  # 合并年和月
            else:
                return ""  # 无法提取日期，返回原URL
        else:
            # 提取年份和月份
            digits_y, digits_m = match.groups()

        # 将2位年份转换为4位（假设2000年之后）
        year = int(digits_y)
        year_full = 2000 + year if year < 50 else 1900 + year

        # 构建期待日期
        try:
            # 确保月份在有效范围（1-12）
            month = int(digits_m)
            if 1 <= month <= 12:
                # return f"{year_full}.{month:02d}"  # 直接格式化字符串
                return f"{year_full}"
            else:
                return f"{year_full}"  # 无效月份，只发年份
        except ValueError:
            return f"{year_full}-{digits_m.zfill(2)}"  # 异常处理，保持原始数字