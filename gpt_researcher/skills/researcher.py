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
from ..actions.query_processing import plan_research_outline, get_search_results
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

        # 获取当前检索器的类型
        retriever_type = self.researcher.cfg.retrievers[0]
        self.logger.info(f"Current retriever type: {retriever_type}")
        
        # 检查是否是PubMed检索器
        if retriever_type == "pubmed_dian":
            self.logger.info(f"Using PubMed Dian retriever")

        outline = await plan_research_outline(
            query=query,
            search_results=search_results,
            agent_role_prompt=self.researcher.role,
            cfg=self.researcher.cfg,
            parent_query=self.researcher.parent_query,
            report_type=self.researcher.report_type,
            cost_callback=self.researcher.add_costs,
            retriever_type=retriever_type  # 传递检索器类型
        )
        self.logger.info(f"Research outline planned: {outline}")
        return outline

    async def conduct_research(self):
        """Runs the GPT Researcher to conduct research"""
        from langdetect import detect
        async def is_english(text: str) -> bool:
            """检测文本是否为英文"""
            try:
                # 快速检查中/日/韩文字符
                if re.search(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7a3]', text):
                    return False
                return detect(text) == 'en'
            except Exception as e:
                print(f"Error detecting language: {e}")
                return False

        async def translate_to_english(text: str) -> str:
            """使用 Qwen 模型翻译文本"""
            try:
                qwen_local = GenericLLMProvider.from_provider("qwen_local")
                prompt = f"请将以下内容准确翻译成英文（保留专业术语）：\n\n{text}\n\n只需返回翻译结果，不要添加任何解释。"
                messages = [{"role": "user", "content": prompt}]
                response = await qwen_local.get_chat_response(messages, stream=False)
                return response.strip()
            except Exception as e:
                print(f"Translation error: {e}")
                return text  # 失败时返回原文

        async def main(query):
            if not await is_english(query):
                print(f"Translating non-English query: {query}")
                translated = await translate_to_english(query)
                print(f"Translated to: {translated}")
                return translated
            else:
                print("Query is already in English.")
                return query

        # 运行主函数
        self.researcher.query = await main(self.researcher.query)

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
            "pubmed": 5,
            "arxiv": 3,
            "tavily": 2
        }
        return source_scores.get(source.lower(), 0)

<<<<<<< HEAD
=======

>>>>>>> e3607eead185b24851ddf9082210abea2f9137c8
    async def _get_context_by_web_search(self, query, scraped_data: list = [], query_domains: list = []):
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
                self.logger.info(f"子查询 {idx + 1} - '{sub_queries[idx]}' 的结果:")
                self.logger.info(f"内容长度: {len(content)} 字符")
                self.logger.info("-" * 50)
        
        # 处理所有子查询并计算得分
        try:
            all_scored_items = []
            for idx, sub_query in enumerate(sub_queries):
                if not context[idx]:
                    continue
                    
                # 计算主查询和子查询的相似度
                query_similarity = await self._calculate_query_similarity(query, sub_query)
                
                # 解析上下文内容
                content_blocks = re.split(r'\n(?=Source: https?://)', context[idx].strip())
                
                # 处理每个内容块
                for block_idx, block in enumerate(content_blocks):
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
                    
                    # 确定来源类型
                    source_type = "tavily"
                    if 'arxiv' in url.lower():
                        source_type = "arxiv"
                    # elif 'ncbi' in url.lower() or 'pubmed' in url.lower():
                    elif self.researcher.cfg.retrievers[0] == "pubmed_dian":
                        source_type = "pubmed"
                    
                    # 计算来源权威性得分
                    source_authority_score = self._calculate_source_authority_score(source_type)
                    
                    # 计算上下文排序得分
                    context_rank_score = self._calculate_context_rank_score(len(content_blocks), block_idx)
                    
                    if source_type == "pubmed" or source_type == "tavily":

                        # 提取期刊信息
                        journal_info = await self._extract_journal_info_from_url(url)
                        journal_name = journal_info.get("journal_name", "")
                        impact_factor = 0
                        
                        # 查找期刊影响因子
                        if journal_df is not None and journal_name:
                            try:
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
                                    issn = journal_info['issn']
                                    journal_match = journal_df[(journal_df['ISSN'] == issn) | (journal_df['EISSN'] == issn)]
                                
                                if not journal_match.empty:
                                    impact_factor = float(journal_match.iloc[0]['JIF']) if 'JIF' in journal_match.columns and not pd.isna(journal_match.iloc[0]['JIF']) else 0
                                    self.logger.info(f"期刊 '{journal_name}' 的影响因子 (JIF): {impact_factor}")
                            except Exception as e:
                                self.logger.warning(f"期刊影响因子查询错误: {e}")
                    else:
                        journal_name = ""
                        impact_factor = 0   
                    # 标准化影响因子得分到0-1范围
                    # 假设最高影响因子为100（可根据实际情况调整）
                    max_impact_factor = 503.1
                    normalized_impact_factor = min(impact_factor / max_impact_factor, 1.0)
                    
                    # 计算总得分 (w1=0.1, w2=0.2, w3=0.2, w4=0.5)
                    total_score = (
                        0.1 * query_similarity +       # 主查询与子查询相关度
                        0.2 * context_rank_score +     # 上下文排序得分
                        0.2 * source_authority_score + # 来源权威性得分
                        0.5 * normalized_impact_factor # 期刊影响因子得分
                    )
                    
                    all_scored_items.append({
                        'content': content_text,
                        'source': url,
                        'title': title,
                        'journal_name': journal_name,
                        'source_type': source_type,
                        'similarity_score': query_similarity,
                        'context_rank_score': context_rank_score,
                        'source_authority_score': source_authority_score,
                        'impact_factor': impact_factor,
                        'normalized_impact_factor': normalized_impact_factor,
                        'score': total_score
                    })
            
            # 记录排序前的得分情况
            self.logger.info("\n排序前的得分情况:")
            for idx, item in enumerate(all_scored_items):
                self.logger.info(f"项目 {idx + 1}:")
                self.logger.info(f"  来源: {item['source']}")
                self.logger.info(f"  期刊名称: {item['journal_name']}")
                self.logger.info(f"  来源类型: {item['source_type']}")
                self.logger.info(f"  标题: {item['title']}")
                self.logger.info(f"  内容长度: {len(item['content'])}")
                self.logger.info(f"  相似度得分: {item['similarity_score']:.4f}")
                self.logger.info(f"  上下文排名得分: {item['context_rank_score']:.4f}")
                self.logger.info(f"  来源权威性得分: {item['source_authority_score']:.4f}")
                self.logger.info(f"  影响因子: {item['impact_factor']}")
                self.logger.info(f"  标准化影响因子得分: {item['normalized_impact_factor']:.4f}")
                self.logger.info(f"  总分: {item['score']:.4f}")
                self.logger.info("-" * 50)
            
            # 按得分排序
            all_scored_items.sort(key=lambda x: x['score'], reverse=True)
            
            # 记录排序后的结果
            self.logger.info("\n排序后的结果:")
            for idx, item in enumerate(all_scored_items):
                self.logger.info(f"排名 {idx + 1}:")
                self.logger.info(f"  来源: {item['source']}")
                self.logger.info(f"  期刊名称: {item['journal_name']}")
                self.logger.info(f"  来源类型: {item['source_type']}")
                self.logger.info(f"  标题: {item['title']}")
                self.logger.info(f"  内容长度: {len(item['content'])}")
                self.logger.info(f"  相似度得分: {item['similarity_score']:.4f}")
                self.logger.info(f"  上下文排名得分: {item['context_rank_score']:.4f}")
                self.logger.info(f"  来源权威性得分: {item['source_authority_score']:.4f}")
                self.logger.info(f"  影响因子: {item['impact_factor']}")
                self.logger.info(f"  标准化影响因子得分: {item['normalized_impact_factor']:.4f}")
                self.logger.info(f"  总分: {item['score']:.4f}")
                self.logger.info("-" * 50)
            
            # 设置阈值或取前N个结果
<<<<<<< HEAD
            # threshold = 0.4  # 默认阈值
            # max_results = 10  # 最大结果数量
            threshold = 0.0  # 默认阈值
            max_results = 40  # 最大结果数量
=======
            threshold = 0.4  # 默认阈值
            max_results = 10  # 最大结果数量
>>>>>>> e3607eead185b24851ddf9082210abea2f9137c8
            
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
                    # f"Journal Name: {item['journal_name']}\n"
                    f"Title: {item['title']}\n"
                    f"Content: {item['content']}\n"
                )
                formatted_context.append(formatted_block)
            
<<<<<<< HEAD

            
            # 将filtered_items转换为分类格式
            classified_items = {
                "arxiv": [],
                "pubmed": [],
                "tavily": []
            }

            for item in filtered_items:
                source = item['source']
                parsed_block = {
                    "source": source,
                    "JournalName": item['journal_name'],
                    "title": item['title'],
                    "content": item['content']
                }
                
                # 使用相同的分类逻辑
                if item['source_type'] == "pubmed":
                    classified_items['pubmed'].append(parsed_block)
                elif 'arxiv' in source.lower():
                    classified_items['arxiv'].append(parsed_block)
                else:
                    classified_items['tavily'].append(parsed_block)

                # if 'arxiv' in source.lower():
                #     classified_items['arxiv'].append(parsed_block)
                # elif 'ncbi' in source.lower():
                #     classified_items['pubmed'].append(parsed_block)
                # else:
                #     classified_items['tavily'].append(parsed_block)


                # 来源类型: item['source_type']

            # 移除空分类
            classified_items = {key: value for key, value in classified_items.items() if value}

            # 转换为JSON字符串
            classified_json = json.dumps(classified_items, ensure_ascii=False, indent=2)
            await stream_output(
                    "logs", "subquery_context_window", f"{classified_json}", self.researcher.websocket
                )

=======
>>>>>>> e3607eead185b24851ddf9082210abea2f9137c8
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
            # content 进行分类
            result = await self.classify_content(content)

            if content and self.researcher.verbose:
                print(f"Content found 12345")
                # await stream_output(
                #     "logs", "subquery_context_window", f"{result}", self.researcher.websocket
                # )
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