import json_repair
from ..utils.llm import create_chat_completion
from ..prompts import generate_search_queries_prompt
from typing import Any, List, Dict
from ..config import Config
import logging
import json

logger = logging.getLogger(__name__)

async def get_search_results(query: str, retriever: Any, query_domains: List[str] = None) -> List[Dict[str, Any]]:
    """
    Get web search results for a given query.
    
    Args:
        query: The search query
        retriever: The retriever instance
    
    Returns:
        A list of search results
    """
    search_retriever = retriever(query, query_domains=query_domains)
    return search_retriever.search()

async def generate_sub_queries(
    query: str,
    parent_query: str,
    report_type: str,
    context: List[Dict[str, Any]],
    cfg: Config,
    cost_callback: callable = None
) -> List[str]:
    """
    Generate sub-queries using the specified LLM model.
    
    Args:
        query: The original query
        parent_query: The parent query
        report_type: The type of report
        max_iterations: Maximum number of research iterations
        context: Search results context
        cfg: Configuration object
        cost_callback: Callback for cost calculation
    
    Returns:
        A list of sub-queries
    """
    gen_queries_prompt = generate_search_queries_prompt(
        query,
        parent_query,
        report_type,
        max_iterations=cfg.max_iterations or 3,
        context=context
    )

    try:
        response = await create_chat_completion(
            model=cfg.strategic_llm_model,
            messages=[{"role": "user", "content": gen_queries_prompt}],
            temperature=0.6,
            llm_provider=cfg.strategic_llm_provider,
            max_tokens=None,
            llm_kwargs=cfg.llm_kwargs,
            reasoning_effort="high",
            cost_callback=cost_callback,
        )
    except Exception as e:
        logger.warning(f"Error with strategic LLM: {e}. Retrying with max_tokens={cfg.strategic_token_limit}.")
        logger.warning(f"See https://github.com/assafelovic/gpt-researcher/issues/1022")
        try:
            response = await create_chat_completion(
                model=cfg.strategic_llm_model,
                messages=[{"role": "user", "content": gen_queries_prompt}],
                temperature=1,
                llm_provider=cfg.strategic_llm_provider,
                max_tokens=cfg.strategic_token_limit,
                llm_kwargs=cfg.llm_kwargs,
                cost_callback=cost_callback,
            )
            logger.warning(f"Retrying with max_tokens={cfg.strategic_token_limit} successful.")
        except Exception as e:
            logger.warning(f"Retrying with max_tokens={cfg.strategic_token_limit} failed.")
            logger.warning(f"Error with strategic LLM: {e}. Falling back to smart LLM.")
            response = await create_chat_completion(
                model=cfg.smart_llm_model,
                messages=[{"role": "user", "content": gen_queries_prompt}],
                temperature=cfg.temperature,
                max_tokens=cfg.smart_token_limit,
                llm_provider=cfg.smart_llm_provider,
                llm_kwargs=cfg.llm_kwargs,
                cost_callback=cost_callback,
            )

    return json_repair.loads(response)

async def plan_research_outline(
    query: str,
    search_results: List[Dict[str, Any]],
    agent_role_prompt: str,
    cfg: Config,
    parent_query: str,
    report_type: str,
    cost_callback: callable = None,
    retriever_type: str = None
) -> List[str]:
    """
    Plan the research outline by generating sub-queries.
    
    Args:
        query: Original query
        retriever: Retriever instance
        agent_role_prompt: Agent role prompt
        cfg: Configuration object
        parent_query: Parent query
        report_type: Report type
        cost_callback: Callback for cost calculation
        retriever_type: Type of retriever being used
    
    Returns:
        A list of sub-queries
    """
    
    sub_queries = await generate_sub_queries(
        query,
        parent_query,
        report_type,
        search_results,
        cfg,
        cost_callback
    )

    logger.info(f"in function plan_research_outline -- parent_query: {parent_query},query: {query}, sub_queries: {sub_queries}")
    return sub_queries

async def generate_pubmed_sub_queries(
    query: str,
    cfg: Config,
    cost_callback: callable = None
) -> List[str]:
    """
    Generate specialized sub-queries for PubMed based on medical terms and basic search strategies
    
    Args:
        query: Original query
        cfg: Configuration object
        cost_callback: Callback for cost calculation
    
    Returns:
        A list of sub-queries in PubMed search format
    """
    logger.info(f"Generating PubMed sub-queries for query: {query}")
    
    prompt = f"""As a medical literature search expert, please analyze the following topic and generate simple search queries:
    Topic: {query}
    
    Requirements:
    1. Extract the main medical term(s) from the topic
    2. For each medical term, add ONE most relevant related term
    3. Keep each query short and simple (2-3 words maximum)
    4. Use standard Boolean operators (AND) in uppercase
    5. DO NOT use any special characters or field tags
    
    Example:
    Input: "What are the latest treatments for Chronic Myeloid Leukemia?"
    Output: ["Withdrawal Chronic Myeloid Leukemia", "Treatment Chronic Myeloid Leukemia"]
    
    Return a valid JSON array of strings:
    [
        "term1 AND term2",
        "term3 AND term4"
    ]
    """
    
    try:
        response = await create_chat_completion(
            model=cfg.smart_llm_model,
            messages=[
                {"role": "system", "content": "You are a medical search expert who creates simple, effective search strings. Focus on extracting medical terms and adding one relevant term. Keep queries short and clear."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            llm_provider=cfg.smart_llm_provider,
            websocket=None,
            max_tokens=cfg.smart_token_limit,
            llm_kwargs=cfg.llm_kwargs,
            cost_callback=cost_callback,
        )
        
        # 预处理响应，移除可能的 ```json 标记
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        
        # 使用 json_repair.loads 替代 json.loads
        sub_queries = json_repair.loads(response)
        logger.info(f"Generated PubMed sub-queries: {sub_queries}")
        return sub_queries
    except Exception as e:
        logger.error(f"Error generating PubMed sub-queries: {e}")
        logger.error(f"Original response: {response}")
        
        # If parsing fails, return a simplified version of the original query
        simplified_query = query.replace(" ", " AND ")
        return [simplified_query]