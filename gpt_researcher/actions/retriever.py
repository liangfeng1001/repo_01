from typing import List, Type
from ..config.config import Config

def get_retriever(retriever):
    """
    Gets the retriever
    Args:
        retriever: retriever name

    Returns:
        retriever: Retriever class

    """
    print(f"Getting retriever for: {retriever}")
    match retriever:
        case "google":
            from gpt_researcher.retrievers import GoogleSearch

            retriever = GoogleSearch
        case "searx":
            from gpt_researcher.retrievers import SearxSearch

            retriever = SearxSearch
        case "searchapi":
            from gpt_researcher.retrievers import SearchApiSearch

            retriever = SearchApiSearch
        case "serpapi":
            from gpt_researcher.retrievers import SerpApiSearch

            retriever = SerpApiSearch
        case "serper":
            from gpt_researcher.retrievers import SerperSearch

            retriever = SerperSearch
        case "duckduckgo":
            from gpt_researcher.retrievers import Duckduckgo

            retriever = Duckduckgo
        case "bing":
            from gpt_researcher.retrievers import BingSearch

            retriever = BingSearch
        case "arxiv":
            from gpt_researcher.retrievers import ArxivSearch

            retriever = ArxivSearch
        case "bioRxiv":
            from gpt_researcher.retrievers import bioRxivSearch

            retriever = bioRxivSearch
        case "tavily":
            from gpt_researcher.retrievers import TavilySearch

            retriever = TavilySearch
        case "exa":
            from gpt_researcher.retrievers import ExaSearch

            retriever = ExaSearch
        case "semantic_scholar":
            from gpt_researcher.retrievers import SemanticScholarSearch

            retriever = SemanticScholarSearch
        case "pubmed_central":
            from gpt_researcher.retrievers import PubMedCentralSearch

            retriever = PubMedCentralSearch
        case "pubmed_dian":
            from gpt_researcher.retrievers import PubmedDianSearch

            retriever = PubmedDianSearch
            print(f"Found PubmedDianSearch retriever")
        case "custom":
            from gpt_researcher.retrievers import CustomRetriever

            retriever = CustomRetriever

        case _:
            retriever = None
            print(f"No matching retriever found for: {retriever}")

    return retriever


def get_retrievers(headers, cfg):
    """
    Determine which retriever(s) to use based on headers, config, or default.

    Args:
        headers (dict): The headers dictionary
        cfg (Config): The configuration object

    Returns:
        list: A list of retriever classes to be used for searching.
    """
    print(f"Getting retrievers with headers: {headers}")
    print(f"Config retrievers: {cfg.retrievers}")
    
    # Check headers first for multiple retrievers
    if headers.get("retrievers"):
        retrievers = headers.get("retrievers").split(",")
        print(f"Using retrievers from headers: {retrievers}")
    # If not found, check headers for a single retriever
    elif headers.get("retriever"):
        retrievers = [headers.get("retriever")]
        print(f"Using single retriever from headers: {retrievers}")
    # If not in headers, check config for multiple retrievers
    elif cfg.retrievers:
        retrievers = cfg.retrievers
        print(f"Using retrievers from config: {retrievers}")
    # If not found, check config for a single retriever
    elif cfg.retriever:
        retrievers = [cfg.retriever]
        print(f"Using single retriever from config: {retrievers}")
    # If still not set, use default retriever
    else:
        retrievers = [get_default_retriever().__name__]
        print(f"Using default retriever: {retrievers}")

    # Convert retriever names to actual retriever classes
    retriever_classes = [get_retriever(r) or get_default_retriever() for r in retrievers]
    print(f"Final retriever classes: {[r.__name__ for r in retriever_classes]}")
    return retriever_classes


def get_default_retriever(retriever):
    from gpt_researcher.retrievers import TavilySearch

    return TavilySearch