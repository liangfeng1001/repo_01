import asyncio
import pprint

from dotenv import load_dotenv

from gpt_researcher.actions.retriever import get_retrievers
from gpt_researcher.config.config import Config

# Load environment variables from .env file
load_dotenv()


async def test_scrape_data_by_query():
    # Initialize the Config object
    config = Config()

    # Retrieve the retrievers based on the current configuration
    retrievers = get_retrievers({}, config)
    print("Retrievers:", retrievers)

    sub_query = "design patterns for autonomous ai agents"

    # Iterate through all retrievers
    for retriever_class in retrievers:
        # Instantiate the retriever with the sub-query
        retriever = retriever_class(sub_query)

        # Perform the search using the current retriever
        search_results = await asyncio.to_thread(
            retriever.search, max_results=3
        )

        print("Search results:")
        pprint.pprint(search_results, indent=4, width=80)

if __name__ == "__main__":
    asyncio.run(test_scrape_data_by_query())