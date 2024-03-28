from exa_py import Exa
from typing import List

def init_search_client()->Exa:
    return Exa("14036498-24af-4ed7-a72e-23eb683a186a")


def search_docs(client:Exa,keyword:str,num_results:int=100)->List[str]:
    response = client.search_and_contents(
        keyword,
        num_results=num_results,
        use_autoprompt=True,
        text={"include_html_tags": False,"max_characters":7500}
    )
    return [result.text for result in response.results]