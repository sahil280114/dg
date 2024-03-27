from enum import Enum
from typing import List
import json
import tqdm
import concurrent
from model import inference,format_prompt_keyword_expansion
import json_repair

class Type(Enum):
    Keyword = 1
    WebDocument = 2
    UserDocument = 3
    WebCrawl = 4

class SeedDataPoint:
    def __init__(self, content: str, type: Type) -> None:
        self.content = content
        self.type = type

    def to_dict(self):
        return {"content": self.content, "type": self.type.name}


def save_seed_data(seed_data: List[SeedDataPoint], path: str) -> None:
    serializable_data = [data_point.to_dict() for data_point in seed_data]
    with open(f'{path}.json', 'w') as f:
        json.dump(serializable_data, f, indent=4)



def keywords_to_seed_data(model,use_case:str,keywords: List[str]) -> List[SeedDataPoint]:
    """Takes in a list of keywords from the users, expands them across each keyword"""
    expanded_keywords = []
    keyword_prompts = [format_prompt_keyword_expansion(use_case, keyword) for keyword in keywords]
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as executor:
        tasks = [executor.submit(inference,model,prompt) for prompt in keyword_prompts]
        for future in tqdm.tqdm(concurrent.futures.as_completed(tasks),total=len(tasks),desc="Expanding keywords"):
            result = future.result()
            result_json = json_repair.loads(result)
            for term in result_json['search_terms']:
                expanded_keywords.append(SeedDataPoint(term, Type(1)))
    return expanded_keywords

def web_documents_to_seed_data(web_documents: List[str]) -> List[SeedDataPoint]:
    """Takes in a list of web documents and converts them to seed data"""
    return [SeedDataPoint(document, Type(2)) for document in web_documents]