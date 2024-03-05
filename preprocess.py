import json
from bs4 import BeautifulSoup
from bs4.element import Tag
from typing import List

with open("data/train_data.json", "r") as f:
    train_data = json.load(f)

# The block elements we want to extract text from
BLOCKS = ["p", "h", "h1", "h2", "h3", "h4", "h5", "li", "u", "div"]

def _extract_blocks(parent_tag) -> List:
    """
    parent_tag:
    return:
    """
    extracted_blocks = []
    for tag in parent_tag:
        if tag.name in BLOCKS:
            extracted_blocks.append(tag)
            continue
        if isinstance(tag, Tag):
            if len(tag.contents) > 0:
                inner_blocks = _extract_blocks(tag)
                if len(inner_blocks) > 0:
                    extracted_blocks.extend(inner_blocks)
    return extracted_blocks

def to_plaintext(html_text: str) -> str:
    soup = BeautifulSoup(html_text, features="lxml")
    extracted_blocks = _extract_blocks(soup.body)
    extracted_blocks_texts = [block.get_text().strip() for block in extracted_blocks]
    return "\n".join(extracted_blocks_texts)

# Create a new json file with preprocessed text
processed_train_data = []
for data in train_data:
    html = data["content"]["fullTextHtml"]
    fullText = to_plaintext(html)
    # Copy the data point in a new dictionary
    d = data.copy()
    # remove the fullTextHtml key
    d["content"].pop("fullTextHtml", None)
    # add a new key for the processed text
    d["content"]["fullText"] = fullText
    processed_train_data.append(d)

with open("processed_data/processed_train_data.json", "w") as f:
    json.dump(processed_train_data, f)

print("Data preprocessing finished successfully!")