import json
from typing import List

from bs4 import BeautifulSoup
from bs4.element import Tag

# The block elements we want to extract text from
BLOCKS = ["p", "h", "h1", "h2", "h3", "h4", "h5", "li", "u", "div"]


def _extract_blocks(parent_tag: str) -> List:
    """This function recursively travels the element tree to find the
    block elements inside other elements. We want to extract the blocks
    corresponding to the tags defined in the 'BLOCKS' list. If a tag name
    is in the 'BLOCKS' list, we add it to the output list.

    Args:
        parent_tag: Content of an HTML tag.

    Returns:
        List: A list of all tags inside the parent tag.
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
    """This dunction extract the texts from all block elements inside the
    HTML body, strips them of left and right whtie spaces and joins them
    together with a single new line.

    Args:
        html_text: Raw HTML text.

    Returns:
        str: Text extracted from the HTML body, without the tags, link urls, images, etc.
    """
    soup = BeautifulSoup(html_text, features="lxml")
    extracted_blocks = _extract_blocks(soup.body)
    extracted_blocks_texts = [block.get_text().strip() for block in extracted_blocks]
    return "\n".join(extracted_blocks_texts)


def preprocess_data(
    path_to_data: str, test: bool = False, path_to_taxonomy_mappings: str = None
):
    """Creates a new json dataset with preprocessed text and a list of probabilities
    for all class labels.

    Args:
        data: Path to the dataset in json format.
        test: If True, it means the data is for training and has the 'labels' key.
        taxonomy_mappings: Path to the taxonomy mappings in json format.
        Required if test=False.
    """
    with open(path_to_data, "r") as f:
        data = json.load(f)
    if path_to_taxonomy_mappings:
        with open(path_to_taxonomy_mappings, "r") as f:
            taxonomy_mappings = json.load(f)
        # Get the reverse taxonomy mappings
        if taxonomy_mappings:
            rev_taxonomy = {v: k for k, v in taxonomy_mappings.items()}
    processed_data = []
    for e in data:
        if not test:
            # for training data, if there are no labels assigned, discard it!
            if e["labels"]:
                html = e["content"]["fullTextHtml"]
                fullText = to_plaintext(html)
                d = {}
                d["title"] = e["content"]["title"]
                # add a new key for the processed text
                d["fullText"] = fullText
                # Substitute the 'labels' with a vector of confidences for all labels.
                # Labels are ordered according to the taxonomy mapping.
                all_labels_probs = [0] * len(rev_taxonomy)
                for l_ in e["labels"]:
                    all_labels_probs[int(rev_taxonomy[l_[0]])] = l_[1]
                d["labels"] = all_labels_probs
                processed_data.append(d)
        else:
            html = e["content"]["fullTextHtml"]
            fullText = to_plaintext(html)
            d = {}
            d["title"] = e["content"]["title"]
            # add a new key for the processed text
            d["fullText"] = fullText
            processed_data.append(d)
    return processed_data


if __name__ == "__main__":
    processed_train_data = preprocess_data(
        "data/train_data.json",
        test=False,
        path_to_taxonomy_mappings="data/taxonomy_mappings.json",
    )

    with open("processed_data/processed_train_data.json", "w") as f:
        json.dump(processed_train_data, f)

    print("Data preprocessing finished successfully!")
