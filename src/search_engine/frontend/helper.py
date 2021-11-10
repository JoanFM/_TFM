import requests
import os

repo_banner_file = os.path.abspath("./eah.svg")


class UI:
    about_block = """

    ### About

    This is a crossmodal search using  engine using [Jina's neural search framework](https://github.com/jina-ai/jina/).

    - [Repo](https://github.com/JoanFM/TFM_Sparse_Embeddings)
    - [Dataset](https://www.kaggle.com/hsankesara/flickr-image-dataset)
    """

    css = f"""
<style>
    .reportview-container .main .block-container{{
        max-width: 1200px;
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }}
    .reportview-container .main {{
        color: "#111";
        background-color: "#eee";
    }}
</style>
"""


headers = {"Content-Type": "application/json"}


def search_by_text(query: str, endpoint: str, top_k: int) -> dict:
    """search_by_text.

    :param query:
    :type query: str
    :param endpoint:
    :type endpoint: str
    :param top_k:
    :type top_k: int
    :rtype: dict
    """
    data = '{"data":["' + query + '"]}'

    response = requests.post(endpoint, headers=headers, data=data)
    content = response.json()

    matches = content["data"]["docs"][0]["matches"]
    return matches
