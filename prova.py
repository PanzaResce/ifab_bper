from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer

# Load HTML
loader = AsyncChromiumLoader(["https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022"])
html = loader.load()

# Transform
bs_transformer = BeautifulSoupTransformer()
docs_transformed = bs_transformer.transform_documents(
    html, tags_to_extract=["p", "li", "div", "a", "span"]
)

print(docs_transformed)