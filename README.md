# Simple semantic document search from file directory

This is a simple semantic search application to index all file inside a specified directory. This app does not generate answer based on the retrieved document, it's only intend to retrieve documents chunks to then be used in ChatGPT or Claude for answer generation.
If you need answer generation (RAG), you can use [Verba](https://github.com/weaviate/Verba).

The application has been built as a quick & dirty project with `HuggingFace` model, `langchain` and `chromadb`.

Supported file are : 
- `pdf`
- `markdown (md)`
- `html`
- `Libre office (odf)`
- `text (txt)`
- `Microsoft word (docx/dox)`
- `Microsoft powerpoint (pptx/ppt)`

# Usage

```bash
poetry run python main.py --data_dir "./data"
```

See help for more commands :

```bash
> main.py --help
usage: main.py [-h] [--user_prompt USER_PROMPT] [--doc_limit DOC_LIMIT]
               [--model_name MODEL_NAME]

Semantic search : query file database using sentence transformers and vector stores

options:
  -h, --help            show this help message and exit
  --user_prompt USER_PROMPT
                        Prompt that is passed to the query console
  --doc_limit DOC_LIMIT
                        Limit the number of documents to return in response to
                        the query
  --model_name MODEL_NAME
                        Name of the model to use for embedding generation
  --data_dir DATA_DIR   Path of the file dir
  --log_file LOG_FILE   Filename for the log file
  --doc_types DOC_TYPES
                        Extension of document, separated by / (e.g.
                        pdf/md/txt). Supported :
                        pdf/md/html/txt/odt/docx/doc/pptx/ppt

```

## Output example

```bash
----------
Context:
**********
Quote 1:
{'page': 6, 'source': './data/2003.13045v2.pdf'}
and our approach can provide reliable supervision.
Usage of Augmentation. As we mentioned above, almost
all of the unsupervised learning approaches avoid using a
heavy combination of augmentations. As a reference, we
**********

**********
Quote 2:
{'page': 2, 'source': './data/2011.09157v2.pdf'}
2. Method
2.1. Background
For self-supervised representation learning, the break-
through approaches are MoCo-v1/v2 [17, 3] and Sim-
CLR [2], which both employ contrastive unsupervised
learning to learn good representations from unlabeled data.
**********

From documents provided as context, answer to the following query: self supervised
```

Default model is `sentence-transformers/all-mpnet-base-v2`.

Result of the search is automatically copied to clipboard.

If a database exists then no more file will be indexed. Tha app is made to index once a whole directory.

You can tweak parameters passed to the embeddings model inside function `initialize_embeddings` which simply use `HuggingFaceEmbeddings` from `langchain`.


# Installation

```bash
poetry install
```
