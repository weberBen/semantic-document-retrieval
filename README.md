# Simple Semantic Document Search from File Directory

This is a simple semantic search application designed to index files in a specified directory. The purpose of this app is to retrieve document chunks, which can then be used in other tools like ChatGPT or Claude for answer generation.

## Note

This app does not generate answers based on the retrieved documents. If you need a solution that performs Retrieval-Augmented Generation (RAG), consider using [Verba](https://github.com/weaviate/Verba).

Semantic search is challenging to implement effectively, as it relies on numerous factors, including:

- Text splitting: Determining where to split text is crucial for optimal results. Various tools can assist in this, such as the [LangChain Document Transformers](https://python.langchain.com/docs/how_to/#text-splitters), the [NLTK Text Splitter](https://python.langchain.com/api_reference/text_splitters/nltk/langchain_text_splitters.nltk.NLTKTextSplitter.html), and the [SpaCy Text Splitter](https://python.langchain.com/api_reference/text_splitters/spacy/langchain_text_splitters.spacy.SpacyTextSplitter.html).
- Query type: Whether the query is phrased as a question or a set of keywords influences which models are best suited for processing.
- (Re)Ranking models: Using the appropriate model for re-ranking retrieved chunks is vital for improving the accuracy of results (see this [guide to re-ranking techniques](https://medium.com/@srikanthdongalajsr/understanding-reranking-techniques-advantages-and-disadvantages-eb0b157f0b6b)).
- Separate models for query and document processing: Utilizing distinct models for query understanding and document analysis can enhance performance.
- Local vs Cloud models: The choice between using local versus cloud-hosted models impacts latency, scalability, accuracy, and cost.
- Combining search approaches: Merging semantic search with traditional full-text search can yield more comprehensive results.

For many, the straightforward solution is to use proprietary software that provides an all-in-one approach, such as [NotebookLM](https://notebooklm.google/), [Slite](https://slite.com/), [Notion](https://www.notion.so/product/ai), and [Confluence](https://www.atlassian.com/software/confluence/resources/guides/best-practices/atlassian-ai).


The application leverages `HuggingFace` models, `langchain`, and `chromadb` to provide efficient and scalable semantic search. 

## Supported File Types
The app supports the following file formats:
- **PDF**
- **Markdown (`.md`)**
- **HTML**
- **LibreOffice (`.odf`)**
- **Text (`.txt`)**
- **Microsoft Word (`.docx/.doc`)**
- **Microsoft PowerPoint (`.pptx/.ppt`)**

## Usage

Run the application using the following command:

```bash
poetry run python main.py --data_dir "./data" --doc_types "pdf/txt"
```

This command will:
1. Create a `chroma_langchain_db` folder for the persistent database.
2. Log interactions in a file (default: `prompt.log`, but you can change the filename).

To view additional options, use the help command:

```bash
python main.py --help
```

### Example Command Options:

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
  --delete_doc DELETE_DOC
                        Delete document by filename
```

The default embedding model is `sentence-transformers/all-mpnet-base-v2`.
You can check model performance depending on your use case on the [Hugging Face MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard).

### Output Example

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


### Behavior:

- Results of the search are automatically copied to your clipboard.
- To add a new document, simply place it inside the data directory. Any document with the same file path will be ignored (even if the content is different). However, if the document has the same content but a different file path, it will still be added to the database. We do not perform content base duplicate detection.
- You can delete a document by its file path with `--delete_doc`
- You can adjust embedding model parameters in the `initialize_embeddings` function, which uses `HuggingFaceEmbeddings` from the `langchain` library.

## Installation

To install the dependencies, run the following command:

```bash
poetry install
```

**Note**: At the time of writing, there is a known [bug](https://github.com/chroma-core/chroma/issues/2513) with `Chromadb` (`v0.5.4`) on Windows. The workaround is to create a Python virtual environment (`venv`) using `Python 3.10`.
