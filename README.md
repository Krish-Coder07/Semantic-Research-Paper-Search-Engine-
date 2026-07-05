# Semantic-Research-Paper-Search-Engine
A small project that lets you search through machine learning research papers using plain English instead of exact keywords, and get a short summary, keywords, and named entities for each result.

Built on top of the [ML-ArXiv-Papers dataset](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers), which contains titles and abstracts of machine learning papers from arXiv.

## What it does

- Takes a search query like "deep learning for medical imaging" and finds the most relevant papers by meaning, not just matching words
- Generates a short AI summary of each paper's abstract instead of showing the full text
- Pulls out the main keywords from each paper
- Groups papers into topic clusters based on their content
- Picks out named entities from each abstract - things like organizations, dates, and (for medical papers) terms like diseases, symptoms, and procedures
- Puts all of this together into a single readable "card" for each paper: title, summary, keywords, and entities

## How it works

1. **Data loading and cleaning** - loads the dataset, checks for missing values, and looks at how long the titles and abstracts are
2. **Embeddings** - each paper's title and abstract are turned into a vector using a sentence embedding model (`all-MiniLM-L6-v2`), so that papers with similar meaning end up close to each other numerically
3. **Search** - a FAISS index is built from these vectors so we can quickly find the closest matches to any query
4. **Summarization** - `distilbart-cnn-12-6` is used to shorten each abstract into a couple of sentences
5. **Keyword extraction** - KeyBERT pulls out short phrases that best represent each abstract
6. **Clustering** - KMeans groups the papers into topics, and the top TF-IDF terms for each cluster are used to label what that group is about
7. **Named entity recognition** - two models are used together:
   - a general-purpose spaCy model for things like organizations, dates, and numbers
   - a biomedical model for medical terms, filtered down to keep only the labels that are actually reliable, since the model tends to misclassify plain English words when the text isn't medical

## Why it caches things

Re-running the embedding step and the entity recognition step on the full set of papers takes a while, so the notebook checks Google Drive first. If the results from a previous run are already saved there, it loads them directly instead of recomputing everything. If not, it runs the full pipeline once and saves the results for next time.

The biomedical entity extraction step also saves its progress every so often, so if the notebook gets interrupted partway through, it can pick up from where it left off instead of starting over.

## Notes on the entity recognition

The biomedical model used here (`d4data/biomedical-ner-all`) is trained on clinical text, not machine learning papers. When it sees ML-specific words it hasn't seen much of during training, it sometimes labels them incorrectly and with high confidence, which is a known limitation of applying a domain-specific model outside its intended domain. To keep the results usable, the code:

- only keeps a handful of labels that turned out to be reliable in practice
- drops very generic labels that matched almost anything
- filters out common non-medical words that kept showing up by mistake

This means results are much cleaner for genuinely medical papers, and mostly empty (correctly) for papers that aren't medical at all.

## Requirements

- Python
- pandas, matplotlib
- sentence-transformers
- faiss-cpu
- transformers (pinned below version 5, since the summarization pipeline used here was removed in that version)
- keybert
- scikit-learn
- spacy (with the `en_core_web_sm` model)

## Running it

The notebook is meant to be run in Google Colab, since it uses Google Drive to cache results between sessions. Run the cells from top to bottom. The first run will take longer since it has to build the embeddings and search index from scratch; later runs will load the cached files instead.

## Possible next steps

- Add a proper trend analysis, which would need a dataset that includes publication dates
- Add a search filter based on entities (for example, only show papers that mention a specific model or dataset)
- Package the search and summarization into a simple web app instead of a notebook
