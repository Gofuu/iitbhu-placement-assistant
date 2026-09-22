"""
Chunks the parsed policy markdown (data/processed/policy/) and the forum
threads (loaded live from data/raw/forum/) and embeds both into a persistent
Chroma vector store, carrying source metadata through for citations.

Run: python -m src.ingest.build_vectorstore
"""
import json
import shutil

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  # replaces the deprecated langchain_community.vectorstores.Chroma (same on-disk store)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.config import PROCESSED_POLICY_DIR, VECTORSTORE_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
from src.ingest.parse_forum import load_forum_docs

HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def load_policy_docs() -> list[Document]:
    docs = []
    for md_path in sorted(PROCESSED_POLICY_DIR.glob("*.md")):
        meta_path = md_path.with_suffix("").with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        text = md_path.read_text(encoding="utf-8")
        docs.append(Document(page_content=text, metadata=meta))
    return docs


def chunk_policy_docs(docs: list[Document]) -> list[Document]:
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    all_chunks = []
    for doc in docs:
        try:
            header_chunks = header_splitter.split_text(doc.page_content)
        except Exception:
            header_chunks = [doc]

        for hc in header_chunks:
            hc.metadata.update(doc.metadata)
            for sub in char_splitter.split_documents([hc]):
                all_chunks.append(sub)
    return all_chunks


def chunk_forum_docs(docs: list[Document]) -> list[Document]:
    # forum threads have no markdown headers to split on, just size-limit them
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = char_splitter.split_documents(docs)

    # Make every chunk self-identifying by stamping the company/year header onto
    # each one. Without this, only the FIRST chunk of a thread contains the
    # company name (it's at the top of the thread text), so a query like
    # "<company> interview experience" retrieves that first chunk -- which is
    # usually just the empty "Share your Interview Experience here" prompt --
    # while the chunks holding the actual experience, which don't mention the
    # company by name, rank too low to reach top-k. Prepending the header means
    # a mid-experience chunk still says which company it's about and ranks
    # properly for company-specific queries.
    for c in chunks:
        company = c.metadata.get("company")
        if company and not c.page_content.lstrip().startswith("Company:"):
            kind = c.metadata.get("kind", "")
            year = c.metadata.get("year", "")
            c.page_content = f"Company: {company} ({kind} {year}) interview experience:\n{c.page_content}"
    return chunks


def main():
    policy_docs = load_policy_docs()
    forum_docs = load_forum_docs()

    if not policy_docs and not forum_docs:
        print(
            "Nothing to embed. Run `python -m src.ingest.parse_policy` first, "
            "and make sure data/raw/forum/ has chunk_*.json files."
        )
        return

    chunks = []
    if policy_docs:
        policy_chunks = chunk_policy_docs(policy_docs)
        print(f"Policy: {len(policy_docs)} documents -> {len(policy_chunks)} chunks")
        chunks.extend(policy_chunks)
    if forum_docs:
        forum_chunks = chunk_forum_docs(forum_docs)
        print(f"Forum: {len(forum_docs)} threads -> {len(forum_chunks)} chunks")
        chunks.extend(forum_chunks)

    # Wipe any existing store first so a rebuild is a clean replace, not an append.
    # Chroma adds to whatever is already in persist_directory, so without this a
    # second run leaves BOTH the old and new chunks in place -- duplicates that
    # pollute retrieval (e.g. old pre-fix forum chunks competing with new ones).
    if VECTORSTORE_DIR.exists():
        shutil.rmtree(VECTORSTORE_DIR)
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Embedding {len(chunks)} chunks total with {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )
    # langchain-community's Chroma (>=0.4.x) auto-persists to persist_directory
    # as documents are added — no manual .persist() call needed or supported.
    print(f"Vector store built at {VECTORSTORE_DIR} ({len(chunks)} chunks)")
    return vectorstore


if __name__ == "__main__":
    main()
