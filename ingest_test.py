import sys
import os

import chromadb
import pymupdf4llm
from langchain_text_splitters import MarkdownTextSplitter
from sentence_transformers import SentenceTransformer


def get_embedding_model():
    return SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


def get_chroma_collection():
    chroma_client = chromadb.PersistentClient()
    chroma_collection = chroma_client.get_or_create_collection(
        name="final-project-book-v1"
    )

    return chroma_collection


def start_ingest():
    print("1. membaca pdf")
    pdf_path = "data/buku-proyek-akhir.pdf"

    if not os.path.exists(pdf_path):
        print(f"error: File {pdf_path} tidak ditemukan!")
        return

    markdown_text = pymupdf4llm.to_markdown(pdf_path)
    print("berhasil membaca pdf halaman")

    print("\n2. melakukan chunking")
    markdown_text_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = markdown_text_splitter.create_documents([markdown_text])
    text_chunks = [chunk.page_content for chunk in chunks]
    print(f"teks berhasil dipotong sebanyak {len(chunks)} bagian")

    print("\n3. membuat embedding")
    embedding_model = get_embedding_model()
    chunk_vector = embedding_model.encode(text_chunks)
    for i, (chunk, vector) in enumerate(zip(text_chunks, chunk_vector)):
        if i > 10:
            break

        print(f"\n> teks ke {i}\n")
        print("-" * 20)
        print(chunk)
        print("-" * 20)
        print(f"\n> ukuran dimensi vektor: {len(vector)}")
        print(f"\n> sampel 5 angka pertama: {vector[:5]}\n")
        print("=" * 20)

    print("\n4. menyimpan ke vector database")
    chrome_collection = get_chroma_collection()
    chrome_collection.add(
        ids=[f"page-{i + 1}_id-{chunk.id}" for i, (chunk) in enumerate(chunks)],
        embeddings=chunk_vector,
        documents=text_chunks,
    )
    print("vector berhasil disimpan ke chromadb")


def query(question: str):
    print("1. pertanyaan:\n")
    print("-" * 20)
    print(f'"{question}"')
    print("-" * 20)

    embedding_model = get_embedding_model()
    question_vector = embedding_model.encode(question)

    print("\n2. Mencari dokumen di chromadb")
    chroma_collection = get_chroma_collection()

    results = chroma_collection.query(query_embeddings=[question_vector], n_results=3)

    retrieved_docs = results["documents"][0]
    context = "\n---\n".join(retrieved_docs)

    print("\n ----- hasil ----- \n")
    print(context)


if __name__ == "__main__":
    query(sys.argv[1])
    # start_ingest()
