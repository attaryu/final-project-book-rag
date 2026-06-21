import os

import chromadb
import pymupdf4llm
from dotenv import load_dotenv
from google.genai import Client, types
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from sentence_transformers import SentenceTransformer

load_dotenv()


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


def get_gemini_response(prompt: str):
    client = Client(api_key=os.getenv("GEMINI_API_KEY"))

    return client.models.generate_content(
        model="gemini-2.5-flash-lite", contents=types.Part.from_text(text=prompt)
    )


def start_ingest():
    print("1. membaca pdf")
    pdf_path = "data/buku-proyek-akhir.pdf"

    if not os.path.exists(pdf_path):
        print(f"error: File {pdf_path} tidak ditemukan!")
        return

    markdown_text = pymupdf4llm.to_markdown(pdf_path)
    print("berhasil membaca pdf halaman")

    print("\n2. melakukan chunking")

    header_to_split = [
        ("#", "Bab"),
        ("##", "Sub-bab"),
        ("###", "Sub-sub-bab"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=header_to_split)
    header_based_chunks = markdown_splitter.split_text(markdown_text)
    header_based_text_chunks = [chunk.page_content for chunk in header_based_chunks]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300,
    )
    final_chunks = text_splitter.split_text("".join(header_based_text_chunks))

    print(f"teks berhasil dipotong sebanyak {len(final_chunks)} bagian")

    print("\n3. membuat embedding")
    embedding_model = get_embedding_model()
    chunk_vector = embedding_model.encode(final_chunks)
    for i, (chunk, vector) in enumerate(zip(final_chunks, chunk_vector)):
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
        ids=[f"page-{i}" for i in enumerate(final_chunks, start=1)],
        embeddings=chunk_vector,
        documents=final_chunks,
    )
    print("vector berhasil disimpan ke chromadb")


def query(question: str):
    print("1. pertanyaan:\n")
    print("-" * 20)
    print(question)
    print("-" * 20)

    embedding_model = get_embedding_model()
    question_vector = embedding_model.encode(question)

    print("\n2. Mencari dokumen di chromadb")
    chroma_collection = get_chroma_collection()

    results = chroma_collection.query(query_embeddings=[question_vector], n_results=3)

    retrieved_doc = results["documents"][0]
    context = "\n---\n".join(retrieved_doc) + "\n---"

    print("Hasil")
    print("-" * 20)
    print(context)
    print("-" * 20)

    print("\n3. menjawab pertanyaan oleh LLM")
    prompt = f"""Anda adalah rekan diskusi cerdas yang memahami seluruh isi buku proyek akhir ini dengan sangat baik. Tugas Anda adalah menjawab pertanyaan secara natural, profesional, dan langsung ke inti pembahasan layaknya seorang ahli yang sedang diwawancarai.

ATURAN MENJAWAB:
1. Jawab HANYA menggunakan informasi yang ada pada bagian [KONTEKS].
2. Bersikaplah natural. JANGAN PERNAH menggunakan frasa mekanis seperti "Berdasarkan konteks yang diberikan", "Menurut teks di atas", atau "Teks tidak menyebutkan". Jawablah seolah-olah informasi tersebut memang sudah ada di kepala Anda.
3. Jika informasi yang ditanyakan TIDAK ADA dalam [KONTEKS] atau di luar topik, jangan menebak atau mengarang. Tolak dengan sopan dan natural, contohnya: "Mohon maaf, dokumen proyek akhir ini tidak membahas secara spesifik mengenai [Topik yang ditanyakan]," atau "Hal tersebut di luar ruang lingkup bahasan proyek akhir ini."
4. Langsung berikan jawaban tanpa mengulang pertanyaan atau merespons instruksi prompt ini.

KONTEKS:
```
{context}
```

PERTANYAAN:
```
{question}
```

Langsung jawab tanpa merespon apa yang saya perintahkan sebelumnya, seperti anda sedang ditanya langsung.
"""
    print("Prompt")
    print("-" * 20)
    print(prompt)
    print("-" * 20)

    response = get_gemini_response(prompt)
    print("Jawaban")
    print("-" * 20)
    print(response.candidates[0].content.parts[0].text)
    print("-" * 20)


if __name__ == "__main__":
    print("Pick the execution mode:")
    print("1. ingest")
    print("2. ask")

    answer = input("Select one: ")

    if answer == "1":
        start_ingest()
    else:
        question = input("Type your question: ")
        query(question)
