
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import dotenv
from langchain_community.embeddings import DashScopeEmbeddings
dotenv.load_dotenv()

# 1. PDF加载配置
def load_pdfs(pdf_dir):
    documents = []
    for filename in os.listdir(pdf_dir):
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(os.path.join(pdf_dir, filename))
            docs = loader.load()
            # 添加文件来源元数据
            for doc in docs:
                doc.metadata.update({"source": filename})
            documents.extend(docs)
    return documents

# 2. 中文优化文本切分
def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？", "；"],
        length_function=len
    )
    return text_splitter.split_documents(docs)

# 3. 向量化与存储
def vectorize_and_store(splits, persist_dir="./chroma_db"):
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
    )  
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="health_docs",
        collection_metadata={"hnsw:space": "cosine"}
    )
    vectorstore.persist()
    return vectorstore

if __name__ == "__main__":
    # 配置PDF目录路径
    pdf_directory = "data"
    
    # 执行全流程
    raw_docs = load_pdfs(pdf_directory)
    splits = split_documents(raw_docs)
    db = vectorize_and_store(splits)
    
    print(f"成功处理 {len(splits)} 个文本块，已存储到ChromaDB")