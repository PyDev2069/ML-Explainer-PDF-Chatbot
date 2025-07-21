# module import section
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from pdf2image import convert_from_path
import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
import shutil
import tempfile
import re
from fallBack import get_fallback_answer
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
FAISS_INDEX_PATH = os.path.join(tempfile.gettempdir(), "faiss_ml_index")
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)
def clean_unicode(text):
    return re.sub(r'[\ud800-\udfff]', '', text)
def create_faiss_index():
    print("[INFO] Processing PDF and creating FAISS index...")
    loader = PyPDFLoader("ml.pdf")
    raw_text_docs = loader.load()
    cleaned_text_docs = []
    for doc in raw_text_docs:        
        doc.page_content = clean_unicode(doc.page_content)
        cleaned_text_docs.append(doc)    
    ocr_text = ""
    try:
        images = convert_from_path("ml.pdf", dpi=300)
        for img in images:
            raw_text = pytesseract.image_to_string(img)
            clean_text = clean_unicode(raw_text)
            ocr_text += clean_text
    except Exception as e:
        print(f"[WARN] OCR failed: {e}")
    ocr_doc = Document(page_content=ocr_text)    
    all_docs = cleaned_text_docs + [ocr_doc]    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)    
    db = FAISS.from_documents(chunks, embeddings)    
    db.save_local(FAISS_INDEX_PATH)
    print(f"[INFO] FAISS index saved to {FAISS_INDEX_PATH}")
    return db
if os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss")):
    print("[INFO] Loading FAISS index from disk...")
    db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    db = create_faiss_index()
retriever = db.as_retriever()
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.4
)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
while True:
    try:
        query = input("\n YOU :  ")
        if query.lower() in ['exit', 'quit']:
            break
        result = qa.invoke(query)
        result_text = result.get("result") if isinstance(result, dict) else result
        if result_text is None or "i don't know" in result_text.lower() or "not in document" in result_text.lower():
            fallBack = get_fallback_answer(query)
            final_response = fallBack if fallBack else "Sorry, I don't know the answer"
        else:
            final_response = result_text
        print("Gemini : ", final_response)

    except UnicodeEncodeError:
        print("Gemini : (Unicode Error in output)")

    except Exception as e:
        print(f"[ERROR] Something went wrong: {e}")