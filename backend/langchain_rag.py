import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
import json

from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "_", name)

def get_domain(url):
    return urlparse(url).netloc

def get_page_name(url, is_html=True):
    parsed = urlparse(url)
    path = parsed.path
    if path.endswith('/'):
        path = path[:-1]
    name = path.split('/')[-1]
    if not name:
        name = "index"
    if is_html and not name.endswith('.html'):
        name += ".html"
    return sanitize_filename(name)

def is_in_queue(url, q):
    for item in q:
        if item["url"] == url:
            return True
    return False

def crawl_and_index(start_url: str, training_id: str, yield_sse_func):
    """
    Crawls the start_url, downloads HTML and binary files to a training-specific directory,
    then uses LangChain to load, chunk, embed, and save them in a Chroma DB.
    """
    # 1. Catch redirect
    try:
        initial_response = requests.head(start_url, allow_redirects=True, timeout=10)
        if initial_response.url != start_url:
            start_url = initial_response.url
    except requests.RequestException as e:
        print(f"Failed to check initial URL: {e}")

    start_domain = get_domain(start_url)
    save_dir = os.path.join("data", training_id)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    queue = [{"url": start_url, "visited": False}]

    yield_sse_func({"status": "crawling", "message": f"Crawling website {start_url}..."})

    pages_crawled = 0
    while True:
        unvisited = [item for item in queue if not item["visited"]]
        if not unvisited:
            break
        
        current_item = unvisited[0]
        current_url = current_item["url"]
        
        try:
            response = requests.get(current_url, timeout=10)
            if response.status_code == 200:
                content_type = response.headers.get('Content-Type', '').lower()
                is_html = 'text/html' in content_type
                
                page_name = get_page_name(current_url, is_html=is_html)
                filepath = os.path.join(save_dir, page_name)
                
                if is_html:
                    soup = BeautifulSoup(response.content, "html.parser")
                    
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(soup.prettify())
                    
                    pages_crawled += 1
                    
                    for link in soup.find_all("a"):
                        href = link.get("href")
                        if not href:
                            continue
                        
                        full_url = urljoin(current_url, href)
                        full_url = full_url.split('#')[0]
                        
                        if get_domain(full_url) == start_domain:
                            if is_in_queue(full_url, queue):
                                continue
                            queue.append({"url": full_url, "visited": False})
                else:
                    with open(filepath, "wb") as f:
                        f.write(response.content)
            
        except requests.RequestException as e:
            pass
            
        current_item["visited"] = True

    # 2. Loading with LangChain
    yield_sse_func({"status": "chunking", "message": "Loading and chunking documents..."})

    pdf_loader = PyPDFDirectoryLoader(path=save_dir)
    pdf_docs = pdf_loader.load()

    html_loader = DirectoryLoader(
        save_dir,
        glob="**/*.html",
        loader_cls=BSHTMLLoader
    )
    html_docs = html_loader.load()

    all_docs = pdf_docs + html_docs

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name='cl100k_base',
        chunk_size=300,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(all_docs)

    if not chunks:
        yield_sse_func({"status": "error", "message": "No usable content found on the page."})
        return 0, pages_crawled

    # 3. Embed and store
    yield_sse_func({"status": "embedding", "message": f"Embedding and saving {len(chunks)} chunks..."})
    
    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    
    vectorstore = Chroma.from_documents(
        chunks,
        embedding_model,
        collection_name=f"{training_id}_ChromaDB",
        persist_directory=save_dir
    )

    yield_sse_func({"status": "generating", "message": "Generating suggested questions..."})
    try:
        from langchain_core.messages import HumanMessage
        import json
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        sample_text = " ".join([c.page_content for c in chunks[:5]])[:3000]
        prompt = f"Based on the following content from a website, generate exactly 4 short, relevant questions a user might ask about it. Format the output strictly as a JSON array of 4 strings, with no markdown formatting or backticks.\n\nContent:\n{sample_text}"
        resp = llm.invoke([HumanMessage(content=prompt)])
        
        content = resp.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
            
        suggestions = json.loads(content.strip())
        if not isinstance(suggestions, list) or len(suggestions) < 4:
            raise ValueError("Invalid suggestion format")
    except Exception as e:
        print(f"Suggestion generation error: {e}")
        suggestions = ["What is the main topic of this site?", "Can you summarize the content?", "Who is the intended audience?", "What are the key takeaways?"]

    return len(chunks), pages_crawled, suggestions[:4]

def chat_with_rag(user_message: str, training_id: str, llm):
    """
    Given a user message and a training ID, load the ChromaDB vector store,
    retrieve relevant documents, and ask the LLM to generate an answer.
    """
    save_dir = os.path.join("data", training_id)
    
    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = Chroma(
        collection_name=f"{training_id}_ChromaDB",
        persist_directory=save_dir,
        embedding_function=embedding_model
    )
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    relevant_document_chunks = retriever.invoke(user_message)
    context_list = [d.page_content for d in relevant_document_chunks]
    context_for_query = ". ".join(context_list)
    
    qna_system_message = """
You are a helpful assistant which can describe the content of a given web page.

Based on the RAG information, describe what the given content is about.
"""

    qna_user_message_template = """
You are given web page content information.

Use the provided context to answer the user's question accurately.
If the answer is not available in the context, clearly say so.

Context:
{context_for_query}

User message:
{user_message}

### Instructions
- Provide a concise answer relating to the question
- Keep the answer professional and easy to understand
"""

    formatted_prompt = f"""[INST]{qna_system_message}\n
                {'user'}: {qna_user_message_template.format(context_for_query=context_for_query, user_message=user_message)}
                [/INST]"""

    response = llm.invoke(formatted_prompt)
    return response.content
