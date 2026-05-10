import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re

def sanitize_filename(name):
    # Replace characters not suitable for filenames
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

# Initial setup
START_URL = "https://feketegabor.com"
BASE_DIR = "/Users/gaborfekete/my-projects/ai/production/week2/twin/scripts/crawler/data"

start_domain = get_domain(START_URL)

# Create a folder named as the original URL
folder_name = sanitize_filename(START_URL.replace("https://", "").replace("http://", ""))
save_dir = os.path.join(BASE_DIR, folder_name)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Initialize the queue with the starting URL
# We keep track of URLs and a boolean flag indicating if they have been visited
queue = [{"url": START_URL, "visited": False}]

def is_in_queue(url, q):
    for item in q:
        if item["url"] == url:
            return True
    return False

print(f"Starting crawl at {START_URL}")
print(f"Saving data to {save_dir}")

while True:
    # Find the next unvisited URL in the queue
    unvisited = [item for item in queue if not item["visited"]]
    if not unvisited:
        print("All URLs in the queue have been visited.")
        break
        
    current_item = unvisited[0]
    current_url = current_item["url"]
    
    print(f"\nVisiting: {current_url}")
    
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
                
                print(f"Saved HTML content to {filepath}")
                
                # Collect links and add to queue
                for link in soup.find_all("a"):
                    href = link.get("href")
                    if not href:
                        continue
                    
                    # Resolve relative URLs
                    full_url = urljoin(current_url, href)
                    
                    # Remove fragment (#) to avoid crawling the same page for different sections
                    full_url = full_url.split('#')[0]
                    
                    # To prevent crawling the entire internet, we only crawl the starting domain
                    if get_domain(full_url) == start_domain:
                        # If it's already in the queue, it's a loop. Skip it to avoid infinite loops.
                        if is_in_queue(full_url, queue):
                            continue
                        
                        # Add new URL to the queue
                        queue.append({"url": full_url, "visited": False})
                        print(f"Added to queue: {full_url}")
            else:
                # Save binary file (e.g., pdf, docx, zip)
                with open(filepath, "wb") as f:
                    f.write(response.content)
                print(f"Downloaded binary file to {filepath}")
        else:
            print(f"Failed URL (Status: {response.status_code})")
            
    except requests.RequestException as e:
        print(f"Failed to fetch {current_url}: {e}")
        
    # Mark the current URL as visited
    current_item["visited"] = True

print("\nCrawling finished.")


#Loading the `config.json` file
import json
import os
from langchain_community.vectorstores import Chroma

# Load the JSON file and extract values
file_name = "config.json"
with open(file_name, 'r') as file:
    config = json.load(file)
    os.environ['OPENAI_API_KEY'] = config["API_KEY"]
    os.environ["OPENAI_BASE_URL"] = config["OPENAI_API_BASE"]

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="gpt-4o-mini",                      # "gpt-4o-mini" to be used as an LLM
    temperature=0,                # Set the temprature to 0
    max_tokens=5000,                 # Set the max_tokens = 5000, so that the long response will not be clipped off
    top_p=0.95,
    frequency_penalty=1.2,
    stop_sequences=['INST']
)

print("\n Pdf files:")
ai_initiative_pdf_paths = [f"./data/{folder_name}/{file}" for file in os.listdir(f"./data/{folder_name}") if file.endswith(".pdf")]
for i, path in enumerate(ai_initiative_pdf_paths, start=1):
    print(f"{i}. {path}")

print("\n Html files:")
ai_initiative_html_paths = [f"./data/{folder_name}/{file}" for file in os.listdir(f"./data/{folder_name}") if file.endswith(".html")]
for i, path in enumerate(ai_initiative_html_paths, start=1):
    print(f"{i}. {path}")


from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, BSHTMLLoader
PyPDFDirectoryLoader
loader = PyPDFDirectoryLoader(path = f"./data/{folder_name}")

# Defining the text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter      #  Helpful in splitting the PDF into smaller chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name='cl100k_base',
    chunk_size=300,
    chunk_overlap=50
)

pdf_docs = loader.load()

# Load HTML
html_loader = DirectoryLoader(
    f"./data/{folder_name}",
    glob="**/*.html",
    loader_cls=BSHTMLLoader
)
html_docs = html_loader.load()

# Combine documents
all_docs = pdf_docs + html_docs

# Split into chunks
ai_initiative_chunks = text_splitter.split_documents(all_docs)

# Total length of all the chunks
print(f"Total number of chunks = {len(ai_initiative_chunks)}")


from langchain_openai import OpenAIEmbeddings
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

#  Creating a Vectorstore, storing all the above created chunks using an embedding model
vectorstore = Chroma.from_documents(
    ai_initiative_chunks,
    embedding_model,
    collection_name=f"{folder_name}_ChromaDB"
)

# Creating an retriever object which can fetch ten similar results from the vectorstore
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


user_message = "What are the ai case studies working on ?"

# Building the context for the query using the retrieved chunks
relevant_document_chunks = retriever.invoke(user_message)
context_list = [d.page_content for d in relevant_document_chunks]
context_for_query = ". ".join(context_list)

print(f"Number of retrieved chunks: {len(relevant_document_chunks)}")

qna_system_message = """
You are a helpful assistant which can describe the content of a given web page.

Based on the RAG information, describe what the given content is about.
"""

# Write an user message template which can be used to attach the context and the questions
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

# Format the prompt
formatted_prompt = f"""[INST]{qna_system_message}\n
                {'user'}: {qna_user_message_template.format(context_for_query=context_for_query, user_message=user_message)}
                [/INST]"""

# Make the LLM call
resp = llm.invoke(formatted_prompt)
resp.content
print(f"Response: {resp.content}")
from langchain.tools import tool
# Define RAG function
def RAG(user_message):
    print("\n" + "=" * 50)
    print("User message:", user_message)
    print("\n" + "=" * 50)
    """
    Args:
    user_message: Takes a user input for which the response should be retrieved from the vectorDB.
    Returns:
    relevant context as per user query.
    """
    relevant_document_chunks = retriever.invoke(user_message)
    context_list = [d.page_content for d in relevant_document_chunks]
    context_for_query = ". ".join(context_list)



    # Combine qna_system_message and qna_user_message_template to create the prompt
    prompt = f"""[INST]{qna_system_message}\n
                {'user'}: {qna_user_message_template.format(context_for_query=context_for_query, user_message=user_message)}
                [/INST]"""

    # Quering the LLM
    try:
        response = llm.invoke(prompt)

    except Exception as e:
        response = f'Sorry, I encountered the following error: \n {e}'

    return response.content

print(RAG("What is Findipend about?"))
print(RAG("What are the deep learning case studies?"))