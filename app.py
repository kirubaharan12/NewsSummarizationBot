#Web / Flask
from flask import Flask, render_template, request, url_for, jsonify, redirect
from werkzeug.utils import secure_filename

#Environment / config
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

#Web Scraping
from bs4 import BeautifulSoup as soup
from urllib.request import urlopen

#News
from newspaper import Article
import feedparser

#PDF / files / IO
import io
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from werkzeug.utils import secure_filename  # (also under web)


#NLP / tokenizers / summarizers / text processing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist
from nltk.tokenize.treebank import TreebankWordDetokenizer
import spacy
#Embedding / LLM / LangChain / GenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter, SpacyTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.tools import DuckDuckGoSearchResults
from google.genai import types


# alternative/third-party Google GenAI integrations present:
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from google import genai

#Media (image/audio)
import speech_recognition as sr

#Utilities / general
import re
import textwrap
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import tiktoken
import os

api_key = os.getenv("GOOGLE_API_KEY")
nlp_model = spacy.load('en_core_web_sm')
load_dotenv(find_dotenv())
search_engine = DuckDuckGoSearchResults()
text_splitter = SpacyTextSplitter(chunk_size=200)
client = genai.Client(api_key=api_key)
text_model = 'gemini-2.0-flash'
embedding_model = 'models/gemini-embedding-001'
temperature = 0.5


llm = GoogleGenerativeAI(model=text_model,api_key=api_key)
llm.temperature = 0.1
pdf_folder_path = "/Users/kiruba-23243/Downloads/Profit and Loss"
loader = PyPDFDirectoryLoader(
    path = pdf_folder_path,
    glob = "**/[!.]*.pdf",
    silent_errors = False,
    load_hidden = False,
    recursive = False,
    extract_images = False,
    password = None,
    mode = "page",
    headers = None,
    extraction_mode = "plain",
)

pdf_index = VectorstoreIndexCreator(
        embedding=GoogleGenerativeAIEmbeddings(model=embedding_model),
        text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)).from_documents(loader.load())

pdf_chain = RetrievalQA.from_chain_type(llm=llm,
                            chain_type="stuff",
                            retriever=pdf_index.vectorstore.as_retriever(),
                            input_key="question")

app = Flask(__name__)
nltk.download('punkt')
nltk.download('stopwords')
app.config['SECRET_KEY']='579128bb0b13ce0c676dfde280ba245'
app.template_folder = 'templates'


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def summarize_pdf1(pdf_file_path):
    loader = PyPDFLoader(pdf_file_path)
    docs = loader.load_and_split()
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(docs)   
    return summary



#FUCTION FOR FETCHING THE TRENDING NEWS
def fetch_top_news():
    site = 'https://news.google.com/news/rss'
    op = urlopen(site)
    rd = op.read()
    op.close()
    sp_page = soup(rd, 'xml')
    news_list = sp_page.find_all('item')
    return news_list

#FUNCTION FOR FETCHING THE NEWS FOR TOPIC
def fetch_news_search_topic(topic):
    site = 'https://news.google.com/rss/search?q={}'.format(topic)
    op = urlopen(site)
    rd = op.read()
    op.close()
    sp_page = soup(rd, 'xml')
    news_list = sp_page.find_all('item')
    return news_list

#FUNCTION FOR FETCHING THE NEWS FOR SOME CATEGORY
def fetch_category_news(topic):
    site = 'https://news.google.com/news/rss/headlines/section/topic/{}'.format(topic)
    op = urlopen(site)
    rd = op.read()
    op.close()
    sp_page = soup(rd, 'xml')
    news_list = sp_page.find_all('item')
    return news_list



#FUCTION FOR RETURN THE NEWS DAT
def fetch_news_data(list_of_news,news_quantity):
   
    news_quantity = 5  # Example value

    news_list = []
    c=0

    for c, news in enumerate(list_of_news, start=1):
        if c > news_quantity:
            break

        news_data = Article(news.link.text)
        try:
            news_data.download()
            news_data.parse()
            news_data.nlp()
        except Exception as e:
            print("Error:", e)
            continue

        news_info = {
            'index': c,
            'title': news.title.text,
            'summary': news_data.summary,
            'source': news.source.text,
            'link': news.link.text,
            'published_date': news.pubDate.text,
          
        }
        news_list.append(news_info)

    return news_list

UPLOAD_FOLDER = ''
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/')
def index():
    return render_template('index.html')
# Define the route for the home page
@app.route('/home', methods=['GET', 'POST'])
def home():
    title = None
    summary = None
    
    if request.method == 'POST':
        url = request.form['url']
        title, summary = summarize_article_from_url(url)
        
    return render_template('index.html', title=title, summary=summary)

def summarize_article_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()
        original_title = article.title
        summary = article.summary

        return original_title, summary
    except Exception as e:
        print("Error summarizing the article:", str(e))
        return None, None
 
@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'GET':
        news_list = fetch_top_news()
        news_list=fetch_news_data(news_list,5)
        return render_template('result.html',news_list=news_list, title="top news")
    elif request.method == 'POST':
        topic = request.form.get('topic')
        category = request.form.get('category')
        if topic:
            news_list = fetch_news_search_topic(topic)
            news_list=fetch_news_data(news_list,5)
            return render_template('result.html',news_list=news_list, title=topic)
        elif category:
            news_list = fetch_category_news(category)
            news_list=fetch_news_data(news_list,5)
            return render_template('result.html',news_list=news_list, title=category)
    return render_template('result.html',news_list=news_list)

@app.route('/speech', methods=["GET", "POST"])
def speechText():
    transcript = ""
    
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        
        file = request.files["file"]
        
        if file.filename == "":
            return redirect(request.url)
        
        try:
            recognizer = sr.Recognizer()
            audioFile = sr.AudioFile(file)
            
            with audioFile as source:
                data = recognizer.record(source)
            
            transcript = recognizer.recognize_google(data, key=None)
        except sr.UnknownValueError:
            transcript = "Could not understand audio"
        except sr.RequestError as e:
            transcript = f"Error: {str(e)}"
    
    # Render the index.html template with the transcript
    return render_template('index.html', transcript=transcript)

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/process_user_input_bot1', methods=['POST'])
def process_user_input_bot1():
    user_message = request.json.get('userMessage')
    query = user_message or ""
    TEXT = search_engine.run(query)
    texts = text_splitter.split_text(TEXT)

    if not texts:
        return jsonify({'chatbotResponse': "I couldn't find relevant information to answer that right now."})

    # Embed all passages in a single batch
    embed_cfg = types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
    emb_resp = client.models.embed_content(
        model=embedding_model,
        contents=texts,
        config=embed_cfg
    )
    passage_vectors = np.array([e.values for e in emb_resp.embeddings], dtype=float)

    # Embed the query
    q_resp = client.models.embed_content(
        model=embedding_model,
        contents=query,
        config=embed_cfg
    )
    q_vec = np.array(q_resp.embeddings[0].values, dtype=float).reshape(1, -1)

    # Select best passage via cosine similarity
    sims = cosine_similarity(passage_vectors, q_vec).ravel()
    best_idx = int(np.argmax(sims))
    passage = texts[best_idx]

    # Build prompt and generate answer
    escaped = passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = textwrap.dedent("""You are a helpful and informative bot that answers questions using text from the reference passage included below. \
    Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
    However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
    strike a friendly and conversational tone. \
    If the passage is irrelevant to the answer, you may ignore it.
    QUESTION: '{query}'
    PASSAGE: '{relevant_passage}'

        ANSWER:
    """).format(query=query, relevant_passage=escaped)

    answer = client.models.generate_content(
        contents=prompt,
        model=text_model,
        config=types.GenerateContentConfig(
        temperature=0.1)
    )
    chatbot1_response = answer.text
    return jsonify({'chatbotResponse': chatbot1_response})

@app.route('/process_user_input_bot2', methods=['POST'])
def process_user_input_bot2():
    user_message = request.json.get('userMessage')
     

    # Dummy response for Chatbot 2 (you can replace this with your chatbot logic)
    chatbot2_response = pdf_chain.run(user_message)

    return jsonify({'chatbotResponse': chatbot2_response})

@app.route('/summarize', methods=['POST'])
def summarize_pdf():
    try:
        pdf_file = request.files['pdf_file']
        if pdf_file:
            pdf_reader = PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            # Tokenize the text into sentences
            sentences = sent_tokenize(text)
            
            # Remove stopwords and perform stemming
            stop_words = set(stopwords.words('english'))
            word_tokens = word_tokenize(text)
            filtered_text = [PorterStemmer().stem(word) for word in word_tokens if word.lower() not in stop_words]
            filtered_text = TreebankWordDetokenizer().detokenize(filtered_text)
            
            # Calculate word frequency
            fdist = FreqDist(filtered_text.split())
            most_common_words = fdist.most_common(10)  # Adjust the number of most common words
            
            # Sort sentences based on the presence of most common words
            ranked_sentences = [(sent, sum(1 for word in word_tokenize(sent) if word.lower() in most_common_words)) for sent in sentences]
            ranked_sentences.sort(key=lambda x: -x[1])
            
            # Extract the top sentences for summary (you can adjust the sentence_count for desired summary length)
            summary = " ".join(sent for sent, _ in ranked_sentences[:5])
            
            return render_template('index.html', summary=summary)
        else:
            return redirect(url_for('index'))
    except Exception as e:
        return render_template('index.html', error=str(e))
@app.route('/abs_summarize', methods=['POST','GET'])
def chatbot1():
    summarize = ""

    if request.method == 'POST':
        # Check if the POST request has a file attached
        if 'pdf_file' in request.files:
            pdf_file = request.files['pdf_file']
            if pdf_file.filename != '':
                # Securely save the uploaded file to the upload folder
                filename = secure_filename(pdf_file.filename)
                pdf_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

                # Call your function to summarize the uploaded PDF file
                pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                summarize = summarize_pdf1(pdf_path)  # You should replace this with your PDF summarization logic

    return render_template('index.html', summarize=summarize)


if __name__ == '__main__':
    app.run(debug=True)
