import os
import fitz  # PyMuPDF
from flask import Flask, render_template, request, session
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)
app.secret_key = '123abc'  # Needed for Flask session

# Set Groq API Key
GROQ_API_KEY = "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"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
llm = ChatGroq(model="llama3-70b-8192", groq_api_key=GROQ_API_KEY)

# Updated prompt and chain using LangChain v0.1.17+
prompt = ChatPromptTemplate.from_template(
    "Given the following PDF content:\n\n{context}\n\nAnswer the question:\n{question}"
)
chain = prompt | llm  # This replaces deprecated LLMChain

def extract_text_from_pdf(file):
    text = ""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text

@app.route("/", methods=["GET", "POST"])
def index():
    answer = None

    if request.method == "POST":
        if 'pdf' in request.files:
            file = request.files['pdf']
            context = extract_text_from_pdf(file)
            session['context'] = context
        elif 'question' in request.form:
            question = request.form['question']
            context = session.get('context', '')
            if context:
                result = chain.invoke({"context": context, "question": question})
                answer = result.content if hasattr(result, "content") else result
            else:
                answer = "Please upload a PDF first."

    return render_template("index.html", answer=answer, context=session.get('context'))

if __name__ == "__main__":
    app.run(port=9000)
