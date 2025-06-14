import os
from flask import Flask, render_template, request
from qa_chain import extract_text_from_pdf, create_qa_chain_from_text

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

qa_chain = None

@app.route("/", methods=["GET", "POST"])
def index():
    global qa_chain
    answer = None

    if request.method == "POST":
        try:
            if 'pdf' in request.files:
                file = request.files['pdf']
                if file.filename == '':
                    answer = "⚠️ No file selected."
                else:
                    # Save uploaded file temporarily
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                    file.save(file_path)

                    # Extract text and create QA chain
                    text = extract_text_from_pdf(file_path)
                    qa_chain = create_qa_chain_from_text(text)

                    answer = "PDF uploaded and processed successfully. You can ask questions now."
            elif 'question' in request.form:
                question = request.form['question']
                if not qa_chain:
                    answer = "⚠️ Please upload a PDF first."
                else:
                    answer = qa_chain.run(question)
        except Exception as e:
            print(f"Exception occurred: {e}")
            answer = f"⚠️ Server error: {e}"

    return render_template("index.html", answer=answer)


if __name__ == "__main__":
    app.run(debug=True)
