from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        text = request.form['text']
        labels = request.form['labels'].split(',')
        result = classifier(text, labels)
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
