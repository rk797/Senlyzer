from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from module.sentiment import Senlyzer


app = Flask(__name__)
senlyzer = Senlyzer()

# model_name = "finiteautomata/bertweet-base-sentiment-analysis"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name)
# sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    input_text = request.form['text']
    senlyzer.text_input(input_text)
    score, sentiment = senlyzer.text_sentiment() 
    result = {
        'score': float(score[0]), 
        'sentiment': sentiment
    }
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
