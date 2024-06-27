from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from module.sentiment import Senlyzer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

senlyzer = Senlyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        input_text = request.form['text']
        print(f"Received text: {input_text}")
        senlyzer.text_input(input_text)
        text_score, text_sentiment = senlyzer.text_sentiment()

        image_score, image_sentiment = None, None
        if 'image' in request.files:
            image = request.files['image']
            if image.filename != '':
                filename = secure_filename(image.filename)
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image.save(image_path)
                print(f"Image saved to {image_path}")
                senlyzer.image_input(image_path)
                image_score, image_sentiment = senlyzer.image_sentiment()
                os.remove(image_path)
                print(f"Image processed and deleted: {image_path}")
        
        combined_score, combined_sentiment = senlyzer.combined_sentiment(text_score[0], image_score)

        result = {
            'text_result': {
                'score': float(text_score[0]),
                'sentiment': text_sentiment
            },
            'image_result': {
                'score': float(image_score) if image_score is not None else None,
                'sentiment': 'NONE'
            }
        }

        return jsonify({'result': result})

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
