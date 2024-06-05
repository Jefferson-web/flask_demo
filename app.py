from flask import Flask, jsonify
from flask import request
from paddleocr import PaddleOCR,draw_ocr
from PIL import Image
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

app = Flask(__name__)

ocr = PaddleOCR(use_angle_cls=True, lang='es')

@app.route('/ping', methods=['POST'])
def ping():
    return 'pong'

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    image = Image.open(file)
    image = np.array(image)
    result = ocr.ocr(image, cls=True)
    full_text = ""
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            text = line[1][0]
            full_text += text + " "
    
    documents = [full_text]
    # Procesar TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

    # Obtener las palabras más relevantes
    top_words = tfidf_df.T.nlargest(10, 0).index.tolist()
    top_words_tfidf = tfidf_df.T.nlargest(10, 0).values.flatten().tolist()

    # Crear un diccionario de palabras y sus valores TF-IDF
    top_words_dict = {word: tfidf for word, tfidf in zip(top_words, top_words_tfidf)}

    print(documents)

    return jsonify({ 'message': 'Ok', 'status': True, 'data': full_text.strip(), 'tfidf': top_words_dict })

if __name__ == '__main__':
    app.run(port=4000)