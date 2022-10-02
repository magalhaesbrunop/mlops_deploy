from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
from textblob import TextBlob
import pickle
import os

colunas = ['tamanho', 'ano', 'garagem']
model = pickle.load(open('MLOPS_DEPLOY/models/model.sav', 'rb'))

app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME')
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('BASIC_AUTH_PASSWORD')

basic_auth = BasicAuth(app)

@app.route("/")
def home():
    return "Minha API."

@app.route("/sentimento/<frase>")
@basic_auth.required
def sentiment(frase):
    tb = TextBlob(frase)
    tb_en = tb.translate(from_lang="pt_br", to="en")
    polarity = tb_en.sentiment.polarity
    return f"polaridade: {polarity}"

@app.route("/cotacao/", methods=['POST'])
@basic_auth.required
def cotacao():
    dados = request.get_json()
    dados_input = [dados[col] for col in colunas]
    preco = model.predict([dados_input])
    return jsonify(preco=preco[0])

app.run(debug=True, host='0.0.0.0')