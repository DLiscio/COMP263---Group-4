from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def homepage():
    return render_template('home.html')

@app.route("/supervised-model")
def supervisedModel():
    return render_template('supervised.html')

@app.route("/unsupervised-model")
def unsupervisedModel():
    return render_template('unsupervised.html')

@app.route("/sota-model")
def sotaModel():
    return render_template('sota.html')

# Run Flask App
if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=8501, debug=True)