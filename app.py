from flask import Flask
from src.logger import logging


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    logging.info("We are testing our second method of logging")
    return "Welcome to Bhang Bhosda Bhaiyya Madarjaat"


if __name__ == '__main__':
    app.run(debug=True, port=8080)
