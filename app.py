import json
import flask
import pandas as pd
import sklearn
from flask import request
from flask_cors import CORS, cross_origin
import numpy as np
import pickle
from scipy.spatial import distance_matrix

HEROKU_ON = True


DATA_LOADED = False

if HEROKU_ON:
    path = ''
else:
    path = 'D:\\best_recall\\'

file = open(path + "vectorizer_test.pickle",'rb')
vectorizer_encoder = pickle.load(file)
file.close()

file = open(path + "kmeans_test.pickle",'rb')
kmeans = pickle.load(file)
file.close()

DATA_LOADED = True

app = flask.Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

# A welcome message to test our server
@app.route('/')
def index():
    return "<h1>Best recall</h1>"


@app.route('/ping')
def ping():

    global DATA_LOADED

    if DATA_LOADED:
        result = {'status': 'ok'}
    else:
        result = {'status': 'data not loaded'}
    return result

@app.route('/test_output')
def test():

    with open('test_input.txt', encoding="utf-8") as f:
        test_input = f.readlines()

    vect_repr = vectorizer_encoder.transform(test_input)

    bag_of_critical_words = vectorizer_encoder.inverse_transform(vect_repr)[0].tolist()

    return flask.jsonify({'result': bag_of_critical_words})

@app.route('/analyze', methods=['POST'])
def query():
    data = request.json

    input = data['input']

    vect_repr = vectorizer_encoder.transform([input])

    first_feat = vect_repr.nnz
    second_feat = len(np.unique(vect_repr.toarray()))
    third_feat = np.sum(vect_repr.toarray())
    result = kmeans.predict([[first_feat, second_feat, third_feat]])

    centroids = kmeans.cluster_centers_
    dist_mat = pd.DataFrame(distance_matrix([[first_feat, second_feat, third_feat]], centroids))
    dist_mat['scores'] = (first_feat + second_feat + third_feat) / 120#1 - dist_mat[1] / (dist_mat[0] + dist_mat[1])

    bag_of_critical_words = vectorizer_encoder.inverse_transform(vect_repr)[0].tolist()

    total_score = (first_feat + second_feat + third_feat)

    rating = 0

    if total_score < 10:
        rating = 0.1
    elif total_score < 20:
        rating = 0.4
    elif total_score < 30:
        rating = 0.7
    else:
        rating = 0.9

    return flask.jsonify({
        'risk_rating': rating,
        'bag_of_words': bag_of_critical_words
    })


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)

