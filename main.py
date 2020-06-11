#!/usr/bin/python3
import json

from flask import Flask, request

from run import Runner

app = Flask(__name__)

app.config["runner"] = Runner.initialize()


@app.route("/ping")
def ping():
    return "OK"


@app.route("/similarity")
def run_similarity():
    query = request.args.get("query")
    count = int(request.args.get("count", 10))
    runner: Runner = app.config["runner"]
    return json.dumps(runner.cosine_similarity(count, query))


@app.route("/parse_new_file", methods=["POST"])
def update_data():
    file = request.files['file']
    data = file.read().decode()
    title = request.args.get("title", file.filename)
    runner: Runner = app.config["runner"]
    runner.process_new_text(title, data)
    return {}


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)

# Q = run.cosine_similarity(int(sys.argv[1]), str(sys.argv[2]))
