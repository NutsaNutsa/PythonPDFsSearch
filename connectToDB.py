#!/usr/bin/python3
print("gamshvebi")


import pymongo

uri = "mongodb://127.0.0.1:27017"
client = pymongo.MongoClient(uri)
database = client['examples']
print("ok here")