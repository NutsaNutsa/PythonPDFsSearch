#!/usr/bin/python3
import fillDB
import sys
import pymongo

Q = fillDB.cosine_similarity(int(sys.argv[1]), str(sys.argv[2]))