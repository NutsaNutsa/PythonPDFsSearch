# README


### To run the program, first install the environment
```console
$ python3 --version  # Should be higher than3.6
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
$ python main.py 
```

### Check if the service is running
`GET /ping`


### Request
```
GET /similarity?query={{query}}&count={{count}}
Query params: 
    - query: keyword or list of keywords to search for
    - count: number of top documents matching query    
```


### Add new file
```REST
POST /parse_new_file?title={{file_name}}
headers:
    Content-type: multipart/form-data;
body:
    file: <file>
```