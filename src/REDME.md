API定义:


## Example

#### example1:
```python
#!/usr/bin/python 
# -*-coding=utf-8 -*- 

import json 
import urllib 
import urllib.request
#import urllib2 
#import  http


def post(server_url, params): 
    #data = urllib.urlencode(params) 
    #data = urllib.parse.urlencode(params)
    data = urllib.parse.urlencode(params).encode('utf-8')
    #request = urllib.request.Request(server_url, data) 

    #return json.loads(urllib.request.urlopen(request, timeout=10).read()) 
    return json.loads(urllib.request.urlopen(server_url,data).read().decode("utf-8")) 




if __name__ == "__main__": 
    # 图片的URL 
    image_url = "https://ss0.bdstatic.com/70cFvHSh_Q1YnxGkpoWK1HF6hhy/it/u=2637851855,2992656155&fm=200&gp=0.jpg" 
    # http服务的URL 
    url = "http://127.0.0.1:6060/image?" 
    print (post(url, {"image_url": image_url}))
```


