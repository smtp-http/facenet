#!/usr/bin/python 
# -*-coding=utf-8 -*- 

import sys 
import traceback 
#from StringIO 
try:
    from StringIO import StringIO 
except ImportError:
	from io import StringIO
import io
import requests 
import tornado 
import tornado.ioloop 
import tornado.web 
from tornado.escape import json_encode 
from PIL import Image 
import imp


imp.reload(sys) 
#sys.setdefaultencoding('utf8') 

class Handler(tornado.web.RequestHandler):
    def post(self): 
        result = {} 
        image_url = self.get_argument("image_url", default="") 
        print (image_url)
        if not image_url: 
            result["msg"] = "no image url" 
        else: 
            result["msg"] = self.process(image_url) 

        self.write(json_encode(result)) 
    def process(self, image_url): 
        """
        根据image_url获取图片进行处理
        :param image_url: 图片url
        """
        try: 
            response = requests.get(image_url) 
            if response.status_code == requests.codes.ok: 
                obj = Image.open(
                	io.BytesIO(response.content))
                # TODO 根据obj进行相应的操作 
                return "ok" 
            else: 
                return "get image failed." 
        except Exception as e: 
            print (traceback.format_exc() )
            return str(e)


class ImageServer(object): 
    def __init__(self, port): 
        self.port = port 
    def process(self): 
        app = tornado.web.Application([(r"/image?", Handler)], ) 
        app.listen(self.port) 
        tornado.ioloop.IOLoop.current().start() 








if __name__ == "__main__": 
    server_port = "6060"
    server = ImageServer(server_port) 
    print ("begin server" )
    server.process()
