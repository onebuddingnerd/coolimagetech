
from http.server import BaseHTTPRequestHandler, HTTPServer
import time
import urllib
import pickle
import os

class MyServer(BaseHTTPRequestHandler):

	
    def do_GET(self):
    	return 1


    def do_POST(self):
    	return 1




# Serve the webpage forever

hostName = "localhost"
serverPort = 8080

if __name__ == "__main__":        
    webServer = HTTPServer((hostName, serverPort), MyServer)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")
    print("Server stopped.")