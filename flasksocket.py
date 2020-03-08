from flask import Flask, render_template
from flask_sockets import Sockets
import datetime
import time
import random


def socket_process_jp(queue):
    app = Flask(__name__)
    sockets = Sockets(app)

    @app.route('/')
    def index():
        return "hello this is socket only port"


    @sockets.route('/')
    def echo_socket(ws):
        while not ws.closed:
            text = queue.get()
            ws.send(text)  #发送数据
            time.sleep(1)


    from gevent import pywsgi
    from geventwebsocket.handler import WebSocketHandler
    server = pywsgi.WSGIServer(('0.0.0.0', 5001), app, handler_class=WebSocketHandler)
    print('server start')
    server.serve_forever()

def socket_process_cn(queue):
    app = Flask(__name__)
    sockets = Sockets(app)

    @app.route('/')
    def index():
        return "hello this is socket only port"


    @sockets.route('/')
    def echo_socket(ws):
        while not ws.closed:
            text = queue.get()
            ws.send(text)  #发送数据
            time.sleep(1)


    from gevent import pywsgi
    from geventwebsocket.handler import WebSocketHandler
    server = pywsgi.WSGIServer(('0.0.0.0', 5002), app, handler_class=WebSocketHandler)
    print('server start')
    server.serve_forever()

def http_process():
    app = Flask(__name__)

    @app.route('/')
    def index():
        return render_template('socket.html', name="字幕")

    app.run(host="0.0.0.0", port=5000)


if __name__ == "__main__":
    import signal
    import sys
    
    
    from multiprocessing import Process, Array, Value, Semaphore, Queue
    queue = Queue(10)
    p1 = Process(target=socket_process_cn, args=(queue,))
    p1.start()
    

    p2 = Process(target=http_process)
    p2.start()


    while 1:
        import time
        time.sleep(1)
        queue.put("test {}".format(time.time()))