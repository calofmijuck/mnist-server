import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import numpy as np
import cv2
import base64
import urllib
from cnn.simple_conv_net import SimpleConvNet

WIDTH, HEIGHT = 28, 28
DIMENSION = (WIDTH, HEIGHT)

SAVED_PARAMS = "cnn/params.pkl"


def base64ToArray(uri):
    encoded_data = uri.split(",")[1]
    data = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(grayscale, DIMENSION, interpolation=cv2.INTER_LINEAR)

    array = (255 - resized) / 255.0
    return array


class S(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "http://www.zxcvber.com")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST")
        self.end_headers()

    def _html(self, message):
        content = f"<html><body><h1>{message}</h1></body></html>"
        return content.encode("utf8")

    def do_GET(self):
        self._set_headers()
        self.wfile.write(self._html("Hi!"))

    def do_HEAD(self):
        self._set_headers()

    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        data_string = self.rfile.read(content_length)

        result = handle_request(data_string)
        self._set_headers()
        self.wfile.write(bytes(result, "utf-8"))

    def do_OPTIONS(self):
        self._set_headers()


def handle_request(data_string):
    data = urllib.parse.unquote_to_bytes(data_string).decode("utf-8")
    data_arr = base64ToArray(data)
    return json.dumps({"ans": int(predict(data_arr))})


def run(server_class=HTTPServer, handler_class=S, addr="localhost", port=8000):
    server_address = (addr, port)
    httpd = server_class(server_address, handler_class)

    print(f"Starting httpd server on {addr}:{port}")
    httpd.serve_forever()


def init_model():
    global network
    network = SimpleConvNet()
    network.load_params(SAVED_PARAMS)


def predict(x):
    x = np.array(x).reshape(1, 1, 28, 28)
    return np.argmax(network.predict(x))


if __name__ == "__main__":
    init_model()
    parser = argparse.ArgumentParser(description="Run a simple HTTP server")
    parser.add_argument(
        "-l",
        "--listen",
        default="localhost",
        help="Specify the IP address on which the server listens",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8000,
        help="Specify the port on which the server listens",
    )
    args = parser.parse_args()
    run(addr=args.listen, port=args.port)
