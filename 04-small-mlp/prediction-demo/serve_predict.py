# -*- coding: utf-8 -*-
"""为预测可视化页面提供静态文件和本地推理 API。"""

import argparse
import base64
import io
import json
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image


CHAPTER_DIR = Path(__file__).resolve().parent.parent
if str(CHAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(CHAPTER_DIR))

from model import CHECKPOINT_PATH, load_model
from predict_image import image_to_tensor, normalize_image, predict


BASE_DIR = Path(__file__).parent
MAX_IMAGE_BYTES = 5 * 1024 * 1024


def image_as_data_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def create_handler(model, metadata):
    class PredictionHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(BASE_DIR), **kwargs)

        def do_GET(self):
            if self.path == "/":
                self.path = "/predict.html"
            super().do_GET()

        def do_POST(self):
            if self.path != "/api/predict":
                self.send_error(404)
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0 or content_length > MAX_IMAGE_BYTES * 2:
                    raise ValueError("图片为空或过大，最大支持 5 MB")

                payload = json.loads(self.rfile.read(content_length))
                data_url = payload.get("image", "")
                if "," not in data_url:
                    raise ValueError("没有收到有效图片")

                encoded = data_url.split(",", 1)[1]
                image_bytes = base64.b64decode(encoded, validate=True)
                if len(image_bytes) > MAX_IMAGE_BYTES:
                    raise ValueError("图片过大，最大支持 5 MB")

                with Image.open(io.BytesIO(image_bytes)) as image:
                    normalized = normalize_image(image)
                predicted_digit, probabilities = predict(
                    model, image_to_tensor(normalized)
                )
                response = {
                    "prediction": predicted_digit,
                    "probabilities": probabilities.tolist(),
                    "processed_image": image_as_data_url(normalized.resize((280, 280))),
                    "test_accuracy": metadata.get("test_accuracy"),
                }
                self.send_json(200, response)
            except Exception as error:
                self.send_json(400, {"error": str(error)})

        def send_json(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            if args and str(args[1]) == "200":
                return
            super().log_message(format, *args)

    return PredictionHandler


def main():
    parser = argparse.ArgumentParser(description="启动 MNIST 图片预测页面")
    parser.add_argument("--port", type=int, default=8004)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    args = parser.parse_args()

    model, metadata = load_model(args.checkpoint)
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port), create_handler(model, metadata)
    )
    print(f"预测页面：http://127.0.0.1:{args.port}")
    print("按 Ctrl+C 停止")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务器已停止")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
