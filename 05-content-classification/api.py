# -*- coding: utf-8 -*-
"""FastAPI 图片分类接口。"""

import io
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from inference import predict_pil
from model import DEFAULT_CHECKPOINT, load_model


MAX_IMAGE_BYTES = 5 * 1024 * 1024


def create_app(checkpoint_path=DEFAULT_CHECKPOINT):
    app = FastAPI(title="Content Classification API", version="0.1.0")
    runtime = {}

    def get_runtime():
        if not runtime:
            model, metadata = load_model(Path(checkpoint_path), device="cpu")
            runtime.update(model=model, metadata=metadata)
        return runtime["model"], runtime["metadata"]

    @app.get("/health")
    def health():
        _, metadata = get_runtime()
        return {
            "status": "ok",
            "model": metadata.get("backbone", "resnet18"),
            "threshold": metadata["threshold"],
        }

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        if file.content_type and not file.content_type.startswith("image/"):
            raise HTTPException(status_code=415, detail="只支持图片文件")

        content = await file.read(MAX_IMAGE_BYTES + 1)
        if len(content) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="图片不能超过 5 MB")

        try:
            with Image.open(io.BytesIO(content)) as image:
                rgb_image = image.convert("RGB")
        except (UnidentifiedImageError, OSError) as error:
            raise HTTPException(status_code=400, detail="无法读取图片") from error

        model, metadata = get_runtime()
        result = predict_pil(model, rgb_image, metadata, device="cpu")
        return {"filename": file.filename, **result}

    return app


app = create_app()
