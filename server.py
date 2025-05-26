from fastapi import FastAPI, HTTPException, UploadFile, File, Query
import numpy as np
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests
import json
from server_utils import resize_image, make_classification, make_explain, decode_image
from ava_aesthetic_predictor.make_prediction import make_prediction_pretrained_ava
from loguru import logger
from PIL import Image
import io
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageFilter
import io
import base64
import numpy as np
from ultralytics import YOLO
from cv2analytics import PhotoQualityAnalyzer

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/classify")
async def classify(
    file: UploadFile = File(...),
):
    # try:
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    resized_image = resize_image(image)

    model = YOLO("path_to_model")
    model.cpu()

    estetica_result = make_classification(model, resized_image)
    logger.info(f"{estetica_result=}")
    ava_estetica_result = make_prediction_pretrained_ava(image)
    logger.info(f"{ava_estetica_result=}")

    explained_image = make_explain(
        model,
        resized_image,
    )

    analyzer = PhotoQualityAnalyzer(image)
    is_blured, blur_fm = analyzer.check_blur()
    is_dark_gray, is_bright_gray, dark_share_gray, bright_share_gray = (
        analyzer.check_exposure_on_gray()
    )
    is_dark_hsv, is_bright_hsv, dark_share_hsv, bright_share_hsv = (
        analyzer.check_exposure_on_hsv()
    )
    is_noised, noise_mse = analyzer.check_noise()

    person_count, image_with_persons = analyzer.person_detection()

    is_good_horizon, _, image_with_horizontal_lines = analyzer.check_horizon()

    result_response = {
        "status": "success",
        "estetica_result": estetica_result,  # текст
        "ava_estetica_result": ava_estetica_result,  # текст
        "yolo_explained_image": decode_image(explained_image),  # изображение
        "image_with_horizontal_lines": decode_image(
            image_with_horizontal_lines
        ),  # изображение
        "image_with_persons": decode_image(image_with_persons),  # изображение
        "person_count": person_count,  # int
        "check_blur": bool(is_blured),  # bool
        "is_dark_gray": bool(is_dark_gray),  # bool
        "is_bright_gray": bool(is_bright_gray),  # bool
        "is_dark_hsv": bool(is_dark_hsv),  # bool
        "is_bright_hsv": bool(is_bright_hsv),  # bool
        "is_noised": bool(is_noised),  # bool
        "is_good_horizon": bool(is_good_horizon),  # bool
    }
    for k in result_response.keys():
        if k in [
            "person_count",
            "check_blur",
            "is_dark_gray",
            "is_bright_gray",
            "is_dark_hsv",
            "is_bright_hsv",
            "is_noised",
            "is_good_horizon",
        ]:
            print(k, result_response[k])
    return JSONResponse(result_response)
    # except Exception as e:
    #     raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8010)
