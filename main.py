from concurrent.futures import ThreadPoolExecutor
import time

from fastapi import FastAPI
from loguru import logger

from trackers.eye_tracker import track_eye
from trackers.head_pose_estimation import detect_head_pose
from trackers.mouth_opening_detector import mouth_opening_detector
from trackers.person_and_phone import detect_phone_and_person


log_format = "{time} | {level}: {message}"
logger.add("logs/app.log", format=log_format, level="INFO")


app = FastAPI()


@app.post("/analyze-video")
def read_root(video_url: str = None):
    start_time = time.time()
    logger.info("Starting video analysis")

    res_dict = {}

    with ThreadPoolExecutor() as executor:
        future_phone = executor.submit(detect_phone_and_person, video_url, res_dict)
        future_eye = executor.submit(track_eye, video_url, res_dict)
        future_head = executor.submit(detect_head_pose, video_url, res_dict)
        future_mouth = executor.submit(mouth_opening_detector, video_url, res_dict)

        res_dict = future_phone.result()
        res_dict = future_eye.result()
        res_dict = future_head.result()
        res_dict = future_mouth.result()
        
    logger.info(f"Res dict: {res_dict}")
    logger.info(f"Proctoring completed in {round(time.time() - start_time, 2)} seconds")

    return {"message": "success", "data": res_dict}
