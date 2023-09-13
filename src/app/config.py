from pydantic import BaseSettings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class Settings(BaseSettings):
    mongodb: str
    db: str
    yolo_weights: str
    yolo_cfg: str
    encodings: str
    classes: str
    
    class Config:
        env_file = "../.env"

settings = Settings() # type: ignore