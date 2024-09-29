from pydantic import BaseModel


class RequestCalculate(BaseModel):
    name_video: str
