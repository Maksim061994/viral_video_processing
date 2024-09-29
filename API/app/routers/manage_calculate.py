from fastapi import APIRouter
from app.config.settings import get_settings
from app.handlers.calculate.schemas import RequestCalculate
from app.handlers.calculate.manager import ManagerViralVideo
from fastapi import HTTPException



manage_calculate_routers = APIRouter()
settings = get_settings()


@manage_calculate_routers.post("/video")
async def run_calculate(request: RequestCalculate):
    """
    Запуска задачи на расчет видео
    :param request:
    :return:
    """
    response = ManagerViralVideo(request)
    try:
        result = response.process()
    except Exception as err:
        raise HTTPException(status_code=500, detail=err)
    return result
