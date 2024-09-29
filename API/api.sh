#!/bin/sh -ex
#celery -A app.main_celery.app_celery beat -l debug &
gunicorn app.main_app:app -k uvicorn.workers.UvicornWorker &

#celery -A app.main_celery.app_celery worker -E --pool=prefork -O fair -c 4 -l INFO &
#celery -A app.main_celery.app_celery flower &

tail -f /dev/null
