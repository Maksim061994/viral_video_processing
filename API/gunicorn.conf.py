import multiprocessing


# -b
bind = "0.0.0.0:8000"

# -w
workers = 1 # multiprocessing.cpu_count() * 1 + 2
worker_connections = 100
# threads = 1
max_requests = 3000

# -t
timeout = 6000

# логирование
accesslog = '-'

# worker_class=uvicorn.workers.UvicornWorker
debug = False
logfile = '/var/log/gunicorn/debug.log'
loglevel = 'debug'
