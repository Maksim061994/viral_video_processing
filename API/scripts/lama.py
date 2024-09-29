import os

# если залезет
os.environ['PARAM_N_CTX'] = 4096

logged_model = f'runs😕ddb306a6b60140009c41d569eea39f1c/base_llama3_8b'
model_loaded = mlflow.pyfunc.load_model(logged_model)
unwrapped_model = model_loaded.unwrap_python_model()