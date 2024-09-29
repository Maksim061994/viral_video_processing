from app.config.settings import get_settings
import requests
import json


settings = get_settings()


class LlamaConnector:

    def __init__(self):
        self.url_llama_model = settings.url_llama_model
        self.user_login_llama = settings.user_login_llama
        self.password_login_llama = settings.password_login_llama
        self.token = self.authorize()

    def authorize(self):
        payload = {
            "login": "vniizht",
            "password": "@Z&WSSv?5|AVhTeD"
        }
        headers = {
            'Content-Type': 'application/json',
        }

        response = requests.post(self.url_llama_model + "/users/login", headers=headers, json=payload)

        return json.loads(response.text)["token"]

    def predict(self, messages):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.token}'
        }
        response = requests.post(
            self.url_llama_model + '/models/predict',
            data=json.dumps({"data_for_predict": messages}),
            headers=headers
        )
        return json.loads(response.text)["result"]








