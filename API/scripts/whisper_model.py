import whisper

class Whisper_model:

    def __init__(self, weights="weights/large-v3.pt", DEVICE='cuda'):
        self.model = whisper.load_model(weights, device=DEVICE)

    def predict(self, path="data/new_video.mp3"):
        result = self.model.transcribe(path, language='ru', verbose=False, beam_size=5, best_of=5)
        return result
