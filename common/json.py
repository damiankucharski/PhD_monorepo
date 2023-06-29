import json

class Json:
    @staticmethod
    def load(path):
        with open(path, "r") as file:
            return json.load(file)

    @staticmethod
    def save(path, dictionary):
        with open(path, "w") as file:
            json.dump(dictionary, file)