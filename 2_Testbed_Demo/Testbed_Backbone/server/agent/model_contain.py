
## Since the tool class cannot pass parameters, global variables are used to pass the model and the corresponding knowledge base introduction
class ModelContainer:
    def __init__(self):
        self.MODEL = None
        self.DATABASE = None

model_container = ModelContainer()
