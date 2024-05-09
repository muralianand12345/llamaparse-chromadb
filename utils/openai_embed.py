from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

class OpenAIEmbed:
    def __init__(self, embedding_model, generator_model):
        self.embedding_model = embedding_model
        self.generator_model = generator_model

    def init_embedding(self):
        llm = OpenAI(model=self.generator_model)
        embed_model = OpenAIEmbedding(model=self.embedding_model)
        return llm, embed_model
    
