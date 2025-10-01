from langchain_openai import OpenAIEmbeddings
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import requests
import json
import re
from crisprgpt.safety import WARNING_PRIVACY, contains_identifiable_genes
from util import get_logger
import dotenv
import os

dotenv.load_dotenv()
logger = get_logger(__name__)


class FakeChatOpenAI:  ## For debug purpose only
    def __init__(self, **kwargs):
        pass

    def __call__(self, inputs):
        logger.info("FakeChatOpenAI Called")
        response = input()
        return AIMessage(content=response)


class IdentifiableGeneError(ValueError): 
    pass


class OpenAIChat:
    google_api_key = os.getenv("GOOGLE_API_KEY")

    # Single model: Gemini 2.5 Flash with JSON output
    model = ChatGoogleGenerativeAI(
        google_api_key=google_api_key,
        model="gemini-2.5-flash",
        temperature=0.9,
        model_kwargs={
            "generation_config": {
                "response_mime_type": "application/json"
            }
        }
    )

    @classmethod
    def chat(cls, request, use_GPT4=True, use_GPT4_turbo=False):
        if contains_identifiable_genes(request):
            raise IdentifiableGeneError(WARNING_PRIVACY)
        
        response = cls.model.invoke(request).content
        logger.info(response)

        # Postprocessing
        response = response.lstrip("```json")
        response = response.lstrip("```")
        response = response.rstrip("```")
        response = response.strip()

        json_response = json.loads(response)
        return json_response

    @classmethod
    def QA(cls, request, use_GPT4=False):
        return "QA is not supported in the lite version."