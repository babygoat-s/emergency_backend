"""LangChain LCEL prompt/model/parser pipeline; the Chroma retriever stays local."""

import json

import httpx
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import tracing_context

from app.rag.enrichment import Extraction


class CancellableTransport(httpx.BaseTransport, httpx.AsyncBaseTransport):
    """Use HTTPX for both SDK paths, including cancellation during fallback.

    Delegating to clients preserves standard proxy/certificate environment settings.
    The installed SDK's aiohttp error path can leave response.json unawaited on timeout.
    """

    def __init__(self):
        self.sync = httpx.Client()
        self.asynchronous = httpx.AsyncClient()

    def handle_request(self, request):
        return self.sync.send(request, stream=True)

    async def handle_async_request(self, request):
        return await self.asynchronous.send(request, stream=True)

    def close(self):
        self.sync.close()

    async def aclose(self):
        await self.asynchronous.aclose()


class GroundedChain:
    def __init__(self, settings, system, schema):
        self.settings, self.system, self.schema = settings, system, schema
        self.models = {}
        self.schema_disabled = set()

    def model(self, name, key):
        if name not in self.models:
            self.models[name] = ChatGoogleGenerativeAI(
                model=name,
                api_key=key,
                vertexai=False,
                timeout=max(10, self.settings.rag_llm_attempt_timeout_seconds),
                max_retries=0,
                max_output_tokens=4096,
                client_args={"transport": CancellableTransport()},
            )
        return self.models[name]

    async def invoke(self, name, key, payload):
        model = self.model(name, key)
        system = self.system
        if name in self.schema_disabled:
            system += "\nJSON 출력 형식:\n" + json.dumps(self.schema, ensure_ascii=False)
            output = model.bind(response_mime_type="application/json") | PydanticOutputParser(
                pydantic_object=Extraction
            )
        else:
            output = model.with_structured_output(
                self.schema, method="json_schema"
            ) | RunnableLambda(Extraction.model_validate, name="validate_grounded_output")
        prompt = ChatPromptTemplate.from_messages([("system", "{system}"), ("human", "{payload}")])
        chain = (
            RunnableLambda(
                lambda data: {
                    "system": system,
                    "payload": json.dumps(data, ensure_ascii=False),
                },
                name="format_retrieved_manual_context",
            )
            | prompt
            | output
        )
        # Only the configured model API receives dialogue; never auto-enable third-party traces.
        with tracing_context(enabled=False):
            result = await chain.ainvoke(payload, config={"callbacks": []})
        return result.model_dump_json()

    async def close(self):
        for model in self.models.values():
            await model.client.aio.aclose()
            model.client.close()
