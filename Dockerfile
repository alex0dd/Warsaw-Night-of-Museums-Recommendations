FROM python:3.12

WORKDIR /workspace

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY data ./data
COPY document_extraction ./document_extraction
COPY rag_question.py ./
COPY utils.py ./

ENV LLM_MODEL=gpt-4-turbo
ENV OPENAI_KEY_PATH=keys/openai_key.json

CMD [ "python", "/workspace/rag_question.py", "--top_recommendations", "5", "--interests", "technology, sports, music", "--output", "/workspace/outputs/suggestions.html"]