# Warsaw Nights of Museums Recommendations

## Intro

The Long Night of Museums is an annual event held in various Polish cities, including Warsaw. On this night, numerous museums and cultural institutions open their doors to the public, offering unique events.

In 2024, 309 institutions are participating in this initiative, presenting a wide array of events to choose from.

To optimize my time and ensure I attend the events that most appeal to me, I decided to leverage my professional expertise in Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and parsing.

## Running

Build the docker container:
```
docker build -t museums_suggestions:latest .
```

Run the docker container:
```
docker run --rm -v $(pwd)/keys:/workspace/keys -v $(pwd)/outputs:/workspace/outputs museums_suggestions python rag_question.py --top_recommendations 5 --interests "culture, art, military, psychology, politcs" --output /workspace/outputs/suggestions.html
```

The output will be stored in `CURRENT_DIRECTORY/outputs/suggestions.html` file.