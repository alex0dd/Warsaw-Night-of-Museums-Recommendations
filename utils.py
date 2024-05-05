import json

from langchain_core.documents.base import Document

import document_extraction.extract_pdf_data as parser


def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def load_documents(fname):
    documents = parser.load_file(fname)
    documents = parser.parse_document_elements(documents)
    langchain_documents = []
    for document in documents:
        ldoc = Document(
            page_content=parser.event_entry_to_string(document),
            metadata={
                "title": document["title"],
                "organizer": document["organizer"],
                "site": document["site"],
                "district": document["address"]["district"],
                "street": document["address"]["street"],
            },
        )
        langchain_documents.append(ldoc)
    return langchain_documents


def write_suggestions_stdout(output):
    print(f"Question:\n{output['query']}\n\n")
    print(f"Answer:\n{output['result']}\n\n")
    print("Sources:")
    for source in output["source_documents"]:
        metadata = source.metadata
        print(
            f"{metadata['title']} - {metadata['organizer']} - {metadata['site']} - {metadata['district']} - {metadata['street']}"
        )


def write_suggestions_file(output, fname):
    sources = [
        f"<li>{source.metadata['title']} - {source.metadata['organizer']} - <a href='{source.metadata['site']}'>{source.metadata['site']}</a> - {source.metadata['district']} - {source.metadata['street']}</li>"
        for source in output["source_documents"]
    ]
    sources = "\n".join(sources)
    contents = f"""
    <html>
        <head>
            <title>Museums suggestions</title>
            <meta charset="UTF-8">
            <script type="module" src="https://md-block.verou.me/md-block.js"></script>
        </head>
        <body>
            <h1>Night of Museums Suggestions</h1>
            <md-block>
            {output['result']}
            </md-block>
            <h2>Sources</h2>
            <ul>{sources}</ul>
        </body>
    </html>
    """
    with open(fname, "w") as f:
        f.write(contents)
