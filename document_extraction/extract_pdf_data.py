import re

from py_pdf_parser.loaders import load_file


def has_time(text):
    """
    Returns a tuple containing a tuple (from, to) and whether the tuple is found or not (None if not found)
    """
    regexp_time = r"from (\d\d:\d\d) to (\d\d:\d\d)"
    matched = re.match(regexp_time, text, re.IGNORECASE)
    return matched, matched is not None


def has_delimiter(text):
    document_finish_delimiter = "The organizer of the event is responsible for the content and reliability of the information provided"
    return text.strip() in document_finish_delimiter.strip()


"""
offsets:
0 = time
1 = audience + title + category
2 = address place title
3 = address street
4 = address city
5 = address district
6 = organizer
7 = site
8 = social media site
9 = "Event details:" string
10-10+K = K description lines
10+K = "Practical information:" string
10+K+1 - 10+K+1+M = M lines of decription
10+K+1+M + 1 = "The organizer of the event is responsible for the content and reliability of the information provided" string
"""


def parse_document_elements(document):
    documents_list = []
    parsed_document = None
    details_start_idx = -1

    elements = document.elements

    for i in range(len(elements)):
        from_to, is_time = has_time(elements[i].text())
        if is_time:
            # Reinitialize the parsing process
            details_start_idx = -1
            parsed_document = {"time_range": {"from": from_to[0], "to": from_to[1]}}

            org_str = elements[i + 1].text().split("\n")
            parsed_document["audience"] = org_str[0]
            parsed_document["title"] = org_str[1]
            parsed_document["categories"] = org_str[2]
            parsed_document["address"] = {
                "place": elements[i + 2].text(),
                "street": elements[i + 3].text(),
                "city": elements[i + 4].text(),
                "district": elements[i + 5].text(),
            }
            parsed_document["organizer"] = (
                elements[i + 6].text().split("\n")[-1]
            )  # exclude "Organizer:"
            parsed_document["site"] = elements[i + 7].text()
            if "Event details" not in elements[i + 8].text():
                parsed_document["social_media"] = elements[i + 8].text()
                details_start_idx = i + 9
            else:
                parsed_document["social_media"] = None
                details_start_idx = i + 8

        else:
            is_delimiter = has_delimiter(elements[i].text())
            if is_delimiter:
                parsed_document["details"] = "\n".join(
                    [
                        elements[doc_details_idx].text()
                        for doc_details_idx in range(details_start_idx, i)
                    ]
                )
                # Flush previous entry (finish parsing process)
                if parsed_document["address"]["city"] == "Warszawa":
                    documents_list.append(parsed_document)

    return documents_list


def event_entry_to_string(event_entry):
    return f"""Title: {event_entry['title']}
Organizer: {event_entry['organizer']}
Address: {event_entry['address']['place']}, {event_entry['address']['street']}, {event_entry['address']['district']}, {event_entry['address']['city']}
Site: {event_entry['site']}
Social media: {event_entry['social_media']}
{event_entry['details']}
    """.strip()


def serialize_single_text_file(entries, filename):
    file_content = []
    delimiter = "\n\n" + "=" * 8 + "\n\n"
    for entry in entries:
        file_content.append(event_entry_to_string(entry))
    file_content = delimiter.join(file_content)
    with open(filename, "w") as f:
        f.write(file_content)


if __name__ == "__main__":
    document = load_file("original_program.pdf")
    parsed_documents = parse_document_elements(document)
    serialize_single_text_file(parsed_documents, "museums.txt")
