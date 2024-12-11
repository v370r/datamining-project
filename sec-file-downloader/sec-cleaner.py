import re
from bs4 import BeautifulSoup

def remove_html_in_document(html_content):
    # Parse the content of the second <document> tag separately
    soup_inside_document = BeautifulSoup(html_content, 'lxml')

    # Replace <p> with newlines to preserve paragraph breaks
    for p_tag in soup_inside_document.find_all('p'):
        p_tag.insert_before('\n')

    # Extract the text, with paragraph breaks preserved
    text = soup_inside_document.get_text('\n', strip=True)
    return text

def extract_content(html_content):
    # Define the regular expression pattern to match text between "Item 1" and "Item 2"
    p1 = r"(Item\s*(?:&#160;)?1\..*?Item\s*(?:&#160;)?2\.)"
    p2 = r"(Item\s*(?:&#160;)?7\..*?Item\s*(?:&#160;)?8\.)"

    # Search for the pattern in the HTML content
    p1_matches = re.findall(p1, html_content, re.DOTALL)  # re.DOTALL allows . to match across newlines match newlines
    p2_matches = re.findall(p2, html_content, re.DOTALL)

    # If a match is found, return the matched content
    if p1_matches or p2_matches:
        return p1_matches + p2_matches
    else:
        return []

# Read your HTML file
with open('./data/sec-edgar-filings/NVDA/10-K/0001045810-24-000029/full-submission.txt', 'r', encoding='utf-8') as file:
    html_content = file.read()

    soup = BeautifulSoup(html_content, 'lxml')
    docs = soup.find_all('document')
    num_docs = len(docs)

    for i in range(num_docs):
        doc = docs[i].decode_contents()
        # Call the function to remove HTML tags and preserve formatting
        cleaned_text = remove_html_in_document(doc)
        
        # Call the function to extract content between "Item 1" and "Item 2"
        extracted_contents = extract_content(cleaned_text)

        for j in range(len(extracted_contents)):
            if len(extracted_contents[j]) != 0:
                # Optionally, write the result to a new file
                with open(f'./clean_data/sec-edgar-filings/NVDA/10-K/0001045810-24-000029/full-submission-{i}-{j}.txt', 'w', encoding='utf-8') as output_file:
                    output_file.write(extracted_contents[j])
