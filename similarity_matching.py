import os
import re
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_features(text):
    invoice_number = re.findall(r'Invoice Number:\s*(\w+)', text)
    date = re.findall(r'Date:\s*(\d{2}/\d{2}/\d{4})', text)
    amount = re.findall(r'Amount:\s*\$?(\d+\.\d{2})', text)
    features = {
        'invoice_number': invoice_number[0] if invoice_number else 'N/A',
        'date': date[0] if date else 'N/A',
        'amount': amount[0] if amount else 'N/A',
        'keywords': set(text.split()),
        'structure': extract_structure(text)
    }
    return features

def extract_structure(text):
    structure = {}
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if 'Invoice Number:' in line:
            structure['invoice_number_line'] = i
        if 'Date:' in line:
            structure['date_line'] = i
        if 'Amount:' in line:
            structure['amount_line'] = i
    return structure

def calculate_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0, 1]

def calculate_structural_similarity(struct1, struct2):
    similarity_score = 0
    total_elements = len(struct1)
    for key in struct1:
        if key in struct2 and struct1[key] == struct2[key]:
            similarity_score += 1
    return similarity_score / total_elements

def load_existing_invoices(directory):
    database = []
    for file in os.listdir(directory):
        if file.endswith('.pdf'):
            text = extract_text_from_pdf(os.path.join(directory, file))
            features = extract_features(text)
            database.append({'file': file, 'text': text, 'features': features})
    return database

def find_most_similar_invoice(input_invoice_path, database):
    input_text = extract_text_from_pdf(input_invoice_path)
    input_features = extract_features(input_text)

    most_similar_invoice = None
    highest_combined_similarity_score = 0

    for invoice in database:
        content_similarity_score = calculate_cosine_similarity(input_text, invoice['text'])
        structural_similarity_score = calculate_structural_similarity(input_features['structure'], invoice['features']['structure'])
        combined_similarity_score = (content_similarity_score + structural_similarity_score) / 2

        if combined_similarity_score > highest_combined_similarity_score:
            highest_combined_similarity_score = combined_similarity_score
            most_similar_invoice = invoice

    return most_similar_invoice, highest_combined_similarity_score, input_features

def main():
    # Load existing invoices
    database = load_existing_invoices('invoices')

    # Input invoice path
    input_invoice_path = 'input_invoice.pdf'

    # Find the most similar invoice
    most_similar_invoice, combined_similarity_score, input_features = find_most_similar_invoice(input_invoice_path, database)

    # Output the most similar invoice and similarity score
    if most_similar_invoice:
        matching_keywords = input_features['keywords'].intersection(most_similar_invoice['features']['keywords'])
        print(f"Most similar invoice: {most_similar_invoice['file']}")
        print(f"Combined similarity score: {combined_similarity_score}")
        print("Matching features:")
        print(f"  Invoice Number: {input_features['invoice_number']} (input) vs {most_similar_invoice['features']['invoice_number']} (matched)")
        print(f"  Date: {input_features['date']} (input) vs {most_similar_invoice['features']['date']} (matched)")
        print(f"  Amount: {input_features['amount']} (input) vs {most_similar_invoice['features']['amount']} (matched)")
        print(f"  Keywords: {', '.join(matching_keywords)}")
        print(f"  Structural Similarity: {combined_similarity_score}")
    else:
        print("No similar invoices found.")

if __name__ == "__main__":
    main()
