import os
from pathlib import Path
from typing import Dict, List

from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM

# Constants
FINANCIAL_INSTITUTIONS = [
    "DBS",
    "Maribank",
    "Amex",
    "Citi",
    "Maybank",
]  # Add more institutions as needed

INVESTMENT_STATEMENT_TYPE = [
    "Dividend Letter",
    "Notice of Payment",
    "Monthly Statement",
]  # Add more sub-categories as needed
UTILITY_INSTITUTIONS = ["StarHub", "Senoko", "SP"]


def load_pdf_chunks(pdf_path: str, chunk_size: int = 1000) -> List[Document]:
    # Load the PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
    chunks = text_splitter.split_documents(pages)

    return chunks


def extract_pdf_category(chunks: List[Document], llm: OllamaLLM) -> str:
    # Create prompt template for category extraction
    prompt_template = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: What type of document is this? Choose only one of the following:
    bank_statement, manulife_investment, utility_bill. Reply with just the category name.
    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context"])

    # Create a question-answering chain
    chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)

    response = chain.invoke({"input_documents": chunks})

    return response["output_text"]


def extract_info_from_pdf(pdf_path: str, llm: OllamaLLM) -> Dict[str, str]:
    """
    Extract key information from a PDF using LLM.

    Args:
        pdf_path: Path to the PDF file
        llm: Initialized LLM instance

    Returns:
        Dictionary with extracted information
    """
    chunks = load_pdf_chunks(pdf_path)
    category = extract_pdf_category(chunks, llm)

    if category == "bank_statement":
        additional_questions = {
            "institution": (
                "What is the name of the company or institution that issued this document? "
                "Reply one of the following: "
                f"{', '.join(FINANCIAL_INSTITUTIONS)}"
            ),
            "month": ("What is the statement month? Reply in YYYYMM format " "(e.g., 202305)."),
        }
    elif category == "manulife_investment":
        additional_questions = {
            "statement_type": (
                "What are the sub-categories of this investment statement? "
                "Reply one of the following: "
                f"{', '.join(INVESTMENT_STATEMENT_TYPE)}"
            ),
            "payment_amount": (
                "If this is a payment statement, what is the total payment amount? "
                "Reply with just the number (no currency symbol)."
            ),
            "month": (
                "What is the statement month or letter month? Reply in YYYYMM format "
                "(e.g., 202305)."
            ),
        }
    elif category == "utility_bill":
        additional_questions = {
            "institution": (
                "What is the name of the company or institution that issued this bill? "
                "Reply one of the following: "
                f"{', '.join(UTILITY_INSTITUTIONS)}"
            ),
            "month": (
                "What is the month of the bill? Reply with just the month in YYYYMM format "
                "(e.g., 202305)."
            ),
            "amount": (
                "What is the total bill amount? Reply with just the number (no currency symbol)."
            ),
            "billing_period": (
                "What is the billing period for this bill? Reply with just the period in MMDD-MMDD format "  # noqa: E501
                "(e.g., 0412-0512, 0213-0312)."
            ),
        }
    else:
        # raise exception
        raise ValueError(f"Unknown document category: {pdf_path}")

    # Create prompt template for information extraction
    prompt_template = """
    Use the following pieces of context to answer the question at the end.
    Be precise and concise in your answer. Only extract the exact information requested.
    If you don't know the answer, just say "unknown", don't try to make up an answer.

    {context}

    Question: {question}
    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)

    # Initialize results dictionary with category
    results = {"category": category}

    # Ask additional questions based on document type
    for key, question in additional_questions.items():
        response = chain.invoke({"input_documents": chunks, "question": question})
        results[key] = response["output_text"].strip()

    return results


def generate_filename(info: Dict[str, str], original_file: str) -> str:
    """
    Generate a standardized filename based on extracted information.

    Args:
        info: Dictionary with extracted information
        original_file: Original filename

    Returns:
        Suggested new filename
    """
    category = info["category"].lower()

    # Get original file extension
    _, extension = os.path.splitext(original_file)

    # Generate filename based on category
    if category == "bank_statement":
        return f"{info["month"]}_{info["institution"]}{extension}"
    elif category == "manulife_investment":
        if info["statement_type"] == "Monthly Statement":
            return f"{info['month']}_{info['statement_type'].replace(' ', '')}{extension}"  # noqa: E501
        else:
            return f"{info['month']}_{info['statement_type'].replace(' ', '')}_{info['payment_amount']}{extension}"  # noqa: E501
    elif category == "utility_bill":
        return f"{info['month']}_{info['institution']}_{info['billing_period']}_{info['amount']}{extension}"  # noqa: E501
    else:
        raise ValueError(f"Unknown category for filename generation: {category}")


def main():
    # Initialize the LLM
    llm = OllamaLLM(model="qwen2.5:7b")

    # Get all PDF files in the current directory
    current_dir = Path(".")
    pdf_files = list(current_dir.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in the current directory.")
        return

    print(f"Found {len(pdf_files)} PDF files. Processing...")

    rename_mapping = {}
    for pdf_file in pdf_files:
        print(f"\nAnalyzing: {pdf_file.name}")
        # Extract information
        info = extract_info_from_pdf(str(pdf_file), llm)
        generated_filename = generate_filename(info, pdf_file.name)
        rename_mapping[pdf_file.name] = generated_filename

    for original, new in rename_mapping.items():
        print(f"Original: {original} -> Suggested: {new}")
        os.rename(original, new)


if __name__ == "__main__":
    main()
