import os
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM

# Constants
INVESTMENT_STATEMENT_TYPE = [
    "Dividend Letter",
    "Notice of Payment",
    "Monthly Statement",
]  # Add more sub-categories as needed


def get_pdf_files_in_current_dir() -> List[Path]:
    """
    Get all PDF files in the current directory.

    Returns:
        List of Path objects for PDF files.
    """
    current_dir = Path(".")
    pdf_files = list(current_dir.glob("*.pdf"))

    if not pdf_files:
        return []

    return pdf_files


def get_all_subdirectories(destination_dir: Path) -> List[Path]:
    subdirs = []
    for subdir in destination_dir.rglob("*"):
        if subdir.is_dir():
            subdirs.append(subdir)
    return subdirs


def _load_pdf_chunks(pdf_path: str, chunk_size: int = 1000) -> List[Document]:
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

    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    chain = create_stuff_documents_chain(llm, prompt)
    response = chain.invoke({"context": chunks})

    return response


def extract_info_from_pdf(pdf_path: str, llm: OllamaLLM) -> Dict[str, str]:
    """
    Extract key information from a PDF using LLM.

    Args:
        pdf_path: Path to the PDF file
        llm: Initialized LLM instance

    Returns:
        Dictionary with extracted information
    """
    chunks = _load_pdf_chunks(pdf_path)
    category = extract_pdf_category(chunks, llm)

    if category == "bank_statement":
        additional_questions = {
            "institution": (
                "What is the name of the company or institution that issued this document? "
                "Reply one of the following: "
                f"{os.getenv('FINANCIAL_INSTITUTIONS')}"
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
                f"{os.getenv('UTILITY_INSTITUTIONS')}"
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

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = create_stuff_documents_chain(llm, prompt)

    # Initialize results dictionary with category
    results = {"category": category}

    # Ask additional questions based on document type
    for key, question in additional_questions.items():
        response = chain.invoke({"context": chunks, "question": question})
        results[key] = response

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


def determine_file_destination(
    file_name: str, subdirs: List[Path], llm: OllamaLLM, destination: Path
) -> Optional[Path]:
    """
    Use LLM to determine the best directory to move a file to based on its name.

    Args:
        file_name: Name of the file to move
        subdirs: List of available subdirectories
        llm: Initialized LLM instance

    Returns:
        Path object of the best matching directory
    """
    # Get directory context
    directory_context = get_directory_context(subdirs, destination)

    # Convert paths to relative paths for better readability
    subdir_names = [str(subdir.relative_to(destination)) for subdir in subdirs]

    # Create prompt for directory selection
    prompt = f"""
    Given a file named "{file_name}", analyze the filename and determine which directory would be the most appropriate to move this file to.

    {directory_context}

    Available directories (choose from one of these exact paths):
    {chr(10).join(f"- {subdir}" for subdir in subdir_names)}

    Analyze the filename for:
    1. Date patterns (YYYYMM format like 202503)
    2. Institution names (Amex, Citi, HSBC, etc.)
    3. Document type (statements, bills, insurance, etc.)
    4. File naming conventions that match the directory structure

    Based on the filename "{file_name}", which directory path would be most appropriate?

    Reply with ONLY the exact directory path from the list above.
    If no directory is clearly appropriate, reply with "NO_MATCH".
    """

    # Get LLM response
    response = llm.invoke(prompt)
    suggested_dir = response.strip()

    # Find the matching directory
    if suggested_dir == "NO_MATCH":
        return None

    # Find the exact match in the subdirs list
    for subdir in subdirs:
        if str(subdir.relative_to(Path(destination))) == suggested_dir:
            return subdir

    # If no exact match found, try partial matching
    for subdir in subdirs:
        subdir_path = str(subdir.relative_to(Path(destination)))
        if (
            suggested_dir.lower() in subdir_path.lower()
            or subdir_path.lower() in suggested_dir.lower()
        ):
            print(f"Found partial match: {subdir_path} for suggestion: {suggested_dir}")
            return subdir

    print(f"No match found for LLM suggestion: {suggested_dir}")
    return None


def move_file_to_destination(file_path: Path, destination_dir: Path) -> bool:
    """
    Move a file to the specified destination directory.

    Args:
        file_path: Path to the file to move
        destination_dir: Destination directory

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create destination directory if it doesn't exist
        destination_dir.mkdir(parents=True, exist_ok=True)

        # Move the file
        destination_file = destination_dir / file_path.name
        file_path.rename(destination_file)

        print(f"Successfully moved {file_path.name} to {destination_dir}")
        return True
    except Exception as e:
        print(f"Error moving {file_path.name}: {e}")
        return False


def get_directory_context(subdirs: List[Path], destination: Path) -> str:
    """
    Generate a context string describing the directory structure for the LLM.

    Args:
        subdirs: List of subdirectories

    Returns:
        Formatted string describing the directory structure
    """
    context = "Directory structure:\n"

    # Group directories by their parent structure
    dir_tree = {}
    for subdir in subdirs:
        parts = subdir.relative_to(Path(destination)).parts
        current_level = dir_tree

        for part in parts:
            if part not in current_level:
                current_level[part] = {}
            current_level = current_level[part]

    def format_tree(tree, level=0):
        result = []
        for key, subtree in tree.items():
            indent = "  " * level
            result.append(f"{indent}- {key}")
            if isinstance(subtree, dict) and subtree:
                result.extend(format_tree(subtree, level + 1))
        return result

    return "\n".join([context] + format_tree(dir_tree))


def handle_file_movement(
    llm: OllamaLLM,
    renamed_pdf_files: List[Path],
    subdirs: List[Path],
    destination: Path,
):
    # Interactive mode (default)
    print("Processing files interactively...")

    # Process each PDF file
    for pdf_file in renamed_pdf_files:
        print(f"\nAnalyzing: {pdf_file.name}")

        # Determine the best destination directory
        destination_dir = determine_file_destination(pdf_file.name, subdirs, llm, destination)

        if destination_dir:
            print(f"LLM suggests moving to: {destination_dir}")

            # Ask user for confirmation
            user_input = input(f"Move {pdf_file.name} to {destination_dir}? (y/n): ")

            if user_input.lower() in ["y", "yes"]:
                success = move_file_to_destination(pdf_file, destination_dir)
                if success:
                    print(f"✓ Moved {pdf_file.name} successfully")
                else:
                    print(f"✗ Failed to move {pdf_file.name}")
            else:
                print(f"Skipped moving {pdf_file.name}")
        else:
            print(f"LLM could not determine a suitable directory for {pdf_file.name}")

    print("\nProcessing complete!")


def rename_pdf_from_content(llm: OllamaLLM, pdf_files: List[Path]) -> None:
    for pdf_file in pdf_files:
        info = extract_info_from_pdf(str(pdf_file), llm)
        generated_filename = generate_filename(info, pdf_file.name)

        print(f"Original: {pdf_file} -> Suggested: {generated_filename}")
        os.rename(pdf_file, generated_filename)


def main():
    load_dotenv()

    llm = OllamaLLM(model=os.getenv("OLLAMA_LLM_MODEL"))
    destination_dir = Path(os.getenv("DESTINATION_DIR", "~/Documents")).expanduser().resolve()
    pdf_files = get_pdf_files_in_current_dir()

    if not pdf_files:
        print("No PDF files found in the current directory, skipping processing.")
        return

    rename_pdf_from_content(llm, pdf_files)

    subdirs = get_all_subdirectories(destination_dir)
    renamed_pdf_files = get_pdf_files_in_current_dir()

    handle_file_movement(llm, renamed_pdf_files, subdirs, destination_dir)


if __name__ == "__main__":
    main()
