# 📄 Context-Aware RAG Chatbot for PDF Q&A 🧠

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Ask questions about your PDF documents interactively! This tool uses Retrieval-Augmented Generation (RAG) to answer based **only** on the content of the PDF you provide, leveraging powerful LLMs via OpenRouter.

## ✨ Features

*   **📄 Interactive PDF Loading:** Specify any PDF file on your computer via its path.
*   **💬 Conversational Q&A:** Ask multiple questions about the loaded PDF in a single session.
*   **🧠 Contextual Answers:** Generates responses based *only* on the PDF content using RAG.
*   **🔍 Optional Source Display:** Choose to view the specific text chunks used by the LLM to formulate the answer.
*   **🤖 Powered by OpenRouter:** Leverages powerful LLMs (like `qwen/qwen3-30b-a3b:free`) via OpenRouter's API.
*   **🚀 Efficient Embeddings:** Uses `sentence-transformers/all-MiniLM-L6-v2` for fast and effective semantic search.
*   **🔌 Custom LLM Integration:** Uses a tailored LlamaIndex LLM class for reliable OpenRouter communication.

## 💡 How it Works

When you run the script and provide a PDF:

1.  **PDF Parsing:** The script reads your specified PDF file and extracts all available text content using the `PyPDF2` library.
2.  **Text Chunking:** The extracted text is broken down into smaller, overlapping chunks (e.g., ~500 words each) using LlamaIndex's `SentenceSplitter`. This makes the text manageable for analysis.
3.  **Embedding Generation:** Each text chunk is converted into a numerical vector (an "embedding") using the `sentence-transformers/all-MiniLM-L6-v2` model. These embeddings capture the semantic meaning of the chunks.
4.  **Index Building:** The text chunks and their corresponding embeddings are stored in an in-memory `VectorStoreIndex` (provided by LlamaIndex). This index allows for efficient searching based on semantic similarity.
5.  **Querying Loop:** The script enters an interactive mode:
    *   **User Question:** You enter a question about the PDF.
    *   **Query Embedding:** Your question is also converted into an embedding using the same sentence transformer model.
    *   **Retrieval:** The script searches the vector index to find the text chunks whose embeddings are most similar (semantically closest) to your question's embedding. These are the most relevant parts of the PDF.
    *   **Context Augmentation:** The retrieved relevant text chunks are combined with your original question.
    *   **LLM Generation:** This combined information (context chunks + question) is sent as a prompt to the LLM (`qwen/qwen3-30b-a3b:free` via OpenRouter).
    *   **Answer Synthesis:** The LLM generates an answer based *only* on the provided context chunks and your question.
    *   **Display:** The generated answer is printed to your terminal. You can optionally view the source chunks used.
6.  **Loop or Exit:** You can ask another question or type `quit`/`exit` to end the session.

## ⚙️ Setup

1.  **Clone the Repository (or download files):**
    ```bash
    # If cloning from GitHub later:
    # git clone <repository-url>
    # cd pdf_rag_chatbot
    ```
    *(Ensure you have Python 3.9+ installed)*

2.  **Create and Activate Virtual Environment:**
    ```bash
    # Create venv (use python3 if python points to python2)
    python -m venv .venv

    # Activate (Windows CMD/PowerShell)
    .\.venv\Scripts\activate

    # Activate (macOS/Linux)
    # source .venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(This installs LlamaIndex, PyPDF2, Sentence-Transformers, Requests, etc.)*

4.  **Set Up Environment Variables:**
    *   Create a file named `.env` in the project root directory (where `main.py` is).
    *   Add your OpenRouter API key to it. **Replace** `<YOUR_OPENROUTER_API_KEY>` with your actual key:
      ```plaintext
      OPENROUTER_API_KEY=<YOUR_OPENROUTER_API_KEY>
      ```
    *   The script will **not** run without a valid key in this file.

## ▶️ Usage

1.  **Activate your virtual environment** (if not already active):
    ```bash
    # Windows: .\.venv\Scripts\activate
    # macOS/Linux: source .venv/bin/activate
    ```

2.  **Run the main script:**
    ```bash
    python main.py
    ```

3.  **Follow the prompts:**
    *   The script will first ask you to: `Enter the full path to your PDF file: `
        *   Provide the complete path (e.g., `C:\Users\YourName\Documents\report.pdf` or `/home/user/docs/paper.pdf`).
    *   It will then process the PDF (parse, chunk, embed, index). This might take a moment, especially the first time when the embedding model is downloaded.
    *   Once ready, it will say `--- Ready to Answer Questions ---` and prompt you with `Query: `.
    *   Type your question about the PDF and press Enter.
    *   The script will generate and display the answer.
    *   It will ask if you want to see the source text chunks used (`Show source text chunks? (yes/no, default: no):`). Type `yes` if you want to see them.
    *   It will then prompt you for the next `Query: `.

4.  **To stop**, type `quit` or `exit` at the `Query: ` prompt and press Enter, or press `Ctrl+C`.

## 🛠️ Technologies Used

*   **Python 3.9+**
*   **LlamaIndex:** Core RAG framework orchestration.
*   **OpenRouter:** API gateway providing access to LLMs (using `qwen/qwen3-30b-a3b:free`).
*   **Sentence Transformers (`all-MiniLM-L6-v2`):** Text embedding model via Hugging Face.
*   **PyPDF2:** Extracting text from PDF documents.
*   **Requests:** Making API calls in the custom LLM class.
*   **python-dotenv:** Loading environment variables (API keys).

## License

This project is licensed under the MIT License. Visit [https://opensource.org/licenses/MIT](https://opensource.org/licenses/MIT) for details.