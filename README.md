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
    git clone https://github.com/073harshith073/interactive-pdf-qa
    cd interactive-pdf-qa
    ```
    *(Ensure you have Python 3.9+ installed)*

2.  **Create and Activate Virtual Environment:**
    ```bash
    # Create venv (use python3 if python points to python2)
    python -m venv .venv

    # Activate (Windows CMD/PowerShell)
    .\.venv\Scripts\activate

    # Activate (macOS/Linux)
    source .venv/bin/activate
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

##  보안 관련 고려사항

*   **API 키 관리**: 이 애플리케이션은 `.env` 파일에 저장된 OpenRouter API 키가 필요합니다. 이 파일을 안전하게 보호하고 버전 제어(예: Git)에 커밋하지 마십시오. `.gitignore` 파일은 이를 방지하도록 이미 설정되어 있습니다.

*   **전송 중 데이터**: OpenRouter API와의 모든 통신은 HTTPS를 통해 이루어지므로, 사용자의 기계와 OpenRouter 서버 간의 데이터는 전송 중에 암호화됩니다.

*   **저장 데이터**: 성능 향상을 위해 이 도구는 처리된 PDF의 텍스트와 생성된 벡터 인덱스를 로컬 `.cache` 디렉터리에 저장합니다.
    *   **이 캐시는 암호화되지 않습니다.** 민감한 정보가 포함된 PDF를 처리하는 경우, 파일 시스템 권한 설정이나 전체 디스크 암호화와 같은 운영 체제 수준의 보안 조치를 사용하여 이 디렉토리를 보호해야 합니다.

*   **의존성**: 항상 신뢰할 수 있는 소스에서 `requirements.txt`에 나열된 의존성을 설치하십시오.

## 🛠️ 사용된 기술

*   **Python 3.9+**
*   **LlamaIndex**: 핵심 RAG 프레임워크 오케스트레이션.
*   **OpenRouter**: LLM에 대한 액세스를 제공하는 API 게이트웨이 (`qwen/qwen3-30b-a3b:free` 사용).
*   **Sentence Transformers (`all-MiniLM-L6-v2`)**: Hugging Face를 통한 텍스트 임베딩 모델.
*   **PyPDF2**: PDF 문서에서 텍스트 추출.
*   **Requests**: 사용자 정의 LLM 클래스에서 API 호출.
*   **python-dotenv**: 환경 변수(API 키) 로딩.
*   **pytest**: 단위 및 통합 테스트용.
*   **reportlab**: 테스트용 샘플 PDF 생성용.

## 라이선스

This project is licensed under the MIT License. Visit [https://opensource.org/licenses/MIT](https://opensource.org/licenses/MIT) for details.
