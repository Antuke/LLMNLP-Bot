# NLP Course Chatbot 🤖

A RAG-based conversational AI assistant designed for the **University of Salerno NLP and LLM course**. This chatbot helps students with questions about the syllabus, teachers, schedule, exams, and materials, while using **Guardrails** to ensure the conversation stays on topic.

## ✨ Features

*   **RAG (Retrieval-Augmented Generation):** Retrieves relevant course information from the `./data` directory using a hybrid search (Semantic + Keyword).
*   **Topic Guardrails:** Filters out off-topic queries using configurable classifiers:
    *   **BERT:** Fine-tuned DistilBERT model (Default).
    *   **Classic:** SVM or Logistic Regression on Ollama embeddings.
*   **Context Awareness:** Maintains conversation history and context for natural interactions.
*   **Configurable Models:** Easily switch between different LLMs and guardrail modes via CLI.

## 🛠️ Prerequisites

1.  **Python 3.8+**
2.  **Ollama**: Required for embeddings and the LLM.
    *   [Download Ollama](https://ollama.com/download)

## 🚀 Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Pull required Ollama models:**
    The project uses `qwen3:8b` (or your preferred model) for chat and `nomic-embed-text` for embeddings.
    ```bash
    ollama pull qwen3:8b
    ollama pull nomic-embed-text
    ```

## 💻 Usage

Run the chatbot using the default settings (Model: `qwen3:8b`, Guardrail: `bert`):

```bash
python main.py
```

### CLI Arguments

You can customize the chatbot's behavior using command-line arguments:

| Argument | Description | Default | Options |
| :--- | :--- | :--- | :--- |
| `--model` | Specify the Ollama model to use for chat. | `qwen3:8b` | Any model available in your `ollama list` |
| `--mode` | Specify the topic classifier (guardrail) mode. | `bert` | `bert`, `svm`, `logistic` |

**Examples:**

Use a different model:
```bash
python main.py --model llama3
```

Use the SVM classifier for guardrails:
```bash
python main.py --mode svm
```

Combine arguments:
```bash
python main.py --model mistral --mode logistic
```

## ⚙️ Configuration

*   **System Prompt:** The chatbot's persona and instructions are defined in `system_prompt.txt`.
*   **Data:** Place your course documents (text files) in the `data/` directory to be indexed by the RAG system.

## 📂 Project Structure

*   `main.py`: Entry point. Handles CLI args, initialization, and the chat loop.
*   `chatbot.py`: Core logic. Manages Ollama interaction and conversation history.
*   `myrag.py`: RAG implementation (Hybrid Search).
*   `topic_classifier.py`: Wrapper for the classic (SVM/Logistic) guardrails.
*   `classifiers/`: Contains training scripts and the fine-tuned BERT model.

## 📄 Design 

Detailed information about the project's design choices, methodology, and implementation can be found in **Chapter 5** of the project report: `report_llm.pdf`.


