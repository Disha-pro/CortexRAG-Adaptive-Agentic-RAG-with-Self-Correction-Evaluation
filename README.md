# CortexRAG-Adaptive-Agentic-RAG-with-Self-Correction-Evaluation

**Self-Correcting Retrieval • Multimodal Document Understanding • Hallucination Detection • RAG Evaluation • Full Observability**

A production-style **Agentic RAG** system built with **LangGraph**, featuring intelligent routing, self-correction, multimodal PDF processing (text + tables), groundedness evaluation, and a modern Streamlit interface.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-green)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B)
![Groq](https://img.shields.io/badge/LLM-Groq-orange)
![Pinecone](https://img.shields.io/badge/VectorDB-Pinecone-purple)

---

## ✨ Key Features

- **Agentic Workflow** powered by LangGraph
- **Intelligent Routing** (Private Knowledge Base vs Direct Answer)
- **Self-Correcting Retrieval**
  - Evidence Grading
  - Query Rewriting
  - Automatic Web Fallback (Tavily)
- **Multimodal PDF Ingestion**
  - Text extraction
  - Table extraction
  - Page-level metadata
- **Hallucination / Groundedness Detection**
- **RAG Evaluation Metrics** (LLM-as-a-Judge)
  - Faithfulness
  - Answer Relevance
  - Context Relevance
  - Groundedness
- **Full Observability**
  - Agent execution path
  - Node-level trace
  - Detailed evaluation scores
- **Modern Dark Theme UI** (Neon Green aesthetic)
- **Source Citations** with page numbers

---

