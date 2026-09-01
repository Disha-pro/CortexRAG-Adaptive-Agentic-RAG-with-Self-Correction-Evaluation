# ============================================================
# Multimodal Adaptive Agentic RAG
# Neon Green Dark Theme + Observability + Streamlit
# ============================================================

import os
import time
from datetime import datetime
from typing import List, Literal, Optional, Dict, Any
from typing_extensions import TypedDict

import streamlit as st
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langgraph.graph import StateGraph, START, END

# Load .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

try:
    import fitz
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

# -------------------- Config --------------------
INDEX_NAME = "industry-agentic-rag-kb"
NAMESPACE = "multimodal-agentic-rag"
MAX_RETRIES = 1
TOP_K = 6
TEXT_MODEL = "openai/gpt-oss-20b"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# ============================================================
# CUSTOM CSS - Neon Green Dark Theme
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp {
        background-color: #000000;
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }

    section[data-testid="stSidebar"] {
        background-color: #0a0a0a;
        border-right: 1px solid #1a1a1a;
    }

    h1, h2, h3 {
        color: #00FF41 !important;
        font-weight: 700 !important;
    }

    .neon-title {
        color: #00FF41;
        text-shadow: 0 0 10px #00FF41, 0 0 20px #00FF41, 0 0 40px #00FF41;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: 1px;
    }

    .subtitle {
        color: #a0a0a0;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        background: #0d0d0d;
        border: 1px solid #1f1f1f;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 0 15px rgba(0, 255, 65, 0.08);
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #00FF41;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #888;
        margin-top: 4px;
    }

    .stChatMessage {
        background-color: #0d0d0d;
        border-radius: 12px;
        border: 1px solid #1a1a1a;
        padding: 1rem;
    }

    .stButton > button {
        background-color: #00FF41;
        color: #000000;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #39FF14;
        box-shadow: 0 0 20px rgba(0, 255, 65, 0.5);
        color: #000;
    }

    .streamlit-expanderHeader {
        background-color: #0d0d0d;
        color: #00FF41 !important;
        border-radius: 8px;
    }

    .agent-path {
        background: #0a0a0a;
        border: 1px solid #00FF41;
        border-radius: 10px;
        padding: 12px 16px;
        color: #00FF41;
        font-family: monospace;
        font-size: 0.95rem;
        box-shadow: 0 0 12px rgba(0, 255, 65, 0.15);
    }

    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0a0a0a;
    }
    ::-webkit-scrollbar-thumb {
        background: #00FF41;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- Helpers --------------------
def load_keys():
    if not os.getenv("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = st.sidebar.text_input("GROQ API Key", type="password") or ""
    if not os.getenv("TAVILY_API_KEY"):
        os.environ["TAVILY_API_KEY"] = st.sidebar.text_input("Tavily API Key", type="password") or ""
    if not os.getenv("PINECONE_API_KEY"):
        os.environ["PINECONE_API_KEY"] = st.sidebar.text_input("Pinecone API Key", type="password") or ""

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )

@st.cache_resource
def get_llms():
    return (
        ChatGroq(model=TEXT_MODEL, temperature=0),
        ChatGroq(model=VISION_MODEL, temperature=0)
    )

# -------------------- Structured Models --------------------
class RouteDecision(BaseModel):
    route: Literal["kb", "direct"]

class EvidenceGrade(BaseModel):
    grade: Literal["good", "weak"]
    relevance_score: float = Field(ge=0.0, le=1.0)

class HallucinationCheck(BaseModel):
    grounded: bool
    faithfulness_score: float
    hallucination_detected: bool
    unsupported_claims: List[str] = []
    explanation: str

class RAGEvaluation(BaseModel):
    context_relevance: float
    answer_relevance: float
    faithfulness: float
    groundedness: Literal["PASS", "FAIL"]
    hallucination_detected: bool
    correctness: Optional[float] = None
    summary: str

class AgentState(TypedDict):
    question: str
    current_query: str
    kb_docs: List[Document]
    web_results: str
    kb_grade: str
    kb_relevance_score: float
    web_grade: str
    answer: str
    source_used: str
    retry_count: int
    evaluation: Optional[Dict]
    citations: List[str]
    grounded: bool
    trace: List[Dict[str, Any]]

def log_trace(state: AgentState, node: str, message: str, extra: Dict = None):
    event = {
        "node": node,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "message": message,
        "extra": extra or {}
    }
    if state.get("trace") is None:
        state["trace"] = []
    state["trace"].append(event)
    return state

# -------------------- PDF Extraction --------------------
def extract_pdf_multimodal(pdf_bytes, filename: str) -> List[Document]:
    if not HAS_PYMUPDF:
        return []
    docs = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page_num, page in enumerate(doc, 1):
        text = page.get_text("text").strip()
        if text:
            docs.append(Document(
                page_content=text,
                metadata={"source": filename, "document": filename, "page": page_num, "content_type": "text"}
            ))
        try:
            tables = page.find_tables()
            if tables.tables:
                for t_idx, table in enumerate(tables.tables):
                    try:
                        table_str = table.to_pandas().to_markdown(index=False)
                    except:
                        table_str = str(table.extract())
                    docs.append(Document(
                        page_content=f"[TABLE]\n{table_str}",
                        metadata={"source": filename, "document": filename, "page": page_num, "content_type": "table"}
                    ))
        except:
            pass
    doc.close()
    return docs

@st.cache_resource
def get_vectorstore_and_retriever(_embeddings):
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    existing = [i["name"] for i in pc.list_indexes()]
    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME, dimension=384, metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)
    index = pc.Index(INDEX_NAME)
    vectorstore = PineconeVectorStore(index=index, embedding=_embeddings, namespace=NAMESPACE)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K, "namespace": NAMESPACE})
    return vectorstore, retriever

# -------------------- Graph Creation --------------------
def create_graph(llm, retriever, web_search):
    router_llm = llm.with_structured_output(RouteDecision, method="json_mode")
    grader_llm = llm.with_structured_output(EvidenceGrade, method="json_mode")
    hallucination_llm = llm.with_structured_output(HallucinationCheck, method="json_mode")
    eval_llm = llm.with_structured_output(RAGEvaluation, method="json_mode")

    def route_question(state: AgentState):
        decision = router_llm.invoke(f"""
You are a router for an Agentic RAG system.

Route to "kb" if the question is about technical topics, documents, attention mechanism, RAG, AI concepts, or needs knowledge from documents.
Route to "direct" only for pure greetings like hi, hello, thanks, bye.

Question: {state['question']}

Return your decision as valid JSON.
Example: {{"route": "kb"}}
""")
        state = log_trace(state, "Router", f"→ {decision.route.upper()}", {"route": decision.route})
        state["source_used"] = decision.route
        state["current_query"] = state["question"]
        return state

    def route_after_router(state: AgentState) -> Literal["retrieve_kb", "direct_answer"]:
        return "retrieve_kb" if state["source_used"] == "kb" else "direct_answer"

    def retrieve_kb(state: AgentState):
        docs = retriever.invoke(state["current_query"])
        state = log_trace(state, "Retriever", f"Retrieved {len(docs)} chunks", {"num_docs": len(docs)})
        state["kb_docs"] = docs
        return state

    def grade_kb_evidence(state: AgentState):
        context = "\n\n".join(f"[{d.metadata.get('content_type')}] {d.page_content[:350]}" for d in state["kb_docs"])
        grade = grader_llm.invoke(f"""
You are an evidence grader.

Question: {state['question']}

Evidence:
{context}

Can this evidence answer the question?
Return "good" if yes, "weak" if no.
Also give a relevance_score between 0 and 1.

Return valid JSON only.
Example: {{"grade": "good", "relevance_score": 0.85}}
""")
        state = log_trace(state, "KB Grader", f"{grade.grade.upper()} ({grade.relevance_score:.2f})", {
            "grade": grade.grade, "score": grade.relevance_score
        })
        state["kb_grade"] = grade.grade
        state["kb_relevance_score"] = grade.relevance_score
        return state

    def decide_after_kb(state: AgentState) -> Literal["generate_from_kb", "rewrite_or_web"]:
        if state["kb_grade"] == "good" and state.get("kb_relevance_score", 0) >= 0.55:
            return "generate_from_kb"
        return "rewrite_or_web"

    def rewrite_or_web_router(state: AgentState):
        return state

    def rewrite_or_web_decision(state: AgentState) -> Literal["rewrite_query", "search_web"]:
        return "rewrite_query" if state["retry_count"] < MAX_RETRIES else "search_web"

    def rewrite_query(state: AgentState):
        rewritten = llm.invoke(f"""
Rewrite the question for better retrieval.
Keep the original intent.
Return only the rewritten query.

Original: {state['question']}
""").content.strip()
        state = log_trace(state, "Query Rewriter", rewritten)
        state["current_query"] = rewritten
        state["retry_count"] += 1
        return state

    def search_web(state: AgentState):
        result = web_search.invoke({"query": state["current_query"]})
        if isinstance(result, dict):
            lines = []
            if result.get("answer"):
                lines.append(result["answer"])
            for item in result.get("results", []):
                lines.append(f"{item.get('title', '')}\n{item.get('content', '')}\n{item.get('url', '')}")
            web_text = "\n\n".join(lines)
        else:
            web_text = str(result)
        state = log_trace(state, "Tavily Web Search", f"{len(web_text)} characters")
        state["web_results"] = web_text
        state["source_used"] = "web"
        return state

    def grade_web_evidence(state: AgentState):
        grade = grader_llm.invoke(f"""
You are an evidence grader.

Question: {state['question']}

Web evidence:
{state['web_results'][:1800]}

Can this web evidence answer the question?
Return "good" or "weak" and a relevance_score between 0 and 1.

Return valid JSON only.
Example: {{"grade": "good", "relevance_score": 0.78}}
""")
        state = log_trace(state, "Web Grader", grade.grade.upper())
        state["web_grade"] = grade.grade
        return state

    def decide_after_web(state: AgentState) -> Literal["generate_from_web", "answer_insufficient"]:
        return "generate_from_web" if state["web_grade"] == "good" else "answer_insufficient"

    def generate_from_kb(state: AgentState):
        citations, context_parts = [], []
        for i, doc in enumerate(state["kb_docs"], 1):
            meta = doc.metadata
            cite = f"[{i}] {meta.get('document')} — Page {meta.get('page')} ({meta.get('content_type')})"
            citations.append(cite)
            context_parts.append(f"{cite}\n{doc.page_content}")
        context = "\n\n".join(context_parts)
        answer = llm.invoke(f"""
Answer the question using ONLY the provided context.
Be accurate and cite the sources at the end.

Question: {state['question']}

Context:
{context}
""").content
        state = log_trace(state, "Generator (Private KB)", "Answer generated")
        state["answer"] = answer
        state["source_used"] = "private_kb"
        state["citations"] = citations
        return state

    def generate_from_web(state: AgentState):
        answer = llm.invoke(f"""
Answer using ONLY the web search results.
Mention that the answer is based on web search.

Question: {state['question']}

Web results:
{state['web_results']}
""").content
        state = log_trace(state, "Generator (Web)", "Answer generated from web")
        state["answer"] = answer
        state["source_used"] = "web_search"
        state["citations"] = ["Tavily Web Search"]
        return state

    def direct_answer(state: AgentState):
        answer = llm.invoke(f"Respond briefly and naturally to: {state['question']}").content
        state = log_trace(state, "Direct Answer", "Simple response")
        state["answer"] = answer
        state["source_used"] = "direct"
        state["citations"] = []
        return state

    def answer_insufficient(state: AgentState):
        state["answer"] = "I could not find enough reliable evidence in the knowledge base or web search to answer confidently."
        state["source_used"] = "insufficient"
        state = log_trace(state, "Insufficient Evidence", "No confident answer")
        return state

    def check_hallucination(state: AgentState):
        if state["source_used"] in ["direct", "insufficient"]:
            state["grounded"] = True
            return state
        context = "\n".join(d.page_content for d in state.get("kb_docs", [])) or state.get("web_results", "")
        check = hallucination_llm.invoke(f"""
You are a strict groundedness evaluator.

Question: {state['question']}
Context: {context[:2800]}
Answer: {state['answer']}

Check if the answer is fully supported by the context.
Return valid JSON with:
- grounded (true/false)
- faithfulness_score (0 to 1)
- hallucination_detected (true/false)
- unsupported_claims (list)
- explanation

Example:
{{"grounded": true, "faithfulness_score": 0.92, "hallucination_detected": false, "unsupported_claims": [], "explanation": "All claims are supported."}}
""")
        state = log_trace(state, "Hallucination Check", f"Grounded={check.grounded} | Faithfulness={check.faithfulness_score:.2f}")
        state["grounded"] = check.grounded
        state["evaluation"] = {
            "faithfulness": check.faithfulness_score,
            "hallucination_detected": check.hallucination_detected,
            "unsupported_claims": check.unsupported_claims,
            "explanation": check.explanation
        }
        return state

    def evaluate_rag(state: AgentState):
        if state["source_used"] in ["direct", "insufficient"]:
            return state
        context = "\n".join(d.page_content for d in state.get("kb_docs", [])) or state.get("web_results", "")
        eval_result = eval_llm.invoke(f"""
You are an expert RAG evaluator.

Score the following on a scale of 0 to 1:

Question: {state['question']}
Context: {context[:2200]}
Answer: {state['answer']}

Return valid JSON with these fields:
- context_relevance
- answer_relevance
- faithfulness
- groundedness ("PASS" or "FAIL")
- hallucination_detected (true/false)
- correctness (null)
- summary

Example:
{{"context_relevance": 0.88, "answer_relevance": 0.91, "faithfulness": 0.94, "groundedness": "PASS", "hallucination_detected": false, "correctness": null, "summary": "Good quality answer."}}
""")
        final = eval_result.model_dump()
        if state.get("evaluation"):
            final.update(state["evaluation"])
        state = log_trace(state, "RAG Evaluation", "Metrics computed")
        state["evaluation"] = final
        return state

    workflow = StateGraph(AgentState)
    nodes = {
        "route_question": route_question,
        "retrieve_kb": retrieve_kb,
        "grade_kb_evidence": grade_kb_evidence,
        "rewrite_or_web_router": rewrite_or_web_router,
        "rewrite_query": rewrite_query,
        "search_web": search_web,
        "grade_web_evidence": grade_web_evidence,
        "generate_from_kb": generate_from_kb,
        "generate_from_web": generate_from_web,
        "direct_answer": direct_answer,
        "answer_insufficient": answer_insufficient,
        "check_hallucination": check_hallucination,
        "evaluate_rag": evaluate_rag,
    }
    for name, func in nodes.items():
        workflow.add_node(name, func)

    workflow.add_edge(START, "route_question")
    workflow.add_conditional_edges("route_question", route_after_router, {
        "retrieve_kb": "retrieve_kb",
        "direct_answer": "direct_answer"
    })
    workflow.add_edge("retrieve_kb", "grade_kb_evidence")
    workflow.add_conditional_edges("grade_kb_evidence", decide_after_kb, {
        "generate_from_kb": "generate_from_kb",
        "rewrite_or_web": "rewrite_or_web_router"
    })
    workflow.add_conditional_edges("rewrite_or_web_router", rewrite_or_web_decision, {
        "rewrite_query": "rewrite_query",
        "search_web": "search_web"
    })
    workflow.add_edge("rewrite_query", "retrieve_kb")
    workflow.add_edge("search_web", "grade_web_evidence")
    workflow.add_conditional_edges("grade_web_evidence", decide_after_web, {
        "generate_from_web": "generate_from_web",
        "answer_insufficient": "answer_insufficient"
    })

    for n in ["generate_from_kb", "generate_from_web", "direct_answer", "answer_insufficient"]:
        workflow.add_edge(n, "check_hallucination")
    workflow.add_edge("check_hallucination", "evaluate_rag")
    workflow.add_edge("evaluate_rag", END)

    return workflow.compile()

# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(
    page_title="Multimodal Adaptive Agentic RAG",
    page_icon="🟢",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown('<div class="neon-title">MULTIMODAL ADAPTIVE AGENTIC RAG</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Self-Correcting Retrieval  •  Multimodal Documents  •  Hallucination Detection  •  Full Observability</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    load_keys()
    st.markdown("---")
    st.markdown("### 📄 Upload PDF")
    uploaded_file = st.file_uploader("Add multimodal knowledge", type=["pdf"], label_visibility="collapsed")
    st.markdown("---")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            with st.expander("🔍 Observability & Evaluation", expanded=False):
                meta = msg["meta"]
                st.markdown("**Agent Execution Path**")
                path = " → ".join(meta.get("agent_path", []))
                st.markdown(f'<div class="agent-path">{path}</div>', unsafe_allow_html=True)

                eval_data = meta.get("evaluation") or {}
                cols = st.columns(4)
                metrics = [
                    ("Faithfulness", eval_data.get("faithfulness")),
                    ("Answer Relevance", eval_data.get("answer_relevance")),
                    ("Context Relevance", eval_data.get("context_relevance")),
                    ("Grounded", "PASS" if meta.get("grounded") else "FAIL"),
                ]
                for col, (label, value) in zip(cols, metrics):
                    with col:
                        display = f"{value:.2f}" if isinstance(value, float) else str(value)
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{display}</div>
                            <div class="metric-label">{label}</div>
                        </div>
                        """, unsafe_allow_html=True)

                if meta.get("citations"):
                    st.markdown("**Citations**")
                    for c in meta["citations"]:
                        st.markdown(f"- `{c}`")

if prompt := st.chat_input("Ask a question about your documents or Agentic RAG..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.status("Agent is reasoning...", expanded=True)
        try:
            embeddings = get_embeddings()
            llm, _ = get_llms()
            vectorstore, retriever = get_vectorstore_and_retriever(embeddings)
            web_search = TavilySearch(max_results=4, include_answer=True)

            if uploaded_file:
                with status:
                    st.write("Ingesting multimodal PDF...")
                pdf_docs = extract_pdf_multimodal(uploaded_file.read(), uploaded_file.name)
                if pdf_docs:
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                    chunks = splitter.split_documents(pdf_docs)
                    vectorstore.add_documents(chunks)
                    st.toast(f"Ingested {len(chunks)} chunks", icon="✅")

            app = create_graph(llm, retriever, web_search)

            initial: AgentState = {
                "question": prompt,
                "current_query": prompt,
                "kb_docs": [],
                "web_results": "",
                "kb_grade": "",
                "kb_relevance_score": 0.0,
                "web_grade": "",
                "answer": "",
                "source_used": "",
                "retry_count": 0,
                "evaluation": None,
                "citations": [],
                "grounded": True,
                "trace": []
            }

            with status:
                st.write("Running agent graph...")
            result = app.invoke(initial)
            status.update(label="Completed", state="complete", expanded=False)

            st.markdown(result["answer"])

            meta = {
                "source_used": result["source_used"],
                "grounded": result.get("grounded"),
                "citations": result.get("citations", []),
                "evaluation": result.get("evaluation"),
                "agent_path": [t["node"] for t in result.get("trace", [])],
                "full_trace": result.get("trace", [])
            }

            with st.expander("🔍 Observability • Agent Path • Evaluation", expanded=True):
                st.markdown("**Agent Execution Path**")
                path = " → ".join(meta["agent_path"])
                st.markdown(f'<div class="agent-path">{path}</div>', unsafe_allow_html=True)

                eval_data = meta.get("evaluation") or {}
                cols = st.columns(4)
                for col, (label, key) in zip(cols, [
                    ("Faithfulness", "faithfulness"),
                    ("Answer Rel.", "answer_relevance"),
                    ("Context Rel.", "context_relevance"),
                    ("Grounded", None)
                ]):
                    with col:
                        if key:
                            val = eval_data.get(key)
                            display = f"{val:.2f}" if isinstance(val, float) else "—"
                        else:
                            display = "PASS" if meta.get("grounded") else "FAIL"
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{display}</div>
                            <div class="metric-label">{label}</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown(f"**Source Used:** `{meta['source_used']}`")
                
                if meta.get("citations"):
                    st.markdown("**Citations**")
                    for c in meta["citations"]:
                        st.markdown(f"- {c}")

                with st.expander("Full Execution Trace"):
                    for t in meta["full_trace"]:
                        st.markdown(f"**[{t['timestamp']}] {t['node']}** — {t['message']}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "meta": meta
            })

        except Exception as e:
            status.update(label="Error", state="error")
            st.error(f"Error: {str(e)}")
            st.exception(e)

st.markdown("---")
st.caption("🟢 Multimodal Adaptive Agentic RAG  •  LangGraph + Pinecone + Groq + Tavily  •  Full Observability")