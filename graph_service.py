from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import json
import os

load_dotenv()

class Plan(BaseModel):
    use_llm: bool
    use_rag: bool
    use_web: bool
    llm_query: str
    rag_query: str
    web_query: str

class Evaluation(BaseModel):
    approved: bool
    feedback: str

class GraphState(BaseModel):
    messages: list = Field(default_factory=list)
    question: str = ""
    plan: Plan | None = None
    llm_result: str = ""
    rag_result: str = ""
    web_results: str = ""
    evidence: dict = Field(default_factory=dict)
    source_contribution: dict = Field(default_factory=dict)
    draft_answer: str = ""
    evaluation: Evaluation | None = None
    retry_count: int = 0
    final_answer: str = ""

class Message(BaseModel):
    role: str
    content: str
    timestamp: str = ""

llm = None

def set_llm(model):
    global llm
    llm = model

def print_color(text, color="green"):
    colors = {"red": "\033[91m", "green": "\033[92m", "yellow": "\033[93m", "blue": "\033[94m", "reset": "\033[0m"}
    print(f"{colors.get(color, '')}{text}{colors['reset']}")

def orchestrator_node(state: GraphState) -> GraphState:
    state.plan = Plan(
        use_llm=True,
        use_rag=True,
        use_web=True,
        llm_query=state.question,
        rag_query=state.question,
        web_query=state.question
    )
    return state

def llm_worker_node(state):
    global llm
    if llm is None:
        from langchain_groq import ChatGroq
        llm = ChatGroq(model="llama-3.1-8b-instant")
    
    prompt = ChatPromptTemplate.from_template("Answer based on knowledge:\n\nQuestion: {question}\n\nAnswer:")
    chain = prompt | llm
    query = state.get("llm_query") or state["question"]
    result = chain.invoke({"question": query})
    return {"llm_result": result.content}

def rag_worker_node(state):
    from app.ingestion.vector_store import similarity_search_with_scores
    
    query = state.get("rag_query") or state["question"]
    results = similarity_search_with_scores(query, k=4)
    
    MIN_SCORE = 0.7
    relevant_docs = [(doc, score) for doc, score in results if score >= MIN_SCORE]
    
    if not relevant_docs:
        print_color("[RAG] No relevant documents found", "yellow")
        return {"rag_result": ""}
    
    print_color(f"[RAG] Found {len(relevant_docs)} relevant docs (min score: {MIN_SCORE})", "green")
    context = "\n\n".join([doc.page_content for doc, _ in relevant_docs])
    return {"rag_result": context}

def web_worker_node(state):
    global llm
    if llm is None:
        from langchain_groq import ChatGroq
        llm = ChatGroq(model="llama-3.1-8b-instant")
    
    tool = TavilySearch(max_results=3)
    query = state.get("web_query") or state["question"]
    results = tool.invoke(query)
    
    parsed = json.loads(results) if isinstance(results, str) else results
    
    content_parts = []
    for r in parsed.get("results", [])[:3]:
        title = r.get("title", "Unknown")
        url = r.get("url", "")
        text = r.get("content", "")[:400]
        content_parts.append(f"Source: {title}\nURL: {url}\nContent: {text}")
    
    return {"web_results": "\n---\n".join(content_parts)}

def rag_validator_node(state: GraphState) -> GraphState:
    global llm
    if llm is None:
        from langchain_groq import ChatGroq
        llm = ChatGroq(model="llama-3.1-8b-instant")
    
    if not state.rag_result:
        print_color("[RAG Validator] No RAG content to validate", "yellow")
    else:
        validation_prompt = ChatPromptTemplate.from_template(
            "Question: {question}\n\nRAG Document:\n{rag_content}\n\n"
            "Is this document relevant to answering the question? Respond with ONLY: RELEVANT or IRRELEVANT"
        )
        chain = validation_prompt | llm
        
        response = chain.invoke({
            "question": state.question,
            "rag_content": state.rag_result[:1000]
        })
        
        if "IRRELEVANT" in response.content.upper():
            print_color("[RAG Validator] ❌ Content not relevant - clearing RAG", "red")
            state.rag_result = ""
        else:
            print_color("[RAG Validator] ✓ Content is relevant", "green")
    
    state.evidence = {
        "llm": state.llm_result or "",
        "rag": state.rag_result or "",
        "web": state.web_results or ""
    }
    
    llm_len = len(state.llm_result) if state.llm_result else 0
    rag_len = len(state.rag_result) if state.rag_result else 0
    web_len = len(state.web_results) if state.web_results else 0
    
    total = llm_len + rag_len + web_len
    if total > 0:
        state.source_contribution = {
            "llm_pct": llm_len * 100 // total,
            "rag_pct": rag_len * 100 // total,
            "web_pct": web_len * 100 // total
        }
    
    return state

def generator_node(state: GraphState) -> GraphState:
    global llm
    if llm is None:
        from langchain_groq import ChatGroq
        llm = ChatGroq(model="llama-3.1-8b-instant")
    
    history = ""
    if state.messages:
        history = "\n".join([f"{m['role']}: {m['content']}" for m in state.messages])
    
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful AI assistant.\n\n"
        "Conversation History:\n{history}\n\n"
        "Current Question: {question}\n\n"
        "Available Information:\n"
        "- LLM Knowledge: {llm}\n"
        "- RAG Documents: {rag}\n"
        "- Web Search: {web}\n\n"
        "Instructions:\n"
        "- Answer the current question based on the available information\n"
        "- You can refer to the conversation history for context\n"
        "- Do NOT make up facts not present in the available information\n"
        "- If information is not available, say you don't know\n\n"
        "Answer:"
    )
    chain = prompt | llm
    result = chain.invoke({
        "history": history or "No previous conversation",
        "llm": (state.evidence.get("llm") or "")[:2000] or "No LLM knowledge",
        "rag": (state.evidence.get("rag") or "")[:2000] or "No documents",
        "web": (state.evidence.get("web") or "")[:2000] or "No web results",
        "question": state.question
    })
    state.draft_answer = result.content
    state.retry_count += 1
    return state

def evaluator_node(state: GraphState) -> GraphState:
    global llm
    if llm is None:
        from langchain_groq import ChatGroq
        llm = ChatGroq(model="llama-3.1-8b-instant")
    
    prompt = ChatPromptTemplate.from_template(
        "Evaluate the following answer for quality:\n\nQuestion: {question}\n\nAnswer: {answer}\n\n"
        "Respond with ONLY: APPROVED or REJECTED"
    )
    chain = prompt | llm
    result = chain.invoke({"question": state.question, "answer": state.draft_answer})
    
    response_text = result.content.strip().upper()
    
    if response_text.startswith("APPROVED"):
        approved = True
    elif response_text.startswith("REJECTED"):
        approved = False
    else:
        approved = "yes" in response_text or "true" in response_text
    
    state.evaluation = Evaluation(approved=approved, feedback=result.content)
    
    if approved:
        state.final_answer = state.draft_answer

    return state

def route_workers(state: GraphState):
    plan = state.plan
    sends = []
    if plan.use_llm:
        sends.append(Send("llm_worker", {"question": state.question}))
    if plan.use_rag:
        sends.append(Send("rag_worker", {"question": state.question}))
    if plan.use_web:
        sends.append(Send("web_worker", {"question": state.question}))
    
    if not sends:
        sends.append(Send("llm_worker", {"question": state.question}))
    
    return sends

def save_message_node(state: GraphState) -> GraphState:
    if state.final_answer:
        state.messages = state.messages + [{"role": "assistant", "content": state.final_answer}]
    return state

def route_after_evaluation(state: GraphState):
    if state.evaluation is not None and not state.evaluation.approved and state.retry_count < 3:
        return "generator"
    return "save_message"

def create_graph():
    builder = StateGraph(GraphState)
    
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("llm_worker", llm_worker_node)
    builder.add_node("rag_worker", rag_worker_node)
    builder.add_node("web_worker", web_worker_node)
    builder.add_node("rag_validator", rag_validator_node)
    builder.add_node("generator", generator_node)
    builder.add_node("evaluator", evaluator_node)
    builder.add_node("save_message", save_message_node)
    
    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges("orchestrator", route_workers, ["llm_worker", "rag_worker", "web_worker"])
    builder.add_edge("llm_worker", "rag_validator")
    builder.add_edge("rag_worker", "rag_validator")
    builder.add_edge("web_worker", "rag_validator")
    builder.add_edge("rag_validator", "generator")
    builder.add_edge("generator", "evaluator")
    builder.add_conditional_edges("evaluator", route_after_evaluation, ["generator", "save_message"])
    builder.add_edge("save_message", END)
    
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)

graph = create_graph()