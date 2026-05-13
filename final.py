from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage,ToolMessage
load_dotenv()
if os.getenv("GROQ_API_KEY"):
    print("GROQ_API_KEY is set.")
else:
    raise ValueError("GROQ_API_KEY is not set")
llm = ChatGroq(model="llama-3.1-8b-instant",)

from pydantic import BaseModel, Field
from langgraph.types import Send
from langchain_tavily import TavilySearch
from langchain_chroma import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings
import sys
import json
def print_color(text, color="green"):
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "reset": "\033[0m"
    }
    print(f"{colors.get(color, '')}{text}{colors['reset']}")
class Plan(BaseModel):
    use_llm:bool
    use_rag:bool
    use_web:bool
    llm_query:str
    rag_query:str
    web_query:str

class Evaluation(BaseModel):
    approved:bool
    feedback:str

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

def orchestrator_node(state: GraphState)-> GraphState:
    prompt=ChatPromptTemplate.from_template(
        "Analyze the following question and decide which tools to use:\n\n"
        "Question: {question}\n\n"
        "Consider:\n"
        "- use_llm: For general knowledge questions\n"
        "- use_rag: If the question requires up-to-date or specialized information\n"
        "- use_web: If the question is about current events or specific facts\n\n"
        "Return a JSON-like response with use_llm, use_rag, use_web (true/false)\n"
        "and the specific query to send to each tool."
    )
    chain=prompt|llm
    response=chain.invoke({"question":state.question})
    content=response.content.lower()
    state.plan = Plan(
        use_llm=True,
        use_rag=("rag" in content and "true" in content),
        use_web=("web" in content and "true" in content),
        llm_query=state.question,
        rag_query=state.question,
        web_query=state.question
    )
    return state

def llm_worker_node(state):
    llm=ChatGroq(model="llama-3.1-8b-instant",temperature=0.7)
    prompt=ChatPromptTemplate.from_template("Answer the following question based on your knowledge:\n\nQuestion: {question}\n\nAnswer:")
    chain =prompt| llm
    query=state.get("llm_query") or state["question"]
    result=chain.invoke({"question":query})
    return {"llm_result":result.content}

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

def collector_node(state: GraphState) -> GraphState:
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
    else:
        state.source_contribution = {"llm_pct": 0, "rag_pct": 0, "web_pct": 0}
    
    return state

def rag_validator_node(state: GraphState) -> GraphState:
    if not state.rag_result:
        print_color("[RAG Validator] No RAG content to validate", "yellow")
    else:
        validation_prompt = ChatPromptTemplate.from_template(
            "Question: {question}\n\n"
            "RAG Document:\n{rag_content}\n\n"
            "Is this document relevant to answering the question? "
            "Respond with ONLY: RELEVANT or IRRELEVANT"
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
    MAX_CHARS = 2000
    
    llm_text = state.evidence.get("llm", "")[:MAX_CHARS]
    rag_text = state.evidence.get("rag", "")[:MAX_CHARS]
    web_text = str(state.evidence.get("web", ""))[:MAX_CHARS]
    
    prompt = ChatPromptTemplate.from_template(
        "Based ONLY on the following evidence, answer the question. "
        "If evidence is insufficient, say 'I don't have enough information'. "
        "Do NOT make up facts, names, or URLs. Only cite sources that appear in the evidence below.\n\n"
        "LLM Evidence: {llm}\n\n"
        "RAG Evidence: {rag}\n\n"
        "Web Evidence: {web}\n\n"
        "Question: {question}\n\n"
        "Answer concisely (do not fabricate sources):"
    )
    chain = prompt | llm
    result = chain.invoke({
        "llm": llm_text or "No LLM evidence available",
        "rag": rag_text or "No RAG evidence available",
        "web": web_text or "No Web evidence available",
        "question": state.question
    })
    state.draft_answer = result.content
    state.retry_count += 1
    return state

def evaluator_node(state: GraphState) -> GraphState:
    prompt = ChatPromptTemplate.from_template(
        "Evaluate the following answer for quality and accuracy:\n\n"
        "Question: {question}\n\n"
        "Answer: {answer}\n\n"
        "Respond with ONLY: APPROVED or REJECTED\n"
        "and a brief reason."
    )
    chain = prompt | llm
    result = chain.invoke({
        "question": state.question,
        "answer": state.draft_answer
    })
    
    response_text = result.content.strip().upper()
    
    if response_text.startswith("APPROVED"):
        approved = True
    elif response_text.startswith("REJECTED"):
        approved = False
    else:
        approved = "yes" in response_text or "true" in response_text
    
    state.evaluation = Evaluation(
        approved=approved,
        feedback=result.content
    )
    
    if approved:
        state.final_answer = state.draft_answer

    return state

def route_workers(state: GraphState):
    plan = state.plan
    sends = []
    if plan.use_llm:
        sends.append(Send("llm_worker", {"question": state.question, "llm_result": ""}))
    if plan.use_rag:
        sends.append(Send("rag_worker", {"question": state.question, "rag_result": ""}))
    if plan.use_web:
        sends.append(Send("web_worker", {"question": state.question, "web_results": ""}))
    
    if not sends:
        sends.append(Send("llm_worker", {"question": state.question, "llm_result": ""}))
    
    return sends


def route_after_evaluation(state: GraphState):

    evaluation = state.evaluation
    retry_count = state.retry_count

    if evaluation is not None and not evaluation.approved and retry_count < 3:
        return "generator"

    return END

from langgraph.graph import StateGraph, START, END

builder=StateGraph(GraphState)

builder.add_node("orchestrator",orchestrator_node)
builder.add_node("llm_worker",llm_worker_node)
builder.add_node("rag_worker",rag_worker_node)
builder.add_node("web_worker",web_worker_node)
builder.add_node("collector",collector_node)
builder.add_node("rag_validator",rag_validator_node)
builder.add_node("generator",generator_node)
builder.add_node("evaluator",evaluator_node)

builder.add_edge(START, "orchestrator")
builder.add_conditional_edges("orchestrator",route_workers, path_map=[
        "llm_worker",
        "rag_worker",
        "web_worker",
    ],)
builder.add_edge("llm_worker", "collector")
builder.add_edge("rag_worker", "collector")
builder.add_edge("web_worker", "collector")
builder.add_edge("collector", "rag_validator")
builder.add_edge("rag_validator", "generator")
builder.add_edge("generator", "evaluator")
builder.add_conditional_edges("evaluator",route_after_evaluation, path_map=[
        "generator",
        END,
    ],)

from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config={"configurable":{"thread_id":"user-4"}}

initial_state = GraphState(
    question=input("what is your question : ")
)
result = graph.invoke(initial_state, config=config)
print("\n" + "="*60)
print("DEBUG INFO:")
print("="*60)
llm_result = result.get('llm_result', '')
rag_result = result.get('rag_result', '')
web_result = result.get('web_results', '')
print(f"LLM result length: {len(llm_result)}")
print(f"RAG result length: {len(rag_result)}")
print(f"Web result length: {len(web_result)}")
print(f"\nDraft answer: {result.get('draft_answer', 'EMPTY')[:200]}")
print(f"Evaluation: {result.get('evaluation')}")
print(f"Final answer: {result.get('final_answer', 'EMPTY')[:200]}")
if not web_result:
    print_color("\n⚠️ Web search returned no results - check TAVILY_API_KEY", "red")
print(graph.get_graph().draw_mermaid())

print_color("\n" + "="*60, "blue")
print_color("FINAL ANSWER:", "green")
print_color("="*60, "blue")
answer = result.get("final_answer") or result.get("draft_answer") or "No answer generated"
print(answer)

print_color("\n" + "="*60, "blue")
print_color("SOURCES & CONTRIBUTIONS:", "yellow")
print_color("="*60, "blue")

contrib = result.get("source_contribution", {})
print(f"\n  LLM (internal knowledge): {'✓' if result['llm_result'] else '✗'} - {contrib.get('llm_pct', 0)}%")
print(f"  RAG (your documents):    {'✓' if result['rag_result'] else '✗'} - {contrib.get('rag_pct', 0)}%")
print(f"  Web (internet):          {'✓' if result['web_results'] else '✗'} - {contrib.get('web_pct', 0)}%")

if not result['rag_result']:
    print_color("\n  Note: RAG returned no relevant documents. Consider adding relevant documents to data/uploads/", "red")
if not result['web_results']:
    print_color("\n  Note: Web search returned no results.", "red")