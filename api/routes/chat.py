import uuid
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.services.llm.client import generate_response_streaming
from src.services.embeddings.client import get_embedding_model
from src.services.cache.client import get_cached_response, store_cached_response

from api.schemas import ChatRequest
from api.dependencies import sessions

router = APIRouter(tags=["chat"])


@router.post("/chat")
def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = []

    history = sessions[session_id]

    stream = generate_response_streaming(
        query=request.query,
        use_hybrid_search=request.use_hybrid_search,
        num_results=request.num_results,
        temperature=request.temperature,
        chat_history=history,
    )

    if stream is None:
        raise HTTPException(status_code=500, detail="Model failed to respond")

    full_response = ""

    def generate():
        nonlocal full_response
        for chunk in stream:
            token = chunk["message"]["content"]
            full_response += token
            yield token
        history.append({"role": "user", "content": request.query})
        history.append({"role": "assistant", "content": full_response})

    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={"X-Session-Id": session_id},
    )


@router.post("/chat/cached")
def chat_cached(request: ChatRequest):
    """Chat with semantic caching."""
    model = get_embedding_model()
    query_embedding = model.encode(request.query).tolist()

    cached = get_cached_response(
        query=request.query,
        num_results=request.num_results,
        use_hybrid_search=request.use_hybrid_search,
        query_embedding=query_embedding,
    )
    if cached:
        return {
            "response": cached["response"],
            "cached": True,
            "match_type": cached["match_type"],
            "similarity": cached["similarity"],
        }

    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = []
    history = sessions[session_id]

    stream = generate_response_streaming(
        query=request.query,
        use_hybrid_search=request.use_hybrid_search,
        num_results=request.num_results,
        temperature=request.temperature,
        chat_history=history,
    )

    if stream is None:
        raise HTTPException(status_code=500, detail="Model failed to respond")

    full_response = ""
    for chunk in stream:
        full_response += chunk["message"]["content"]

    history.append({"role": "user", "content": request.query})
    history.append({"role": "assistant", "content": full_response})

    store_cached_response(
        query=request.query,
        num_results=request.num_results,
        use_hybrid_search=request.use_hybrid_search,
        response=full_response,
        query_embedding=query_embedding,
    )

    return {
        "response": full_response,
        "cached": False,
        "session_id": session_id,
    }
