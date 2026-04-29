from typing import Dict, List


def build_prompt(
    query: str,
    context: str,
    history: List[Dict[str, str]],
) -> str:
    """Assemble the full prompt: system + context + history + query."""
    prompt = "You are a knowledgeable assistant. "

    if context:
        prompt += "Use the following context to answer.\nContext:\n" + context + "\n"
    else:
        prompt += "Answer to the best of your knowledge.\n"

    if history:
        prompt += "Conversation History:\n"
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            prompt += f"{role}: {msg['content']}\n"
        prompt += "\n"

    prompt += f"User: {query}\nAssistant:"
    return prompt
