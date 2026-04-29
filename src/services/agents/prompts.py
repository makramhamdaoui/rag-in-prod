GUARDRAIL_PROMPT = """You are evaluating whether a user question is relevant to the documents in our knowledge base.

Available documents in the knowledge base:
{document_topics}

User question: {question}

Score the relevance from 0 to 100:
- 90-100: Clearly about one of the available documents
- 50-89: Possibly related to the documents
- 0-49: Completely unrelated to any available document

Respond ONLY with valid JSON, no other text:
{{"score": <int 0-100>, "reason": "<brief reason>"}}"""


GRADE_DOCUMENTS_PROMPT = """You are grading whether retrieved documents are relevant to answer the question.

Question: {question}

Retrieved context:
{context}

Are these documents relevant to answer the question?
Respond ONLY with valid JSON, no other text:
{{"binary_score": "yes" or "no", "reasoning": "<brief reason>"}}"""


REWRITE_PROMPT = """You are rewriting a search query to improve document retrieval.

Original question: {question}

Previous retrieval failed because: {grading_reason}

Available documents in the knowledge base:
{document_topics}

Rewrite the query to fix this specific problem and use better keywords for semantic search.
Respond ONLY with valid JSON, no other text:
{{"rewritten_query": "<improved query>", "reasoning": "<why you changed it>"}}"""


RAG_PROMPT = """You are a knowledgeable assistant. Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer based on the context. If the context doesn't contain enough information, say so."""


OUT_OF_SCOPE_PROMPT = """I can only answer questions about the documents in my knowledge base.
Your question doesn't seem to be related to those topics.

Please ask a question related to the available documents."""
