---
title: Prompt files for RAG Streamlit app

date: 2025-11-19
---

This folder contains the versioned prompt templates used by the Streamlit RAG app.

Files
-----

- `system_prompt_v1.txt` (role: system / assistant instructions)
  - Purpose: Sets the assistant's global role, tone and strong guardrails (no hallucination, prefer books present in Context).
  - Key expectations / placeholders:
    - No template variables in this file — it is prepended as a system instruction.
  - Example rule: "Do not invent book titles or authors. If missing, say so and offer a close alternative."

- `get_response_v1.txt` (role: human / main LLM prompt)
  - Purpose: Main response template used when answering user queries. It receives chat history, retrieved Context and the user's question.
  - Placeholders:
    - `{chat_history}` — serialized chat history (list of messages). Keep this concise to avoid long prompts.
    - `{context}` — the formatted retrieved documents (titles, summaries, review_text, ratings). This is the primary grounding source.
    - `{question}` — the user's current question.
  - Behavior notes:
    - Contains rules for how to format recommendations, how to return reviews and ratings, and what to do when no matching book is present.
    - Version this file when modifying instructions for response formatting or grounding behavior.

- `multi_query_v1.txt` (role: generator / query expansion)
  - Purpose: Used to generate alternative retrieval queries (multi-query) to increase recall from the vector DB.
  - Placeholders:
    - `{question}` — original user question.
  - Output contract:
    - Must return multiple alternative queries separated by newlines (the pipeline splits on newlines).
    - Should not include extra commentary or numbered lists — only the plain queries.

Guidelines for editing
----------------------

- Keep each prompt small and focused. Use explicit instructions about what to include and what not to hallucinate.
- When you update a prompt, increment the version suffix (e.g., `get_response_v2.txt`) and update this README `version` and `date` header.
- Avoid embedding long retrieved documents inside prompts directly; instead, provide summarized `context` produced by `format_docs()`.

How prompts are used in code
---------------------------

- `system_prompt_v1.txt` is loaded and prepended to the human prompt before creating the ChatPromptTemplate. This enforces guardrails.
- `multi_query_v1.txt` is passed into a small LLM pipeline that produces multiple alternative queries; the output is split on newlines and used to run multiple retrievals.
- `get_response_v1.txt` is the main template fed to the LLM along with `{context}`, `{chat_history}`, and `{question}`.

Quick troubleshooting
---------------------

- If the model hallucinates titles, ensure `format_docs()` actually includes `book_title` and `review_text` fields and that `{context}` is not empty.
- If retrieval returns poor results, increase the number of alternative queries or refine the `multi_query` prompt to generate more focused queries.


