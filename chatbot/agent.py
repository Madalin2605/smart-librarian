import os, json
from typing import List, Optional
from openai import OpenAI
from chatbot.retriever import search_books
from tools.summary_tool import get_summary_by_title


openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

# Tool definition for the summary retrieval function
# This is the OpenAI "function calling" format
summary_tool_definition = {
    "type": "function",
    "function": {
        "name": "get_summary_by_title",
        "description": "Returns the complete summary of a book, given the exact title.",
        "parameters": {
            "type": "object",
            "properties": {"title": {"type": "string", "description": "The exact title of the book (e.g: '1984')"}},
            "required": ["title"],
        },
    },
}


def choose_title_llm(query: str, candidates: List[str], model: str = "gpt-4o-mini") -> Optional[str]:
    """
    Model returns exactly ONE title from candidates or 'NONE'.
    """
    if not candidates:
        return None
    
    system = (
        "You are a strict selector. From the provided list of titles, choose EXACTLY ONE that best matches "
        "the user query. If none fit, reply with EXACTLY 'NONE'. Reply with the title string only."
    )

    titles_block = "\n".join(f"- {t}" for t in candidates)
    user = f"User query: {query}\n\nTitles:\n{titles_block}\n\nAnswer with one title or NONE."

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )

    raw = (resp.choices[0].message.content or "").strip()

    if raw == "NONE":
        return None
    
    return raw if raw in candidates else None


def run_agent(user_query: str, model: str = "gpt-4o-mini") -> str:
    """
    Agent that finds and summarizes a book based on user query.
    """
    # 1) Retrieve candidates
    results = search_books(user_query)
    matched_titles = [m["title"] for m in results.get("metadatas", [[]])[0]]
    print("Matched titles:", matched_titles)

    if not matched_titles:
        return "Nu am gasit nicio carte relevanta in baza de date."

    # 2) Choose ONE title (no tools here)
    chosen = choose_title_llm(user_query, matched_titles, model=model)
    print("Chosen title:", chosen)

    if not chosen:
        return "Nu am gasit o potrivire suficient de buna pentru cererea ta."

    # 3) Force the tool call to get the summary for that exact title
    messages = [
        {
            "role": "system",
            "content": (
                "Call the function get_summary_by_title with EXACTLY the provided title. "
                "Do not ask questions. Do not output any text other than the function call."
            ),
        },
        {
            "role": "user",
            "content": json.dumps({"title": chosen}, ensure_ascii=False),
        },
    ]

    # Force the tool call here (no 'auto'):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=[summary_tool_definition],
        tool_choice={"type": "function", "function": {"name": "get_summary_by_title"}},
        temperature=0,
    )

    msg = response.choices[0].message

    # It must contain a tool call now
    if not msg.tool_calls:
        # Defensive fallback – but with forced tool_choice this shouldn't happen
        summary = get_summary_by_title(chosen)
        return f"Recomandare: {chosen}\n\n{summary}"

    tool_call = msg.tool_calls[0]
    args = json.loads(tool_call.function.arguments or "{}")
    title_arg = args.get("title") or chosen
    summary = get_summary_by_title(title_arg)

    return f"Recomandare: {chosen}\n\n{summary}"
