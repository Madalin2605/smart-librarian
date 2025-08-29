import os
from openai import OpenAI
from chatbot.retriever import search_books
from tools.summary_tool import get_summary_by_title


openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

# Define the tool schema that tells the LLM what function it can call.
# This is the OpenAI "function calling" format.
summary_tool_definition = {
    "type": "function",
    "function": {
        "name": "get_summary_by_title",
        "description": "Returneaza rezumatul complet al unei carti, dat fiind titlul exact.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Titlul exact al cartii (ex: '1984')"
                }
            },
            "required": ["title"]
        }
    }
}


def run_agent(user_query: str, model: str = "gpt-4o-mini"):
    """
    Main entry point: takes a user query, finds relevant books,
    asks GPT to recommend one, and (if needed) calls the summary tool.
    """
    # --- Step 1: Retrieve candidate books from ChromaDB ---
    results = search_books(user_query)
    matched_titles = [m["title"] for m in results["metadatas"][0]]
    print(f"Matched titles: {matched_titles}")

    # --- Step 2: Build context prompt for the model ---
    # We tell GPT which books it is allowed to choose from
    context = "\n".join([f"- {title}" for title in matched_titles])
    print(f"Context for LLM:\n{context}")

    messages = [
        {
            "role": "system",
            "content": (
                "Esti un asistent AI care recomanda carti. Tinand cont de cerinta utilizatorului, recomanda una "
                "dintre urmatoarele carti:\n" + context +
                "\n Raspunde doar cu titlul cartii. Dupa recomandare, apeleaza functia get_summary_by_title cu titlul"
                "\n recomandat, pentru a obtine un rezumat complet."
            )
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    # --- Step 3: Call the OpenAI model with tool definition ---
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=[summary_tool_definition],
        tool_choice="auto"
    )

    message = response.choices[0].message
    
    # --- Step 4: Check if the model decided to call a tool ---
    if message.tool_calls:
        tool_call = message.tool_calls[0]
        title_arg = tool_call.function.arguments
        import json
        args = json.loads(title_arg)
        summary = get_summary_by_title(args["title"])

        return message.content + f"\n\n Rezumat complet:\n{summary}"
    else:
        return message.content
