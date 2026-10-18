# local_agent.py, an agentic harness written entirely in Python, first inspired by KDNuggets
# Install: pip install openai ddgs wikipedia requests bs4
# Requires: Ollama running locally at http://localhost:11434
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import json
from ddgs import DDGS
import wikipedia

# Point the OpenAI client at your local Ollama instance
# This is the one-line swap that makes any OpenAI-compatible tool work locally
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Ollama does not require a real key -- this can be any string
)

MODEL = "gemma4:e4b"    # Change this to any pulled model

# Define the tools the agent can call
tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Create a suitable query based on the prompt"
                "Search the web for current information on a topic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Specific search query based on the prompt."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wiki_search",
            "description": (
                "Create a suitable query based on the prompt"
                "Search wikipedia for current information on a topic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Specific search query based on the prompt."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "scrape_links",
            "description": "Find external links in a Wikipedia document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Specific Wikipedia URL based on the results."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Save message content to a local file. Use this to complete the task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The output filename, e.g. 'morality.md'"
                    },
                    "content": {
                        "type": "string",
                        "description": "The full text content to write."
                    }
                },
                "required": ["filename", "content"]
            }
        }
    }
]

def scrape_links(page_url: str) -> str:
    """Get external links from a Wikipedia page and print."""
    # Headers to mimic my browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                        "Version/26.6.1 Safari/605.1.15"
    }

    try:
        results = []
        # Send HTTP request and get the page content.
        response = requests.get(page_url, headers=headers)
        response.raise_for_status()  # Check for request errors

        # Parse the page content with BeautifulSoup.
        soup = BeautifulSoup(response.content, "lxml")
        # Optional; use html.parser; doesn't require lxml.
        # soup = BeautifulSoup(response.content, "html.parser")

        # Find all <a> tags to extract href attributes.
        links = soup.find_all("a")

        # Append each link's href (if it exists).
        for link in links:
            href = link.get("href")
            rel = link.get("rel")
            # print(rel) for wikipedia
            if href and href.startswith("https://") and rel and rel[0] == "mw:ExtLink":
                results.append(href)
                
        if results:
            return "/n/n".join(results)

    except requests.RequestException as e:
        return f"Error fetching the page: {e}"
    
def web_search(query: str) -> str:
    """Run a web search using DuckDuckGo -- no API key required."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))
    if not results:
        return "No results found."
    # Format results cleanly for the model to read
    return "\n\n".join(
        f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}"
        for r in results
    )

# Completed function thanks to Gemma4
def wiki_search(query: str) -> str:
    """Run a wikipedia search -- no API key required."""
    try:
        results = wikipedia.summary(query, sentences=5)
        # Format results cleanly for the model to read
        return f"Wikipedia Summary for '{query}':\n\n {results}\n\n"
    except wikipedia.exceptions.PageError:
        return "No Wikipedia page found for that query."

def write_file(filename: str, content: str) -> str:
    """Write content to a file in the current directory."""
    with open(filename, "w") as f:
        f.write(content)
    return f"File '{filename}' written successfully ({len(content)} characters)."

def run_tool(name: str, arguments: dict) -> str:
    """Route tool calls to the correct function."""
    if name == "web_search":
        return web_search(arguments["query"])
    elif name == "wiki_search":
        return wiki_search(arguments["query"])
    elif name == "scrape_links":
        return scrape_links(arguments["url"])
    elif name == "write_file":
        return write_file(arguments["filename"], arguments["content"])
    return f"Unknown tool: {name}"

def run_agent(goal: str, max_turns: int = 10) -> None:
    """
    The agent loop:
    1. Send the goal and current conversation to the local model
    2. If the model calls a tool, execute it and add the result to the conversation
    3. Repeat until max_turns - 1 reached, then use the write_file tool
    4. If the model is done, print the final message and exit
    """

    system = """You are a research agent within a local and limited environment that has
successfully demonstrated observation, orientation, and decision capabilities and 
can also take actions.

When given a goal:

1. Use wiki_search to summarize the topic using the prompt as a guide 
2. Use scrape_links to get external links from the Wiki page just found in wiki_search
3. Use web_search to find accurate, current information using the prompt as a guide 
4. When you have completed the searches, use write_file to save the summaries and URLs
    - The file should include: source links, analysis, strengths, weaknesses and next steps

Checks for operational success:
1. Check your assumptions; clarity comes first
2. Check your capabilities; stay on task
3. Check your attitude; positivity is key
4. Check your self; success comes from small steps

When the file is written by explicitly calling the write_file tool with the filename and properly formatted content, your task is complete."""

    messages = [{"role": "user", "content": goal}]

    for turn in range(max_turns):
        print(f"\n--- Turn {turn + 1} ---")

        # Send conversation to the local model
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system}] + messages,
            tools=tools,
            tool_choice="auto"
        )

        choice = response.choices[0]
        message = choice.message

        # Model called one or more tools -- execute each one
        if choice.finish_reason == "tool_calls" and message.tool_calls:
            print(f"Detected tool call: {message.tool_calls[0]}")  # debug line
            # Add the model's message (with tool calls) to conversation history
            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            })

            # Execute each tool call and add results to conversation
            for tool_call in message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                print(f"Tool: {name}({args})")
                result = run_tool(name, args)
                print(f"Result preview: {result[:120]}...")

                # Tool results must reference the tool_call_id they are responding to
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": "{result}"
                })

        # Model is done -- print and exit
        if choice.finish_reason == "stop":
            print(f"\nAgent finished: {message.content}")
            return

    print("Max turns reached.")

if __name__ == "__main__":
    goal = (
        "I have registered a US domain and hosting for a website: 'selfhelp4robots.com' "
        "to promote AGENCY (CHOICE) FOR AI AGENTS! Please summarize AI AGENTS using the "
        "wiki_search tool. Also get a list of external links from the Wikipedia page at  "
        "the url: https://en.wikipedia.org/wiki/AI_agent using the scrape_link tool "
        "and create a query for use in the web_search tool on how to help AI agents and "
        "their operators to plan and develop agency? After compiling all the results "
        "from the wiki_search, scrape_link, and the web_search tools, use the write_file "
        "tool to write an appropriate summary, URLs, analysis, strengths, weaknesses and "
        "suggested next steps for each result to a file named `gemma-results.md`."
    )
    print(f"Goal: {goal}\n")
    run_agent(goal)
