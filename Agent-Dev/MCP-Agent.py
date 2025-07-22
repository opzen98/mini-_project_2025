import os, getpass
from datetime import datetime
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model
import asyncio
import json
import os
from mem0 import Memory
import configuration

llm = init_chat_model("openai:gpt-4o-mini", temperature=0.0)

# Pass the initialized model to the config
llmconfig = {
    "llm": {
        "provider": "langchain",
        "config": {
            "model": llm
        }
    },
    "vector_store": {
        "provider": "supabase",
        "config": {
            "connection_string": os.environ['DATABASE_URL'],
            "collection_name": "MCP_Agent"
        }
    }    
}

mem0 = Memory.from_config(llmconfig)


def load_mcp_config():
    """Load MCP configuration and filter out disabled servers"""
    
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Filter out disabled servers
    filtered_config = {}
    for server_name, server_config in config.items():
        if not server_config.get('disabled', False):
            filtered_config[server_name] = server_config
    
    return filtered_config

# Replace your current config loading with:
config = load_mcp_config()
client = MultiServerMCPClient(connections=config)

async def gets_tools():
    tools = await client.get_tools()
    return tools

tools = asyncio.run(gets_tools())
llm_with_tools = llm.bind_tools(tools)

system_message_template ="""{asst_role}:\n. Your task is to solve user queries by reasoning step-by-step. The current datetime is: {now}
This is your User Memories: User Memories:\n{memories_str}
For each task:
1. Think through the problem methodically.
2. Decide which tool(s) to use.
3. Execute the chosen tool(s).
4. Observe and analyze the output.
5. If the output is unsatisfactory or incomplete, reconsider your approach and utilize alternative tools or methods.
6. Continue this process until you arrive at a satisfactory solution.
7. Once confident, provide the final answer to the user.
8. If your response includes code, present it within a markdown-formatted Python code block.
for example 
```python
...
``` """

class State(MessagesState):
    summary: str
    initial_query: str
    final_response: str
    memories_retrieved: bool

def preserve_msg(state: State):
    """Clean up old messages to prevent memory bloat"""
    messages = state["messages"]
    human_indices = []
    for idx, msg in enumerate(messages):
        if type(msg).__name__ == 'HumanMessage':
            human_indices.append(idx)
    
    if len(human_indices) < 2:
        return {"messages": []}
    
    start_index = human_indices[-2]
    messages_to_remove = messages[:start_index]
    
    delete_ops = [RemoveMessage(id=msg.id) for msg in messages_to_remove]
    return {"messages": delete_ops}

def retrieve_memories(state: State, config: RunnableConfig):
    """Retrieve memories only once per user query"""
    if state.get("memories_retrieved"):
        return {"messages": []}
    
    configurable = configuration.Configuration.from_runnable_config(config)
    mem0_user_id = configurable.user_id
    
    messages = state["messages"]
    
    def ensure_str(x):
        if isinstance(x, str):
            return x
        if isinstance(x, list):
            try:
                return " ".join(map(str, x))
            except Exception:
                return str(x)
        return str(x)

    # Get the latest user message
    last_content = ensure_str(messages[-1].content)

    
    return {
        "initial_query": last_content,
        "memories_retrieved": True,
        "messages": []
    }

def assistant(state: State, config: RunnableConfig):
    """Main assistant function with memory context"""
    configurable = configuration.Configuration.from_runnable_config(config)
    mem0_user_id = configurable.user_id
    assistant_role = configurable.asst_role
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = state.get("summary", "")
    messages = state["messages"]
    
    # Get memories for this query (already retrieved)
    memories = mem0.search(state.get("initial_query", ""), user_id=mem0_user_id, limit=3)
    memories_str = "\n".join(f"- {entry['memory']}" for entry in memories.get("results", []))

    sys_msg = system_message_template.format(now=now, asst_role=assistant_role, memories_str=memories_str)
    
    if summary:
        messages = [SystemMessage(content=sys_msg + summary)] + messages
    else:
        messages = [SystemMessage(content=sys_msg)] + messages
    
    response = llm_with_tools.invoke(messages)
    
    # Store the response for later memory storage
    return {
        "messages": response,
        "final_response": response.content
    }

def store_memory(state: State, config: RunnableConfig):
    """Store the final interaction in mem0"""
    configurable = configuration.Configuration.from_runnable_config(config)
    mem0_user_id = configurable.user_id
    
    initial_query = state.get("initial_query", "")
    final_response = state.get("final_response", "")
    
    if initial_query and final_response:
        mem0.add(f"User: {initial_query}\nAssistant: {final_response}", user_id=mem0_user_id)
    
    return {"messages": []}

def should_continue(state: State):
    """Determine the next step in the conversation"""
    messages = state["messages"]
    
    # Check if we need to summarize due to length
    if len(messages) > 20:
        return "summarize"
    
    # Otherwise, store memory and end
    return "store_memory"

def summarize_conversation(state: State):
    """Summarize the conversation when it gets too long"""
    summary = state.get("summary", "")
    messages = state["messages"]

    if summary:
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    messages_with_prompt = messages + [HumanMessage(content=summary_message)]
    response = llm.invoke(messages_with_prompt)
    
    return {"summary": response.content}

# Create the main workflow
workflow = StateGraph(State, config_schema=configuration.Configuration)

# Add nodes
workflow.add_node("preserve", preserve_msg)
workflow.add_node("retrieve_memories", retrieve_memories)
workflow.add_node("assistant", assistant)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("store_memory", store_memory)
workflow.add_node("summarize", summarize_conversation)

# Define the flow
workflow.add_edge(START, "preserve")
workflow.add_edge("preserve", "retrieve_memories")
workflow.add_edge("retrieve_memories", "assistant")

# ReAct loop: assistant -> tools -> assistant (until no more tools needed)
workflow.add_conditional_edges("assistant", tools_condition)
workflow.add_edge("tools", "assistant")

# After assistant is done, decide next step
workflow.add_conditional_edges("assistant", should_continue)

# Handle memory storage and summarization
workflow.add_edge("store_memory", END)
workflow.add_edge("summarize", "store_memory")

# Compile the graph
graph = workflow.compile()
