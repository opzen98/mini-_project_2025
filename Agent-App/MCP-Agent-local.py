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
from dotenv import load_dotenv
# Change this import to use AsyncPostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# Fix for Windows event loop compatibility with psycopg
import sys
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment variables from .env file
load_dotenv()

DB_URI = "postgresql://postgres:432306@localhost:5432/MCPAgent?sslmode=disable"

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

async def assistant(state: State, config: RunnableConfig):
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
    
    # Use await since llm_with_tools should be async
    response = await llm_with_tools.ainvoke(messages)
    
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
    if len(messages) > 10:
        return "summarize"
    
    # Otherwise, store memory and end
    return "__end__"

async def summarize_conversation(state: State):
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
    response = await llm.ainvoke(messages_with_prompt)
    
    return {"summary": response.content}

async def get_chat_history(graph, thread_id):
    all_states = [s for s in graph.get_state_history(thread_id)]
    return all_states[-1:]  # Return last 2 states for simplicity

async def test_streaming():
    print("Testing MCP Agent Streaming...")
    print("Type 'quit' to exit")
    
    try:
        # Use async context manager for PostgreSQL checkpointer
        async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
            # Setup the database schema (uncomment if needed)
            # await checkpointer.setup()

                        
            # Create the main workflow
            workflow = StateGraph(State, config_schema=configuration.Configuration)

            # Add nodes for React Agent
            workflow.add_node("preserve", preserve_msg)
            workflow.add_node("assistant", assistant)
            workflow.add_node("tools", ToolNode(tools))

            workflow.add_edge(START, "preserve")
            workflow.add_edge("preserve", "assistant")
            workflow.add_conditional_edges("assistant", tools_condition)
            workflow.add_edge("tools", "assistant")

            react_graph = workflow.compile(checkpointer=True)

            build = StateGraph(State, config_schema=configuration.Configuration)

            build.add_node("react_agent", react_graph)
            build.add_node("retrieve_memories", retrieve_memories)
            build.add_node("store_memory", store_memory)
            build.add_node("summarize", summarize_conversation)

            # Define the flow
            build.add_edge(START, "retrieve_memories")
            build.add_edge("retrieve_memories", "react_agent")
            build.add_edge("react_agent", "store_memory")
            #build.add_conditional_edges("store_memory", should_continue)
            build.add_conditional_edges(
                "store_memory", 
                should_continue,
                {
                    "summarize": "summarize",
                    "__end__": END
                }
            )

            build.add_edge("summarize", END)

            # Compile the graph
            graph = build.compile(checkpointer=checkpointer)

            # Get configuration
            config = configuration.Configuration()
            print(f"Loaded config - User ID: {config.user_id}")
            
            thread_id = "test-thread-122"
            run_config = {
                "configurable": {
                    "user_id": config.user_id,
                    "asst_role": config.asst_role,
                    "thread_id": thread_id
                }
            }
            
            # Continuous conversation loop
            while True:
                query = input("\nEnter your query: ")
                
                if query.lower() in ['quit', 'exit']:
                    print("Goodbye!")
                    break
                
                # Create fresh state for new query
                initial_state = {
                    "messages": [HumanMessage(content=query)],
                    "summary": "",
                    "initial_query": "",
                    "final_response": "",
                    "memories_retrieved": False
                }
                
                print(f"\nStreaming response for thread: {thread_id}")
                print("-" * 50)
                
                # Wait for completion and show only final result
                print("\n🤖 Processing...")
                
                # Run the graph and wait for completion
                final_result = await graph.ainvoke(initial_state, run_config)
                for m in final_result["messages"][-1:]:
                    m.pretty_print()
                    
    except Exception as e:
        print(f"Error in test_streaming: {e}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    asyncio.run(test_streaming())
