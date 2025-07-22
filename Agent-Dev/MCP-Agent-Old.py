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

# Load environment variables from .env file
load_dotenv()

llm = init_chat_model("openai:gpt-4o-mini", temperature=0.0)
# Set necessary environment variables for your chosen LangChain provider

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

with open('config.json', 'r') as f:
    config = json.load(f)

client = MultiServerMCPClient(connections=config)

async def gets_tools():
    tools = await client.get_tools()
    return tools
    #llm_with_tools = llm.bind_tools(tools)
    # rest of the code

tools = asyncio.run(gets_tools())

    # rest of the code
#tools = await client.get_tools()
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
 #   mem0_user_id: str

def assistant(state: State, config: RunnableConfig):

    configurable = configuration.Configuration.from_runnable_config(config)
    mem0_user_id = configurable.user_id
    assistant_role = configurable.asst_role

    #mem0_user_id = config.get("user_id", "default-user")
    #mem0_user_id = config.get("user_id", "default-user")
    #assistant_role = config.get("asst_role", "You are an intelligent assistant equipped with a suite of tools, including code execution capabilities.")
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = state.get("summary", "")
    message = state["messages"]
    #user_id = state.get("user_id", "default_user")
    def ensure_str(x):
        if isinstance(x, str):
            return x
        if isinstance(x, list):
            # if it’s a list of strings, join them; otherwise, fallback to a generic repr
            try:
                return " ".join(map(str, x))
            except Exception:
                return str(x)
        return str(x)

    last_content = ensure_str(message[-1].content)
    # Retrieve relevant memories
    memories = mem0.search(last_content, user_id=mem0_user_id, limit=3)
    memories_str = "\n".join(f"- {entry['memory']}" for entry in memories.get("results",[]))

    sys_msg = system_message_template.format(now=now, asst_role=assistant_role, memories_str=memories_str)
    
    if summary:
        messages = [SystemMessage(content= sys_msg + summary)] + message
    else:
        messages = [SystemMessage(content= sys_msg)] + message
    
    response = llm_with_tools.invoke(messages)
    # # Store the interaction in Mem0
    mem0.add(f"User: {last_content}\nAssistant: {response.content}", user_id=mem0_user_id)
    return {"messages": response}  

def preserve_msg(state: State):
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


# Determine whether to end or summarize the conversation
def should_continue(state: State):
    
    """Return the next node to execute."""
    
    messages = state["messages"]
    
    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 20:
        return "summarize_conversation"
    
    # Otherwise we can just end
    return END


def summarize_conversation(state: State):
    
    # First get the summary if it exists
    summary = state.get("summary", "")

    # Create our summarization prompt 
    if summary:
        
        # If a summary already exists, add it to the prompt
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
        
    else:
        # If no summary exists, just create a new one
        summary_message = "Create a summary of the conversation above:"

    # Add prompt to our history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    
    response = llm.invoke(messages)
    
    # Delete all but the 2 most recent messages and add our summary to the state 
    #delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": state["messages"]}

# Define a sub graph
workflow = StateGraph(State, config_schema=configuration.Configuration)
workflow.add_node("preserve", preserve_msg)
workflow.add_node("assistant", assistant)
workflow.add_node("tools", ToolNode(tools))
workflow.add_edge(START, "preserve")
workflow.add_edge("preserve", "assistant")
workflow.add_conditional_edges("assistant", tools_condition)
workflow.add_edge("tools", "assistant")
#workflow.add_edge("preserve", "assistant")
#memory = MemorySaver()
graph3 = workflow.compile()

# Define the main conversation flow
conv_flow = StateGraph(State, config_schema=configuration.Configuration)
conv_flow.add_node("react", graph3)
conv_flow.add_node(summarize_conversation)

# Set the entrypoint as conversation
conv_flow.add_edge(START, "react")
conv_flow.add_conditional_edges("react", should_continue)
conv_flow.add_edge("summarize_conversation", END)
#memory1 = MemorySaver()

# Compile
graph = conv_flow.compile()
#display(Image(graph2.get_graph(xray=1).draw_mermaid_png()))



