import os, getpass
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langgraph.prebuilt import tools_condition, ToolNode
#from IPython.display import Image, display
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.chat_models import init_chat_model
import asyncio
import json

llm = init_chat_model("openai:gpt-4o-mini", temperature=0.0)

with open('../../config.json', 'r') as f:
    config = json.load(f)

client = MultiServerMCPClient(connections=config)

async def gets_tools():
    tools = await client.get_tools()
    return tools
    #llm_with_tools = llm.bind_tools(tools)
    # rest of the code

tools = asyncio.run(gets_tools())

    # rest of the code
# tools = await client.get_tools()
llm_with_tools = llm.bind_tools(tools)


class State(MessagesState):
    summary: str

def assistant(state: State):
    sys_msg = "You are a helpful assistant."
    summary = state.get("summary", "")
    message = state["messages"]
    if summary:
        messages = [SystemMessage(content= sys_msg + summary)] + message
    else:
        messages = [SystemMessage(content= sys_msg)] + message
    
    response = llm_with_tools.invoke(messages)
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
workflow = StateGraph(State)
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
conv_flow = StateGraph(State)
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



