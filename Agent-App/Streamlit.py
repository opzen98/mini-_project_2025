import streamlit as st
import asyncio
import json
import os
import sys
import traceback
from datetime import datetime
from typing import Optional
import uuid

# Add current directory to path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Page configuration
st.set_page_config(
    page_title="MCP Agent Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "user_id" not in st.session_state:
    st.session_state.user_id = "Zen"
if "graph" not in st.session_state:
    st.session_state.graph = None
if "initialization_error" not in st.session_state:
    st.session_state.initialization_error = None

def check_dependencies():
    """Check if all required dependencies are available"""
    missing_deps = []
    
    try:
        import langchain_openai
    except ImportError:
        missing_deps.append("langchain_openai")
    
    try:
        import langgraph
    except ImportError:
        missing_deps.append("langgraph")
    
    try:
        import langchain_mcp_adapters
    except ImportError:
        missing_deps.append("langchain_mcp_adapters")
    
    try:
        import mem0
    except ImportError:
        missing_deps.append("mem0")
    
    try:
        from dotenv import load_dotenv
    except ImportError:
        missing_deps.append("python-dotenv")
    
    return missing_deps

def check_files():
    """Check if required files exist"""
    missing_files = []
    
    if not os.path.exists('.env'):
        missing_files.append('.env')
    
    if not os.path.exists('config.json'):
        missing_files.append('config.json')
    
    if not os.path.exists('configuration.py'):
        missing_files.append('configuration.py')
    
    return missing_files

def check_environment():
    """Check if required environment variables are set"""
    from dotenv import load_dotenv
    load_dotenv()
    
    missing_env = []
    
    if not os.getenv('OPENAI_API_KEY'):
        missing_env.append('OPENAI_API_KEY')
    
    if not os.getenv('DATABASE_URL'):
        missing_env.append('DATABASE_URL')
    
    return missing_env

@st.cache_resource
def initialize_agent():
    """Initialize the agent and return the compiled graph"""
    try:
        # Check dependencies first
        missing_deps = check_dependencies()
        if missing_deps:
            error_msg = f"Missing dependencies: {', '.join(missing_deps)}"
            st.error(error_msg)
            return None, error_msg
        
        # Check files
        missing_files = check_files()
        if missing_files:
            error_msg = f"Missing files: {', '.join(missing_files)}"
            st.error(error_msg)
            return None, error_msg
        
        # Import after dependency check
        from langchain_openai import ChatOpenAI
        from langgraph.graph import MessagesState, StateGraph, START, END
        from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
        from langgraph.prebuilt import tools_condition, ToolNode
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain_core.runnables import RunnableConfig
        from langchain.chat_models import init_chat_model
        from mem0 import Memory
        from configuration import Configuration
        from dotenv import load_dotenv
        from langgraph.checkpoint.memory import MemorySaver
        
        # Load environment variables
        load_dotenv()
        
        # Check environment variables
        missing_env = check_environment()
        if missing_env:
            error_msg = f"Missing environment variables: {', '.join(missing_env)}"
            st.error(error_msg)
            return None, error_msg
        
        # Initialize LLM
        llm = init_chat_model("openai:gpt-4o-mini", temperature=0.0)
        
        # Configure mem0
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
        
        # Load MCP configuration
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        client = MultiServerMCPClient(connections=config)
        
        # Get tools asynchronously
        async def get_tools():
            tools = await client.get_tools()
            return tools
        
        # Create event loop for async operations
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        tools = loop.run_until_complete(get_tools())
        llm_with_tools = llm.bind_tools(tools)
        
        # System message template
        system_message_template = """{asst_role}:\n. Your task is to solve user queries by reasoning step-by-step. The current datetime is: {now}
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
            tool_calls_log: list
        
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
            
            configurable = Configuration.from_runnable_config(config)
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
            
            # Search memories
            memories = mem0.search(last_content, user_id=mem0_user_id, limit=3)
            memories_str = "\n".join(f"- {entry['memory']}" for entry in memories.get("results", []))
            
            return {
                "initial_query": last_content,
                "memories_retrieved": True,
                "messages": []
            }
        
        def assistant(state: State, config: RunnableConfig):
            """Main assistant function with memory context"""
            configurable = Configuration.from_runnable_config(config)
            mem0_user_id = configurable.user_id
            assistant_role = configurable.asst_role
            
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            summary = state.get("summary", "")
            messages = state["messages"]
            
            # Get memories for this query
            memories = mem0.search(state.get("initial_query", ""), user_id=mem0_user_id, limit=3)
            memories_str = "\n".join(f"- {entry['memory']}" for entry in memories.get("results", []))
            
            sys_msg = system_message_template.format(now=now, asst_role=assistant_role, memories_str=memories_str)
            
            if summary:
                messages = [SystemMessage(content=sys_msg + summary)] + messages
            else:
                messages = [SystemMessage(content=sys_msg)] + messages
            
            response = llm_with_tools.invoke(messages)
            
            # Log tool calls if any
            tool_calls_log = state.get("tool_calls_log", [])
            if hasattr(response, 'tool_calls') and response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_calls_log.append({
                        "name": tool_call.get("name", "unknown"),
                        "args": tool_call.get("args", {}),
                        "timestamp": datetime.now().isoformat()
                    })
            
            return {
                "messages": response,
                "final_response": response.content,
                "tool_calls_log": tool_calls_log
            }
        
        def store_memory(state: State, config: RunnableConfig):
            """Store the final interaction in mem0"""
            configurable = Configuration.from_runnable_config(config)
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
        
        # Create the workflow
        workflow = StateGraph(State, config_schema=Configuration)
        
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
        
        # ReAct loop: assistant -> tools -> assistant
        workflow.add_conditional_edges("assistant", tools_condition)
        workflow.add_edge("tools", "assistant")
        
        # After assistant is done, decide next step
        workflow.add_conditional_edges("assistant", should_continue)
        
        # Handle memory storage and summarization
        workflow.add_edge("store_memory", END)
        workflow.add_edge("summarize", "store_memory")
        
        # Compile the graph
        memory1 = MemorySaver()
        graph = workflow.compile(checkpointer=memory1)
        
        return graph, None
        
    except Exception as e:
        error_msg = f"Failed to initialize agent: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        st.error(error_msg)
        return None, error_msg

def run_agent(graph, user_input, user_id, thread_id, assistant_role="helpful assistant"):
    """Run the agent with the given input and return the response"""
    try:
        # Import here to avoid issues if not available
        from langchain_core.runnables import RunnableConfig
        from langchain_core.messages import HumanMessage
        
        # Create configuration
        config = RunnableConfig(
            configurable={
                "user_id": user_id,
                "thread_id": thread_id,
                "asst_role": assistant_role
            }
        )
        
        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "summary": "",
            "initial_query": "",
            "final_response": "",
            "memories_retrieved": False,
            "tool_calls_log": []
        }
        
        # Run the graph
        result = graph.invoke(initial_state, config=config)
        
        return result, None
        
    except Exception as e:
        error_msg = f"Error running agent: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        st.error(error_msg)
        return None, error_msg

def main():
    st.title("🤖 MCP Agent Chat")
    st.markdown("Chat with your MCP-enabled AI agent")
    
    # Show system status
    with st.expander("🔍 System Status", expanded=False):
        st.markdown("### Dependency Check")
        missing_deps = check_dependencies()
        if missing_deps:
            st.error(f"❌ Missing dependencies: {', '.join(missing_deps)}")
            st.code(f"pip install {' '.join(missing_deps)}")
        else:
            st.success("✅ All dependencies available")
        
        st.markdown("### File Check")
        missing_files = check_files()
        if missing_files:
            st.error(f"❌ Missing files: {', '.join(missing_files)}")
        else:
            st.success("✅ All required files present")
        
        st.markdown("### Environment Variables")
        if os.path.exists('.env'):
            from dotenv import load_dotenv
            load_dotenv()
            missing_env = check_environment()
            if missing_env:
                st.error(f"❌ Missing environment variables: {', '.join(missing_env)}")
            else:
                st.success("✅ All environment variables set")
        else:
            st.warning("⚠️ .env file not found")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # User ID input
        user_id = st.text_input("User ID", value=st.session_state.user_id)
        st.session_state.user_id = user_id
        
        # Assistant role
        assistant_role = st.text_area(
            "Assistant Role", 
            value="You are an intelligent assistant equipped with a suite of tools, including code execution capabilities.",
            height=100
        )
        
        # Thread management
        if st.button("New Conversation"):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()
        
        st.markdown(f"**Thread ID:** `{st.session_state.thread_id[:8]}...`")
        
        # Clear chat
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        
        # Force reinitialize
        if st.button("Reinitialize Agent"):
            st.cache_resource.clear()
            st.session_state.graph = None
            st.session_state.initialization_error = None
            st.rerun()
    
    # Initialize agent
    if st.session_state.graph is None:
        with st.spinner("Initializing agent..."):
            graph, error = initialize_agent()
            st.session_state.graph = graph
            st.session_state.initialization_error = error
    
    # Show initialization error if any
    if st.session_state.initialization_error:
        st.error("Agent initialization failed:")
        st.code(st.session_state.initialization_error)
        st.stop()
    
    if st.session_state.graph is None:
        st.error("Failed to initialize agent. Please check the System Status above.")
        st.stop()
    
    # Success message
    st.success("✅ Agent initialized successfully!")
    
    # Chat interface
    st.header("Chat")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show tool calls if any
            if message["role"] == "assistant" and "tool_calls" in message:
                if message["tool_calls"]:
                    with st.expander("🔧 Tool Calls", expanded=False):
                        for tool_call in message["tool_calls"]:
                            st.json(tool_call)
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result, error = run_agent(
                    st.session_state.graph, 
                    prompt, 
                    st.session_state.user_id, 
                    st.session_state.thread_id,
                    assistant_role
                )
                
                if error:
                    st.error(f"Error: {error}")
                elif result:
                    # Extract response and tool calls
                    response = result.get("final_response", "No response generated")
                    tool_calls_log = result.get("tool_calls_log", [])
                    
                    # Display response
                    st.markdown(response)
                    
                    # Display tool calls if any
                    if tool_calls_log:
                        with st.expander("🔧 Tool Calls", expanded=False):
                            for tool_call in tool_calls_log:
                                st.json(tool_call)
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "tool_calls": tool_calls_log
                    })
                else:
                    st.error("Failed to generate response")
    
    # Show current state info
    with st.expander("📊 Session Info", expanded=False):
        st.json({
            "user_id": st.session_state.user_id,
            "thread_id": st.session_state.thread_id,
            "messages_count": len(st.session_state.messages),
            "graph_initialized": st.session_state.graph is not None,
            "initialization_error": st.session_state.initialization_error is not None
        })

if __name__ == "__main__":
    main()