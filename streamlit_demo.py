"""
Streamlit Demo for TikZ Agent

A web interface for generating, compiling, and reviewing TikZ diagrams using AI.
"""

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import asyncio
import os
import requests
import json
import webbrowser
from urllib.parse import urlparse
from typing import Optional, Dict, Any
from io import BytesIO
from datetime import datetime
# Import TikZ Agent components
from TikzAgent.workflow import TikzAgentWorkflow
from TikzAgent.latex_tools import Figure, load_pdf_to_figure, convert_pdf_figure_to_png
from langgraph.checkpoint.memory import InMemorySaver
from PIL import Image

# Import language models
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None


# OpenRouter constants
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_BASE = "https://openrouter.ai"


# Utility functions
def get_url():
    """Get the current app URL."""
    try:
        # Try to get URL from Streamlit context
        from streamlit.web import cli as stcli
        return "http://localhost:8501"  # Default for local development
    except:
        return "http://localhost:8501"


def url_to_hostname(url):
    """Extract hostname from URL."""
    return urlparse(url).netloc or "localhost:8501"


def open_page(url):
    """Open a page in a new tab."""
    webbrowser.open_new_tab(url)


# OpenRouter functions
def get_available_models():
    """Get available models from the OpenRouter API."""
    try:
        response = requests.get(OPENROUTER_API_BASE + "/models")
        response.raise_for_status()
        models = json.loads(response.text)["data"]
        
        # Filter models that support image input, structured outputs, and tools
        return [
            model["id"] for model in models
            if "image" in model.get("architecture", {}).get("input_modalities", [])
            and "structured_outputs" in model.get("supported_parameters", [])
            and "tools" in model.get("supported_parameters", [])
        ]
    except requests.exceptions.RequestException as e:
        st.error(f"Error getting models from API: {e}")
        return []


def handle_model_selection(available_models, selected_model, default_model):
    """Handle the model selection process."""
    # Determine the index of the selected model
    if selected_model and selected_model in available_models:
        selected_index = available_models.index(selected_model)
    elif default_model in available_models:
        selected_index = available_models.index(default_model)
    else:
        selected_index = 0
    
    selected_model = st.selectbox(
        "Select a model", available_models, index=selected_index
    )
    return selected_model


def exchange_code_for_api_key(code: str):
    """Exchange authorization code for API key."""
    st.info(f"Exchanging code for API key...")
    try:
        response = requests.post(
            OPENROUTER_API_BASE + "/auth/keys",
            json={"code": code},
        )
        response.raise_for_status()
        # Clear query params
        st.query_params.clear()
        api_key = json.loads(response.text)["key"]
        st.session_state["api_key"] = api_key
        st.success("Successfully connected to OpenRouter!")
        st.rerun()
    except requests.exceptions.RequestException as e:
        st.error(f"Error exchanging code for API key: {e}")


def handle_openrouter_auth(default_model="gpt-4"):
    """Handle OpenRouter authentication and model selection."""
    # Check for authorization code in query params
    code = st.query_params.get("code", None)
    if code:
        exchange_code_for_api_key(code)
    
    # Get stored API key and model
    api_key = st.session_state.get("api_key")
    selected_model = st.query_params.get("model", None) or st.session_state.get("model", None)
    url = url_to_hostname(get_url())
    
    if not api_key:
        if st.button(
            "🔗 Connect OpenRouter",
            use_container_width=True,
            help="Click to authenticate with OpenRouter"
        ):
            auth_url = f"{OPENROUTER_BASE}/auth?callback_url=http://{url}"
            open_page(auth_url)
            st.info("Please complete authentication in the opened browser tab, then return here.")
    else:
        st.success("✅ Connected to OpenRouter")
        if st.button("🚪 Log out", use_container_width=True):
            if "api_key" in st.session_state:
                del st.session_state["api_key"]
            st.rerun()
    
    # Model selection
    available_models = get_available_models() if api_key else []
    if available_models:
        selected_model = handle_model_selection(available_models, selected_model, default_model)
        st.session_state["model"] = selected_model
        st.query_params["model"] = selected_model
    else:
        selected_model = default_model
    
    return api_key, selected_model


@st.dialog("Draw reference figure", width="large")
def edit_figure():
    st.session_state.reference_figure = None
    container_canvas = st.container()
    cols3 = st.columns(4)
    cols = st.columns(4)
    cols2 = st.columns([1,3])
    
    drawing_mode = cols[0].selectbox(
        "Drawing tool:",
        ("freedraw", "line", "rect", "circle", "transform", "polygon", "point"),
    )
    if drawing_mode == "point":
        point_display_radius = cols[1].slider("Point display radius: ", 1, 25, 3)
        stroke_width = 2
    else:
        stroke_width = cols[1].slider("Stroke width: ", 1, 25, 3)
        point_display_radius = 0
       
    stroke_color = cols[2].color_picker("Stroke color hex: ")
    bg_color = cols[3].color_picker("Background color hex: ", "#eee")
    bg_image = cols2[1].file_uploader("Background image:", type=["png", "jpg"])
    use_image = cols3[0].button("Use image")
    clear_canvas = cols3[1].button("Clear canvas")
    use_previous_as_background = cols3[2].button("Use previous")
    ratio = cols2[0].number_input("Ratio", min_value=0.5, max_value=2.0, value=1.5, step=0.1)
    
    if clear_canvas:
        st.session_state.reference_figure = None
        st.rerun()
        
    # Create a canvas component
    with container_canvas:

        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            background_image=Image.open(bg_image) if bg_image else None,
            update_streamlit=True,
            height=600/ratio,
            width=600,
            drawing_mode=drawing_mode,
            point_display_radius=point_display_radius if drawing_mode == "point" else 0,
            display_toolbar=True,
            key="full_app",
            initial_drawing=st.session_state.get("last_canvas", None),
        )
        
        if canvas_result.image_data is not None and use_image:
            st.session_state.last_canvas = canvas_result.json_data
            img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
            
            # Convert to PNG using BytesIO
            png_buffer = BytesIO()
            img.save(png_buffer, format='PNG')
            png_data = png_buffer.getvalue()
            
            # Create Figure object and store in session state
        
            st.session_state.reference_figure = Figure(
                type="png",
                data=png_data,
                latex_code=""
            )
            st.rerun()
                
         

def display_message(message: dict) -> None:
    """Parse the message and update the session state."""
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown("**User request:**\n\n" + message["content"])
            if message.get("reference_figure", None):
                st.markdown("**Following figure was added for reference:**")
                display_figure(message["reference_figure"])
    elif message["role"] == "finalizer":
        with st.chat_message("assistant"):
            st.markdown("**Response:**")
            st.markdown(message["review_result"])
            
            col_tmp1, col_tmp2 = st.columns(2)
            with col_tmp1:  
                with st.popover("Generated Tikz code", icon="️💻"):
                    st.markdown("```latex\n" + st.session_state.workflow.latex_code_body % message.get("tikz_code", "") + "\n```")
            with col_tmp2:
                with st.popover("️Generated figure", icon="🏙️"):
                    if message.get("compiled_figure", None):
                        display_figure(message["compiled_figure"])
    elif message["role"] == "reviewer":
        with st.expander("👁️ Reviewer"):
            st.markdown("**Review result:**")
            st.markdown(message["review_result"])
            st.markdown("**Verdict: " + ("needs revision" if message["needs_revision"] else "approved") + "**")
    elif message["role"] == "compiler":
        with st.expander("🔧 Compiler"):
            st.markdown("**Response:**")
            st.markdown(message["compilation_result"])
            if message.get("compiled_figure", None):
                st.markdown("**Compiled figure:**")
                display_figure(message["compiled_figure"])
    elif message["role"] == "generator":
        with st.expander("🎨 Generator"):
            st.markdown("**Iteration: " + str(message["iteration_count"]) + "**")
            st.markdown("**Scratchpad:**")
            st.markdown(message.get("scratch_pad", ""))
            st.markdown("**TikZ code:**")
            st.markdown("```latex\n" + st.session_state.workflow.latex_code_body % message.get("tikz_code", "") + "\n```")

def display_figure(figure: Figure) -> None:
    """Display the figure."""
    if figure.type == "pdf":
        st.image(BytesIO(convert_pdf_figure_to_png(figure).data), width=400)
    elif figure.type == "png":
        st.image(BytesIO(figure.data), width=400)
    else:
        st.error("Unknown figure type: " + figure.type)


def initialize_llm(provider: str, api_key: str, model_name: str) -> Optional[Any]:
    """Initialize the selected LLM provider."""
    try:
        if provider == "OpenRouter" and ChatOpenAI:
            return ChatOpenAI(
                model=model_name,
                temperature=0.7,
                openai_api_key=api_key,
                openai_api_base=OPENROUTER_API_BASE,
                default_headers={
                    "HTTP-Referer": get_url(),
                    "X-Title": "TikZ Agent Demo"
                }
            )
        elif provider == "OpenAI" and ChatOpenAI:
            return ChatOpenAI(
                model=model_name,
                temperature=0.7,
                openai_api_key=api_key
            )
        elif provider == "Anthropic" and ChatAnthropic:
            return ChatAnthropic(
                model=model_name,
                temperature=0.7,
                anthropic_api_key=api_key
            )
        elif provider == "Google" and ChatGoogleGenerativeAI:
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0.7,
                google_api_key=api_key
            )
        else:
            return None
    except Exception as e:
        st.error(f"Failed to initialize {provider} LLM: {str(e)}")
        return None


def main():
    
    # State preparation
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "code_history" not in st.session_state:
        st.session_state.code_history = []
    if "figure_history" not in st.session_state:
        st.session_state.figure_history = []
    if "reference_figure" not in st.session_state:
        st.session_state.reference_figure = None
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(datetime.now())
    if "checkpointer" not in st.session_state:
        st.session_state.checkpointer = InMemorySaver()
    if "workflow" not in st.session_state:
        st.session_state.workflow = None
    if "current_llm_settings" not in st.session_state:
        st.session_state.current_llm_settings = None
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(datetime.now())

    st.set_page_config(
        page_title="TikZ Agent Demo",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    

    
    st.title("🎨 TikZ Agent Demo")
    
    
    # Sidebar for configuration
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # LLM Provider Selection with OpenRouter as first option
        provider = st.selectbox(
            "Select LLM Provider",
            ["OpenRouter", "OpenAI", "Anthropic", "Google"],
            index=0
        )
        
        if provider == "OpenRouter":
            st.subheader("🔗 OpenRouter Authentication")
            
            # Handle OpenRouter authentication
            api_key, model_name = handle_openrouter_auth(default_model="openai/gpt-4")
            
        else:
            # Model selection based on provider
            if provider == "OpenAI":
                model_options = ["gpt-4.1", "o1", "o3", "gpt-4o"]
                default_model = "gpt-4.1"
            elif provider == "Anthropic":
                model_options = ["claude-sonnet-4-20250514", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
                default_model = "claude-3-sonnet-20240229"
            elif provider == "Google":
                model_options = ["gemini-pro", "gemini-1.5-pro", "gemini-2.5-flash"]
                default_model = "gemini-pro"
            
            model_name = st.selectbox(
                "Select Model",
                model_options,
                index=0 if default_model == model_options[0] else 
                      model_options.index(default_model) if default_model in model_options else 0
            )
            
            # API Key input for other providers
            api_key = st.text_input(
                f"{provider} API Key",
                type="password",
                help=f"Enter your {provider} API key"
            )
            if not api_key:
                api_key = os.getenv(f"{provider.upper()}_API_KEY") or os.getenv("OPENAI_API_KEY")
        
        # Max iterations
        st.session_state.max_iterations = st.slider(
            "Max Iterations",
            min_value=1,
            max_value=16,
            value=8,
            help="Maximum number of revision iterations"
        )
        
        # Create LLM settings tuple for comparison
        current_settings = (provider, model_name, api_key)
        
        # Create/update workflow only if LLM settings changed or workflow doesn't exist
        if (st.session_state.current_llm_settings != current_settings or 
            st.session_state.workflow is None) and api_key:
            
            # Initialize LLM
            llm = initialize_llm(provider, api_key, model_name)
            
            if llm:
                try:
                    # Create workflow with the new LLM
                    st.session_state.workflow = TikzAgentWorkflow(
                        llm=llm, 
                        checkpointer=st.session_state.checkpointer,
                        max_iterations=st.session_state.max_iterations
                    )
                    st.session_state.current_llm_settings = current_settings
                    st.success(f"✅ Workflow initialized with {provider} {model_name}")
                except Exception as e:
                    st.error(f"❌ Failed to initialize workflow: {str(e)}")
                    st.session_state.workflow = None
            else:
                st.error(f"❌ Failed to initialize {provider} LLM")
                st.session_state.workflow = None
        elif not api_key:
            if provider == "OpenRouter":
                st.info("🔗 Please connect to OpenRouter above to get started")
            else:
                st.warning("⚠️ Please provide an API key to initialize the workflow")
            st.session_state.workflow = None
        
        if st.button("Clear session", use_container_width=True):
            st.session_state.messages = []
            st.session_state.code_history = []
            st.session_state.figure_history = []
            st.session_state.reference_figure = None
            st.session_state.thread_id = str(datetime.now())

        st.markdown("""---""")
        
        container_state_debug = st.empty()
        with container_state_debug:
            with st.popover("Graph state", icon="🔍").container(height=400):
                if st.session_state.workflow:
                    st.write(st.session_state.workflow.get_state(st.session_state.thread_id))
                else:
                    st.write("No workflow initialized")
        
    # Main content area
   
    
    tab_chat, tab_reference, tab_generated = st.tabs(["Chat", "Reference figure", "Generated figures and code"])

    with tab_reference:
        if st.session_state.reference_figure:
            display_figure(st.session_state.reference_figure)
        else:
            st.markdown("No reference figure provided")

    with tab_generated:
        option_map = {
            0: ":material/chevron_left:",
            1: ":material/chevron_right:",
        }
        if len(st.session_state.figure_history) > 0 and st.session_state.get('figure_index', -1) >= 0:
 
            selection = st.segmented_control(
                "Figure navigation",
                options=option_map.keys(),
                format_func=lambda option: option_map[option],
                selection_mode="single",
            )
            if selection == 1:
                st.session_state.figure_index += 1
                if st.session_state.figure_index >= len(st.session_state.figure_history):
                    st.session_state.figure_index = len(st.session_state.figure_history) - 1
            elif selection == 0:
                st.session_state.figure_index -= 1
                if st.session_state.figure_index < 0:
                    st.session_state.figure_index = 0
            
            figure = st.session_state.figure_history[st.session_state.figure_index]
            cols_tmp = st.columns([1, 1])
            with cols_tmp[0]:
                st.write(f"**Figure ({st.session_state.get('figure_index', 0)+1}/{len(st.session_state.figure_history)}):**")
                display_figure(figure)
            with cols_tmp[1]:
                st.write(f"**Latex code:**")
                with st.container(border=True, height=600):
                    st.markdown("```latex\n" + st.session_state.workflow.latex_code_body % figure.latex_code + "\n```")
        else:
            st.markdown("No figures generated yet")
        
    with tab_chat:
        # Messages container with fixed height
        messages_container = st.container(height=600)
        with messages_container:
            # Chat window
            for message in st.session_state.messages:
                display_message(message)

        
        col_input, col_button = st.columns([4, 1])
        with col_input:
            prompt = st.chat_input("What is on your mind?")
        with col_button:
            if st.button("🛠️ Create figure" if not st.session_state.reference_figure else "🛠️ Edit figure", use_container_width=True):
                res = edit_figure()
                
        
        # Chat input after the messages container
        if prompt:
            # Display user message in chat message container
            user_message = {"role": "user", "content": prompt}
            if st.session_state.reference_figure:
                user_message["reference_figure"] = st.session_state.reference_figure
            st.session_state.messages.append(user_message)
            
            with messages_container:
                display_message(user_message)
            
            for msgs in st.session_state.workflow.stream_sync(prompt, st.session_state.reference_figure, thread_id=st.session_state.thread_id):
                for key in msgs.keys():
                    dct = msgs[key].copy()
                    dct["role"] = key
                    with messages_container:
                        display_message(dct)
                    st.session_state.messages.append(dct)
                    
                    with container_state_debug:
                        with st.popover("Graph state", icon="🔍").container(height=400):
                            if st.session_state.workflow:
                                st.write(st.session_state.workflow.get_state(st.session_state.thread_id))
                            else:
                                st.write("No workflow initialized")
            
            state = st.session_state.workflow.get_state(st.session_state.thread_id)
            st.session_state.code_history.append(st.session_state.workflow.latex_code_body % state.get("tikz_code", ""))
            st.session_state.figure_history = state["figure_history"]
            st.session_state.figure_index = len(st.session_state.figure_history) - 1
            
            

if __name__ == "__main__":
    main() 