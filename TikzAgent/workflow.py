"""
LangGraph agentic AI workflow for TikZ generation, compilation, and review.
"""

import asyncio
import operator
import base64
from typing import Dict, Any, List, Optional, Literal, Annotated
from dataclasses import dataclass
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models import BaseLLM
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict
from TikzAgent.latex_tools import Figure, compile_latex_to_pdf







class TikzWorkflowState(TypedDict):
    """Typed state for the workflow."""
    messages: Annotated[List[BaseMessage], operator.add]
    figure_history: Annotated[List[Figure], operator.add]
    reference_figure: Optional[Figure]
    tikz_code: str
    scratch_pad: str
    compiled_figure: Optional[Figure]
    compilation_result: str
    review_result: str
    needs_revision: bool
    iteration_count: int
    max_iterations: int

class ReviewResult(BaseModel):
    quality: Literal["EXCELLENT", "GOOD", "ACCEPTABLE", "NEEDS_IMPROVEMENT"] = Field(description="The quality of the TikZ code and compilation results")
    decision: Literal["APPROVED", "NEEDS_REVISION"] = Field(description="Whether the TikZ code is approved or needs revision")
    feedback: str = Field(description="Feedback on the quality of the TikZ code and compilation results. Use markdown to format the feedback.")

class GenerationResult(BaseModel):
    tikz_code: str = Field(description="The TikZ code to be compiled")
    scratch_pad: str = Field(description="A scratch pad for the workflow to use")
    
class TikzAgentWorkflow:
    """LangGraph workflow for TikZ generation, compilation, and review."""
    
    latex_code_body = """\\documentclass{standalone}
\\usepackage{tikz}
\\usepackage{amsmath}
\\usepackage{amssymb}
\\usepackage{amsthm}
\\usepackage{amsfonts}
\\usepackage{amscd}
\\usepackage{amsthm}
\\usepackage{amssymb}
\\usepackage{amsthm}
\\usepackage{tikz-cd}
\\usetikzlibrary{arrows,shapes,positioning,shadows,trees,calc}

\\begin{document}
\\begin{tikzpicture}
%s
\\end{tikzpicture}
\\end{document}
"""
    
    def __init__(self, llm: BaseLLM, max_iterations: int = 16, checkpointer = InMemorySaver()):
        """
        Initialize the TikZ workflow.
        
        Args:
            llm: The language model to use for generation and review
            max_iterations: Maximum number of revision iterations
        """
        self.llm = llm
        self.max_iterations = max_iterations
        self.checkpointer = checkpointer
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create the graph
        workflow = StateGraph(TikzWorkflowState)
        
        # Add nodes
        workflow.add_node("generator", self._generator_node)
        workflow.add_node("compiler", self._compiler_node)
        workflow.add_node("reviewer", self._reviewer_node)
        workflow.add_node("finalizer", self._finalizer_node)
        
        # Add edges
        workflow.add_edge("generator", "compiler")
        workflow.add_edge("compiler", "reviewer")
        workflow.add_edge("finalizer", END)
        
        # Add conditional edge from reviewer
        workflow.add_conditional_edges(
            "reviewer",
            self._should_continue,
            {
                "continue": "generator",
                "end": "finalizer"
            }
        )
        
        # Set entry point
        workflow.set_entry_point("generator")
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    async def _generator_node(self, state: TikzWorkflowState) -> Dict[str, Any]:
        """Generate TikZ code based on the user request."""
        print(f"🎨 Generator Node - Iteration {state['iteration_count'] + 1}")
        
        user_request = state["messages"][-1].content if state["messages"] else ""
        reference_fig = state.get("reference_figure", None)
        
        # Create the generation prompt
        if state["iteration_count"] == 0:
            prompt = """
You are an expert TikZ code generator. Generate clean, well-commented TikZ code based on the user's request.
                
Guidelines:
- Use proper TikZ syntax and commands
- Include necessary packages in comments
- Make the code readable and well-structured
- Focus on creating visually appealing diagrams
- Return ONLY the TikZ code that goes inside \begin{tikzpicture} \end{tikzpicture}
- Do NOT include \documentclass, \begin{document}, or other LaTeX document structure
- The code will be wrapped in tikzpicture environment automatically

You must provide a structured response with:
- tikz_code: The TikZ code to be compiled (without document structure)
- scratch_pad: Any notes, reasoning, or thoughts about your approach"""

            # First iteration - use original request with multimodal content
            human_message_content = [
                {"type": "text", "text": f"Original request: {user_request}"},
            ]
            
            # Add reference figure if available
            if reference_fig:
                human_message_content.append({"type": "text", "text": "Reference figure is attached:"})
                if reference_fig.type == "png":
                    human_message_content.append({"type": "image_url", "image_url": {"url": f"data:image/{reference_fig.type};base64,{base64.b64encode(reference_fig.data).decode('utf-8')}"}})
                else:
                    human_message_content.append({"type": "file", "file": {"filename": "reference.pdf", "file_data": f"data:application/pdf;base64,{base64.b64encode(reference_fig.data).decode('utf-8')}"}})
            else:
                human_message_content.append({"type": "text", "text": "No reference figure was provided."})
            
            # Add last generated figure from figure history if available
            if state.get("figure_history") and len(state["figure_history"]) > 0:
                last_figure = state["figure_history"][-1]
                human_message_content.append({"type": "text", "text": "Last generated figure from history is attached:"})
                human_message_content.append({"type": "file", "file": {"filename": "last_generated.pdf", "file_data": f"data:application/pdf;base64,{base64.b64encode(last_figure.data).decode('utf-8')}"}})
            
            # Add conversation history
            if state.get("messages") and len(state["messages"]) > 1:
                conversation_history = "\n".join([f"{msg.__class__.__name__}: {msg.content}" for msg in state["messages"][:-1]])
                human_message_content.append({"type": "text", "text": f"Conversation history:\n{conversation_history}"})
            
            human_message_content.append({"type": "text", "text": "Please provide your structured response with the TikZ code and any scratch pad notes:"})
            
            generation_messages = [
                SystemMessage(content=prompt),
                HumanMessage(content=human_message_content)
            ]
        else:
            prompt = """
You are an expert TikZ code generator. Revise the existing TikZ code based on the compilation results and review feedback.
                
Guidelines:
- Fix any compilation errors mentioned
- Address the reviewer's concerns
- Improve the code quality and visual appeal
- Return ONLY the TikZ code that goes inside \begin{tikzpicture} \end{tikzpicture}
- Do NOT include \documentclass, \begin{document}, or other LaTeX document structure
- The code will be wrapped in tikzpicture environment automatically

You must provide a structured response with:
- tikz_code: The revised TikZ code to be compiled (without document structure)
- scratch_pad: Any notes, reasoning, or thoughts about your revisions"""

            # Subsequent iterations - use review feedback with multimodal content
            original_request = state["messages"][-1].content if state["messages"] else ""
            
            human_message_content = [
                {"type": "text", "text": f"Original request: {original_request}"},
                {"type": "text", "text": f"Current TikZ code: {state['tikz_code']}"},
                {"type": "text", "text": f"Compilation result: {state['compilation_result']}"},
                {"type": "text", "text": f"Review feedback: {state['review_result']}"},
            ]
            
            # Add reference figure if available
            if reference_fig:
                human_message_content.append({"type": "text", "text": "Reference figure is attached:"})
                if reference_fig.type == "png":
                    human_message_content.append({"type": "image_url", "image_url": {"url": f"data:image/{reference_fig.type};base64,{base64.b64encode(reference_fig.data).decode('utf-8')}"}})
                else:
                    human_message_content.append({"type": "file", "file": {"filename": "reference.pdf", "file_data": f"data:application/pdf;base64,{base64.b64encode(reference_fig.data).decode('utf-8')}"}})
            else:
                human_message_content.append({"type": "text", "text": "No reference figure was provided."})
            
            # Add conversation history
            if state.get("messages") and len(state["messages"]) > 1:
                conversation_history = "\n".join([f"{msg.__class__.__name__}: {msg.content}" for msg in state["messages"][:-1]])
                human_message_content.append({"type": "text", "text": f"Conversation history:\n{conversation_history}"})
            
            # Add previous scratchpad if available
            if state.get("scratch_pad"):
                human_message_content.append({"type": "text", "text": f"Previous scratchpad notes: {state['scratch_pad']}"})
            
            human_message_content.append({"type": "text", "text": "Please provide your structured response with the revised TikZ code and any scratch pad notes:"})
            
            generation_messages = [
                SystemMessage(content=prompt),
                HumanMessage(content=human_message_content)
            ]
            
            
        llm = self.llm.with_structured_output(GenerationResult)
        
        # Generate TikZ code
        response = await llm.ainvoke(generation_messages)
        tikz_code = response.tikz_code.strip()
        scratch_pad = response.scratch_pad.strip()
        
        # Update state
        new_state = {
            "tikz_code": tikz_code,
            "scratch_pad": scratch_pad,
            "iteration_count": state["iteration_count"] + 1
        }
        
        return new_state
    
    async def _compiler_node(self, state: TikzWorkflowState) -> Dict[str, Any]:
        """Compile and validate the TikZ code."""
        print("🔧 Compiler Node - Compiling TikZ code")
        
        tikz_code = state["tikz_code"]
        
        # Create a complete LaTeX document with the TikZ code
        latex_document = self.latex_code_body % tikz_code

        # Actually compile the LaTeX document
        success, message, figure = compile_latex_to_pdf(latex_document)
        
        if success and figure:
            # Compilation successful
            compilation_result = f"✅ Compilation successful: {message}"
            print(f"✅ TikZ code compiled successfully")
            
            # Update state with successful compilation
            new_state = {
                "compiled_figure": figure,
                "compilation_result": compilation_result,
            }
        else:
            # Compilation failed
            compilation_result = f"❌ Compilation failed:\n{message}"
            print(f"❌ TikZ code compilation failed: {message}")
            
            # Update state with compilation failure - return raw error without LLM analysis
            new_state = {
                "compiled_figure": None,
                "compilation_result": compilation_result,
            }
        
        return new_state
    
    async def _reviewer_node(self, state: TikzWorkflowState) -> Dict[str, Any]:
        """Review the TikZ code and compilation results."""
        print("👁️ Reviewer Node - Reviewing output quality")
        
        # Check if compilation was successful
        compilation_successful = state["compiled_figure"] is not None
        original_request = state["messages"][-1].content if state["messages"] else ""
        llm = self.llm.with_structured_output(ReviewResult)
        
        if compilation_successful:
            fig = state.get("compiled_figure", None)
            reference_fig = state.get("reference_figure", None)
            
            # Create review messages for successful compilation
            review_messages = [
                SystemMessage(content="""You are a TikZ quality reviewer. Your primary responsibility is to evaluate how well the generated TikZ diagram corresponds to the user's request and any reference image provided.

CORE EVALUATION CRITERIA:
1. **Request Correspondence**: Analyze how accurately the generated image fulfills the user's specific request
2. **Reference Image Matching**: If a reference image is provided, compare the generated image against it
3. **Orientative vs. Exact References**: Detect if the user's language indicates the reference is orientative (e.g., "something like...", "similar to...", "inspired by...") vs. exact replication
4. **Technical Quality**: Assess TikZ code quality and visual appeal

APPROVAL STANDARDS:
- Only APPROVE if there is a VERY GOOD match between the generated image and the user's request
- If a reference image is provided, the generated image must match BOTH the request AND the reference appropriately
- For orientative references, allow reasonable creative interpretation while maintaining core similarities
- For exact references, require high fidelity to the reference image

QUALITY GRADING:
- EXCELLENT: Perfect or near-perfect match to request (and reference if provided)
- GOOD: Strong match with minor acceptable differences
- ACCEPTABLE: Adequate match but with some notable gaps
- NEEDS_IMPROVEMENT: Poor match with significant issues

FEEDBACK REQUIREMENTS:
- If approved: Provide a clear conclusion summarizing why the image successfully matches the request
- If not approved: Detail specific mismatches between the generated image and the user's request/reference
- Always explain your reasoning for the quality grade
- For reference images, explicitly state whether they appear orientative or exact based on user language

You must provide a structured response with:
- quality: One of EXCELLENT, GOOD, ACCEPTABLE, or NEEDS_IMPROVEMENT
- decision: Either APPROVED or NEEDS_REVISION
- feedback: Detailed analysis of matches/mismatches and conclusion for the user. Use markdown to format the feedback."""),
                HumanMessage(content=[
                    {"type": "text", "text": f"Original request: {original_request}"},
                    {"type": "text", "text": f"Current TikZ code: {state['tikz_code']}"},
                    {"type": "text", "text": f"Iteration: {state['iteration_count']}/{state['max_iterations']}"},
                    {"type": "text", "text": "The compiled figure is attached."},
                    {"type": "file", "file": {"filename": "compiled.pdf", "file_data": f"data:application/pdf;base64,{base64.b64encode(fig.data).decode('utf-8')}"}},
                    {"type": "text", "text": "The reference figure is attached." if reference_fig else "No reference figure was provided by the user."},
                    *([{"type": "image_url", "image_url": {"url": f"data:image/{reference_fig.type};base64,{base64.b64encode(reference_fig.data).decode('utf-8')}"}} if reference_fig.type == "png" else {"type": "file", "file": {"filename": "reference.pdf", "file_data": f"data:application/pdf;base64,{base64.b64encode(reference_fig.data).decode('utf-8')}"}}] if reference_fig else []),
                    {"type": "text", "text": "Please provide your structured review:"}
                ])
            ]
        else:
            # Create review messages for failed compilation
            review_messages = [
                SystemMessage(content="""You are a TikZ quality reviewer and debugging expert. The TikZ code failed to compile and needs debugging analysis.

DEBUGGING RESPONSIBILITIES:
1. **TikZ Bug Analysis**: Analyze the compilation error to identify specific TikZ-related issues
2. **Common TikZ Errors**: Look for typical problems like:
   - Missing semicolons after TikZ commands
   - Incorrect coordinate syntax
   - Undefined node names or styles
   - Missing or incorrect library imports
   - Syntax errors in path specifications
   - Incorrect use of TikZ commands or options
   - Bracket/brace mismatches in TikZ code
   - Invalid mathematical expressions in coordinates
3. **Code Structure Issues**: Check for LaTeX/TikZ document structure problems
4. **Fixability Assessment**: Determine if the errors can be reasonably fixed

DECISION CRITERIA:
- Always use NEEDS_REVISION when compilation errors occur
- Never give up on compilation errors - always attempt to fix them
- Consider iteration count and remaining attempts (handled by workflow logic)
- Prioritize common, easily fixable TikZ syntax errors

FEEDBACK REQUIREMENTS:
- Identify specific TikZ bugs and syntax errors from the compilation output
- Explain what each error means in TikZ context
- Provide concrete suggestions for fixing identified TikZ issues
- Assess whether the errors indicate fundamental problems or simple syntax mistakes
- Give clear guidance on what needs to be changed in the TikZ code

You must provide a structured response with:
- quality: Should be NEEDS_IMPROVEMENT for compilation failures
- decision: Always NEEDS_REVISION for compilation errors
- feedback: Detailed TikZ bug analysis and specific fix recommendations"""),
                HumanMessage(content=[
                    {"type": "text", "text": f"Original request: {original_request}"},
                    {"type": "text", "text": f"Current TikZ code: {state['tikz_code']}"},
                    {"type": "text", "text": f"Compilation result: {state['compilation_result']}"},
                    {"type": "text", "text": "Please provide your structured review:"}
                ])
            ]
        
        # Review the code
        review_result = await llm.ainvoke(review_messages)
        
        # Determine if revision is needed based on the structured decision
        needs_revision = not compilation_successful or (compilation_successful and review_result.decision == "NEEDS_REVISION")
        
        # Update state - store the feedback as a string for compatibility
        new_state = {
            "review_result": review_result.feedback,
            #"review_result": f"**QUALITY: {review_result.quality}**\n\n**DECISION: {review_result.decision}**\n\n**FEEDBACK:**\n\n {review_result.feedback}",
            "needs_revision": needs_revision,
        }
        
        return new_state
    
    async def _finalizer_node(self, state: TikzWorkflowState) -> Dict[str, Any]:
        """Finalize the workflow by updating messages and figure history."""
        print("🏁 Finalizer Node - Preparing final results")
        
        # Create a new message list with the review result added
        new_messages = []
        if state.get("review_result"):
            new_messages.append(AIMessage(content=f"Review Result:\n{state['review_result']}"))
        
        # Create figure history update
        new_figure_history = []
        if state.get("compiled_figure"):
            new_figure_history.append(state["compiled_figure"])
        
        # Update state with final information
        new_state = {
            "messages": new_messages,
            "figure_history": new_figure_history,
            # Ensure tikz_code and compilation_result are passed forward
            "tikz_code": state.get("tikz_code", ""),
            "compilation_result": state.get("compilation_result", ""),
            "review_result": state.get("review_result", ""),
            "needs_revision": state.get("needs_revision", False),
            "iteration_count": state.get("iteration_count", 0),
            "compiled_figure": state.get("compiled_figure", None)
        }
        
        return new_state
    
    def _should_continue(self, state: TikzWorkflowState) -> Literal["continue", "end"]:
        """Determine whether to continue with revisions or end the workflow."""
        # Check if we've reached max iterations
        if state["iteration_count"] >= state["max_iterations"]:
            print(f"🛑 Maximum iterations ({state['max_iterations']}) reached. Ending workflow.")
            return "end"
        
        # Check if revision is needed
        if state.get("needs_revision", False):
            print("🔄 Revision needed. Continuing to generator.")
            return "continue"
        else:
            print("✅ Review approved. Ending workflow.")
            return "end"
    
    async def run(self, user_request: str, thread_id: str = "1") -> Dict[str, Any]:
        """
        Run the TikZ workflow with the given user request.
        
        Args:
            user_request: The user's request for TikZ code generation
            thread_id: The thread ID for conversation state tracking
            
        Returns:
            Final state of the workflow
        """
        print(f"🚀 Starting TikZ workflow for request: {user_request}")
        
        # Initialize state
        initial_state = {
            "messages": [HumanMessage(content=user_request)],
            "figure_history": [],
            "reference_figure": None,
            "tikz_code": "",
            "scratch_pad": "",
            "compiled_figure": None,
            "compilation_result": "",
            "review_result": "",
            "needs_revision": False,
            "iteration_count": 0,
            "max_iterations": self.max_iterations
        }
        
        # Run the workflow
        final_state = await self.graph.ainvoke(initial_state, config={"configurable": {"thread_id": thread_id}})
        
        print("🎉 Workflow completed!")
        return final_state
    
    def run_sync(self, user_request: str, thread_id: str = "1") -> Dict[str, Any]:
        """
        Synchronous version of run() for convenience.
        
        Args:
            user_request: The user's request for TikZ code generation
            thread_id: The thread ID for conversation state tracking
            
        Returns:
            Final state of the workflow
        """
        return asyncio.run(self.run(user_request, thread_id))
    
    def stream_sync(self, user_request: str, reference_figure: Optional[Figure] = None, thread_id: str = "1"):
        """
        Synchronous streaming version that yields the state after each node call.
        
        Args:
            user_request: The user's request for TikZ code generation
            reference_figure: The reference figure for the workflow
            thread_id: The thread ID for conversation state tracking
            
        Yields:
            The chunk (state) after each node call
        """
        async def _async_stream():
            print(f"🚀 Starting TikZ workflow for request: {user_request}")
            
            # Initialize state
            initial_state = {
                "messages": [HumanMessage(content=user_request)],
                "figure_history": [],
                "reference_figure": reference_figure,
                "tikz_code": "",
                "scratch_pad": "",
                "compiled_figure": None,
                "compilation_result": "",
                "review_result": "",
                "needs_revision": False,
                "iteration_count": 0,
                "max_iterations": self.max_iterations
            }
            
            # Stream through the workflow
            async for chunk in self.graph.astream(initial_state, config={"configurable": {"thread_id": thread_id}}):
                yield chunk
        
        # Run the async generator synchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            async_gen = _async_stream()
            while True:
                try:
                    result = loop.run_until_complete(async_gen.__anext__())
                    yield result
                except StopAsyncIteration:
                    break
        finally:
            loop.close()

    def get_state(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current state for a given thread_id.
        
        Args:
            thread_id: The thread ID to retrieve state for
            
        Returns:
            The current state for the thread, or None if no state exists
        """
        try:
            state = self.graph.get_state(config={"configurable": {"thread_id": thread_id}})
            return state.values if state else None
        except Exception as e:
            print(f"Error retrieving state for thread {thread_id}: {e}")
            return None


def create_tikz_workflow(llm: BaseLLM, max_iterations: int = 16) -> TikzAgentWorkflow:
    """
    Factory function to create a TikZ workflow.
    
    Args:
        llm: The language model to use
        max_iterations: Maximum number of revision iterations
        
    Returns:
        Configured TikZ workflow instance
    """
    return TikzAgentWorkflow(llm, max_iterations)


# Example usage function
async def example_usage():
    """Example of how to use the TikZ workflow."""
    from langchain_openai import ChatOpenAI
    
    # Initialize LLM (replace with your preferred model)
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
    # Create workflow
    workflow = create_tikz_workflow(llm, max_iterations=16)
    
    # Run workflow
    user_request = "Create a TikZ diagram showing a simple neural network with 3 input nodes, 2 hidden layers with 4 nodes each, and 1 output node"
    
    result = await workflow.run(user_request)
    
    print("\n" + "="*50)
    print("FINAL RESULT:")
    print("="*50)
    print(f"TikZ Code:\n{result['tikz_code']}")
    print(f"\nCompilation Result:\n{result['compilation_result']}")
    print(f"\nReview Result:\n{result['review_result']}")
    print(f"\nIterations: {result['iteration_count']}")

def example_streaming_usage():
    """Example of how to use the streaming TikZ workflow."""
    from langchain_openai import ChatOpenAI
    
    # Initialize LLM (replace with your preferred model)
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
    # Create workflow
    workflow = create_tikz_workflow(llm, max_iterations=16)
    
    # Stream workflow
    user_request = "Create a TikZ diagram showing a simple neural network with 3 input nodes, 2 hidden layers with 4 nodes each, and 1 output node"
    
    final_state = None
    
    # Stream through the workflow
    for item in workflow.stream_sync(user_request):
        print(">>>>>>>>>>\nSTREAMED:\n", item)
        final_state = item
    
    final_state = final_state["finalizer"]
    if final_state:
        print("\n" + "="*50)
        print("FINAL RESULT:")
        print("="*50)
        print(f"TikZ Code:\n{final_state['tikz_code']}")
        print(f"\nCompilation Result:\n{final_state['compilation_result']}")
        print(f"\nReview Result:\n{final_state['review_result']}")
        print(f"\nIterations: {final_state['iteration_count']}")

  

if __name__ == "__main__":
    example_streaming_usage() 