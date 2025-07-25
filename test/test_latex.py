from TikzAgent import compile_latex_to_pdf, Figure, convert_pdf_figure_to_png, convert_png_figure_to_pdf, save_figure_to_file
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import base64

latex_code = open("test/sample1.tex", "r").read()

status, message, fig = compile_latex_to_pdf(latex_code)
tmp = convert_pdf_figure_to_png(fig)
save_figure_to_file(tmp, "test.png")
tmp = convert_png_figure_to_pdf(tmp)
save_figure_to_file(tmp, "test2.pdf")

if status:
    print("LaTeX compilation successful")
    print("Number of bytes in pdf_content: ", len(fig.data))
    with open("test.pdf", "wb") as f:
        f.write(fig.data)
        
    llm = ChatOpenAI(model="gpt-4o-mini")

# Invoke the model with a Human message
    response = llm.invoke([SystemMessage(content="You are a helpful assistant that can analyze images and provide a description of the image."), 
                           HumanMessage(content=[
                               {"type": "text", "text": "Analyze the image and provide a description of the image."},
                               {"type": "file", "file": {"filename": "fig.pdf", "file_data": f"data:application/pdf;base64,{base64.b64encode(fig.data).decode('utf-8')}"}}
                               ])])
    print(f"LLM Response: {response.content}")
else:
    print("LaTeX compilation failed")