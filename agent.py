import os
from typing import Dict
from langchain.agents import create_agent
from helper import init_huggingface_llm
from tools import compare_cv_data, extract_pdf_pages, optimize_cv, transform_job_description, write_cover_letter, write_new_cv

from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

login(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=r"Pydantic serializer warnings:")




available_tools = [extract_pdf_pages]

def call_agent(content: str):
    """
    Create and return a LangChain agent for CV extraction.
    
    The agent uses HuggingFace LLM and has access to the CV extraction tool.
    """
    
    # Initialize the main LLM for the agent
    main_llm = init_huggingface_llm()
    
    # Create the agent with the extraction tool
    agent = create_agent(
        model=main_llm,
        tools=available_tools,
        system_prompt="""You are a helpful CV/resume parsing assistant. 
        
          When given CV text, you should:
          1. Use the extract_cv_information tool to analyze and extract structured data
          2. Return the extracted dictionary to the user
          3. Be helpful and explain what information was found

          Always use the tool to extract information rather than trying to parse it yourself."""
      )
    response = agent.invoke({
    "messages": [
        {
            "role": "user", 
            "content": f"Please extract structured information from this CV:\n\n{content}"
        }
      ]
    })
    
    print("✓ Agent created successfully")
    return response['messages'][1].content



def convert_latex_to_pdf(latex_content: str) -> bytes:
    # Convert LaTeX content to PDF
    pdf_bytes = b"%PDF-1.4..."
    return pdf_bytes