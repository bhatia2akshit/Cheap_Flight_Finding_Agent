from __future__ import annotations

import json
from typing import Any, Dict

from langchain_core.tools import tool
from helper import init_huggingface_llm, parse_jsonish
from prompt_service import PromptService
from dotenv import load_dotenv

load_dotenv('.env.local')



@tool
def extract_cv_information(content: str) -> Dict[str, Any]:
    """
    Extract structured information from CV text with dynamic schema.
    
    This tool analyzes the CV text and:
    1. Determines what information is present in the CV
    2. Creates an appropriate schema for that information
    3. Extracts the data into a structured dictionary
    
    Args:
        content: The raw text extracted from a CV/resume
        
    Returns:
        Dictionary with extracted information based on CV content
    """
    
    # Initialize the LLM for extraction
    extraction_llm = init_huggingface_llm()
    extraction_prompt = PromptService.get("cv_extraction")
    extraction_prompt = extraction_prompt.render_messages(content=content)
    # Prompt the LLM to analyze and extract information
    # extraction_prompt =
    
    try:
        # Call the LLM to extract information
        print("  → Calling LLM for extraction...")
        response = extraction_llm.invoke(extraction_prompt)
        
        # Extract JSON from response
        response_text = response if isinstance(response, str) else str(response)
        
        # Try to find JSON in the response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            json_str = response_text[start_idx:end_idx + 1]
            extracted_data = json.loads(json_str)
            print("  ✓ Successfully extracted structured data")
        else:
            # Fallback: return raw response wrapped in a dict
            extracted_data = {
                "raw_extraction": response_text,
                "note": "Could not parse as JSON, returning raw response"
            }
            print("  ⚠ Could not parse JSON, returning raw response")
        
        return extracted_data
        
    except json.JSONDecodeError as e:
        print(f"  ❌ JSON parsing failed: {e}")
        return {
            "error": "JSON parsing failed",
            "raw_response": response_text[:500],
            "message": "The LLM response could not be parsed as JSON"
        }
    except Exception as e:
        print(f"  ❌ Error during extraction: {e}")
        return {
            "error": str(e),
            "message": "An error occurred during extraction"
        }

# @tool
def transform_job_description(content: str) -> dict:
    """
    Transform job description into a json object based on the given prompt.

    Parameters:
    - content (str): The job description.

    Returns:
    - dict: JSON of the job description.
    """
    print('\n\n\n extract job description.')
    prompt_name: str = "job_description_to_dict"
    # print('the content for job description transformation is: ', content)
    prompt = PromptService().get(prompt_name)
    messages = prompt.render_messages(content=content)
    llm_output = litellm_chat(messages=messages)
    parsed = parse_jsonish(llm_output)
    flag=False
    if "llm_output" in parsed:
        flag=True
        parsed["content"] = content
    if flag:
        print(f'comparison done {parsed["llm_output"]}')
        return parsed['llm_output']
    else:
        print(f'comparison done {parsed}')
        return parsed


# @tool
def compare_cv_data(
    content: str,
) -> dict:
    """
    Compares the extracted resume content with a job description using a prompt. Returns a json object with comparison results.

    Parameters:
    - content (str): The content contains both the parsed CV and the provided job description. Separate them with necessary separators, so that LLM is aware of the separation.

    Returns:
    - dict: A json object of comparison results.

    """
    prompt_name: str = "compare_cv_to_job"
    print('\n\n\n compare description vs cv.')

    # print('the content for comparison is: ', content)

    prompt = PromptService().get(prompt_name)
    messages = prompt.render_messages(content=content)
    llm_output = litellm_chat(messages=messages)
    parsed = parse_jsonish(llm_output)
    flag=False
    if "llm_output" in parsed:
        flag=True
        parsed["content"] = content
    if flag:
        print(f'comparison done {parsed["llm_output"]}')
        return parsed['llm_output']
    else:
        print(f'comparison done {parsed}')
        return parsed


# @tool
def optimize_cv(
    content: str,
) -> dict:
    """
    Optimize the given CV based on the provided gap analysis.

    Parameters:
    - content (str): Contains JSON of the original CV and the JSON of gap analysis.

    Returns:
    - dict: JSON output of the optimized CV.
    """
    print('\n\n\n optimize cv.')
    # print(f'the inputs is: {content}')
    prompt_name: str = "optimize_cv"
    prompt = PromptService().get(prompt_name)
    messages = prompt.render_messages(content=content)
    llm_output = litellm_chat(messages=messages)
    parsed = parse_jsonish(llm_output)
    return parsed


# @tool
def write_new_cv(content: str) -> dict:
    """
    Generate the new CV from the optimized CV JSON.

    Parameters:
    - content (str): JSON output of the optimized CV.
    Returns:
    - dict: JSON object of the optimized json.
    """
    print('\n\n\n write new cv.')
    
    print(f'the contents are: {content}')
    prompt_name: str = "write_new_cv"
    prompt = PromptService().get(prompt_name)
    messages = prompt.render_messages(content=content)
    llm_output = litellm_chat(messages=messages)
    parsed = parse_jsonish(llm_output)
    print(f'CV written {parsed}')
    return parsed


# @tool
def write_cover_letter(
    content: str,
) -> dict:
    """
    Generates a cover letter.

    Parameters:
    - content (str): The content contains the newly generated CV and the provided job description. 

    Returns:
    - dict: Parsed JSON output from the LLM (or a wrapper with the raw output).
    """
    print('write cover letter.')
    # print(f'the content for cover letter transformation is: {content}')
    prompt_name: str = "cover_letter"
    prompt = PromptService().get(prompt_name)
    messages = prompt.render_messages(content=content)
    llm_output = litellm_chat(messages=messages)
    parsed = parse_jsonish(llm_output)

    print(f'cover letter done {parsed}')
    return parsed


if __name__ == "__main__":
    # resume_data = extract_pdf_pages("CV_Aksh.pdf")

    cv_parsed = {'full_name': 'Akshit Bhatia', 'email': 'bhatia2akshit@gmail.com', 'phone': '+49 1516 8555138', 'location': 'Bochum, Germany', 'links': ['www.linkedin.com/in/being-akshit-bhatia'], 'summary': 'AI/ML Engineer and Python Backend Developer with 3+ years of experience architecting real-time systems, with integrated AI solutions. Expert in creating production ready web applications, and deployment on Azure and AWS clouds. Proven ability to modernize complex legacy codebases, implement robust CI/CD workflows, and deliver production-grade systems under strict performance constraints.', 'skills': ['Python', 'Java', 'JavaScript', 'TypeScript', 'SQL', 'PyTorch', 'OpenCV', 'Hugging Face Transformers', 'LangChain', 'LangGraph', 'SmolAgent', 'CrewAI', 'PydanticAI', 'FastAPI', 'Flask', 'Django', 'Streamlit', 'React', 'HTML', 'CSS', 'Docker', 'Kubernetes', 'Jenkins', 'Terraform', 'Airflow', 'Redis', 'PostgreSQL', 'Weaviate', 'Git', 'CI/CD', 'Azure', 'AWS', 'Confluence', 'JIRA'], 'experience': [{'company': 'Kumo Clouds Solutions', 'title': 'AI/ML Engineer', 'start_date': 'June 2025', 'end_date': 'December 2025', 'location': 'Cologne, Germany', 'job details': ['Designed and deployed a multi-tenant, real-time interview bot using Python, LLMs, Django, PostgreSQL, and Redis, enabling context-aware conversational AI with low-latency responses.', 'Implemented a retrieval-augmented generation (RAG) pipeline using embedding-based search and Weaviate, extracting structured knowledge from 1,000+ pages of technical documentation to improve information accessibility.', 'Built end-to-end CI/CD pipelines using Jenkins, Terraform, Docker, and Azure, enabling automated builds, infrastructure provisioning, and zero-downtime deployments.']}, {'company': 'Auto1', 'title': 'Python Developer', 'start_date': 'April 2024', 'end_date': 'October 2024', 'location': 'Berlin, Germany (Remote)', 'job details': ['Built an AI-driven news intelligence agent using Python, LangChain, FastAPI, and LLMs, continuously monitoring industry and competitor news and generating sales-oriented summaries via contextual retrieval.', 'Integrated web search tools as agent actions and stored hundreds of articles in a vector database, enabling semantic search and grounded LLM responses.', 'Automated Google Ads campaign management using Python APIs, reducing manual operational effort.', 'Implemented CI/CD workflows using Jenkins and Airflow, ensuring reliable deployments, scheduling, and production monitoring with Sentry.']}, {'company': 'PropertyExpert', 'title': 'ML Engineer - Internship', 'start_date': 'July 2024', 'end_date': 'October 2024', 'location': 'Langenfeld, Germany (Remote)', 'job details': ['Developed an AI image generation pipeline using Vision Transformers, GPT-2 captioning, and diffusion models, generating context-consistent synthetic images for damaged infrastructure scenarios.', 'Trained a ResNet-50-based fake image detection model using PyTorch, classifying real vs. synthetic artifacts to support content-agnostic quality assurance pipelines.']}, {'company': 'Blue Avenir', 'title': 'Python Developer - Internship', 'start_date': 'July 2022', 'end_date': 'February 2023', 'location': 'Dusseldorf, Germany', 'job details': ['Improved predictive analytics pipelines using Python, Pandas, XGBoost, and feature optimization, increasing model accuracy by 6% on structured business datasets.', 'Built a semantic document search pipeline using LLMs and vector embeddings, enabling efficient information retrieval from unstructured documents.']}, {'company': 'WHK', 'title': 'Research Assistant - Computational Social Science', 'start_date': 'April 2021', 'end_date': 'May 2022', 'location': 'Paderborn, Germany', 'job details': ['Trained and fine-tuned transformer-based language models (BART, GPT-2, RoBERTa) using Hugging Face Transformers to generate contextually relevant and persuasive counter-arguments.', 'Developed Flask-based inference APIs in Python, enabling controlled experimentation and evaluation of generated text outputs.']}, {'company': 'Sopra Steria', 'title': 'Software Developer', 'start_date': 'October 2015', 'end_date': 'October 2019', 'location': 'Delhi, India', 'job details': ['Developed and deployed production-grade enterprise applications using Java, EJB, Hibernate, and SQL, ensuring scalability, reliability, and maintainability in large systems.', 'Built a face recognition system using OpenCV and scikit-learn for secure access control, and implemented clustered ticket grouping logic to streamline assignment workflows.', 'Collaborated in agile, distributed teams, contributing to CI pipelines, system upgrades, and production releases using Jenkins and version control tools.']}], 'education': [{'school': 'University of Paderborn, Germany', 'degree': 'Masters', 'field': 'Informatiks', 'start_date': 'October 2019', 'end_date': 'August 2024'}, {'school': 'Indraprastha University, Delhi', 'degree': 'Bachelor of Technology', 'field': 'Computer Science', 'start_date': 'August 2011', 'end_date': 'May 2015'}], 'certifications': ['IBM Certificate: Develop Generative AI Applications: Get Started', 'IBM Certificate: Build RAG Applications: Get Started', 'AWS: Generative AI with Large Language Models'], 'projects': [{'name': 'REACT Based Agentic AI System', 'description': 'Designed and implemented an Agentic AI application using LLMs and tool orchestration, enabling safe tool execution, state tracking, and decision-making across web search, mathematical reasoning, and governance writing, with a lightweight API consumable by automation frameworks such as Trigger.dev.', 'links': [], 'tech': []}, {'name': 'LLM Jokes Better', 'description': 'Fine-tuned Mistral 7B using LoRA and prompt engineering to generate contextually relevant, high-quality humor, demonstrating parameter-efficient LLM fine-tuning, controlled text generation, and applied generative AI system design.', 'links': [], 'tech': []}]}
    job_description = """
Job Description:
Type: Full-time 

Eligibility:  Independent and unrestricted work authorization in the EU  

Travel: 0–25% 

Language: English

About Dawnguard 

Are you passionate about secure cloud architecture and excited to shape the future of cybersecurity with AI?  

Dawnguard’s mission is to redefine cybersecurity with a platform that enables true shift-left security—from day zero to day 10,000. 

We embed security directly into system architecture, before a single line of code is written. Our AI-powered platform automates design validation and generates production-ready Infrastructure as Code (IaC) across AWS, Azure, and GCP. 

At Dawnguard, we believe security should be proactive, collaborative, and cloud-native. We’re rewriting the DNA of cybersecurity—driven by curiosity, integrity, and resilience. 

We start with real customer problems—no tech for tech’s sake. We speak with honesty, even when it’s hard. We think independently, challenge assumptions, and welcome bold ideas that push us forward. We break things to understand them, then build something better. And when we see a problem, we own it. 

If that sounds like you, let’s talk. 

The Role

As a Junior Software Engineer for AI/ML, you’ll be part of the core team building Dawnguard’s platform. You’ll focus on developing AI/ML components that power our cloud architecture engine. Your work will involve training, evaluating, and integrating models; building data pipelines; and supporting the AI reasoning and generation of workflows behind the product. This will all contribute to the security-by-design architecture that powers our product.

You’ll collaborate with founding engineers, product designers, and security experts to:

Build and refine features that transform user intent into cloud architecture recommendations.

Develop intuitive workflows that make complex infrastructure concepts accessible to both experts and non-experts.

Implement and test components of the AI/ML engines powering our platform

Contribute to documentation, testing, and continuous improvement of the platform.

Help improve our ML tooling, datasets, evaluation frameworks, and model reliability.

Responsibilities

Implement features supporting natural-language-driven cloud architecture generation.

Implementing prompt and context engineering techniques within agentic systems

Own data pipelines, evaluation scripts, and experimentation workflows for modeling cloud architecture patterns, constraints, and best practices.

Collaborate with senior AI engineers to integrate trained models into the architecture-generation engine, including orchestration, memory systems, and inference services.

Qualifications

0–2 years of experience in software engineering, machine learning, or related fields.

Proficiency in Python

Experience with cloud platforms (Azure, GCP, AWS) and containerization through Docker.

Understanding of machine learning concepts, LLM and Agentic-driven applications.

Bonus: Experience with vector databases, multi agentic workflows, and bringing features into production

Bonus: Experience with the following programming languages: Go, Node.js, C/C++ and Rust

What You’ll Get   

Competitive salary and equity package.  

Flexible working hours and remote setup.  

Unlimited PTO

Opportunity to shape a category-defining product from the ground up.  
    """

    json_comparison = {'match_score': 85, 'matched_skills': ['Python', 'Azure', 'AWS', 'Docker', 'LLM', 'Agentic-driven applications', 'vector databases', 'multi agentic workflows', 'bringing features into production'], 'missing_skills': ['GCP', 'Go', 'Node.js', 'C/C++', 'Rust'], 'strengths': ['Designed and deployed a multi-tenant, real-time interview bot using Python, LLMs, Django, PostgreSQL, and Redis, enabling context-aware conversational AI with low-latency responses.', 'Implemented a retrieval-augmented generation (RAG) pipeline using embedding-based search and Weaviate, extracting structured knowledge from 1,000+ pages of technical documentation to improve information accessibility.', 'Built an AI-driven news intelligence agent using Python, LangChain, FastAPI, and LLMs, continuously monitoring industry and competitor news and generating sales-oriented summaries via contextual retrieval.', 'Integrated web search tools as agent actions and stored hundreds of articles in a vector database, enabling semantic search and grounded LLM responses.', 'Developed an AI image generation pipeline using Vision Transformers, GPT-2 captioning, and diffusion models, generating context-consistent synthetic images for damaged infrastructure scenarios.'], 'concerns': ['No experience with GCP.', 'No experience with Go, Node.js, C/C++, or Rust.'], 'recommended_resume_edits': ["Add 'GCP' under 'Technologies of Interest'.", 'Highlight any experience or interest in Go, Node.js, C/C++, or Rust.'], 'role_fit_summary': 'The candidate meets most of the core requirements with strong experience in Python, Azure, AWS, Docker, and LLM/Agentic-driven applications. There are no critical gaps, but the candidate lacks experience with GCP and some bonus programming languages. Overall, the candidate is a strong fit for the role.'}

    cv_optimized = optimize_cv(f"json of orignal CV: {cv_parsed}\n\njson of gap analysis: {json_comparison}")
    # print(cv_optimized)
    # print(json.dumps(resume_data, indent=2, ensure_ascii=False))
