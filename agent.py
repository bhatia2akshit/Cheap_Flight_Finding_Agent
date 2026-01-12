import os
from typing import Dict
from langchain.agents import create_agent
from pdfreader import compare_cv_data, extract_pdf_pages, optimize_cv, transform_job_description, write_cover_letter, write_new_cv
# from litellm import completion\
from langchain.chat_models import init_chat_model
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

login(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=r"Pydantic serializer warnings:")


# model = init_chat_model(
#     "mistral-medium-latest", 
#     model_provider="mistralai",
#     temperature=0.4)



cv_path = "CV_Aksh.pdf"

description = """
Location: Remote (CET to CET+3) 

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

# agent = create_agent(
#     model, 
#     tools = tools,
#     system_prompt = f"""You are a recruiter at a large staffing agency. 
#     From the given tools {tools}, please write a new CV and a cover letter which are aligned with the given job description and the original CV.
    
#     """
#     )

# agent.invoke({"messages": [{"role": "user", "content": f"Path to the CV is: {cv_path} and the job description is: {description}"}],
#             #   "user_preferences": {"style": "technical", "verbosity": "detailed"},
#               },
#              )

tools = [extract_pdf_pages, compare_cv_data, write_cover_letter, optimize_cv, write_new_cv]

async def call_agent(model_str: str, cv_content: str, job_desc: str):
    model_obj = model_str.split('/')
    model_provider = model_obj[0]
    model_name = model_obj[1]

    model = init_chat_model(
    model_name,
    model_provider=model_provider,
    temperature=0.7)

    print('model initialized.')
    
    agent = create_agent(
        model,
        tools=tools,
        system_prompt=f"""You are a recruiter at a large staffing agency.
        From the given tools {tools}, please write a new CV and a cover letter which are aligned with the given job description and the original CV.
        """
    )

    agent.invoke({"messages": [{"role": "user", "content": f"Original CV Content: {cv_content} and the job description is: {job_desc}. Please write a new CV and a cover letter as mentioned in the instructions above."}],
                   #   "user_preferences": {"style": "technical", "verbosity": "detailed"},
                   },
                  )
    
    print('everything is done.')

def convert_latex_to_pdf(latex_content: str) -> bytes:
    # Convert LaTeX content to PDF
    pdf_bytes = b"%PDF-1.4..."
    return pdf_bytes


def call_manager(cv_content: str, job_desc: str) -> Dict:
    # convert cv_content and job_desc into json objects

    cv_json = """{
  "full_name": "Akshit Bhatia",
  "email": "bhatia2akshit@gmail.com",
  "phone": "+49 1516 8555138",
  "location": "Bochum, Germany",
  "links": [
    "www.linkedin.com/in/being-akshit-bhatia"
  ],
  "summary": "AI/ML Engineer and Python Backend Developer with 3+ years of experience architecting real-time systems, with integrated AI solutions. Expert in creating production-ready web applications, and deployment on Azure and AWS clouds. Proven ability to modernize complex legacy codebases, implement robust CI/CD workflows, and deliver production-grade systems under strict performance constraints.",

  "skills": {
    "languages": ["Python", "Java", "JavaScript", "TypeScript", "SQL"],
    "tools_frameworks": [
      "PyTorch", "OpenCV", "Hugging Face Transformers", "LangChain", "LangGraph", "SmolAgent", "CrewAI", "PydanticAI",
      "FastAPI", "Flask", "Django", "Streamlit", "React", "HTML", "CSS", "Docker", "Kubernetes", "Jenkins", "Terraform",
      "Airflow", "Redis", "PostgreSQL", "Weaviate", "Git", "CI/CD", "Azure", "AWS", "Confluence", "JIRA"
    ],
    "expertise": [
      "AI/ML System Design", "Real-Time AI Applications", "Agentic AI & LLMs", "Retrieval-Augmented Generation (RAG)",
      "Computer Vision & Image Generation", "Model Fine-Tuning (LoRA)", "ML Pipelines & MLOps", "Cloud-Native Deployment",
      "Multi-Tenant Architectures", "API Design", "DevOps & Automation", "Production Monitoring & Reliability"
    ]
  },

  "experience": [
    {
      "company": "Kumo Clouds Solutions",
      "title": "AI/ML Engineer",
      "start_date": "2025-06-01",
      "end_date": "2025-12-31",
      "location": "Cologne, Germany",
      "job_details": [
        "Designed and deployed a multi-tenant, real-time interview bot using Python, LLMs, Django, PostgreSQL, and Redis, enabling context-aware conversational AI with low-latency responses.",
        "Implemented a retrieval-augmented generation (RAG) pipeline using embedding-based search and Weaviate, extracting structured knowledge from 1,000+ pages of technical documentation to improve information accessibility.",
        "Built end-to-end CI/CD pipelines using Jenkins, Terraform, Docker, and Azure, enabling automated builds, infrastructure provisioning, and zero-downtime deployments."
      ]
    },
    {
      "company": "Auto1",
      "title": "Python Developer",
      "start_date": "2024-04-01",
      "end_date": "2024-10-31",
      "location": "Berlin, Germany (Remote)",
      "job_details": [
        "Built an AI-driven news intelligence agent using Python, LangChain, FastAPI, and LLMs, continuously monitoring industry and competitor news and generating sales-oriented summaries via contextual retrieval.",
        "Integrated web search tools as agent actions and stored hundreds of articles in a vector database, enabling semantic search and grounded LLM responses.",
        "Automated Google Ads campaign management using Python APIs, reducing manual operational effort.",
        "Implemented CI/CD workflows using Jenkins and Airflow, ensuring reliable deployments, scheduling, and production monitoring with Sentry."
      ]
    },
    {
      "company": "PropertyExpert",
      "title": "ML Engineer - Internship",
      "start_date": "2024-07-01",
      "end_date": "2024-10-31",
      "location": "Langenfeld, Germany (Remote)",
      "job_details": [
        "Developed an AI image generation pipeline using Vision Transformers, GPT-2 captioning, and diffusion models, generating context-consistent synthetic images for damaged infrastructure scenarios.",
        "Trained a ResNet-50-based fake image detection model using PyTorch, classifying real vs. synthetic artifacts to support content-agnostic quality assurance pipelines."
      ]
    },
    {
      "company": "Blue Avenir",
      "title": "Python Developer - Internship",
      "start_date": "2022-07-01",
      "end_date": "2023-02-28",
      "location": "Dusseldorf, Germany",
      "job_details": [
        "Improved predictive analytics pipelines using Python, Pandas, XGBoost, and feature optimization, increasing model accuracy by 6% on structured business datasets.",
        "Built a semantic document search pipeline using LLMs and vector embeddings, enabling efficient information retrieval from unstructured documents."
      ]
    },
    {
      "company": "WHK (Computational Social Science)",
      "title": "Research Assistant",
      "start_date": "2021-04-01",
      "end_date": "2022-05-31",
      "location": "Paderborn, Germany",
      "job_details": [
        "Trained and fine-tuned transformer-based language models (BART, GPT-2, RoBERTa) using Hugging Face Transformers to generate contextually relevant and persuasive counter-arguments.",
        "Developed Flask-based inference APIs in Python, enabling controlled experimentation and evaluation of generated text outputs."
      ]
    },
    {
      "company": "Sopra Steria",
      "title": "Software Developer",
      "start_date": "2015-10-01",
      "end_date": "2019-10-31",
      "location": "Delhi, India",
      "job_details": [
        "Developed and deployed production-grade enterprise applications using Java, EJB, Hibernate, and SQL, ensuring scalability, reliability, and maintainability in large systems.",
        "Built a face recognition system using OpenCV and scikit-learn for secure access control, and implemented clustered ticket grouping logic to streamline assignment workflows.",
        "Collaborated in agile, distributed teams, contributing to CI pipelines, system upgrades, and production releases using Jenkins and version control tools."
      ]
    }
  ],

  "education": [
    {
      "school": "University of Paderborn, Germany",
      "degree": "Masters: Informatiks",
      "field": "Computer Science",
      "start_date": "2019-10-01",
      "end_date": "2024-08-31",
      "thesis": {
        "title": "Train Neural Networks to detect Images generated with AI",
        "description": "Developed and deployed two end-to-end computer vision pipelines in Python. The work covered dataset preparation, model training and evaluation, hyperparameter tuning, and performance analysis, with a strong focus on generalization, robustness, and real-world applicability of image authenticity detection."
      }
    },
    {
      "school": "Indraprastha University, Delhi",
      "degree": "Bachelor of Technology",
      "field": "Computer Science",
      "start_date": "2011-08-01",
      "end_date": "2015-05-31",
      "thesis": {
        "title": "InfoRank: an algorithm to rank Reviews",
        "description": "Designed and implemented an entropy-based algorithm to rank large-scale user reviews by informational value, applying statistical methods and feature engineering to text data. The project involved data preprocessing, relevance scoring, and evaluation against baseline ranking approaches."
      }
    }
  ],

  "projects": [
    {
      "name": "REACT Based Agentic AI System",
      "date": "2025-12-01",
      "description": "Designed and implemented an Agentic AI application using LLMs and tool orchestration, enabling safe tool execution, state tracking, and decision-making across web search, mathematical reasoning, and governance writing, with a lightweight API consumable by automation frameworks such as Trigger.dev."
    },
    {
      "name": "LLM Jokes Better",
      "date": "2024-10-01",
      "description": "Fine-tuned Mistral 7B using LoRA and prompt engineering to generate contextually relevant, high-quality humor, demonstrating parameter-efficient LLM fine-tuning, controlled text generation, and applied generative AI system design."
    }
  ],

  "certifications": [
    {
      "name": "IBM Certificate: Develop Generative AI Applications: Get Started",
      "date": "2025-12-01"
    },
    {
      "name": "IBM Certificate: Build RAG Applications: Get Started",
      "date": "2025-12-01"
    },
    {
      "name": "AWS: Generative AI with Large Language Models",
      "date": "2024-10-01"
    }
  ],

  "languages": [
    {
      "language": "English",
      "proficiency": "Professional Proficiency"
    },
    {
      "language": "German",
      "proficiency": "Professional Proficiency"
    }
  ]
}
"""
    # cv_json =  extract_pdf_pages(cv_content)
    job_desc_json = transform_job_description(job_desc)

    # compare cv and jd
    comparison_result = compare_cv_data(content=f"CV: {cv_json}\n\nJob Description: {job_desc_json}")

    # optimize cv
    optimized_cv = optimize_cv(content=f"Gap Analysis: {comparison_result}")

    # write new cv
    new_cv = write_new_cv(content=optimized_cv)

    # write cover letter
    cover_letter = write_cover_letter(content=f"Job Description: {job_desc_json}\n\nNew CV: {new_cv}")

    return {
        "cv": new_cv,
        "cover_letter": cover_letter
    }