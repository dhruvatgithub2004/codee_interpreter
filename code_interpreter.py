import streamlit as st
import os
from crewai import Agent, Task, Crew
# Importing crewAI tools
from crewai_tools import CodeInterpreterTool
from dotenv import load_dotenv
text_holder=st.text_area("Enter your code",value=None)
# text_holder
tools=CodeInterpreterTool()
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv['GROQ_API_KEY']

interpreter=Agent(
    role = "Code Interpreter",
    goal= "To create a seamless platform for interpreting, debugging, and learning code, empowering developers to innovate effortlessly.",
    backstory="The Code Interpreter Agent was born to bridge the gap between coding beginners and professionals by simplifying code execution, debugging, and learning. It provides a seamless, secure environment for writing and running code, offering real-time insights and suggestions. Its mission is to empower creativity and collaboration in programming for all skill levels.",
    tools=[tools],
    llm="groq/llama3-8b-8192",
    verbose=True
)
suggestor= Task(
    description='''
    Identify syntax errors, runtime errors, and logical flaws in the {code}.
    Analyze performance, identify bottlenecks, and suggest efficiency improvementsin the {code}.
    Act as an educational tool, providing step-by-step execution explanations.
    Validate {code} by checking syntax, conduct static analysis, and ensure adherence to language rules.
    Support {code} refactoring for better readability and maintainability.
    Manage library dependencies, explain functionality, and integrate libraries seamlessly of the {code}.
    Provide detailed error explanations, annotate {code} with comments, and generates visualizations like charts or flow diagrams.
        ''',
    expected_output='''
    1.If code is not correct Give the corrected code with optimized time complexity
    2.If code is correct but not optimized then give the most optimized code
    2.Clear, detailed descriptions of errors in the original code, including syntax errors, runtime errors, or logical issues, alongside possible solutions or debugging tips.
    3.Information about execution time, memory usage, or computational efficiency.
    4.Step-by-step explanations or annotations of how the code operates, particularly for educational or debugging contexts.
    5.Warnings about potentially unsafe or malicious code, including harmful system calls if neceessary.
    ''',
    agent=interpreter
)
crew= Crew(
    Agents=[interpreter],
    tasks=[suggestor],
    verbose=True
)
if st.button("interpret"):
    results=crew.kickoff(inputs={'code':text_holder})
    st.write(results.raw if hasattr(results, 'raw') else str(results))
