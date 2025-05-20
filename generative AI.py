from langchain.chains import SequentialChain, LLMChain
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Custom tools
from tools.job_skill_scraper import get_required_skills_from_linkedin
from tools.course_finder import find_courses_from_coursera

# Initialize the language model and memory
llm = ChatOpenAI(temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define the tools the agent can use
tools = [
    Tool(
        name="LinkedInSkillScraper",
        func=get_required_skills_from_linkedin,
        description="Fetch required skills for a given job title from LinkedIn job posts."
    ),
    Tool(
        name="CourseraCourseFinder",
        func=find_courses_from_coursera,
        description="Find online Coursera courses for a list of skills."
    )
]

# Initialize the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    memory=memory,
    verbose=True
)

# Prompt templates for SequentialChain
roadmap_template = PromptTemplate(
    input_variables=["user_background", "skills_and_courses"],
    template="""
You are an expert career coach. Based on the user's background:
{user_background}

And the following skills and courses:
{skills_and_courses}

Create a 6-month personalized learning roadmap.
"""
)

roadmap_chain = LLMChain(llm=llm, prompt=roadmap_template, output_key="roadmap")

sequential_chain = SequentialChain(
    chains=[roadmap_chain],
    input_variables=["user_background", "skills_and_courses"],
    output_variables=["roadmap"],
    verbose=True
)

# Run the full pipeline
if __name__ == "__main__":
    user_goal = input("Describe your current role and the job you want to get: ")
    agent_response = agent.run(user_goal)
    print("\n🔁 Agent reasoning complete.")

    inputs = {
        "user_background": user_goal,
        "skills_and_courses": agent_response
    }
    roadmap_output = sequential_chain(inputs)

    print("\n📋 Personalized Learning Roadmap:\n")
    print(roadmap_output['roadmap'])

# tools/job_skill_scraper.py
import requests
from bs4 import BeautifulSoup

def get_required_skills_from_linkedin(job_title: str) -> str:
    # This is a simplified simulation. In production, you would use LinkedIn's official API or a service like SerpAPI.
    query = job_title.replace(" ", "+")
    url = f"https://www.linkedin.com/jobs/search/?keywords={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        # In production, you'd parse real skills here
        return "Python, Machine Learning, Data Science, TensorFlow, SQL"
    else:
        return "Unable to fetch skills from LinkedIn"

# tools/course_finder.py
import requests

def find_courses_from_coursera(skills: str) -> str:
    skills_list = skills.split(", ")
    results = []
    for skill in skills_list:
        query = skill.replace(" ", "%20")
        url = f"https://api.coursera.org/api/courses.v1?q=search&query={query}&limit=1"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data.get("elements"):
                course = data["elements"][0]
                title = course.get("name", "No Title")
                results.append(f"{skill}: {title} (Coursera)")
            else:
                results.append(f"{skill}: No course found")
        else:
            results.append(f"{skill}: Error fetching course")
    return "\n".join(results)

# chains/roadmap_chain.py (no longer needed separately but kept for legacy)
def generate_roadmap(user_background: str, agent_summary: str) -> str:
    return f"""
User Background: {user_background}

Skills & Learning Resources Identified:
{agent_summary}

