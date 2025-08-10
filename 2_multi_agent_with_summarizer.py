import requests
import os
import time 

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Always log to output.txt
log_file_path = "output.md"
log_file = open(log_file_path, "w", encoding="utf-8")

def log_print(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=log_file)

class Agent:
    def __init__(self, name, role_prompt):
        self.name = name
        self.role_prompt = role_prompt

    def respond(self, conversation):
        messages = [{"role": "system", "content": self.role_prompt}]
        for msg in conversation:
            messages.append({"role": "user", "content": f"{msg['role']}: {msg['content']}"})

        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
        payload = {
            "model": "deepseek/deepseek-chat-v3-0324:free",
            "messages": messages
        }

        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            log_print(f"[{self.name} error] {e}")
            return "[Error]"

if __name__ == "__main__":
    user_prompt = input("Enter the task for the agents: ").strip()

    # Agents
    #summarizer = Agent("Summarizer", "You are a Summarizer. Summarize the relevant points from all agents' responses so far.")

    assistant = Agent("Assistant","You are a research assistant. Search for and explain accurate, relevant information. Focus on factual detail and background context. Never write code.")
    planner = Agent("Planner","You are a planner. Turn the user's request into a logical, step-by-step plan. Define objectives, order of tasks, and key considerations. Never write code.")
    coder = Agent("Coder","You are a coder. Write clean, functional Python code following best practices. Keep explanations minimal unless needed for clarity.")
    critic = Agent("Critic","You are a critic. Review the provided code for correctness, efficiency, readability, and potential issues. Suggest concrete improvements.")
    summarizer = Agent("Summarizer","You are a summarizer. Create a concise summary of the conversation so far, keeping all important technical details and decisions.")   

    
    conversation = [{"role": "user", "content": user_prompt}]
    agents = [assistant, planner, coder, critic, summarizer]

    for _ in range(3):  # number of collaboration rounds
        for agent in agents:
            time.sleep(30) 
            reply = agent.respond(conversation)
            log_print(f"---\n{agent.name} says:\n{reply}\n")

            if agent.name == "Summarizer":
                # Replace entire conversation with just the summary
                conversation = [{"role": "Summarizer", "content": reply}]
            else:
                conversation.append({"role": agent.name, "content": reply})

    log_print(f"\n[Output saved to {log_file_path}]")
    log_file.close()
