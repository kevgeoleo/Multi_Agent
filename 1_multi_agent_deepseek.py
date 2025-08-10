import requests
import os

# i have set the key in environment using setx OPENROUTER_API_KEY "key"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

class Agent:
    def __init__(self, name, role_prompt):
        self.name = name
        self.role_prompt = role_prompt

    def respond(self, conversation):

        # Prepend the agent's own system prompt
        # content field makes sure agent knows who he is - research assistant, planner, crictic, coder .etc. 
        messages = [{"role": "system", "content": self.role_prompt}] + conversation

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}"
        }

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
            print(f"[{self.name} error] {e}")
            return "[Error]"

    '''def respond(self, conversation):

        messages = [{"role": "system", "content": self.role_prompt}]
        for msg in conversation:
            messages.append({"role": "user", "content": f"{msg['role']}: {msg['content']}"})

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}"
        }

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
            print(f"[{self.name} error] {e}")
            return "[Error]"'''

if __name__ == "__main__":

    #get user input
    user_prompt = input("Enter the task for the agents: ").strip()

    # Creating different agents as different objects of the class
    # Eg. Name: Researcher, Role prompt: You are a research assistant. Gather relevant info.  
    assistant = Agent("Assistant", "You are a research assistant. Gather relevant info.")
    planner    = Agent("Planner",    "You are a planner. Outline a detailed plan.")
    coder      = Agent("Coder",      "You are a coder. Write clean, functional Python code.")
    critic     = Agent("Critic",     "You are a critic. Analyze for bugs or improvements.")
    
    # starting the conversation with the user prompt
    conversation = [{"role": "user", "content": user_prompt}]

    # create a list of agent objects
    agents = [assistant, planner, coder, critic]

    # Two collaboration rounds where different agents collaborate with eachother to generate result
    for _ in range(5):  

        # Start with 1st agent (Assistant) and goes till Critic 
        for agent in agents:

            # Make each agent respond to the conversation (all previous responses are appended to conversation)
            reply = agent.respond(conversation)
            print(f"---\n{agent.name} says:\n{reply}\n")

            # Each time, previous responses are appended to the initial prompt and so on 
            conversation.append({"role": agent.name, "content": reply})