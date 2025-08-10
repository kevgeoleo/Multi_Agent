import os
import json
import difflib
from typing import List, Dict
import requests
import time 

log_file_path = "eval_output.md"
log_file = open(log_file_path, "w", encoding="utf-8")

def log_print(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=log_file)

#   CONFIGURATION
#openai.api_key = os.getenv("OPENROUTER_API_KEY")
#openai.api_base = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# MODEL = "deepseek/deepseek-chat-v3-0324:free"
MODEL_MAP = {
                "deepseek_v3": "deepseek/deepseek-chat-v3-0324:free",
                "qwen3": "qwen/qwen3-coder:free",
                "moonshot_kimi_K2": "moonshotai/kimi-k2:free",
                "Gemma_3n": "google/gemma-3n-e2b-it:free",
                "gpt_oss_20b": "openai/gpt-oss-20b:free",
                "zai": "z-ai/glm-4.5-air:free"
            }

#   AGENT CLASS
class Agent:
    def __init__(self, name: str, system_prompt: str):
        self.name = name
        self.system_prompt = system_prompt


    def respond(self, conversation: List[Dict[str, str]]) -> str:
        try:
            # Prepend agent's system prompt
            messages = [{"role": "system", "content": self.system_prompt}] + conversation

            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            }

            if self.name not in MODEL_MAP:
                raise ValueError(f"Unknown agent name: {self.name}")

            payload = {
                "model": MODEL_MAP[self.name],
                "messages": messages
            }

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
            print(f"[Error in {self.name}]: {e}")
            return "[Error]"

    
      

#   AGENTS
deepseek_v3 = Agent("deepseek_v3", "Explain the concept like you are teaching a 5 year old. Be crisp and use as less words as possible.")
zai = Agent("zai", "Explain the concept like you are teaching a 5 year old. Be crisp and use as less words as possible.")
moonshot_kimi_K2 = Agent("moonshot_kimi_K2", "Explain the concept like you are teaching a 5 year old. Be crisp and use as less words as possible.")
gpt_oss_20b = Agent("gpt_oss_20b", "Explain the concept like you are teaching a 5 year old. Be crisp and use as less words as possible.")

agents = [deepseek_v3, zai, moonshot_kimi_K2, gpt_oss_20b]


# MEASUREMENTS
def measure_accuracy(expected: str, actual: str) -> float:

    # compares 2 strings and return similarity 
    return difflib.SequenceMatcher(None, expected.strip(), actual.strip()).ratio()

def measure_efficiency(conversation: List[Dict]) -> int:

    # lesser the length of conversation the better 
    return len(conversation)

def measure_consistency(results: List[str]) -> float:
    if len(results) < 2:
        return 1.0          # task ran only once 
    base = results[0]       # result when task ran the 1st time 
    sims = [difflib.SequenceMatcher(None, base, r).ratio() for r in results[1:]]  # calculate how similar base is to all other strings and generates a list 
    return sum(sims) / len(sims)    # return average similarity score 

def measure_robustness(result1: str, result2: str) -> float:
    return difflib.SequenceMatcher(None, result1, result2).ratio()  # compare original task output and modified task output 



# Run the given AI agent on the task prompt 
def run_agent_on_task(prompt: str, agent) -> List[Dict]:
    time.sleep(30)
    conversation = [{"role": "user", "content": prompt}]
    reply = agent.respond(conversation)
    log_print(f"\n---\n{agent.name} says:\n{reply}\n") 
    conversation.append({"role": "assistant", "content": reply})
    return conversation


# MULTIPLE AGENTS ON MULTIPLE TASKS
def evaluate_tasks(test_tasks: List[Dict]):
    evaluation_results = []

    # iterate over tasks 
    for task in test_tasks:
        #print(f"\n=== Running Task {task['id']}: {task['prompt']} ===\n")
        log_print(f"\n=== Running Task {task['id']}: {task['prompt']} ===\n")  # was print()
        
        # Accuracy & Efficiency
        for agent in agents:

            # Accuracy 
            conversation = run_agent_on_task(task["prompt"],agent)   # reply of the agent and the task prompt 
            final_answer = conversation[-1]["content"]              # last message in the list - just the message doesnt specify the agent  
            expected_output = task.get("expected", "")          # what is the expected output of the task 
            acc = measure_accuracy(expected_output, final_answer) if expected_output else None
            eff = measure_efficiency(conversation)              # lesser the length the better

            # Consistency
            runs = [run_agent_on_task(task["prompt"],agent)[-1]["content"] for _ in range(3)]  # run same task 3 times 
            cons = measure_consistency(runs)                                              # check consistency of each result 

            # Robustness
            modified_prompt = task["prompt"] + " Please explain step-by-step."       # task prompt is modified
            result_original = run_agent_on_task(task["prompt"],agent)[-1]["content"]       # result of original task prompt 
            result_modified = run_agent_on_task(modified_prompt,agent)[-1]["content"]      # result of modified task prompt 
            rob = measure_robustness(result_original, result_modified)               # compare results of original and modified prompt 

            evaluation_results.append({
                "agent": agent.name,
                "task_id": task["id"],
                "accuracy": acc,
                "efficiency": eff,
                "consistency": cons,
                "robustness": rob
            })
    return evaluation_results


if __name__ == "__main__":

    # Example evaluation tasks - Expected field is what we expect and later on we can find accuracy by comparing 
    # Agent output with this expected output 
    test_tasks = [
        {"id": 1, "prompt": "Write a Python script to reverse a string. Just give me a basic code", "expected": "text = \"Hello, world!\"\nreversed_text = text[::-1]\nprint(reversed_text)"},
        {"id": 2, "prompt": "Find the largest prime less than 1000.  Just give me the number", "expected": "997"},
        {"id": 3, "prompt": "Summarize gradient descent in two sentences. Give me a basic explanation", "expected": "Gradient descent is like rolling a ball down a hill to find the lowest point — it keeps moving in the direction that makes things go down the fastest. In machine learning, it’s used to slowly adjust the model’s settings so the predictions get more and more accurate."},
    ]

        # run multi agent conversation on multiple tasks and perform EVALUATION using metrics
    results = evaluate_tasks(test_tasks)
    #print("\n=== Evaluation Results ===")
    log_print("\n=== Evaluation Results ===")
    log_print(json.dumps(results, indent=2))  # write results JSON to file
