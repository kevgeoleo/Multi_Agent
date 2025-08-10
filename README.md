## ✅ Prerequisites

- Python 3.8 or later
- Windows (commands provided are for Windows; minor adjustments may be needed for macOS/Linux)

---

## 🚀 Setup Instructions

### 🔑 Step 1: Get OpenRouter API Key

1. Go to [OpenRouter API Keys](https://openrouter.ai/settings/keys)
2. If you don't have an account, create one first.
3. Click **Create API Key**.
4. Copy and **save your key** securely — you can only view it once.

---

### ⚙️ Step 2: Set Environment Variable

In **VSCode terminal**, run:

```powershell
$env:OPENROUTER_API_KEY = "insert_your_openrouter_api_key_here"
```

---

### 📦 Step 3: Install Dependencies

Install required libraries using:

```bash
pip install -r requirements.txt
```

## 📄 Information for Users

### 1. `1_multi_agent_deepseek.py`

This program initializes 4 instances of the DeepSeek LLM as 4 different agents:

1. Assistant
2. Planner
3. Coder
4. Critic

When the user gives a task as a prompt, these 4 agents work collaboratively to solve it.
By default, **5 rounds of conversation** are set up where each agent gets the complete history of messages to work on before giving its response.

---

### 2. `2_multi_agent_with_summarizer.py`

This is the same as `1_multi_agent_deepseek.py`, except it has an **additional summarizer agent**.
The summarizer compiles the conversation between the first 4 agents as the 5th step, then passes the summary into the next round.
This helps ensure the LLM input word limit is not exceeded.

---

### 3. `3_Evaluate_multiple_models.py`

This script evaluates **4 models**:

1. DeepSeek V3 0324 (free)
2. OpenAI: gpt-oss-20b (free)
3. Z.AI: GLM 4.5 Air (free)
4. MoonshotAI: Kimi K2 (free)

They are tested on **3 hardcoded tasks** for:

- **Accuracy** — compared against expected outputs
- **Efficiency** — shorter responses are better
- **Consistency** — each model responds 3 times to the same prompt; results are compared for similarity
- **Robustness** — each model receives an original and slightly modified prompt; responses are compared

The hardcoded tasks and expected results were generated using OpenAI GPT-5.

**Future Work:**
Automate the step of generating tasks and expected results using an LLM.

```

```
