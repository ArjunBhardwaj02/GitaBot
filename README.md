# 🪷 GitaBot: Self-Reflective RAG & Fine-Tuned LLM

## Architecture Overview
GitaBot is a local, multi-agent AI mentor designed to provide spiritual guidance using the Bhagavad Gita. It utilizes a **Self-Reflective RAG (Retrieval-Augmented Generation)** pipeline orchestrated via LangGraph, and a custom **QLoRA fine-tuned Llama-3 8B** model for persona-driven generation.

## 🛠️ Tech Stack
* **Orchestration:** LangGraph (Stateful Multi-Agent Routing)
* **LLMs:** * *Grader/Router:* Groq API (`llama-3.1-8b-instant`) for strict semantic evaluation.
  * *Generator:* Local custom-trained model (`gita-model.gguf`) served via Ollama.
* **Vector Database:** ChromaDB (Persistent)
* **Embeddings:** Nomic-Embed-Text
* **Memory Management:** SQLite (Checkpointer for conversational state)
* **Frontend:** Streamlit

## 🧠 Engineering Challenges Overcome
1. **The Local Structured Output Failure:** Local quantized models struggled with strict Pydantic JSON enforcement (`binary_score: yes/no`) required by the LangGraph router. 
   * *Solution:* Implemented a Multi-Model Architecture. Offloaded logical routing and document grading to a fast, instruction-following API model, while preserving the local fine-tuned model exclusively for persona generation.
2. **Infinite Graph Routing Loops:** When the Rewriter node flagged a prompt as `OUT_OF_SCOPE`, the graph continuously routed the failure back to the retriever, causing an infinite loop.
   * *Solution:* Engineered Conditional Edges in LangGraph to bypass the Vector DB entirely upon detecting out-of-scope flags, routing directly to the generator for a graceful refusal.
3. **Vector Dilution via Conversational Noise:** User prompts containing greetings and context (e.g., *"Hi, my name is Aju, I'm tired..."*) degraded ChromaDB's cosine similarity scores.
   * *Solution:* Leveraged the Self-RAG loop. If the Grader rejected the initial noisy retrieval, the Query Rewriter node stripped conversational filler to generate a dense, search-optimized query.
4. **Catastrophic Forgetting (LoRA limitation):** Fine-tuning the base model heavily on 3,000 English-only persona examples resulted in the loss of native multilingual capabilities (Hindi/Hinglish).
   * *Solution:* Acknowledged as a trade-off of aggressive QLoRA parameter updates; mitigated via strict System Prompt constraints limiting the model to English output to prevent hallucinations.

## 🚀 Run Locally
1. Clone the repository.
2. Install requirements: `pip install -r requirement.txt`
3. Add your `.env` file with `GROQ_API_KEY`.
4. Load your local model into Ollama: `ollama create gitabot -f Modelfile`
5. Launch the UI: `streamlit run app.py`