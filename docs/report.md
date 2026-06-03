# Project 2 Report Draft

LLM system used: OpenAI GPT-5 Codex in the local coding workspace. I used it to read the project requirement, design the agent system, write code, draft the report, and produce presentation material. All generated content should be checked by the student before submission.

## Q1. Data Agent System

### Task 1: Data Selection Methods

The DataMind-12K dataset is a trajectory dataset for data-analysis agents. Its HuggingFace page shows 12,187 rows, with train and test splits, and fields such as `data_source`, `db_id`, `task_id`, `trajectory`, `prompt`, `ability`, `reward_model`, and `extra_info`. The dataset is therefore suitable for selecting complete agent traces instead of isolated instruction-answer pairs.

After reading the "Data Selection for Instruction-Tuning and Multitask Training" part of *A Survey on Data Selection for Language Models*, I consider the following three methods most effective for data science and mathematical modeling agents.

#### 1. Complexity-Based Filtering

Data science tasks are not equally useful for training. Very short examples may only teach shallow question answering, while overly long or noisy trajectories may waste context length. Complexity-based filtering keeps examples with enough analytical structure:

- multiple reasoning turns;
- SQL, Python, statistics, or modeling operations;
- numerical reasoning;
- validation or final artifact generation such as `result.csv`.

For a Data Agent, complexity is important because the model must learn decomposition: understand the business question, map it to a mathematical abstraction, choose features or metrics, write executable SQL/Python, and verify the answer. In the code, complexity is scored by token length, number of trajectory turns, analytic keywords, code blocks, and numeric density.

#### 2. Trajectory Deduplication and Diversity Balancing

Instruction tuning datasets often contain near-duplicate prompts or templated tasks. Duplicates can make the model overfit to repeated database schemas or question patterns. Deduplication is especially important in DataMind because many records can share the same `db_id` and similar SQL templates.

The implemented method uses:

- exact prompt fingerprinting;
- near-duplicate filtering with normalized prompt similarity;
- source and database balancing to avoid selecting all top-scoring examples from a few easy domains.

This improves coverage across database domains and encourages the fine-tuned agent to generalize.

#### 3. Reward/Quality-Based Selection

For agent trajectories, quality is more than fluent text. Good data should include correct final answers, executable plans, and successful completion signals. DataMind includes a `reward_model` field, so quality can be estimated from existing reward information when available. When explicit reward numbers are unavailable, useful proxies include ground-truth availability, answer length, final-result markers, and penalties for refusal/error patterns.

This is useful for mathematical modeling because a low-quality trace can teach the model to produce plausible but invalid calculations. Reward-based selection prioritizes examples with stronger correctness evidence.

### Task 2: Data Processing Code

The data processing code is in:

- `src/data_agent/selection.py`
- `scripts/prepare_datamind_qwen.py`

Run:

```bash
conda run -n agent python scripts/prepare_datamind_qwen.py --download
```

The script downloads or reads `datamind_12k.json`, computes complexity, quality, and diversity scores, removes duplicates, selects 2,000 training samples and 500 validation samples, and writes Qwen-style chat JSONL files:

- `outputs/data/train_qwen.jsonl`
- `outputs/data/val_qwen.jsonl`
- `outputs/data/selection_stats.json`

Each line has:

```json
{
  "id": "task id",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {
    "source": "...",
    "db_id": "...",
    "score": 0.0
  }
}
```

### Task 3: GPU LoRA Training Plan

The GPU LoRA training script is:

- `scripts/train_sft_gpu.py`

Run:

```bash
conda run -n agent python scripts/train_sft_gpu.py \
  --model models/Qwen3.5-0.8B \
  --train-file outputs/data/train_qwen.jsonl \
  --val-file outputs/data/val_qwen.jsonl \
  --max-steps 500 \
  --max-length 1024 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --fp16 \
  --lora-r 8 \
  --lora-alpha 16 \
  --output-dir outputs/checkpoints/qwen35_08b_data_agent_lora
```

The script uses `transformers`, `torch`, and `peft` with CUDA. It freezes the Qwen3.5-0.8B base model and trains LoRA adapters on attention and MLP projection modules. It loads the selected Qwen-format JSONL data, applies the tokenizer chat template, trains with causal language modeling loss, periodically evaluates validation loss, saves intermediate LoRA adapters, and saves the final adapter under `outputs/checkpoints/qwen35_08b_data_agent_lora/final`. This is still supervised fine-tuning, but it is parameter-efficient and fits a consumer GPU more reliably than full-parameter training.

Recommended hyperparameters for a real GPU run:

- base model: Qwen/Qwen3.5-0.8B, or a manually downloaded local Qwen3.5-0.8B checkpoint directory if HuggingFace is not reachable;
- max sequence length: 512 or 1024 on RTX 4070;
- learning rate: 1e-4 to 2e-4 for LoRA;
- batch size: 1 per GPU with gradient accumulation 16-32;
- epochs: 1-3;
- LoRA parameters: rank 8 or 16, alpha 16 or 32, dropout 0.05;
- validation: evaluate loss on `val_qwen.jsonl`, choose the lowest validation loss adapter;
- training framework: Ray Train for distributed PyTorch, or OpenRLHF SFT script for a production SFT pipeline.

### Task 4: Demo

The demo code is:

- `demo/web_demo.py`

Run:

```bash
conda run -n agent python demo/web_demo.py \
  --base-model models/Qwen3.5-0.8B \
  --adapter-path outputs/checkpoints/qwen35_08b_data_agent_lora/final
```

Demo: http://127.0.0.1:8000

The interface is adapted for data analysis and includes a lightweight tool layer. Users can upload CSV or Excel files; the demo uses pandas to profile schema, missing values, numeric statistics, and categorical top values. The dataset profile is injected into the model context so the agent can produce problem abstraction, data understanding, analysis plan, SQL/Python sketch, modeling choices, validation checks, and final answer format. This makes the demo more than a plain chatbot: it performs local data processing before calling the fine-tuned Qwen agent.

## Q2. Startup Business Plan

Prompt used for LLM-assisted business plan and roadshow design:

```text
You are an experienced startup founder, product strategist, and VC pitch advisor. I want to build a profitable indie startup based on LLM applications. Please help me brainstorm and write a complete business plan for a product named DataPilot.

Product context: DataPilot is an LLM-powered AI data analyst for small teams. Users can connect CSV files, spreadsheets, PostgreSQL/MySQL databases, and SaaS tools such as Stripe or Shopify. They can ask natural-language business questions, and the product generates SQL/Python analysis, mathematical problem abstraction, predictive-model baselines, validation checks, charts, and shareable reports.

Please write a business plan suitable for a university project report. The plan should include: problem statement, target customers, market opportunity, product positioning, core features, user workflow, business model, pricing, go-to-market strategy, competitor survey and comparison, differentiation, moat, financial assumptions, milestones, team plan, risks, and mitigation strategies.

Please also design a roadshow presentation to secure funding. Provide a 10-12 slide structure with slide titles and bullet points. The pitch deck should include problem, solution, market, product demo story, traction assumptions, competitors, business model, technology, roadmap, financial plan, funding ask, and closing vision.

Make the answer realistic for an indie startup, avoid exaggerated claims, and clearly state assumptions that need verification.
```

### Startup Idea

Product name: DataPilot.

DataPilot is an AI data analyst for small teams. It connects to CSV files, spreadsheets, PostgreSQL/MySQL warehouses, and SaaS tools, then turns natural-language questions into auditable data-analysis workflows.

### Customer Pain

Small teams often have business data but no full-time data scientist. Generic LLMs can explain concepts, but they are not connected to the team's schemas, do not reliably execute SQL/Python, and usually lack validation traces. Enterprise BI platforms are powerful but expensive and complex.

### Product

Core features:

- schema-aware natural-language analysis;
- SQL/Python generation and sandbox execution;
- automated metric diagnosis;
- predictive modeling baselines;
- report and dashboard export;
- agent trace and validation log.

Initial wedge: weekly revenue/churn/growth diagnosis for indie SaaS and e-commerce teams.

### Competitor Comparison

ChatGPT and Claude are strong general assistants, but they lack built-in governed data connectors and repeatable analysis workflows.

Tableau, Power BI, and Looker provide strong dashboards and enterprise governance, but they are heavier to configure and less natural for exploratory modeling.

Dataiku and Alteryx support enterprise data science automation, but their price and implementation complexity are high for indie teams.

DataPilot differentiates through narrow vertical workflows, affordable pricing, auditable agent traces, and fast setup.

### Business Model

- Free tier: CSV analysis and limited monthly runs.
- Pro: $29/user/month for connectors and report export.
- Team: $199/month for shared workspace, scheduled jobs, and governance.
- Usage add-on: high-volume agent runs, hosted model inference, and long-running modeling jobs.

### Go-To-Market

Start with founder-led growth:

- Product Hunt launch;
- templates for Shopify, Stripe, PostgreSQL, and Google Sheets;
- SEO content around SaaS and e-commerce metrics;
- partnerships with analytics consultants and agencies;
- public case studies showing time saved.

### Financial Assumptions

Month 6 target: 100 paying teams at an average $80 monthly revenue per account, about $8K MRR.

Month 12 target: 500 paying teams at $100 ARPA, about $50K MRR.

Main costs are model inference, cloud infrastructure, connector maintenance, and customer support.

### Risks and Mitigation

Data security risk: use encryption, least-privilege connectors, audit logs, and optional self-hosting.

Hallucination risk: require executable SQL/Python, validation checks, and confidence labels.

Competition risk: focus on vertical templates and speed for small teams.

## Q2 Task 2: Industrial Architecture for 100,000-Level Concurrency

Prompt used for LLM architecture design:

```text
You are a senior AI system architect. I am building an indie startup product named DataPilot, an LLM-powered data analyst for small teams. Users can connect CSV files, spreadsheets, PostgreSQL/MySQL databases, and SaaS data sources, then ask natural-language questions. The product should generate SQL/Python analysis, mathematical modeling plans, predictive-model baselines, charts, validation checks, and shareable reports.

Please design an industrial-grade system architecture that can support 100,000 concurrent users. The design must include the following modules: LLM engine, agent orchestration, data processing pipeline, SQL/Python tool sandbox, database and storage layer, vector retrieval over schemas/documents, high-concurrency components, monitoring and operation module, security/governance module, billing/quota module, and deployment strategy.

For each module, explain its responsibility, recommended technologies, scaling strategy, failure handling, and how it interacts with other modules. Include caching, message queues, load balancing, autoscaling, rate limiting, observability, audit logs, PII masking, permission control, model fallback, and cost control.

Please also provide: (1) an end-to-end request flow for an interactive data-analysis query, (2) a batch/scheduled report flow, (3) a Mermaid system design diagram, (4) key bottlenecks and mitigations, and (5) a short capacity-planning discussion for 100,000 concurrent users. Make the answer suitable for inclusion in a university project report.
```

The system is designed as a stateless, horizontally scalable service.

Main modules:

- API Gateway: authentication, quota, routing, rate limiting.
- Web/BFF layer: stateless FastAPI or Node service.
- Agent Orchestrator: decomposes tasks, manages tool calls, controls retries and guardrails.
- LLM Engine: fine-tuned Qwen model served by vLLM/TensorRT-LLM, with fallback to external APIs.
- Tool Sandbox: isolated SQL/Python/chart execution.
- Data Processing: async workers for ETL, profiling, exports, and scheduled reports.
- Database: PostgreSQL/MySQL for transactional data, object storage for files, vector database for schema/document retrieval.
- High Concurrency: CDN/WAF, load balancer, Redis cache, Kafka/Pulsar queue, autoscaling app and worker pools.
- Monitoring/Ops: Prometheus, Grafana, logs, tracing, alerting, cost dashboards.
- Security/Governance: PII masking, permission checks, audit logs, policy engine.

See `docs/architecture.mmd` for the system design diagram.

For 100,000 concurrent users, the key principle is to avoid synchronous long jobs. The API layer accepts requests quickly, streams short LLM responses when possible, and sends long analysis jobs to queues. GPU inference nodes scale independently from web nodes. Redis absorbs hot-session traffic, Kafka/Pulsar smooths bursts, and object storage handles generated reports.

## References

- DataMind-12K HuggingFace dataset page.
- Albalak et al., *A Survey on Data Selection for Language Models*.
- QwenLM/Qwen3 GitHub repository and demo example.
- Ray Train documentation.
- OpenRLHF SFT example.
- System Design Primer and ByteByteGo System Design 101.

