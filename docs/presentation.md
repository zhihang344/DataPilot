# DataPilot: AI Data Analyst for Small Teams

## 1. Problem

Small teams own more data than they can analyze. They cannot hire enough data scientists, and generic chatbots do not reliably produce auditable pipelines, SQL, metrics, and models.

## 2. Solution

DataPilot is a data-analysis agent that turns natural questions into end-to-end workflows:

- problem abstraction
- SQL/Python analysis
- model baseline
- validation checks
- exportable report and dashboard

## 3. Product Demo

User asks: "Why did May revenue drop?"

DataPilot connects to warehouse, profiles tables, writes SQL, checks anomalies, segments causes, generates charts, and saves a shareable investigation report.

## 4. Market

Target users:

- indie SaaS companies
- cross-border e-commerce sellers
- product and operations teams
- agencies serving SMEs

Initial wedge: "one-click weekly business diagnosis for SaaS and e-commerce."

## 5. Competitors

ChatGPT/Claude: strong general reasoning, weak governed data workflow.

Dataiku/Alteryx: enterprise-grade, expensive and heavy.

Tableau/Power BI Copilot: strong BI surface, less flexible for modeling and agentic debugging.

DataPilot wins with vertical workflow templates, lower price, and auditable agent traces.

## 6. Business Model

- Free: local CSV analysis, limited runs.
- Pro: $29/user/month.
- Team: $199/month with connectors, scheduled reports, shared workspaces.
- Usage add-on: long-running modeling and hosted inference.

## 7. Go-To-Market

Start from founder-led distribution:

- Product Hunt launch
- templates for Shopify, Stripe, Notion, PostgreSQL
- SEO pages around business metrics
- agency partnerships

## 8. Technology

Fine-tuned Qwen data agent plus tool execution:

- data profiling
- SQL/Python sandbox
- retrieval over schemas and past analyses
- validation and observability
- human approval before write operations

## 9. 100K Concurrency Architecture

API Gateway, stateless app layer, agent orchestrator, Redis, Kafka, vector DB, object storage, GPU serving pool, autoscaling workers, observability and policy enforcement.

## 10. Milestones

- Month 1: MVP with CSV/PostgreSQL, report export.
- Month 2: Stripe/Shopify connectors and scheduled diagnosis.
- Month 3: team workspace, billing, public launch.
- Month 6: 100 paying teams.

## 11. Funding Ask

Raise $500K seed to cover 12 months:

- 3 engineers
- inference and cloud cost
- security review
- initial growth experiments

## 12. Closing

DataPilot makes data science available to every small team, with the rigor of a real analyst and the speed of an AI agent.

