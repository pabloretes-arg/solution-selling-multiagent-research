# Solution Selling Multi-Agent Research

**A small solution-selling team made of three AI agents.** The Researcher Agent gathers facts and news about a company, the Account Strategy Agent turns them into a discovery brief, and the Sales Manager Agent rejects the brief if it mixes your pitch into the client's priorities or states doubtful facts.

Built with [smolagents](https://github.com/huggingface/smolagents) as the multi-agent project for **Unit 2.1** of the [Hugging Face AI Agents Course](https://huggingface.co/learn/agents-course). It evolves my single-agent [account-research-agent](https://github.com/pabloretes-arg/account-research-agent).

## Demo

<!-- Drag the demo video here when editing this file on GitHub -->

## Why a team instead of one agent

The single agent reached a quality ceiling. Code-enforced rules fixed everything that can be counted (dates, number of questions, structure), but not judgment: it kept turning what we sell into the client's "priorities". A prompt rule saying "don't do that" was not enough.

So I split the work the way a real solution-selling team does:

```
You ──► Account Strategy Agent (manager, CodeAgent)
          │  1. writes a plan                          ← planning_interval
          │  2. delegates research to ──► Researcher Agent (CodeAgent)
          │                                 web search + page reading
          │                                 returns facts and dated news, each with a source
          │  3. writes the brief
          ▼
        Review before delivery                         ← final_answer_checks
          ├─ Is it the real brief, not a summary?           (checked in code)
          └─ Sales Manager Agent (ToolCallingAgent)
               pitch disguised as a priority? leading questions? doubtful facts?
               can open a source to verify one fact
               └─ FAIL → the strategist gets the reason, fixes the brief and tries again
```

| Agent | Type | Job | Tools |
|---|---|---|---|
| **Researcher Agent** | `CodeAgent` | Find facts and recent news about the company in the right market | Resilient web search, safe page reader |
| **Account Strategy Agent** | `CodeAgent` (manager) | Plan, delegate, write the brief | `build_discovery_brief`, the Researcher Agent |
| **Sales Manager Agent** | `ToolCallingAgent` | Approve or reject the brief before delivery | Safe page reader (to verify one fact) |

**Why a ToolCallingAgent for the Sales Manager?** It doesn't need to write code: it reads, maybe opens one source, and gives a verdict. JSON tool calls are simpler and more predictable for that. The other two write Python because they combine several searches and tool calls in one step.

**Why the Sales Manager runs as a check, not as a managed agent?** If the strategist could call it, it could also skip it. As a `final_answer_check`, it runs every time the strategist tries to deliver.

## What the brief contains

| Section | What it is | Who it's about |
|---|---|---|
| **Recent signals** | Dated news with sources | Facts about the company |
| **Likely priorities** | Hypotheses to validate in the call | The client's business |
| **Where our offering could fit** | Our angle, to test only after the client talks | Us |
| **Discovery questions** | 2 about the client's goals, 3 about the area where we could help | The conversation |

## Project structure

| File | What it does |
|---|---|
| `researcher_agent.py` | Researcher Agent. Run it alone with `python researcher_agent.py` to test it |
| `accountstrategy_agent.py` | Account Strategy Agent (manager), its instructions and the checks before delivery |
| `salesmanager_agent.py` | Sales Manager Agent. Run it alone with `python salesmanager_agent.py` to test it |
| `model.py` | Shared model setup (with fallback) and the step budget |
| `tools.py` | Search, page reader and the brief builder |
| `app.py` | The web interface |

## Design decisions

- **Separate memories.** The Researcher Agent's long search results stay in its own memory. The strategist only sees the final report, so it works with a short, clean context.
- **Live progress.** The interface shows which agent is working and on what (searching, reading a page, writing, reviewing), with elapsed time.
- **Long-running delegation.** The researcher runs inside the strategist's code block, so that block gets a 10-minute limit instead of the default 30 seconds.
- **Planning.** The strategist writes a plan before starting and reviews it halfway (`planning_interval`).
- **Checks before delivery.** `final_answer_checks` run every time the strategist tries to finish. A rejected brief goes back with the reason. After two rejections it is accepted as is, to avoid endless loops.
- **Critical rules live in code.** Date filter, brief structure and step budgets are enforced in code, not in the prompt (same as the single-agent version).
- **Resilient by design.** Model fallback (`LiteLLMRouterModel`) and search fallback across engines.
- **Model-agnostic.** The model is a line in `.env`.

## Run it locally

Requires Python 3.10+.

```bash
git clone https://github.com/pabloretes-arg/solution-selling-multiagent-research.git
cd solution-selling-multiagent-research
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your API keys to .env
python app.py
```

Open http://127.0.0.1:7860. The progress bar shows which agent is working. The panel *How the team worked* shows the plan and every step of the three agents, including rejected briefs.

## Trade-offs

- **Slower and more expensive.** A team makes more model calls than a single agent: expect 2–4 minutes per brief.
- **An LLM reviewing an LLM is not a guarantee.** The Sales Manager Agent catches the most common mistakes, but the brief still needs a human check before a client meeting.
- **If the reviewer fails** (e.g. the model provider is down), the brief is delivered anyway rather than lost.

## Series

Built while taking the course:

1. [Account Research Agent](https://github.com/pabloretes-arg/account-research-agent): single agent, smolagents
2. **Solution Selling Multi-Agent Research**: multi-agent, smolagents (this repo)
3. LlamaIndex project (coming soon)
4. LangGraph project (coming soon)

## Author

**Pablo Retes**, Buenos Aires.