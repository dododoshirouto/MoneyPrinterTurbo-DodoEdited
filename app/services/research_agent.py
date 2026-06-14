"""
Research Agent — two-thread LLM architecture for pre-script information gathering.

Thread 1 (Orchestrator): decides what to search and when to stop
Thread 2 (Worker):       executes searches and summarizes each result set

The orchestrator returns a JSON action each step:
  {"action": "search", "queries": ["query1", "query2"]}
  {"action": "done",   "summary": "...final research summary..."}
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from app.services import llm as llm_service
from app.services import web_search

_ORCHESTRATOR_SYSTEM_PROMPT = """You are a research orchestrator. Your job is to plan web searches to gather accurate, up-to-date information for a video script.

Given:
- Video subject
- User's research instructions
- Previous search history (if any)

You must respond with a JSON object (no markdown, no explanation):

To search:
{"action": "search", "queries": ["search query 1", "search query 2"]}

To finish (when you have enough information):
{"action": "done", "summary": "...comprehensive research summary in the same language as the video subject..."}

Rules:
- Use 1-3 queries per step
- Queries should be specific and targeted
- Stop when you have sufficient information or have reached the step limit
- The summary must be detailed and directly useful for writing a video script
- Write the summary in the same language as the video subject
"""

_WORKER_SYSTEM_PROMPT = """You are a research analyst. Given web search results, extract and summarize the most relevant facts, statistics, and insights that would be useful for creating a video script.

Be concise but thorough. Focus on:
- Key facts and data points
- Interesting angles and perspectives
- Recent developments (if applicable)
- Surprising or counterintuitive findings

Respond in the same language as the search queries."""


@dataclass
class ResearchStep:
    step: int
    queries: list[str]
    raw_results: list[dict]  # list of {query, results: [{title, url, snippet}]}
    worker_summary: str


@dataclass
class ResearchResult:
    subject: str
    instructions: str
    steps: list[ResearchStep] = field(default_factory=list)
    final_summary: str = ""
    success: bool = False

    def to_context_string(self) -> str:
        """Format research results as a context block for the script generator."""
        if not self.final_summary:
            return ""
        lines = [
            "# Research Context (gathered via web search)",
            f"Subject: {self.subject}",
            "",
            self.final_summary,
        ]
        return "\n".join(lines)

    def to_log_dict(self) -> dict:
        return {
            "subject": self.subject,
            "instructions": self.instructions,
            "success": self.success,
            "final_summary": self.final_summary,
            "steps": [
                {
                    "step": s.step,
                    "queries": s.queries,
                    "worker_summary": s.worker_summary,
                    "raw_results": s.raw_results,
                }
                for s in self.steps
            ],
        }


def _call_orchestrator(
    subject: str,
    instructions: str,
    history: list[ResearchStep],
) -> dict:
    """Ask the orchestrator LLM what to do next. Returns parsed JSON action."""
    history_text = ""
    for step in history:
        history_text += f"\n\n## Step {step.step} searches: {step.queries}\n"
        history_text += f"Summary: {step.worker_summary}"

    user_message = f"""Video subject: {subject}

Research instructions: {instructions if instructions else "Gather comprehensive background information relevant to this subject."}

Search history so far:{history_text if history_text else " (none yet)"}

What should we search next? Or do we have enough information?"""

    prompt = _ORCHESTRATOR_SYSTEM_PROMPT + "\n\n" + user_message
    response = llm_service._generate_response(prompt=prompt)
    if not response:
        return {"action": "done", "summary": ""}

    # Strip markdown code fences if present
    response = re.sub(r"```(?:json)?\s*", "", response).strip().rstrip("```").strip()

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        logger.warning(f"Orchestrator returned non-JSON: {response[:200]}")
        return {"action": "done", "summary": response}


def _call_worker(queries: list[str], all_results: list[dict]) -> str:
    """Ask the worker LLM to summarize search results."""
    results_text = ""
    for item in all_results:
        results_text += f"\n\n### Search: '{item['query']}'\n"
        if item["results"]:
            for r in item["results"]:
                results_text += f"\n**{r['title']}**\n{r['snippet']}\nURL: {r['url']}\n"
        else:
            results_text += "(no results)\n"

    prompt = (
        _WORKER_SYSTEM_PROMPT
        + f"\n\nSearch queries: {queries}\n\nResults:{results_text}\n\nProvide your analysis:"
    )
    response = llm_service._generate_response(prompt=prompt)
    return response or "(worker returned empty response)"


def run_research(
    subject: str,
    instructions: str = "",
    max_steps: int = 3,
    progress_callback=None,
) -> ResearchResult:
    """
    Run the research agent loop.

    Args:
        subject: video subject
        instructions: user's guidance on what to research
        max_steps: maximum number of search rounds
        progress_callback: optional callable(step, total, message) for UI updates

    Returns:
        ResearchResult with final_summary and step log
    """
    result = ResearchResult(subject=subject, instructions=instructions)
    history: list[ResearchStep] = []

    logger.info(f"[research_agent] Starting research: subject='{subject}', max_steps={max_steps}")

    for step_num in range(1, max_steps + 1):
        logger.info(f"[research_agent] Step {step_num}/{max_steps} — calling orchestrator")

        if progress_callback:
            progress_callback(step_num, max_steps, f"Step {step_num}: deciding what to search...")

        action = _call_orchestrator(subject, instructions, history)
        logger.debug(f"[research_agent] Orchestrator action: {action}")

        if action.get("action") == "done":
            result.final_summary = action.get("summary", "")
            result.success = bool(result.final_summary)
            logger.info(f"[research_agent] Orchestrator decided to finish at step {step_num}")
            break

        queries = action.get("queries", [])
        if not queries:
            logger.warning("[research_agent] Orchestrator returned no queries, stopping")
            break

        # Execute searches (one per query)
        all_results = []
        for query in queries:
            if progress_callback:
                progress_callback(step_num, max_steps, f"Step {step_num}: searching '{query}'...")
            logger.info(f"[research_agent] Searching: '{query}'")
            search_results = web_search.search(query)
            all_results.append({
                "query": query,
                "results": [r.to_dict() for r in search_results],
            })

        if progress_callback:
            progress_callback(step_num, max_steps, f"Step {step_num}: analyzing results...")

        # Worker summarizes results
        worker_summary = _call_worker(queries, all_results)
        logger.info(f"[research_agent] Worker summary (step {step_num}): {worker_summary[:200]}...")

        step = ResearchStep(
            step=step_num,
            queries=queries,
            raw_results=all_results,
            worker_summary=worker_summary,
        )
        history.append(step)
        result.steps.append(step)

    else:
        # Reached max_steps without orchestrator saying "done" — force finish
        logger.info("[research_agent] Reached max_steps, requesting final summary")
        if progress_callback:
            progress_callback(max_steps, max_steps, "Generating final research summary...")

        all_summaries = "\n\n".join(
            f"Step {s.step}: {s.worker_summary}" for s in history
        )
        final_prompt = (
            _ORCHESTRATOR_SYSTEM_PROMPT
            + f"\n\nVideo subject: {subject}\n\n"
            + "You have reached the search step limit. Based on all research gathered, "
            + "respond with a 'done' action containing a comprehensive summary.\n\n"
            + f"All research gathered:\n{all_summaries}"
        )
        response = llm_service._generate_response(prompt=final_prompt)
        if response:
            response = re.sub(r"```(?:json)?\s*", "", response).strip().rstrip("```").strip()
            try:
                action = json.loads(response)
                result.final_summary = action.get("summary", response)
            except json.JSONDecodeError:
                # If not JSON, use the response directly as summary
                result.final_summary = response
        else:
            # Fallback: concatenate worker summaries
            result.final_summary = all_summaries

        result.success = bool(result.final_summary)

    logger.info(
        f"[research_agent] Done. steps={len(result.steps)}, "
        f"success={result.success}, summary_len={len(result.final_summary)}"
    )
    return result
