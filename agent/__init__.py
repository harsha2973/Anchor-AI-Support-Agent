"""
Agent execution pipeline module.

Implements the end-to-end support pipeline:
Customer Query -> Intent Classification -> Grounding Retrieval -> Draft Synthesis -> Escalation Policy Check.
"""

from agent.pipeline import SupportAgentPipeline, run
from agent.state import AgentState

__all__ = ["SupportAgentPipeline", "AgentState", "run"]
