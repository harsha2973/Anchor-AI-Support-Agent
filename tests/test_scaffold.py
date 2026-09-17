"""
Basic sanity tests for scaffold integrity, imports, and pipeline execution.
"""

from agent.pipeline import SupportAgentPipeline
from agent.run import setup_demo_store
from intents.taxonomy import IntentCategory, SUPPORT_TAXONOMY


def test_taxonomy_integrity():
    """Verify that all intent categories have registered definitions and examples."""
    for category in IntentCategory:
        assert category in SUPPORT_TAXONOMY, f"Missing taxonomy definition for {category}"
        definition = SUPPORT_TAXONOMY[category]
        assert definition.name
        assert definition.description
        assert len(definition.example_queries) > 0


def test_pipeline_smoke_run():
    """Verify that the end-to-end pipeline processes a query without errors."""
    store = setup_demo_store()
    pipeline = SupportAgentPipeline(store=store)

    state = pipeline.process_query("Where is my package #99887?")
    assert state.query_id
    assert state.intent_prediction is not None
    assert state.draft_response is not None
    assert isinstance(state.should_escalate, bool)
