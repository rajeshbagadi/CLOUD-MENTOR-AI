import pytest
from unittest.mock import MagicMock, patch
from src.rag.chat_memory import ChatMemoryManager


def test_memory_history_truncation():
    """Verify history manager trims conversational blocks when limit threshold is exceeded."""
    # Gemini client mock is not called for simple adds
    memory = ChatMemoryManager(gemini_manager=MagicMock(), max_history_turns=2)
    session_id = "sess_1"

    # Add 3 turns (each turn has user + assistant, total 6 messages)
    for i in range(3):
        memory.add_message(session_id, "user", f"Query {i}")
        memory.add_message(session_id, "assistant", f"Response {i}")

    history = memory.get_history(session_id)
    
    # 2 turns * 2 messages = 4 messages max
    assert len(history) == 4
    
    # Verify the oldest message (Query 0, Response 0) is purged
    assert history[0]["content"] == "Query 1"
    assert history[-1]["content"] == "Response 2"


@patch("src.rag.chat_memory.ChatPromptTemplate")
def test_question_condensation(mock_prompt, monkeypatch):
    """Verify condense_question rephrases follow-up questions containing referencing terms."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="What are firewalls for AWS EC2 instances?")
    
    # Mock GeminiManager to return our mock LLM
    mock_gemini = MagicMock()
    mock_gemini.get_llm.return_value = mock_llm
    
    memory = ChatMemoryManager(gemini_manager=mock_gemini)
    session_id = "sess_2"
    
    # Log context session
    memory.add_message(session_id, "user", "What is AWS EC2?")
    memory.add_message(session_id, "assistant", "EC2 provides virtual servers.")
    
    # Rephrase follow-up query
    standalone = memory.condense_question(session_id, "How do I secure them?")
    
    # Verify result rephrase output matches our mock response
    assert standalone == "What are firewalls for AWS EC2 instances?"
