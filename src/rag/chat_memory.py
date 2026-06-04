from typing import List, Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.model.gemini_client import GeminiClientManager
from src.core.logging_config import setup_logger

logger = setup_logger("chat_memory")


class ChatMemoryManager:
    """Manages conversational session history and query reformulation (condensation) for RAG context awareness."""

    def __init__(self, gemini_manager: GeminiClientManager, max_history_turns: int = 10):
        """Initializes the memory manager.
        
        Args:
            gemini_manager (GeminiClientManager): Manager to load Gemini for query reformulation.
            max_history_turns (int): Maximum number of message turns to persist (prevents context window overflow).
        """
        self.gemini_manager = gemini_manager
        self.max_history_turns = max_history_turns
        
        # In-memory storage mapping session_id -> list of message dicts: [{"role": "user"/"assistant", "content": "..."}]
        self._sessions: Dict[str, List[Dict[str, str]]] = {}

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Retrieves history records for a session, ensuring length constraints.
        
        Args:
            session_id (str): The unique conversation session identifier.
            
        Returns:
            List[Dict[str, str]]: Array of roles and content messages.
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        return self._sessions[session_id]

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Adds a message turn to the session history.
        
        Args:
            session_id (str): Session target.
            role (str): Sender type ('user' or 'assistant').
            content (str): Text payload.
        """
        history = self.get_history(session_id)
        history.append({"role": role, "content": content})
        
        # Limit history to max turns (each turn is a user+assistant pair, so count * 2)
        max_messages = self.max_history_turns * 2
        if len(history) > max_messages:
            trimmed = history[-max_messages:]
            self._sessions[session_id] = trimmed
            logger.debug(f"Trimmed history for session '{session_id}' to last {self.max_history_turns} turns.")
        else:
            self._sessions[session_id] = history

    def clear_session(self, session_id: str) -> None:
        """Purges history records for a session.
        
        Args:
            session_id (str): Session target.
        """
        if session_id in self._sessions:
            self._sessions[session_id] = []
            logger.info(f"Cleared session history for: '{session_id}'")

    def _format_history_for_prompt(self, history: List[Dict[str, str]]) -> str:
        """Formats session history into a structured string block."""
        formatted = []
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
        return "\n".join(formatted)

    def condense_question(self, session_id: str, follow_up_query: str) -> str:
        """Reformulates follow-up queries containing references to prior turns into standalone questions.
        
        This step guarantees context-aware retrieval. For example:
        Prior turn: "What is AWS EC2?"
        Follow-up: "How do I configure its firewalls?"
        Reformulated: "How do I configure security group firewalls for AWS EC2 instances?"
        
        Args:
            session_id (str): Session target.
            follow_up_query (str): Fresh user query.
            
        Returns:
            str: Standalone query text ready for vector search.
        """
        history = self.get_history(session_id)
        if not history:
            logger.debug(f"No history found for session '{session_id}'. Returning query as-is.")
            return follow_up_query

        logger.info(f"Condensing follow-up query for session '{session_id}'")
        try:
            formatted_history = self._format_history_for_prompt(history)
            
            # Formulate the query rewrite prompt
            condensation_instructions = (
                "Given the following conversation history and a follow-up query, "
                "rephrase the follow-up query to be a standalone query that can be understood "
                "without the conversation history. Do NOT answer the query or add any extra commentary. "
                "Only output the rephrased standalone query. If no rephrasing is needed, output the follow-up query as-is."
            )

            prompt_template = ChatPromptTemplate.from_messages([
                ("system", condensation_instructions),
                (
                    "user", 
                    "CHAT HISTORY:\n{history}\n\n"
                    "FOLLOW-UP QUERY: {query}\n\n"
                    "STANDALONE QUERY:"
                )
            ])

            llm = self.gemini_manager.get_llm()
            condensation_chain = prompt_template | llm | StrOutputParser()

            # Execute model query reformulation
            standalone_query = condensation_chain.invoke({
                "history": formatted_history,
                "query": follow_up_query
            })

            cleaned_query = standalone_query.strip()
            logger.info(f"Reformulated: '{follow_up_query}' -> '{cleaned_query}'")
            return cleaned_query

        except Exception as e:
            logger.error(
                f"Query condensation failed for session '{session_id}': {str(e)}. "
                "Falling back to original query."
            )
            return follow_up_query
