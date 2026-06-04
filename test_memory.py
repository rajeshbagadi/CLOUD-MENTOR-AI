import os
import sys
from pathlib import Path

# Add workspace root to Python path
sys.path.append(str(Path(__file__).parent))

from src.model.gemini_client import GeminiClientManager
from src.rag.chat_memory import ChatMemoryManager


def main():
    print("Testing ChatMemoryManager query condensation and context tracking...")
    
    # Check credentials before starting LLM-driven query condensation
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("\n[Warning]: GEMINI_API_KEY is not configured.")
        print("To run query condensation test, setting the API key is required.")
        return

    try:
        # Initialize LLM Manager
        print("\nInitializing Gemini client...")
        gemini_mgr = GeminiClientManager(
            model_name="gemini-1.5-flash",
            temperature=0.0
        )
        
        # Initialize Memory Manager
        print("Initializing ChatMemoryManager...")
        memory_mgr = ChatMemoryManager(gemini_manager=gemini_mgr, max_history_turns=5)
        
        session_id = "test_user_session_123"
        
        # Test 1: Empty history rephrasing (should return query as-is)
        print("\n--- Test 1: Empty history test ---")
        initial_query = "What is AWS EC2?"
        standalone_1 = memory_mgr.condense_question(session_id, initial_query)
        print(f"Initial Query   : '{initial_query}'")
        print(f"Standalone Query: '{standalone_1}'")
        
        # Add message sequence to database logs
        print("\nLogging turns into chat memory...")
        memory_mgr.add_message(session_id, "user", initial_query)
        memory_mgr.add_message(
            session_id, 
            "assistant", 
            "AWS EC2 (Elastic Compute Cloud) provides secure, resizable compute capacity in the cloud as virtual servers."
        )
        
        # Test 2: Follow-up query with referencing pronouns
        print("\n--- Test 2: Follow-up query with references ---")
        follow_up = "How do I secure them?"
        standalone_2 = memory_mgr.condense_question(session_id, follow_up)
        print(f"Follow-up Query  : '{follow_up}'")
        print(f"Standalone Query : '{standalone_2}'")
        
        # Log this turn
        memory_mgr.add_message(session_id, "user", follow_up)
        memory_mgr.add_message(
            session_id, 
            "assistant", 
            "You can secure EC2 instances using security group firewalls, IAM roles, and AWS KMS encryption keys."
        )
        
        # Test 3: Second follow-up query referencing prior answers
        print("\n--- Test 3: Multi-turn context preservation ---")
        follow_up_2 = "Can you explain the first one?"
        standalone_3 = memory_mgr.condense_question(session_id, follow_up_2)
        print(f"Follow-up Query  : '{follow_up_2}'")
        print(f"Standalone Query : '{standalone_3}'")
        
        # Test 4: Clear session
        print("\n--- Test 4: Clearing session memory ---")
        memory_mgr.clear_session(session_id)
        history_after = memory_mgr.get_history(session_id)
        print(f"Current history list length: {len(history_after)}")
        
    except Exception as e:
        print(f"\n[Test Error Occurred]: {e}")


if __name__ == "__main__":
    main()
