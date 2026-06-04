import os
import sys
from pathlib import Path

# Add workspace root to Python path
sys.path.append(str(Path(__file__).parent))

from src.model.gemini_client import GeminiClientManager


def main():
    print("Testing GeminiClientManager validation connectivity...")
    
    # Check current system environment
    api_key_env = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key_env:
        print("\n[Warning]: No GEMINI_API_KEY or GOOGLE_API_KEY was found in environment.")
        print("Please configure this environment variable to perform remote connection testing:")
        print("  Windows Powershell: $env:GEMINI_API_KEY='your_api_key'")
        print("  Windows CMD       : set GEMINI_API_KEY=your_api_key")
        print("  Linux/Mac         : export GEMINI_API_KEY='your_api_key'")
        
        # We can still test local validation error raising
        print("\nTesting missing API key exception behavior...")
        try:
            manager = GeminiClientManager(api_key=None)
            manager.get_llm()
        except ValueError as val_err:
            print(f"Exception successfully captured as expected: {val_err}")
            return
        except Exception as e:
            print(f"Unexpected error: {e}")
            return
        
    try:
        # Load Client Manager
        print("\nInitializing client manager with environment key...")
        manager = GeminiClientManager(
            model_name="gemini-1.5-flash",
            temperature=0.0
        )
        
        # Test connection validity
        is_connected = manager.validate_connection()
        if is_connected:
            print("\nSuccess! Gemini connection validation passed. Ready for chat operations.")
            
            # Simple prompt response test
            print("\nExecuting light test invoke prompt...")
            llm = manager.get_llm()
            resp = llm.invoke("Briefly answer in 3 words: What is AWS EC2?")
            print(f"Response: '{resp.content.strip()}'")
        else:
            print("\nFailure: Validation probe failed. Check network or API key authorization.")
            
    except Exception as e:
        print(f"\n[Test Execution Failed]: {e}")


if __name__ == "__main__":
    main()
