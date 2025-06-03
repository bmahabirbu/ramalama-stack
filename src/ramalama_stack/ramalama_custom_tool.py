import os
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent

# Configuration
LLAMA_STACK_PORT = 8321
INFERENCE_MODEL = os.getenv("INFERENCE_MODEL", "llama3.2")

def create_http_client():
    return LlamaStackClient(
        base_url=f"http://localhost:{LLAMA_STACK_PORT}",
        timeout=2000.0,
    )

client = create_http_client()

# Custom agent that reverses prompts
class ReversePromptAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def create_turn(self, session_id: str, messages: list, stream: bool = False):
        # Extract the latest user message
        user_message = next(
            (msg["content"] for msg in reversed(messages) if msg["role"] == "user"),
            "",
        )
        
        # Reverse the message
        reversed_message = user_message[::-1]
        
        # Return the modified message
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": reversed_message,
                }
            }]
        }

# Create the agent
agent = ReversePromptAgent(
    client=client,
    model=INFERENCE_MODEL,
    instructions="You are a test agent that reverses text prompts.",
)

# Create a session
session_id = agent.create_session("reverse-prompt-test")
print(f"Agent is ready! Session ID: {session_id}")

# Test cases
test_messages = [
    "hello",
    "llama stack",
    "reverse me",
    "12345",
    "!@#$%"
]

print("\nTesting agent with sample messages:")
for message in test_messages:
    # Create turn with test message
    turn_response = agent.create_turn(
        session_id=session_id,
        messages=[{"role": "user", "content": message}],
        stream=False,
    )
    
    # Get the reversed response
    reversed_message = turn_response["choices"][0]["message"]["content"]
    
    # Print results
    print(f"Input: '{message}'")
    print(f"Reversed: '{reversed_message}'")
    print("-" * 40)

print("\nTesting complete!")