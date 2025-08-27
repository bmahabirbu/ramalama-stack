import logging
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types import UserMessage
from typing import cast, Iterator
import os
import json

# ---------------------------
# Setup logging
# ---------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

INFERENCE_MODEL = os.getenv("INFERENCE_MODEL", "qwen2.5")
LLAMA_STACK_PORT = 8321

# ---------------------------
# Create LlamaStack HTTP client
# ---------------------------
def create_http_client():
    return LlamaStackClient(
        base_url=f"http://localhost:{LLAMA_STACK_PORT}",
        timeout=2000.0,
    )

client = create_http_client()

# ---------------------------
# Cleanup previously registered MCP toolgroups
# ---------------------------

for toolgroup in client.toolgroups.list():
    print(f"Unregistering MCP toolgroup: {toolgroup.identifier}")
    client.toolgroups.unregister(toolgroup_id=toolgroup.identifier)

# ---------------------------
# Register your test MCP server
# ---------------------------
client.toolgroups.register(
    toolgroup_id="mcp::random_number",
    provider_id="model-context-protocol",
    mcp_endpoint={"uri": "http://localhost:8000/sse"},
)
# ---------------------------
# Cleanup previous agents
# ---------------------------
for agent_info in client.agents.list().data:
    agent_id = agent_info["agent_id"]
    print(f"Unregistering agent: {agent_id}")
    client.agents.delete(agent_id=agent_id)

# ---------------------------
# Create a new agent with your MCP tool
# ---------------------------
agent = Agent(
    client=client,
    model=INFERENCE_MODEL,
    instructions="You are a helpful assistant",
    enable_session_persistence=False,
    tools=["mcp::random_number"],
    sampling_params={"max_tokens": 2048}
)
print("\nCurrent Agent ID: ", agent.agent_id)

# ---------------------------
# Start a new session
# ---------------------------
session_id = agent.create_session(session_name="mcp_test_session")
print("\nStarted Agent Session: ", session_id)

# ---------------------------
# Run a test turn
# ---------------------------
while True:
    prompt = input("Enter a prompt: ")
    if not prompt:
        break
    turn_response = agent.create_turn(
        messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        session_id=session_id,
    )

    for log in EventLogger().log(turn_response):
        log.print()