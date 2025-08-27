from fastmcp import FastMCP
import random

mcp = FastMCP("random_number")


@mcp.tool(description="returns a random number")
def random_number(session_id: str) -> int:
    number = random.randint(1, 100)
    return number


if __name__ == "__main__":
   mcp.run(transport="sse", host="127.0.0.1", port=8000)


## start mcp service

# uv run python mcp-test-server.py

