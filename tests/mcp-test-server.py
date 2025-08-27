from fastmcp import FastMCP
import subprocess

mcp = FastMCP("terminal_runner")


@mcp.tool(description="Run a shell command and return the output")
def run_command(command: str) -> str:
    try:
        # Run the command, capture stdout and stderr
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        return result.stdout.strip() or "Command executed successfully."
    except subprocess.CalledProcessError as e:
        return f"Error ({e.returncode}): {e.stderr.strip()}"


if __name__ == "__main__":
    mcp.run(transport="sse", host="127.0.0.1", port=8000)



## start mcp service

# uv run python mcp-test-server.py

