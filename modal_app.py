"""Modal deployment configuration for hermes-agent.

Deploys the FastAPI streaming wrapper as a serverless ASGI app on Modal.

Usage:
    modal deploy modal_app.py       # Deploy to Modal
    modal serve modal_app.py        # Local dev with hot-reload
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "fastapi[standard]",
        "uvicorn",
        "openai",
        "python-dotenv",
        "fire",
        "httpx",
        "rich",
        "tenacity",
        "pyyaml",
        "requests",
        "jinja2",
        "pydantic>=2.0",
        "prompt_toolkit",
        "firecrawl-py",
        "fal-client",
        "edge-tts",
        "litellm>=1.75.5",
        "typer",
        "platformdirs",
        "PyJWT[crypto]",
    )
    .add_local_dir(".", "/app", copy=True, ignore=[".git", "__pycache__", "venv", ".venv", "*.pyc"])
)

app = modal.App("hermes-agent", image=image)


@app.function(
    min_containers=0,
    scaledown_window=300,
    timeout=600,
    secrets=[modal.Secret.from_name("hermes-secrets")],
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def web():
    import os
    import sys
    from pathlib import Path

    # Force HERMES_HOME to a known writable path inside the container
    hermes_home = "/tmp/hermes"
    os.environ["HERMES_HOME"] = hermes_home
    Path(hermes_home).mkdir(parents=True, exist_ok=True)
    (Path(hermes_home) / "logs").mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, "/app")
    from serve import app as fastapi_app
    return fastapi_app
