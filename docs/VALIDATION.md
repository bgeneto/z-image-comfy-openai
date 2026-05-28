# Validation performed in this packaging environment

Validated locally:

- Python files compile with `python -m compileall api`.
- `api.workflow.build_zimage_aio_prompt(...)` returns a ComfyUI API graph with the expected checkpoint, sampler, scheduler, and `SaveImage` node.
- Shell scripts are present and executable.

Not validated in this environment:

- Docker build, because Docker is not available in the packaging sandbox.
- Actual GPU inference, because the sandbox does not have NVIDIA runtime, ComfyUI running, or the model checkpoint downloaded.

Expected host validation command:

```bash
cp .env.example .env
docker compose up --build
curl http://localhost:8000/health | jq
./scripts/test_curl.sh
```
