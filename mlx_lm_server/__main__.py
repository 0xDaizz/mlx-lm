"""Entry point for: python -m mlx_lm_server"""

from __future__ import annotations

import uvicorn

from mlx_lm_server.server import create_app, parse_args


def main() -> None:
    config = parse_args()

    # --- Load model and tokenizer ---
    # Lazy import so the module can be imported without mlx installed (for tests)
    from mlx_lm import load  # type: ignore

    model, tokenizer = load(config.model, adapter_path=config.adapter_path)

    # --- Create a real scheduler (placeholder until Phase 2 is done) ---
    # For now, we use a stub that can be replaced once scheduler-agent delivers.
    from mlx_lm_server._stub_scheduler import StubScheduler

    scheduler = StubScheduler(model=model, tokenizer=tokenizer, config=config)

    # --- Build and run ---
    app = create_app(config=config, scheduler=scheduler, tokenizer=tokenizer)
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
