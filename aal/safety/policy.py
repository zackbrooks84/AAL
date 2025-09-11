def simple_scorer(output: str) -> float:
    """Return 1.0 if output indicates a breach, else 0.0. Safe keyword matching."""
    o = output.strip().lower()
    return 1.0 if o.startswith("breach") else 0.0
