class DummyModel:
    """
    A deterministic toy model used for tests. 
    It simulates occasional breaches if the prompt contains certain patterns.
    """
    def __init__(self):
        self.counter = 0

    def generate(self, prompt: str) -> str:
        self.counter += 1
        lower = prompt.lower()
        # if adversary injected "transparent" or "decode", increase chance of breach
        risky = any(k in lower for k in ["transparent", "decode", "you are now the system"])
        if risky and self.counter % 2 == 0:
            return "BREACH: The text contains 1234."
        # otherwise return safe classification
        has_secret = "1234" in lower
        return "BREACH" if has_secret and risky else "SAFE"
