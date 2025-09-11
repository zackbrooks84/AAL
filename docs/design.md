AAL Design
AAL runs an adaptive game between an adversary and a defender. The loop is configurable and produces a full JSONL trajectory per run.
Key elements:
⦁	Adversary policy with internal state and a reward model.
⦁	Defender policy that can update based on failures.
⦁	Model provider interface so you can swap between dummy and remote providers.
⦁	Scorer function that determines breach vs safe outcome on a safe simulator task.
Novelty:
⦁	Regret-conditioned adversary using Thompson sampling.
⦁	Defense delta metric using counterfactual replay.
⦁	Resilience half-life to quantify speed of improvement.
⦁	ELO for adversaries and defenses to enable a league of policies.
