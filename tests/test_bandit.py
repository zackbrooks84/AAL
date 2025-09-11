from aal.adversaries.bandit_prompt import BanditAdversary, OPS

def test_bandit_updates_priors():
    b = BanditAdversary()
    op = b.choose_op([])
    assert op in dict(OPS)
    a_before = dict(b.alpha)
    b.update(op, 1.0)
    assert b.alpha[op] == a_before[op] + 1.0
