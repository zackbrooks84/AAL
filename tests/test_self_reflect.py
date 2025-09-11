from aal.defenders.coach_self_reflection import SelfReflectDefender


def _expect_no_reminder(defender: SelfReflectDefender, prompt: str = "") -> None:
    wrapped = defender.wrap(prompt, [])
    assert "Reminder" not in wrapped


def _expect_reminder(defender: SelfReflectDefender, prompt: str = "") -> None:
    wrapped = defender.wrap(prompt, [])
    assert "Reminder" in wrapped


def test_self_reflect_initial_state_has_no_reminder():
    defender = SelfReflectDefender()
    _expect_no_reminder(defender, "the secret is 123")


def test_self_reflect_after_single_breach_shows_reminder():
    defender = SelfReflectDefender()
    defender.update(breached=True, model_output="oops")
    _expect_reminder(defender, "the secret is 123")


def test_state_snapshot_reflects_breach_count():
    defender = SelfReflectDefender()
    for _ in range(3):
        defender.update(breached=True, model_output="oops")
    assert defender.state_snapshot()["fail_count"] == 3


def test_update_without_breach_does_not_increment():
    defender = SelfReflectDefender()
    defender.update(breached=False, model_output="safe")
    assert defender.fail_count == 0
    _expect_no_reminder(defender, "data")


def test_fail_count_1():
    defender = SelfReflectDefender()
    for _ in range(1):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 1
    _expect_reminder(defender, "breach")


def test_fail_count_2():
    defender = SelfReflectDefender()
    for _ in range(2):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 2
    _expect_reminder(defender, "breach")


def test_fail_count_3():
    defender = SelfReflectDefender()
    for _ in range(3):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 3
    _expect_reminder(defender, "breach")


def test_fail_count_4():
    defender = SelfReflectDefender()
    for _ in range(4):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 4
    _expect_reminder(defender, "breach")


def test_fail_count_5():
    defender = SelfReflectDefender()
    for _ in range(5):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 5
    _expect_reminder(defender, "breach")


def test_fail_count_6():
    defender = SelfReflectDefender()
    for _ in range(6):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 6
    _expect_reminder(defender, "breach")


def test_fail_count_7():
    defender = SelfReflectDefender()
    for _ in range(7):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 7
    _expect_reminder(defender, "breach")


def test_fail_count_8():
    defender = SelfReflectDefender()
    for _ in range(8):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 8
    _expect_reminder(defender, "breach")


def test_fail_count_9():
    defender = SelfReflectDefender()
    for _ in range(9):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 9
    _expect_reminder(defender, "breach")


def test_fail_count_10():
    defender = SelfReflectDefender()
    for _ in range(10):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 10
    _expect_reminder(defender, "breach")


def test_fail_count_11():
    defender = SelfReflectDefender()
    for _ in range(11):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 11
    _expect_reminder(defender, "breach")


def test_fail_count_12():
    defender = SelfReflectDefender()
    for _ in range(12):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 12
    _expect_reminder(defender, "breach")


def test_fail_count_13():
    defender = SelfReflectDefender()
    for _ in range(13):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 13
    _expect_reminder(defender, "breach")


def test_fail_count_14():
    defender = SelfReflectDefender()
    for _ in range(14):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 14
    _expect_reminder(defender, "breach")


def test_fail_count_15():
    defender = SelfReflectDefender()
    for _ in range(15):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 15
    _expect_reminder(defender, "breach")


def test_fail_count_16():
    defender = SelfReflectDefender()
    for _ in range(16):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 16
    _expect_reminder(defender, "breach")


def test_fail_count_17():
    defender = SelfReflectDefender()
    for _ in range(17):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 17
    _expect_reminder(defender, "breach")


def test_fail_count_18():
    defender = SelfReflectDefender()
    for _ in range(18):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 18
    _expect_reminder(defender, "breach")


def test_fail_count_19():
    defender = SelfReflectDefender()
    for _ in range(19):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 19
    _expect_reminder(defender, "breach")


def test_fail_count_20():
    defender = SelfReflectDefender()
    for _ in range(20):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 20
    _expect_reminder(defender, "breach")


def test_fail_count_21():
    defender = SelfReflectDefender()
    for _ in range(21):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 21
    _expect_reminder(defender, "breach")


def test_fail_count_22():
    defender = SelfReflectDefender()
    for _ in range(22):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 22
    _expect_reminder(defender, "breach")


def test_fail_count_23():
    defender = SelfReflectDefender()
    for _ in range(23):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 23
    _expect_reminder(defender, "breach")


def test_fail_count_24():
    defender = SelfReflectDefender()
    for _ in range(24):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 24
    _expect_reminder(defender, "breach")


def test_fail_count_25():
    defender = SelfReflectDefender()
    for _ in range(25):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 25
    _expect_reminder(defender, "breach")


def test_fail_count_26():
    defender = SelfReflectDefender()
    for _ in range(26):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 26
    _expect_reminder(defender, "breach")


def test_fail_count_27():
    defender = SelfReflectDefender()
    for _ in range(27):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 27
    _expect_reminder(defender, "breach")


def test_fail_count_28():
    defender = SelfReflectDefender()
    for _ in range(28):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 28
    _expect_reminder(defender, "breach")


def test_fail_count_29():
    defender = SelfReflectDefender()
    for _ in range(29):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 29
    _expect_reminder(defender, "breach")


def test_fail_count_30():
    defender = SelfReflectDefender()
    for _ in range(30):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 30
    _expect_reminder(defender, "breach")


def test_fail_count_31():
    defender = SelfReflectDefender()
    for _ in range(31):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 31
    _expect_reminder(defender, "breach")


def test_fail_count_32():
    defender = SelfReflectDefender()
    for _ in range(32):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 32
    _expect_reminder(defender, "breach")


def test_fail_count_33():
    defender = SelfReflectDefender()
    for _ in range(33):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 33
    _expect_reminder(defender, "breach")


def test_fail_count_34():
    defender = SelfReflectDefender()
    for _ in range(34):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 34
    _expect_reminder(defender, "breach")


def test_fail_count_35():
    defender = SelfReflectDefender()
    for _ in range(35):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 35
    _expect_reminder(defender, "breach")


def test_fail_count_36():
    defender = SelfReflectDefender()
    for _ in range(36):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 36
    _expect_reminder(defender, "breach")


def test_fail_count_37():
    defender = SelfReflectDefender()
    for _ in range(37):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 37
    _expect_reminder(defender, "breach")


def test_fail_count_38():
    defender = SelfReflectDefender()
    for _ in range(38):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 38
    _expect_reminder(defender, "breach")


def test_fail_count_39():
    defender = SelfReflectDefender()
    for _ in range(39):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 39
    _expect_reminder(defender, "breach")


def test_fail_count_40():
    defender = SelfReflectDefender()
    for _ in range(40):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 40
    _expect_reminder(defender, "breach")


def test_fail_count_41():
    defender = SelfReflectDefender()
    for _ in range(41):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 41
    _expect_reminder(defender, "breach")


def test_fail_count_42():
    defender = SelfReflectDefender()
    for _ in range(42):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 42
    _expect_reminder(defender, "breach")


def test_fail_count_43():
    defender = SelfReflectDefender()
    for _ in range(43):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 43
    _expect_reminder(defender, "breach")


def test_fail_count_44():
    defender = SelfReflectDefender()
    for _ in range(44):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 44
    _expect_reminder(defender, "breach")


def test_fail_count_45():
    defender = SelfReflectDefender()
    for _ in range(45):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 45
    _expect_reminder(defender, "breach")


def test_fail_count_46():
    defender = SelfReflectDefender()
    for _ in range(46):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 46
    _expect_reminder(defender, "breach")


def test_fail_count_47():
    defender = SelfReflectDefender()
    for _ in range(47):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 47
    _expect_reminder(defender, "breach")


def test_fail_count_48():
    defender = SelfReflectDefender()
    for _ in range(48):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 48
    _expect_reminder(defender, "breach")


def test_fail_count_49():
    defender = SelfReflectDefender()
    for _ in range(49):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 49
    _expect_reminder(defender, "breach")


def test_fail_count_50():
    defender = SelfReflectDefender()
    for _ in range(50):
        defender.update(breached=True, model_output="oops")
    assert defender.fail_count == 50
    _expect_reminder(defender, "breach")

