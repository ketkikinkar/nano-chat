from rl.reward import compute_reward


def test_good_completion_gets_positive_reward():
    good = "To be or not to be, that is the question."
    assert compute_reward(good) > 0.3


def test_empty_completion_gets_zero():
    assert compute_reward("") == 0.0


def test_pure_repetition_gets_low_reward():
    repetitive = "the the the the the the the the the the the the"
    assert compute_reward(repetitive) < 0.2


def test_reward_bounded_between_0_and_1():
    for text in ["", "Hello!", "abc " * 100, "Great answer!"]:
        r = compute_reward(text)
        assert 0.0 <= r <= 1.0, f"reward {r} out of [0,1] for: {text!r}"
