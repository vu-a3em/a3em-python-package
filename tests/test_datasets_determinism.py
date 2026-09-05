import numpy as np


def _assert_splits_equal(split_a, split_b):
    for part_a, part_b in zip(split_a, split_b):
        if hasattr(part_a, "equals"):
            assert part_a.equals(part_b)
        else:
            assert np.array_equal(part_a, part_b)


def test_load_data_deterministic_across_instances(make_arden):
    features_a, labels_a = make_arden().load_data(random_state=42, rumble_only=True)
    features_b, labels_b = make_arden().load_data(random_state=42, rumble_only=True)

    assert features_a.equals(features_b)
    assert labels_a.equals(labels_b)


def test_load_data_ml_deterministic_across_instances(make_arden):
    result_a = make_arden().load_data_ml(
        test_split=0.5, random_state=42, rumble_only=True, shuffle=True
    )
    result_b = make_arden().load_data_ml(
        test_split=0.5, random_state=42, rumble_only=True, shuffle=True
    )

    _assert_splits_equal(result_a, result_b)


def test_load_clips_deterministic_across_instances(make_arden):
    clips_a, features_a = make_arden().load_clips(random_state=42, rumble_only=True)
    clips_b, features_b = make_arden().load_clips(random_state=42, rumble_only=True)

    assert len(clips_a) == len(clips_b)
    for clip_a, clip_b in zip(clips_a, clips_b):
        assert np.array_equal(clip_a, clip_b)
    assert features_a.equals(features_b)


def test_load_data_ml_differs_across_random_states(make_arden):
    # Guards against the determinism tests above passing vacuously (e.g. if
    # there were too little data for the random_state to matter).
    x_train_a, x_test_a, _, _ = make_arden().load_data_ml(
        test_split=0.5, random_state=1, rumble_only=True, shuffle=True
    )
    x_train_b, x_test_b, _, _ = make_arden().load_data_ml(
        test_split=0.5, random_state=2, rumble_only=True, shuffle=True
    )

    assert not x_train_a.equals(x_train_b) or not x_test_a.equals(x_test_b)


def test_load_data_ml_deterministic_with_background_noise(make_arden):
    result_a = make_arden().load_data_ml(test_split=0.5, random_state=42, shuffle=True)
    result_b = make_arden().load_data_ml(test_split=0.5, random_state=42, shuffle=True)

    _assert_splits_equal(result_a, result_b)
