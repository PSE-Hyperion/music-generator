def get_loss_weights():
    """
    Set the weights for the feature losses (might be moved to the config module)
    The weights defined are relative. The method calculates the absolute values, that sum up to 1
    """
    loss_weights_relative = {
        'bar_output': 1,
        'position_output': 2,
        'pitch_output': 3,
        'velocity_output': 3,
        'duration_output': 2,
        'tempo_output': 1
    }
    loss_weights_relative_sum = 0
    for key in loss_weights_relative:
        loss_weights_relative_sum += loss_weights_relative[key]

    return {
        key: loss_weights_relative[key] / loss_weights_relative_sum
        for key in loss_weights_relative
    }
