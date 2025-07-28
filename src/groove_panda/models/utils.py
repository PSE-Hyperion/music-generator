def get_loss_weights():
    """
    Set the weights for the feature losses (might be moved to the config module)
    The weights defined are relative. The method calculates the absolute values, that sum up to 1
    """
    loss_weights_relative = {
        'output_bas': 1,
        'output_position': 2,
        'output_pitch': 3,
        'output_velocity': 3,
        'output_duration': 2,
        'output_tempo': 1
    }
    loss_weights_relative_sum = sum(loss_weights_relative.values())

    return {
        key: value / loss_weights_relative_sum
        for key, value in loss_weights_relative
    }
