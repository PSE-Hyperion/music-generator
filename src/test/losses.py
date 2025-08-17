import tensorflow as tf

from groove_panda.models.tf_custom import losses

# Dummy data: 5 Klassen, batch size = 2
y_true = tf.constant([2])  # Labels
y_pred = tf.constant([
    [0.0, 0.001, 0.998, 0.001, 0.0],  # perfektes prediction für 2
], dtype=tf.float32)

loss_fn = losses.SoftCategoricalKLDivergence([0.05, 0.1, 0.7, 0.1, 0.05])

loss = loss_fn(y_true, y_pred)
print("Loss:", loss.numpy())

y_true = tf.constant([2])  # Labels
y_pred = tf.constant([
    [0.05, 0.1, 0.7, 0.1, 0.05],  # perfektes prediction für 2
], dtype=tf.float32)

loss_fn = losses.NormalDistributedCategorialKLDivergence(1, 1e-4)

loss = loss_fn(y_true, y_pred)
print("Loss:", loss.numpy())


y_true = tf.constant([3])
y_pred = tf.constant([[0.1, 0.2, 0.4, 0.2, 0.1]])

loss_fn = losses.SumUnderRange()
loss = loss_fn(y_true, y_pred)
print(f"Expected distance: {loss}")
