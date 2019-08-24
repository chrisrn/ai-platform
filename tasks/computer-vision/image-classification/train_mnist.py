from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import argparse
import numpy as np
import mlflow
import mlflow.tensorflow

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
    """
    Model function for CNN
    :param features:
    :param labels:
    :param mode:
    :return:
    """

    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def get_data():
    """
    Function for downloading mnist data and normalize them
    :return: all the data
    """

    ((train_data, train_labels),
     (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

    train_data = train_data / np.float32(255)
    train_labels = train_labels.astype(np.int32)

    eval_data = eval_data / np.float32(255)
    eval_labels = eval_labels.astype(np.int32)

    return train_data, train_labels, eval_data, eval_labels


def estimator_creation(args, train_data, train_labels):
    """

    :param args:
    :param train_data:
    :param train_labels:
    :return:
    """

    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=args.model_dir)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        shuffle=True)

    return mnist_classifier, train_input_fn


def train_and_evaluate(mnist_classifier, train_input_fn, eval_data, eval_labels):
    """

    :param mnist_classifier:
    :param train_input_fn:
    :param eval_data:
    :param eval_labels:
    :return:
    """
    hooks = []
    if args.early_stopping:
        print("*** Early stopping enabled ***")
        early_stopping = tf.estimator.experimental.stop_if_no_decrease_hook(
            mnist_classifier,
            metric_name='loss',
            max_steps_without_decrease=1000,
            min_steps=100)
        hooks.append(early_stopping)

    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=1,
        hooks=hooks)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    mlflow.log_metric("Evaluation loss", eval_results['loss'], step=1)
    mlflow.log_metric("Evaluation accuracy", eval_results['accuracy'], step=1)
    # print(eval_results)


def main(args):

    # Load data and labels
    train_data, train_labels, eval_data, eval_labels = get_data()

    with mlflow.start_run():
        # Create estimator and input function
        mnist_classifier, train_input_fn = estimator_creation(args, train_data, train_labels)

        # Training and evaluation
        train_and_evaluate(mnist_classifier, train_input_fn, eval_data, eval_labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training parameters')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Number of examples for each training step')
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='Number of epochs to train')
    parser.add_argument('--early_stopping', action="store_true",
                        help='Stop training according to loss value')
    parser.add_argument('--model_dir', type=str, default="/tmp/mnist_convnet_model",
                        help='Directory to save checkpoints and tensorboard events')

    args = parser.parse_args()

    main(args)
