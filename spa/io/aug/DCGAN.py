'''
DCGAN (Deep Convolutional GAN)
'''

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Conv1D, Dense, Dropout, Flatten, BatchNormalization, LeakyReLU, GaussianNoise, ReLU

def make_generator_model(noise_dim, feature_dim, kernel_size, kernel_strides):

    model = tf.keras.Sequential()

    model.add(Input(shape=(noise_dim,)))
    model.add(Reshape([noise_dim, 1]))
    model.add(Conv1D(kernel_size=kernel_size, filters=256, 
                     activation='leaky_relu', strides=kernel_strides))  # (opt) (number of filters and kernel size)
    model.add(Dropout(0.2))  # (opt) (dropout probability)

    model.add(Conv1D(kernel_size=kernel_size, filters=128, strides=kernel_strides))  # (opt) (number of filters and kernel size)
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(Dropout(0.2))  # (opt) (dropout probability)

    model.add(Flatten())                                                    #(opt) (number of nodes in layer)
    model.add(Dense(feature_dim))
    model.compile()

    assert model.output_shape == (None, feature_dim)

    return model


def make_discriminator_model(feature_dim, kernel_size, kernel_strides):

    model = tf.keras.Sequential()

    model.add(Input(shape=(feature_dim,)))
    model.add(Reshape([feature_dim, 1]))
    model.add(Conv1D(kernel_size=kernel_size, filters=256, activation='leaky_relu', strides=kernel_strides))  # (opt) (number of filters and kernel size)
    model.add(Dropout(0.2))  # (opt) (dropout probability)

    model.add(Conv1D(kernel_size=kernel_size, filters=128, strides=kernel_strides))  # (opt) (number of filters and kernel size)
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))

    # model.add(MaxPool1D())
    model.add(Dropout(0.2))  # (opt) (dropout probability)

    model.add(Flatten())
    model.add(Dense(64))  # (opt) (number of nodes in layer)
    model.add(Dense(1))
    model.compile()

    return model


def generate_data(model, num_synthetic_to_gen=1,noise_dim=100):
    """
      Function that takes in the generator model and
      does a prediction and returns it as a numpy array.
    """
    noise_input = tf.random.normal([num_synthetic_to_gen, noise_dim])
    predictions = model(noise_input, training=False)
    predictions = predictions.numpy()
    return predictions


def train_step(data, batch_size, noise_dim, generator, discriminator, generator_optimizer, discriminator_optimizer):
    """
      Function for implementing one training step
      of the GAN model
    """

    def generator_loss(fake_output):
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_output), fake_output)

    def discriminator_loss(real_output, fake_output):
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_output), real_output) + tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake_output), fake_output)

    noise = tf.random.normal([batch_size, noise_dim], seed=1)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_data = generator(noise, training=True)

        real_output = discriminator(data, training=True)
        fake_output = discriminator(generated_data, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    #       acc = calc_accuracy(fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss


def train(dataset, epochs, batch, noise_dim, 
          generator, discriminator, generator_optimizer, discriminator_optimizer, 
          verbose=True):
    """
      Main GAN Training Function
    """
    epochs_gen_losses, epochs_disc_losses, epochs_accuracies = [], [], []
    for epoch in range(epochs):

        gen_losses, disc_losses, accuracies = [], [], []

        for data_batch in dataset:
            gen_loss, disc_loss = train_step(data_batch, batch, noise_dim, generator, discriminator,
                                             generator_optimizer, discriminator_optimizer)
            #       accuracies.append(acc)
            gen_losses.append(gen_loss)
            disc_losses.append(disc_loss)

        epoch_gen_loss = np.average(gen_losses)
        epoch_disc_loss = np.average(disc_losses)
        epoch_accuracy = np.average(accuracies)
        epochs_gen_losses.append(epoch_gen_loss)
        epochs_disc_losses.append(epoch_disc_loss)
        
        if verbose:
            print("Epoch: {}/{}".format(epoch + 1, epochs))
            print("Generator Loss: {}, Discriminator Loss: {}".format(epoch_gen_loss, epoch_disc_loss))
        
    # Draw the model every 2 epochs
    #     if (epoch + 1) % 2 == 0:
    #         draw_training_evolution(generator, epoch+1)

    # Save the model every 2 epochs for the last 2000 epochs
    #     if (epoch + 1) % 2 == 0 and epoch > (numofEPOCHS - 2000):
    #           checkpoint.save(file_prefix = checkpoint_prefix)   # Comment not to save model checkpoints while training

    return epochs_gen_losses, epochs_disc_losses


def expand_dataset(X, y, nobs, X_names=None, epochs=500, batch_size=16, noise_dim=100, verbose=True):
    '''
    Generate equal number of samples for each class.

    nobs : samples to generate per class.
    '''

    if X.shape[1] >= 100:
        kernel_size = 15
        kernel_strides = 2
    else:
        kernel_size = round(X.shape[1]/15)*2+1
        kernel_strides = 1
    
    if verbose:
        print('DCGAN will use kernel size', kernel_size, ' and strides', kernel_strides)
    
    if not X_names:
        X_names = list(range(X.shape[1]))

    synth_data = pd.DataFrame()
    for label in set(y):
        data = pd.concat([pd.DataFrame(X, columns=X_names), pd.DataFrame(y, columns=['label'])], axis=1)
        data = data[data['label'] == label]
        X0 = data.iloc[:, :-1]
        y0 = data.iloc[:, -1]
        X0.columns = X0.columns.astype(float)
        column_labels = X0.columns.tolist
        data_raw = X0.to_numpy()
        data_processed = data_raw
        train_data = data_processed

        data_size = train_data.shape[0]
        batch = batch_size
        train_dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(data_size).batch(batch)
        feature_dim = train_data.shape[1]

        generator = make_generator_model(noise_dim, feature_dim, kernel_size, kernel_strides)
        noise = tf.random.normal([1, noise_dim])
        generated_data = generator(noise, training=False)

        discriminator = make_discriminator_model(feature_dim, kernel_size, kernel_strides)
        decision = discriminator(generated_data)
        generator_optimizer = tf.keras.optimizers.Adam(1e-3)
        discriminator_optimizer = tf.keras.optimizers.Adam(1e-3)
        #     tf.config.experimental_run_functions_eagerly(True)

        _ = train(train_dataset, epochs=epochs, batch=batch_size,
                  noise_dim=noise_dim, generator=generator, discriminator=discriminator,
                  generator_optimizer=generator_optimizer,
                  discriminator_optimizer=discriminator_optimizer, 
                  verbose=verbose)

        generated_batch = generate_data(generator, num_synthetic_to_gen=nobs,noise_dim=noise_dim)
        df = pd.DataFrame(generated_batch, columns= X_names)
        df['label'] = len(df) * [label]
        synth_data = pd.concat([synth_data, df], axis=0)

        if verbose:

            from keras.utils import plot_model
            import IPython.display

            IPython.display.display(IPython.display.HTML('<b>generator</b>'))
            IPython.display.display(plot_model(generator,show_shapes=True))
            IPython.display.display(IPython.display.HTML('<b>discriminator</b>'))
            IPython.display.display(plot_model(discriminator, show_shapes=True))

    synth_data.reset_index(drop=True, inplace=True)
    return synth_data.iloc[:, :-1], synth_data.iloc[:, -1] #, generator_plot, discriminator_plot