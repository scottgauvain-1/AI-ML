#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 14:25:34 2025

@author: sjgauva
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Conv2D, MaxPooling1D, UpSampling1D
from tensorflow.keras.layers import Concatenate, BatchNormalization, Activation
from tensorflow.keras.models import Model

#%% 3072,2 input shape surfnet
class SurfNet:
    def __init__(self, input_shape=(3072, 2), num_frequencies=50):
        """
        Initialize Surf-Net architecture
        
        Parameters:
        - input_shape: Shape of input (time samples, channels)
        - num_frequencies: Number of target frequencies
        """
        self.input_shape = input_shape
        self.num_frequencies = num_frequencies
        self.model = self._build_network()
    
    def _large_kernel_convolution_unit(self, inputs):
        """
        Large kernel convolution unit for extracting long-period features
        """
        # Separate cross-correlation channel
        cc_input = inputs[:, :, 0:1]  # Shape: (batch, 3072, 1)
        
        # Create kernel with matching time dimension
        kernel = tf.random.normal((3072, 1, self.num_frequencies))  # Match input time dimension
        
        # Perform FFT along time dimension (axis 1)
        cc_fft = tf.signal.fft(tf.cast(cc_input, tf.complex64))  # Shape: (batch, 3072, 1)
        kernel_fft = tf.signal.fft(tf.cast(kernel[:, :, :], tf.complex64))  # Shape: (3072, 1, num_frequencies)
        
        # Reshape and expand dimensions for broadcasting
        cc_fft = tf.expand_dims(cc_fft, axis=-1)  # Shape: (batch, 3072, 1, 1)
        kernel_fft = tf.expand_dims(kernel_fft, axis=0)  # Shape: (1, 3072, 1, num_frequencies)
        
        # Broadcast kernel to batch size
        kernel_fft = tf.tile(kernel_fft, [tf.shape(cc_input)[0], 1, 1, 1])
        
        # Convolution in frequency domain
        conv_fft = cc_fft * kernel_fft
        features = tf.math.real(tf.signal.ifft(conv_fft))  # Shape: (batch, 3072, 1, num_frequencies)
        
        # Remove extra dimension
        features = tf.squeeze(features, axis=2)  # Shape: (batch, 3072, num_frequencies)
        
        # Channel normalization
        features_normalized = features / tf.reduce_max(
            tf.abs(features), axis=(1, 2), keepdims=True
        )
        
        # Concatenate with distance channel
        distance_channel = inputs[:, :, 1:2]  # Shape: (batch, 3072, 1)
        return Concatenate(axis=-1)([features_normalized, distance_channel])
    
    def _down_sampling_unit(self, inputs, filters):
        """
        Down-sampling unit for feature extraction
        
        Includes:
        - Convolution
        - Momentum batch normalization
        - Relu 
        - Max pooling
        """
        x = Conv1D(filters, kernel_size=3, padding='same')(inputs)
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        return x
        
    def _up_sampling_unit(self, inputs, skip_connection, filters):
        """
        Up-sampling unit for recovering data length
        """
        # First adjust the filters
        x = Conv1D(filters, kernel_size=3, padding='same')(inputs)
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation('relu')(x)
        
        # Process skip connection
        skip = Conv1D(filters, kernel_size=1, padding='same')(skip_connection)
        
        # Concatenate along temporal dimension
        x = Concatenate(axis=1)([skip, x])
        
        return x
        
    def _output_unit(self, inputs):
        output = Conv1D(self.num_frequencies, kernel_size=1, activation='linear')(inputs)
        output_probs = Activation('sigmoid')(output)
        return output_probs
    
    def _custom_sigmoid(self, x):
        """
        Custom sigmoid activation to map to probabilities
        
        qi(t) = exp(yi(t)) / (1 + exp(yi(t)))
        """
        return tf.sigmoid(x)
    
    def _build_network(self):
        """
        Construct Surf-Net architecture
        
        Follows U-Net inspired design with modifications
        """
        inputs = Input(shape=self.input_shape)
        
        # Large kernel convolution unit
        x = self._large_kernel_convolution_unit(inputs)
        
        # Down-sampling path with skip connections
        down_samples = []
        for filters in [64, 128, 256, 512, 1024, 2048]:
            x = self._down_sampling_unit(x, filters)
            down_samples.append(x)
        
        # Up-sampling path with skip connections
        for i, filters in reversed(list(enumerate([2048, 1024, 512, 256, 128, 64]))):
            x = self._up_sampling_unit(x, down_samples[i], filters)
        
        # Output unit
        outputs = self._output_unit(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def _compute_frequency_weights(self, y_true):
        """
        Compute frequency-dependent weights
        Should return tensor of same shape as y_true
        """
        # For now, return ones of same shape as input
        return tf.ones_like(y_true)

    def compile(self, learning_rate=0.001):
        """
        Compile the model with custom weighted cross-entropy loss
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        def weighted_cross_entropy_loss(y_true, y_pred):
            """
            Weighted cross-entropy loss with frequency-dependent weights
            Shapes:
            y_true, y_pred: (batch_size, 3072, 50)
            """
            # Use tf.keras.backend.binary_crossentropy to preserve dimensions
            loss = tf.keras.backend.binary_crossentropy(
                target=y_true,
                output=y_pred,
                from_logits=False
            )  
            
            # Get weights same shape as target
            weights = self._compute_frequency_weights(y_true)
            
            # Ensure loss has same shape as weights
            loss = tf.reshape(loss, tf.shape(weights))
            
            # Apply weights and reduce
            weighted_loss = loss * weights
            
            # Average over all dimensions
            return tf.reduce_mean(weighted_loss)
        
        self.model.compile(
            optimizer=optimizer,
            loss=weighted_cross_entropy_loss,
            metrics=['accuracy'],
            run_eagerly=True
        )
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=16):
        """
        Train the Surf-Net model
        """
        print("Training data shapes:")
        print("X_train shape:", X_train.shape)
        print("y_train shape:", y_train.shape)
        if X_val is not None:
            print("X_val shape:", X_val.shape)
            print("y_val shape:", y_val.shape)
        
        return self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size
        )
    
    def predict_dispersion_curves(self, X_test):
        """
        Predict dispersion curves from input data
        """
        return self.model.predict(X_test)

# Example usage placeholder
def prepare_data():
    """
    Prepare input data in the format expected by Surf-Net
    
    Input shape: (num_samples, 3072, 2)
    - First channel: Normalized cross-correlation waveform
    - Second channel: Distance information
    
    Output shape: (num_samples, 3072, 50)
    - Probability distributions for 50 target frequencies
    """
    # Generate dummy data
    X = np.random.random((100, 3072, 2))  # Input data
    y = np.random.random((100, 3072, 50))  # Target probabilities
    return X, y

#%% execute

def main():
    # Prepare data
    X_train, y_train = prepare_data()
    X_val, y_val = prepare_data()
    
    # Initialize Surf-Net
    surf_net = SurfNet()
    surf_net.compile()
    
    # Train model
    history = surf_net.train(X_train, y_train, X_val, y_val)
    
    # Predict on validation data
    predictions = surf_net.predict_dispersion_curves(X_val)

if __name__ == "__main__":
    main()