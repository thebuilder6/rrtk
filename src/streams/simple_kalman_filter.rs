// SPDX-License-Identifier: LGPL-3.0-only
/*

     This file is part of Rust Robotics ToolKit.

    Rust Robotics ToolKit is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, version 3.

    Rust Robotics ToolKit is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
*/

//!Kalman filter implementation for use with the stream system.

use std::time::SystemTime;

use crate::streams::*;
use nalgebra::{DMatrix, DVector, Dynamic, Matrix, Vector};


/// State vector of the Kalman filter.
#[derive(Clone, Debug, PartialEq)]
pub struct KalmanState {
    /// The estimated state of the system.
    pub x: DVector<f32>,
    /// The estimated error covariance of the system.
    pub p: DMatrix<f32>,
}

/// Defines the Kalman filter parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct KalmanParameters {
    /// The process noise covariance.
    pub q: DVector<f32>,
    /// The measurement noise covariance.
    pub r: DVector<f32>,
    /// The state transition matrix.
    pub a: DMatrix<f32>,
    /// The measurement matrix.
    pub h: DMatrix<f32>,
}

/// A Kalman filter stream for use with the stream system.
pub struct KalmanFilterStream<E: Copy + Debug> {
    input: InputGetter<f32, E>,
    parameters: KalmanParameters,
    state: KalmanState,
}

impl<E: Copy + Debug> KalmanFilterStream<E> {
    /// Constructor for `KalmanFilterStream`.
    pub fn new(input: InputGetter<f32, E>, parameters: KalmanParameters) -> Self {
        Self {
            input: input,
            parameters: parameters,
            state: KalmanState {
                x: DVector::from_element(2, 0.0), // Initialize state vector with zeros
                p: DMatrix::identity(2, 2), // Initialize covariance matrix as identity
            },
        }
    }
}

impl<E: Copy + Debug> Getter<f32, E> for KalmanFilterStream<E> {
    fn get(&self) -> Output<f32, E> {
        Ok(Some(Datum::new(SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_millis() as i64, self.state.x[0]))) // Return the first element of the state vector (estimated position)
    }
}

impl<E: Copy + Debug> Updatable<E> for KalmanFilterStream<E> {
    fn update(&mut self) -> NothingOrError<E> {
        let process = self.input.borrow().get();
        let process = match process {
            Ok(Some(value)) => value,
            Ok(None) => {
                return Ok(());
            }
            Err(error) => return Err(error),
        };

        // Predict
        let predicted_x = self.parameters.a * self.state.x;
        let predicted_p = self.parameters.a * self.state.p * self.parameters.a.transpose() + self.parameters.q;

        // Update
        let s = (self.parameters.h * predicted_p * self.parameters.h.transpose() + self.parameters.r);
        let s_scalar = s[(0, 0)];
        let z_hat = self.parameters.h * predicted_x;
        let k = (predicted_p * self.parameters.h.transpose()) / s_scalar;
        let x_update = predicted_x + k * (process.value as DVector<f32> - z_hat);
        let p_update = (Matrix::identity() - k * self.parameters.h) * predicted_p;

        // Update state
        self.state.x = x_update;
        self.state.p = p_update;

        Ok(())
    }
}

