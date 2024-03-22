// SPDX-License-Identifier: LGPL-3.0-only
/*
Copyright 2024 UxuginPython on GitHub

     This file is part of Rust Robotics ToolKit.

    Rust Robotics ToolKit is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, version 3.

    Rust Robotics ToolKit is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License along with Rust Robotics ToolKit. If not, see <https://www.gnu.org/licenses/>.
*/
use rrtk::*;
#[test]
fn encoder() {
    struct DummyEncoder {}
    impl DummyEncoder {
        fn new() -> DummyEncoder {
            DummyEncoder {}
        }
    }
    impl Encoder for DummyEncoder {
        fn get_state(&mut self) -> Datum<State> {
            Datum::new(1.0, State::new(2.0, 3.0, 4.0))
        }
    }
    let mut my_encoder = DummyEncoder::new();
    let output = my_encoder.get_state();
    assert_eq!(output.time, 1.0);
    assert_eq!(output.value.position, 2.0);
    assert_eq!(output.value.velocity, 3.0);
    assert_eq!(output.value.acceleration, 4.0);
}
#[test]
fn simple_encoder_position() {
    struct DummySimpleEncoder {
        simple_encoder_data: SimpleEncoderData,
        time: f32,
        pos: f32,
    }
    impl DummySimpleEncoder {
        fn new(start_state: Datum<State>) -> DummySimpleEncoder {
            DummySimpleEncoder {
                simple_encoder_data: SimpleEncoderData::new(MotorMode::POSITION, start_state.clone()),
                time: start_state.time,
                pos: start_state.value.position,
            }
        }
    }
    impl SimpleEncoder for DummySimpleEncoder {
        fn get_simple_encoder_data_ref(&self) -> &SimpleEncoderData {
            &self.simple_encoder_data
        }
        fn get_simple_encoder_data_mut(&mut self) -> &mut SimpleEncoderData {
            &mut self.simple_encoder_data
        }
        fn device_update(&mut self) -> Datum<f32> {
            self.time += 0.1;
            self.pos += 2.0;
            Datum::new(self.time, self.pos)
        }
    }
    let mut my_simple_encoder = DummySimpleEncoder::new(Datum::new(1.0, State::new(2.0, 3.0, 4.0)));
    let output = my_simple_encoder.get_state();
    assert_eq!(output.time, 1.0);
    assert_eq!(output.value.position, 2.0);
    assert_eq!(output.value.velocity, 3.0);
    assert_eq!(output.value.acceleration, 4.0);
    my_simple_encoder.update();
    let output = my_simple_encoder.get_state();
    assert_eq!(output.time, 1.1);
    assert_eq!(output.value.position, 4.0);
    //floating point errors
    assert!(19.999 < output.value.velocity && output.value.velocity < 20.001);
    assert!(169.999 < output.value.acceleration && output.value.acceleration < 170.001);
}
#[test]
fn simple_encoder_velocity() {
    struct DummySimpleEncoder {
        simple_encoder_data: SimpleEncoderData,
        time: f32,
        vel: f32,
    }
    impl DummySimpleEncoder {
        fn new(start_state: Datum<State>) -> DummySimpleEncoder {
            DummySimpleEncoder {
                simple_encoder_data: SimpleEncoderData::new(MotorMode::VELOCITY, start_state.clone()),
                time: start_state.time,
                vel: start_state.value.velocity,
            }
        }
    }
    impl SimpleEncoder for DummySimpleEncoder {
        fn get_simple_encoder_data_ref(&self) -> &SimpleEncoderData {
            &self.simple_encoder_data
        }
        fn get_simple_encoder_data_mut(&mut self) -> &mut SimpleEncoderData {
            &mut self.simple_encoder_data
        }
        fn device_update(&mut self) -> Datum<f32> {
            self.time += 0.1;
            self.vel += 2.0;
            Datum::new(self.time, self.vel)
        }
    }
    let mut my_simple_encoder = DummySimpleEncoder::new(Datum::new(1.0, State::new(2.0, 3.0, 4.0)));
    let output = my_simple_encoder.get_state();
    assert_eq!(output.time, 1.0);
    assert_eq!(output.value.position, 2.0);
    assert_eq!(output.value.velocity, 3.0);
    assert_eq!(output.value.acceleration, 4.0);
    my_simple_encoder.update();
    let output = my_simple_encoder.get_state();
    assert_eq!(output.time, 1.1);
    assert_eq!(output.value.position, 2.4);
    assert_eq!(output.value.velocity, 5.0);
    //floating point errors
    assert!(19.999 < output.value.acceleration && output.value.acceleration < 20.001);
}
#[test]
fn simple_encoder_acceleration() {
    struct DummySimpleEncoder {
        simple_encoder_data: SimpleEncoderData,
        time: f32,
        acc: f32,
    }
    impl DummySimpleEncoder {
        fn new(start_state: Datum<State>) -> DummySimpleEncoder {
            DummySimpleEncoder {
                simple_encoder_data: SimpleEncoderData::new(MotorMode::ACCELERATION, start_state.clone()),
                time: start_state.time,
                acc: start_state.value.acceleration,
            }
        }
    }
    impl SimpleEncoder for DummySimpleEncoder {
        fn get_simple_encoder_data_ref(&self) -> &SimpleEncoderData {
            &self.simple_encoder_data
        }
        fn get_simple_encoder_data_mut(&mut self) -> &mut SimpleEncoderData {
            &mut self.simple_encoder_data
        }
        fn device_update(&mut self) -> Datum<f32> {
            self.time += 0.1;
            self.acc += 2.0;
            Datum::new(self.time, self.acc)
        }
    }
    let mut my_simple_encoder = DummySimpleEncoder::new(Datum::new(1.0, State::new(2.0, 3.0, 4.0)));
    let output = my_simple_encoder.get_state();
    assert_eq!(output.time, 1.0);
    assert_eq!(output.value.position, 2.0);
    assert_eq!(output.value.velocity, 3.0);
    assert_eq!(output.value.acceleration, 4.0);
    my_simple_encoder.update();
    let output = my_simple_encoder.get_state();
    assert_eq!(output.time, 1.1);
    assert_eq!(output.value.position, 2.325);
    assert_eq!(output.value.velocity, 3.5);
    assert_eq!(output.value.acceleration, 6.0);
}
#[test]
fn feedback_motor() {
    struct DummyFeedbackMotor {
        time: f32,
        pos: f32,
        vel: f32,
        acc: f32,
    }
    impl DummyFeedbackMotor {
        fn new(start_state: Datum<State>) -> DummyFeedbackMotor {
            DummyFeedbackMotor {
                time: start_state.time,
                pos: start_state.value.position,
                vel: start_state.value.velocity,
                acc: start_state.value.acceleration,
            }
        }
    }
    impl FeedbackMotor for DummyFeedbackMotor {
        fn get_state(&mut self) -> Datum<State> {
            Datum::new(self.time, State::new(self.pos, self.vel, self.acc))
        }
        fn set_acceleration(&mut self, acceleration: f32) {
            self.time += 1.0;
            self.acc = acceleration;
        }
        fn set_velocity(&mut self, velocity: f32) {
            self.time += 1.0;
            self.acc = 0.0;
            self.vel = velocity;
        }
        fn set_position(&mut self, position: f32) {
            self.time += 1.0;
            self.acc = 0.0;
            self.vel = 0.0;
            self.pos = position;
        }
    }
    let mut my_feedback_motor = DummyFeedbackMotor::new(Datum::new(1.0, State::new(2.0, 3.0, 4.0)));
    let output = my_feedback_motor.get_state();
    assert_eq!(output.time, 1.0);
    assert_eq!(output.value.position, 2.0);
    assert_eq!(output.value.velocity, 3.0);
    assert_eq!(output.value.acceleration, 4.0);
    my_feedback_motor.set_acceleration(5.0);
    let output = my_feedback_motor.get_state();
    assert_eq!(output.time, 2.0);
    assert_eq!(output.value.position, 2.0);
    assert_eq!(output.value.velocity, 3.0);
    assert_eq!(output.value.acceleration, 5.0);
    my_feedback_motor.set_velocity(6.0);
    let output = my_feedback_motor.get_state();
    assert_eq!(output.time, 3.0);
    assert_eq!(output.value.position, 2.0);
    assert_eq!(output.value.velocity, 6.0);
    assert_eq!(output.value.acceleration, 0.0);
    my_feedback_motor.set_position(7.0);
    let output = my_feedback_motor.get_state();
    assert_eq!(output.time, 4.0);
    assert_eq!(output.value.position, 7.0);
    assert_eq!(output.value.velocity, 0.0);
    assert_eq!(output.value.acceleration, 0.0);
}
#[test]
fn servo_motor() {
    struct DummyServoMotor {
        servo_motor_data: ServoMotorData,
        time: f32,
        pos: f32,
        vel: f32,
        acc: f32,
    }
    impl DummyServoMotor {
        fn new(start_state: Datum<State>) -> DummyServoMotor {
            DummyServoMotor {
                servo_motor_data: ServoMotorData::new(start_state.clone()),
                time: start_state.time,
                pos: start_state.value.position,
                vel: start_state.value.velocity,
                acc: start_state.value.acceleration,
            }
        }
    }
    impl ServoMotor for DummyServoMotor {
        fn get_servo_motor_data_ref(&self) -> &ServoMotorData {
            &self.servo_motor_data
        }
        fn get_servo_motor_data_mut(&mut self) -> &mut ServoMotorData {
            &mut self.servo_motor_data
        }
        fn device_get_time(&mut self) -> f32 {
            self.time
        }
        fn device_set_acceleration(&mut self, acceleration: f32) {
            self.time += 1.0;
            self.acc = acceleration;
        }
        fn device_set_velocity(&mut self, velocity: f32) {
            self.time += 1.0;
            self.acc = 0.0;
            self.vel = velocity;
        }
        fn device_set_position(&mut self, position: f32) {
            self.time += 1.0;
            self.acc = 0.0;
            self.vel = 0.0;
            self.pos = position;
        }
    }
    let mut my_servo_motor = DummyServoMotor::new(Datum::new(1.0, State::new(2.0, 3.0, 4.0)));
    let output = my_servo_motor.get_state();
    assert_eq!(output.time, 1.0);
    assert_eq!(output.value.position, 2.0);
    assert_eq!(output.value.velocity, 3.0);
    assert_eq!(output.value.acceleration, 4.0);
    my_servo_motor.set_acceleration(5.0);
    let output = my_servo_motor.get_state();
    assert_eq!(output.time, 2.0);
    assert_eq!(output.value.position, 2.0);
    assert_eq!(output.value.velocity, 3.0);
    assert_eq!(output.value.acceleration, 5.0);
    my_servo_motor.set_velocity(6.0);
    let output = my_servo_motor.get_state();
    assert_eq!(output.time, 3.0);
    assert_eq!(output.value.position, 2.0);
    assert_eq!(output.value.velocity, 6.0);
    assert_eq!(output.value.acceleration, 0.0);
    my_servo_motor.set_position(7.0);
    let output = my_servo_motor.get_state();
    assert_eq!(output.time, 4.0);
    assert_eq!(output.value.position, 7.0);
    assert_eq!(output.value.velocity, 0.0);
    assert_eq!(output.value.acceleration, 0.0);
}
