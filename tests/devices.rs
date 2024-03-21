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
            self.vel+= 2.0;
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
