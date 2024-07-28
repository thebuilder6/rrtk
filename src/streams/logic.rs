// SPDX-License-Identifier: LGPL-3.0-only
/*
Copyright 2024 UxuginPython on GitHub

     This file is part of Rust Robotics ToolKit.

    Rust Robotics ToolKit is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, version 3.

    Rust Robotics ToolKit is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License along with Rust Robotics ToolKit. If not, see <https://www.gnu.org/licenses/>.
*/
//!Logic operations for boolean getters.
use crate::streams::*;
//TODO: make these take arrays of inputs with generic lengths.
//TODO: document these better using that combination table thing
enum AndState {
    DefinitelyFalse, //An input returned false.
    MaybeTrue, //An input returned None and no input has returned false, so we can't assume an
               //output.
    ReturnableTrue, //No input has returned None or false.
}
impl AndState {
    fn none(&mut self) {
        match self {
            AndState::ReturnableTrue => *self = AndState::MaybeTrue,
            _ => (),
        }
    }
}
///Performs an and operation on two boolean getters.
pub struct AndStream<E: Copy + Debug> {
    input1: InputGetter<bool, E>,
    input2: InputGetter<bool, E>,
}
impl<E: Copy + Debug> AndStream<E> {
    ///Constructor for `AndStream`.
    pub fn new(input1: InputGetter<bool, E>, input2: InputGetter<bool, E>) -> Self {
        Self {
            input1: input1,
            input2: input2,
        }
    }
}
impl<E: Copy + Debug> Getter<bool, E> for AndStream<E> {
    fn get(&self) -> Output<bool, E> {
        let gotten1 = self.input1.borrow().get()?;
        let gotten2 = self.input2.borrow().get()?;
        //Never assume the boolean value of a None from an input:
        //To return true, we require that both inputs return true (not None).
        //To return false, we require that at least one input returns false (not None).
        //If neither of these is met, return None.
        let mut time = None;
        let mut and_state = AndState::ReturnableTrue;
        match gotten1 {
            Some(datum) => {
                time = Some(datum.time);
                if !datum.value {
                    and_state = AndState::DefinitelyFalse;
                }
            }
            None => {
                and_state.none();
            }
        }
        match gotten2 {
            Some(datum) => {
                match time {
                    Some(existing) => {
                        if datum.time > existing {
                            time = Some(datum.time);
                        }
                    }
                    None => time = Some(datum.time),
                }
                if !datum.value {
                    and_state = AndState::DefinitelyFalse;
                }
            }
            None => {
                and_state.none();
            }
        }
        let time = match time {
            Some(time) => time,
            None => return Ok(None),
        };
        match and_state {
            AndState::DefinitelyFalse => Ok(Some(Datum::new(time, false))),
            AndState::MaybeTrue => Ok(None),
            AndState::ReturnableTrue => Ok(Some(Datum::new(time, true))),
        }
    }
}
impl<E: Copy + Debug> Updatable<E> for AndStream<E> {
    fn update(&mut self) -> NothingOrError<E> {
        Ok(())
    }
}
