// SPDX-License-Identifier: LGPL-3.0-only
/*
Copyright 2024 UxuginPython on GitHub

     This file is part of Rust Robotics ToolKit.

    Rust Robotics ToolKit is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, version 3.

    Rust Robotics ToolKit is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License along with Rust Robotics ToolKit. If not, see <https://www.gnu.org/licenses/>.
*/
//!Rust Robotics ToolKit
//!A set of algorithms and other tools for robotics in Rust.
//!It is partially `no_std`. It does not currently integrate with any API directly, but this may be added in the future.
#![warn(missing_docs)]
#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::fmt::Debug;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Not, Sub, SubAssign};
mod datum;
#[cfg(feature = "devices")]
pub mod devices;
mod motion_profile;
mod state;
pub mod streams;
pub use datum::*;
pub use motion_profile::*;
pub use state::*;
///RRTK follows the enum style of error handling. This is the error type returned from nearly all
///RRTK types, but you can add your own custom error type using `Other(O)`. It is strongly
///recommended that you use a single `O` type across your crate.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum Error<O: Copy + Debug> {
    ///Returned when a `None` is elevated to an error by a `NoneToError`.
    FromNone,
    ///A custom error of a user-defined type. Not created by any RRTK type but can be propagated by
    ///them.
    Other(O),
}
///A derivative of position: position, velocity, or acceleration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PositionDerivative {
    ///Where you are.
    Position,
    ///How fast you're going.
    Velocity,
    ///How fast how fast you're going's changing.
    Acceleration,
}
///Coefficients for a PID controller.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PIDKValues {
    ///Proportional coefficient.
    pub kp: f32,
    ///Integral coefficient.
    pub ki: f32,
    ///Derivative coefficient.
    pub kd: f32,
}
impl PIDKValues {
    ///Constructor for `PIDKValues`.
    pub const fn new(kp: f32, ki: f32, kd: f32) -> Self {
        Self {
            kp: kp,
            ki: ki,
            kd: kd,
        }
    }
    ///Calculate the control variable using the coefficients given error, its integral, and its
    ///derivative.
    #[inline]
    pub fn evaluate(&self, error: f32, error_integral: f32, error_derivative: f32) -> f32 {
        self.kp * error + self.ki * error_integral + self.kd * error_derivative
    }
}
///A set of PID k-values for controlling each position derivative.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PositionDerivativeDependentPIDKValues {
    ///Use these k-values when controlling position.
    pub position: PIDKValues,
    ///Use these k-values when controlling velocity.
    pub velocity: PIDKValues,
    ///Use these k-values when controlling acceleration.
    pub acceleration: PIDKValues,
}
impl PositionDerivativeDependentPIDKValues {
    ///Constructor for `PositionDerivativeDependentPIDKValues`.
    pub const fn new(position: PIDKValues, velocity: PIDKValues, acceleration: PIDKValues) -> Self {
        Self {
            position: position,
            velocity: velocity,
            acceleration: acceleration,
        }
    }
    ///Get the k-values for a specific position derivative.
    #[inline]
    pub fn get_k_values(&self, position_derivative: PositionDerivative) -> PIDKValues {
        match position_derivative {
            PositionDerivative::Position => self.position,
            PositionDerivative::Velocity => self.velocity,
            PositionDerivative::Acceleration => self.acceleration,
        }
    }
    ///Calculate the control variable using the coefficients for a given position derivative given
    ///error, its integral, and its derivative.
    #[inline]
    pub fn evaluate(
        &self,
        position_derivative: PositionDerivative,
        error: f32,
        error_integral: f32,
        error_derivative: f32,
    ) -> f32 {
        self.get_k_values(position_derivative)
            .evaluate(error, error_integral, error_derivative)
    }
}
///A generic output type when something may return an error, nothing, or something with a
///timestamp.
pub type Output<T, E> = Result<Option<Datum<T>>, Error<E>>;
///Returned from `TimeGetter` objects, which may return either a time or an error.
pub type TimeOutput<E> = Result<i64, Error<E>>;
///Makes `Getter`s easier to work with by containing them in an `Rc<RefCell<_>>`.
pub type InputGetter<T, E> = Rc<RefCell<dyn Getter<T, E>>>;
///Makes `TimeGetter`s easier to work with by containing them in an `Rc<RefCell<_>>`.
pub type InputTimeGetter<E> = Rc<RefCell<dyn TimeGetter<E>>>;
///Returned when something may return either nothing or an error.
pub type NothingOrError<E> = Result<(), Error<E>>;
///An object for getting the absolute time.
pub trait TimeGetter<E: Copy + Debug>: Updatable<E> {
    ///Get the time.
    fn get(&self) -> TimeOutput<E>;
}
///An object that can return a value, like a `Getter`, for a given time.
pub trait History<T, E: Copy + Debug>: Updatable<E> {
    ///Get a value at a time.
    fn get(&self, time: i64) -> Option<Datum<T>>;
}
///A command for a motor to perform: go to a position, run at a velocity, or accelerate at a rate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Command {
    ///Controls whether you go to a position, run at a velocity, or accelerate at a rate.
    pub position_derivative: PositionDerivative,
    ///The position, velocity, or acceleration rate.
    pub value: f32,
}
impl Command {
    ///Constructor for `Command`.
    pub const fn new(position_derivative: PositionDerivative, value: f32) -> Self {
        Self {
            position_derivative: position_derivative,
            value: value,
        }
    }
    ///Get the commanded constant position if there is one. If `position_derivative` is
    ///`PositionDerivative::Velocity` or `PositionDerivative::Acceleration`, this will return
    ///`None` as there is not a constant position.
    pub fn get_position(&self) -> Option<f32> {
        match self.position_derivative {
            PositionDerivative::Position => Some(self.value),
            _ => None,
        }
    }
    ///Get the commanded constant velocity if there is one. If `position_derivative` is
    ///`PositionDerivative::Acceleration`, this will return `None` as there is not a constant
    ///velocity. If `position_derivative` is `PositionDerivative::Position`, this will return 0 as
    ///velocity should be zero with a constant position.
    pub fn get_velocity(&self) -> Option<f32> {
        match self.position_derivative {
            PositionDerivative::Position => Some(0.0),
            PositionDerivative::Velocity => Some(self.value),
            PositionDerivative::Acceleration => None,
        }
    }
    ///Get the commanded constant acceleration if there is one. If `position_derivative` is not
    ///`PositionDerivative::Acceleration`, this will return `None` as there is not a constant
    ///acceleration.
    pub fn get_acceleration(&self) -> f32 {
        match self.position_derivative {
            PositionDerivative::Acceleration => self.value,
            _ => 0.0,
        }
    }
}
impl From<State> for Command {
    fn from(state: State) -> Self {
        if state.acceleration == 0.0 {
            if state.velocity == 0.0 {
                return Command::new(PositionDerivative::Position, state.position);
            } else {
                return Command::new(PositionDerivative::Velocity, state.velocity);
            }
        } else {
            return Command::new(PositionDerivative::Acceleration, state.acceleration);
        }
    }
}
///Something with an `update` method. Mostly for subtraiting.
pub trait Updatable<E: Copy + Debug> {
    ///As this trait is very generic, exactly what this does will be very dependent on the
    ///implementor.
    fn update(&mut self) -> NothingOrError<E>;
}
///Something with a `get` method. Structs implementing this will often be chained for easier data
///processing, with a struct having other implementors in fields which will have some operation
///performed on their output before it being passed on. Data processing Getters with other Getters
///as fields can be referred to as streams, though this is only in naming and trait-wise there is
///no distinction. The other common use for this trait is encoders. These should not be called
///streams.
pub trait Getter<G, E: Copy + Debug>: Updatable<E> {
    ///Get something.
    fn get(&self) -> Output<G, E>;
}
///Internal data needed for following a `Getter` with a `Settable`.
pub struct SettableData<S, E: Copy + Debug> {
    following: Option<InputGetter<S, E>>,
    last_request: Option<S>,
}
impl<S, E: Copy + Debug> SettableData<S, E> {
    ///Constructor for SettableData.
    pub fn new() -> Self {
        Self {
            following: None,
            last_request: None,
        }
    }
}
///Something with a `set` method. Usually used for motors and other mechanical components and
///systems. This trait too is fairly broad.
pub trait Settable<S: Clone, E: Copy + Debug>: Updatable<E> {
    ///Set something, not updating the internal `SettableData`. Due to current limitations of the
    ///language, you must implement this but call `set`. Do not call this directly as it will make
    ///`get_last_request` work incorrectly.
    fn impl_set(&mut self, value: S) -> NothingOrError<E>;
    ///Set something to a value. For example, this could set a motor to a voltage. You should call
    ///this and not `direct_set`.
    fn set(&mut self, value: S) -> NothingOrError<E> {
        self.impl_set(value.clone())?;
        let data = self.get_settable_data_mut();
        data.last_request = Some(value);
        Ok(())
    }
    ///As traits cannot have fields, get functions and separate types are required. All you have to
    ///do is make a field for a corresponding `SettableData` and make this return an immutable
    ///reference to it.
    fn get_settable_data_ref(&self) -> &SettableData<S, E>;
    ///As traits cannot have fields, get functions and separate types are required. All you have to
    ///do is make a field for a corresponding `SettableData` and make this return a mutable
    ///reference to it.
    fn get_settable_data_mut(&mut self) -> &mut SettableData<S, E>;
    ///Begin following a `Getter` of the same type. For this to work, you must have
    ///`update_following_data` in your `Updatable` implementation.
    fn follow(&mut self, getter: InputGetter<S, E>) {
        let data = self.get_settable_data_mut();
        data.following = Some(getter);
    }
    ///Stop following the `Getter`.
    fn stop_following(&mut self) {
        let data = self.get_settable_data_mut();
        data.following = None;
    }
    ///Get a new value from the `Getter` we're following, if there is one, and call `set`
    ///accordingly. You must add this to your `Updatable` implementation if you are following
    ///`Getter`s. This is a current limitation of the Rust language. If specialization is ever
    ///stabilized, this will hopefully be done in a better way.
    fn update_following_data(&mut self) -> NothingOrError<E> {
        let data = self.get_settable_data_ref();
        match &data.following {
            None => {}
            Some(getter) => {
                let new_value = getter.borrow().get()?;
                match new_value {
                    None => {
                        return Ok(());
                    }
                    Some(datum) => {
                        self.set(datum.value)?;
                    }
                }
            }
        }
        Ok(())
    }
    ///Get the argument from the last time `set` was called.
    fn get_last_request(&self) -> Option<S> {
        let data = self.get_settable_data_ref();
        data.last_request.clone()
    }
}
///A fast way to turn anything implementing `Getter` into an `InputGetter`.
pub fn make_input_getter<T: Getter<G, E> + 'static, G, E: Copy + Debug>(
    getter: T,
) -> InputGetter<G, E> {
    Rc::new(RefCell::new(getter)) as Rc<RefCell<dyn Getter<G, E>>>
}
///A fast way to turn anything implementing `TimeGetter` into an `InputTimeGetter`.
pub fn make_input_time_getter<T: TimeGetter<E> + 'static, E: Copy + Debug>(
    time_getter: T,
) -> InputTimeGetter<E> {
    Rc::new(RefCell::new(time_getter)) as Rc<RefCell<dyn TimeGetter<E>>>
}
///Because `Stream`s always return a timestamp (as long as they don't return `Err(_)` or
///`Ok(None)`), we can use this to treat them like `TimeGetter`s.
pub struct TimeGetterFromGetter<T: Clone, E> {
    elevator: streams::converters::NoneToError<T, E>,
}
impl<T: Clone, E> TimeGetterFromGetter<T, E> {
    ///Constructor for `TimeGetterFromGetter`.
    pub fn new(stream: InputGetter<T, E>) -> Self {
        Self {
            elevator: streams::converters::NoneToError::new(Rc::clone(&stream)),
        }
    }
}
impl<T: Clone, E: Copy + Debug> TimeGetter<E> for TimeGetterFromGetter<T, E> {
    fn get(&self) -> TimeOutput<E> {
        let output = self.elevator.get()?;
        let output = output.expect("`NoneToError` made all `Ok(None)`s into `Err(_)`s, and `?` returned all `Err(_)`s, so we're sure this is now an `Ok(Some(_))`.");
        return Ok(output.time);
    }
}
impl<T: Clone, E: Copy + Debug> Updatable<E> for TimeGetterFromGetter<T, E> {
    fn update(&mut self) -> NothingOrError<E> {
        Ok(())
    }
}
///As histories return values at times, we can ask them to return values at the time of now or now
///with a delta. This makes that much easier and is the recommended way of following
///`MotionProfile`s.
pub struct GetterFromHistory<'a, G, E: Copy + Debug> {
    history: &'a mut dyn History<G, E>,
    time_getter: InputTimeGetter<E>,
    time_delta: i64,
}
impl<'a, G, E: Copy + Debug> GetterFromHistory<'a, G, E> {
    ///Constructor such that the time in the request to the history will be directly that returned
    ///from the `TimeGetter` with no delta.
    pub fn new_no_delta(
        history: &'a mut impl History<G, E>,
        time_getter: InputTimeGetter<E>,
    ) -> Self {
        Self {
            history: history,
            time_getter: time_getter,
            time_delta: 0,
        }
    }
    ///Constructor such that the times requested from the `History` will begin at zero where zero
    ///is the moment this constructor is called.
    pub fn new_start_at_zero(
        history: &'a mut impl History<G, E>,
        time_getter: InputTimeGetter<E>,
    ) -> Result<Self, Error<E>> {
        let time_delta = -time_getter.borrow().get()?;
        Ok(Self {
            history: history,
            time_getter: time_getter,
            time_delta: time_delta,
        })
    }
    ///Constructor such that the times requested from the `History` will start at a given time with
    ///that time defined as the moment of construction.
    pub fn new_custom_start(
        history: &'a mut impl History<G, E>,
        time_getter: InputTimeGetter<E>,
        start: i64,
    ) -> Result<Self, Error<E>> {
        let time_delta = start - time_getter.borrow().get()?;
        Ok(Self {
            history: history,
            time_getter: time_getter,
            time_delta: time_delta,
        })
    }
    ///Constructor with a custom time delta.
    pub fn new_custom_delta(
        history: &'a mut impl History<G, E>,
        time_getter: InputTimeGetter<E>,
        time_delta: i64,
    ) -> Self {
        Self {
            history: history,
            time_getter: time_getter,
            time_delta: time_delta,
        }
    }
    ///Set the time delta.
    pub fn set_delta(&mut self, time_delta: i64) {
        self.time_delta = time_delta;
    }
    ///Define now as a given time in the history. Mostly used when construction and use are far
    ///apart in time.
    pub fn set_time(&mut self, time: i64) -> NothingOrError<E> {
        let time_delta = time - self.time_getter.borrow().get()?;
        self.time_delta = time_delta;
        Ok(())
    }
}
impl<G, E: Copy + Debug> Updatable<E> for GetterFromHistory<'_, G, E> {
    fn update(&mut self) -> NothingOrError<E> {
        self.history.update()?;
        self.time_getter.borrow_mut().update()?;
        Ok(())
    }
}
impl<G: Clone, E: Copy + Debug> Getter<G, E> for GetterFromHistory<'_, G, E> {
    fn get(&self) -> Output<G, E> {
        let time = self.time_getter.borrow().get()?;
        Ok(match self.history.get(time + self.time_delta) {
            Some(datum) => Some(Datum::new(time, datum.value)),
            None => None,
        })
    }
}
///Getter for returning a constant value.
pub struct ConstantGetter<T, E: Copy + Debug> {
    settable_data: SettableData<T, E>,
    time_getter: InputTimeGetter<E>,
    value: T,
}
impl<T, E: Copy + Debug> ConstantGetter<T, E> {
    ///Constructor for `ConstantGetter`.
    pub fn new(time_getter: InputTimeGetter<E>, value: T) -> Self {
        Self {
            settable_data: SettableData::new(),
            time_getter: time_getter,
            value: value,
        }
    }
}
impl<T: Clone, E: Copy + Debug> Getter<T, E> for ConstantGetter<T, E> {
    fn get(&self) -> Output<T, E> {
        let time = self.time_getter.borrow().get()?;
        Ok(Some(Datum::new(time, self.value.clone())))
    }
}
impl<T: Clone, E: Copy + Debug> Settable<T, E> for ConstantGetter<T, E> {
    fn get_settable_data_ref(&self) -> &SettableData<T, E> {
        &self.settable_data
    }
    fn get_settable_data_mut(&mut self) -> &mut SettableData<T, E> {
        &mut self.settable_data
    }
    fn impl_set(&mut self, value: T) -> NothingOrError<E> {
        self.value = value;
        Ok(())
    }
}
impl<T: Clone, E: Copy + Debug> Updatable<E> for ConstantGetter<T, E> {
    ///This does not need to be called.
    fn update(&mut self) -> NothingOrError<E> {
        self.update_following_data()?;
        Ok(())
    }
}
///Getter always returning `Ok(None)`.
pub struct NoneGetter;
impl NoneGetter {
    ///Constructor for `NoneGetter`. Since `NoneGetter` is a unit struct, you can use this or just
    ///the struct's name.
    pub fn new() -> Self {
        Self
    }
}
impl<T, E: Copy + Debug> Getter<T, E> for NoneGetter {
    fn get(&self) -> Output<T, E> {
        Ok(None)
    }
}
impl<E: Copy + Debug> Updatable<E> for NoneGetter {
    fn update(&mut self) -> NothingOrError<E> {
        Ok(())
    }
}
impl<E: Copy + Debug> TimeGetter<E> for i64 {
    fn get(&self) -> TimeOutput<E> {
        Ok(*self)
    }
}
impl<E: Copy + Debug> Updatable<E> for i64 {
    fn update(&mut self) -> NothingOrError<E> {
        Ok(())
    }
}
///A place where a device can connect to another.
#[cfg(feature = "devices")]
pub struct Terminal<'a, E: Copy + Debug> {
    settable_data_state: SettableData<Datum<State>, E>,
    settable_data_command: SettableData<Datum<Command>, E>,
    other: Option<&'a RefCell<Terminal<'a, E>>>,
    no_recurse_set_command: bool,
}
#[cfg(feature = "devices")]
impl<E: Copy + Debug> Terminal<'_, E> {
    ///Direct constructor for a `Terminal`. You almost always actually want `RefCell<Terminal>`
    ///however, in which case you should call `new`, which returns `RefCell<Terminal>`.
    pub fn new_raw() -> Self {
        Self {
            settable_data_state: SettableData::new(),
            settable_data_command: SettableData::new(),
            other: None,
            no_recurse_set_command: false,
        }
    }
    ///This constructs a `RefCell<Terminal>`. This is almost always what you want, and what is
    ///needed for connecting terminals. If you do just want a `Terminal`, use `raw_get` instead.
    pub fn new() -> RefCell<Self> {
        RefCell::new(Self::new_raw())
    }
    ///Disconnect this terminal and the one that it is connected to. You can connect terminals by
    ///calling the `rrtk::connect` function.
    pub fn disconnect(&mut self) {
        debug_assert!(!self.no_recurse_set_command);
        match self.other {
            Some(other) => {
                let mut other = other.borrow_mut();
                other.other = None;
                self.other = None;
            }
            None => (),
        }
    }
}
#[cfg(feature = "devices")]
impl<E: Copy + Debug> Settable<Datum<State>, E> for Terminal<'_, E> {
    fn get_settable_data_ref(&self) -> &SettableData<Datum<State>, E> {
        &self.settable_data_state
    }
    fn get_settable_data_mut(&mut self) -> &mut SettableData<Datum<State>, E> {
        &mut self.settable_data_state
    }
    //SettableData takes care of this for us.
    fn impl_set(&mut self, _state: Datum<State>) -> NothingOrError<E> {
        Ok(())
    }
}
#[cfg(feature = "devices")]
impl<E: Copy + Debug> Settable<Datum<Command>, E> for Terminal<'_, E> {
    fn get_settable_data_ref(&self) -> &SettableData<Datum<Command>, E> {
        &self.settable_data_command
    }
    fn get_settable_data_mut(&mut self) -> &mut SettableData<Datum<Command>, E> {
        &mut self.settable_data_command
    }
    fn impl_set(&mut self, command: Datum<Command>) -> NothingOrError<E> {
        match self.other {
            Some(other) => {
                if !self.no_recurse_set_command {
                    let mut other_borrow = other.borrow_mut();
                    other_borrow.no_recurse_set_command = true;
                    other_borrow.set(command)?;
                    other_borrow.no_recurse_set_command = false;
                }
            }
            None => {}
        }
        Ok(())
    }
}
#[cfg(feature = "devices")]
impl<E: Copy + Debug> Getter<State, E> for Terminal<'_, E> {
    fn get(&self) -> Output<State, E> {
        let mut addends: [core::mem::MaybeUninit<Datum<State>>; 2] =
            [core::mem::MaybeUninit::uninit(); 2];
        let mut addend_count = 0usize;
        match self.get_last_request() {
            Some(state) => {
                addends[0].write(state);
                addend_count += 1;
            }
            None => (),
        }
        match self.other {
            Some(other) => match other.borrow().get_last_request() {
                Some(state) => {
                    addends[addend_count].write(state);
                    addend_count += 1;
                }
                None => (),
            },
            None => (),
        }
        unsafe {
            match addend_count {
                0 => return Ok(None),
                1 => return Ok(Some(addends[0].assume_init())),
                2 => {
                    return Ok(Some(
                        (addends[0].assume_init() + addends[1].assume_init()) / 2.0,
                    ))
                }
                _ => unimplemented!(),
            }
        }
    }
}
#[cfg(feature = "devices")]
impl<E: Copy + Debug> Getter<TerminalData, E> for Terminal<'_, E> {
    fn get(&self) -> Output<TerminalData, E> {
        let command: Option<Datum<Command>> = self.get_last_request();
        let state = self.get().expect("Terminal get cannot return Err");
        let (mut time, command) = match command {
            Some(datum_command) => (Some(datum_command.time), Some(datum_command.value)),
            None => (None, None),
        };
        let state = match state {
            Some(datum_state) => {
                time = Some(datum_state.time);
                Some(datum_state.value)
            }
            None => None,
        };
        Ok(match time {
            Some(time) => Some(Datum::new(
                time,
                TerminalData {
                    time: time,
                    command: command,
                    state: state,
                },
            )),
            None => None,
        })
    }
}
#[cfg(feature = "devices")]
impl<E: Copy + Debug> Updatable<E> for Terminal<'_, E> {
    fn update(&mut self) -> NothingOrError<E> {
        <Terminal<'_, E> as Settable<Datum<Command>, E>>::update_following_data(self)?;
        <Terminal<'_, E> as Settable<Datum<State>, E>>::update_following_data(self)?;
        Ok(())
    }
}
///Connect two terminals. Connected terminals should represent a physical connection between
///mechanical devices. This function will automatically disconnect the specified terminals if they
///are connected. You can manually disconnect terminals by calling the `disconnect` method on
///either of them.
#[cfg(feature = "devices")]
pub fn connect<'a, E: Copy + Debug>(
    term1: &'a RefCell<Terminal<'a, E>>,
    term2: &'a RefCell<Terminal<'a, E>>,
) {
    let mut term1_borrow = term1.borrow_mut();
    let mut term2_borrow = term2.borrow_mut();
    term1_borrow.disconnect();
    term2_borrow.disconnect();
    term1_borrow.other = Some(term2);
    term2_borrow.other = Some(term1);
}
///Data that are sent between terminals: A timestamp, an optional command, and a state.
#[cfg(feature = "devices")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerminalData {
    ///Timestamp.
    pub time: i64,
    ///Optional command from the terminal.
    pub command: Option<Command>,
    ///Optional state from the terminal.
    pub state: Option<State>,
}
#[cfg(feature = "devices")]
impl TryFrom<TerminalData> for Datum<Command> {
    type Error = ();
    fn try_from(value: TerminalData) -> Result<Datum<Command>, ()> {
        match value.command {
            Some(command) => Ok(Datum::new(value.time, command)),
            None => Err(()),
        }
    }
}
#[cfg(feature = "devices")]
impl TryFrom<TerminalData> for Datum<State> {
    type Error = ();
    fn try_from(value: TerminalData) -> Result<Datum<State>, ()> {
        match value.state {
            Some(state) => Ok(Datum::new(value.time, state)),
            None => Err(()),
        }
    }
}
///A mechanical device.
#[cfg(feature = "devices")]
pub trait Device<E: Copy + Debug>: Updatable<E> {
    ///Call only the `update` methods of owned terminals and do not update anything else with the
    ///device.
    fn update_terminals(&mut self) -> NothingOrError<E>;
}
