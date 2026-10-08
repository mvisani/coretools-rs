#![doc = include_str!("../README.md")]

pub mod bounded;
mod number;
pub mod probability;
pub mod special;

pub use number::ParseNumberError;

#[cfg(test)]
mod test_util;
