use core::num::{ParseFloatError, ParseIntError};

use thiserror::Error;

/// Why a string is not a number of the expected type.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ParseNumberError {
    #[error(transparent)]
    Float(#[from] ParseFloatError),

    #[error(transparent)]
    Int(#[from] ParseIntError),
}
