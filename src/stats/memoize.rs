//! Caching a function's results by argument.

use core::hash::Hash;
use std::collections::HashMap;

/// A function whose results are cached by argument: each argument is evaluated once.
///
/// The function gets a handle for recursive calls as its first parameter, so recursions are
/// memoised as well (the handle is a `dyn FnMut` because a closure cannot name its own type);
/// functions of several arguments take a tuple.
///
/// ```
/// use coretools_rs::stats::Memoizer;
///
/// let mut fibonacci = Memoizer::new(|fibonacci, n: u64| {
///     if n < 2 { n } else { fibonacci(n - 1) + fibonacci(n - 2) }
/// });
/// assert_eq!(fibonacci.call(80), 23_416_728_348_467_685);
/// assert_eq!(fibonacci.len(), 81);
/// ```
pub struct Memoizer<A, R, F> {
    cache: HashMap<A, R>,
    function: F,
}

impl<A, R, F> Memoizer<A, R, F>
where
    A: Eq + Hash + Clone,
    R: Clone,
    F: Fn(&mut dyn FnMut(A) -> R, A) -> R,
{
    pub fn new(function: F) -> Self {
        Self {
            cache: HashMap::new(),
            function,
        }
    }

    /// The function's result for `argument`, evaluated on the first call only.
    pub fn call(&mut self, argument: A) -> R {
        call_cached(&mut self.cache, &self.function, argument)
    }
}

impl<A, R, F> Memoizer<A, R, F> {
    /// The number of cached results.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Forgets all cached results.
    pub fn clear(&mut self) {
        self.cache.clear();
    }
}

fn call_cached<A, R, F>(cache: &mut HashMap<A, R>, function: &F, argument: A) -> R
where
    A: Eq + Hash + Clone,
    R: Clone,
    F: Fn(&mut dyn FnMut(A) -> R, A) -> R,
{
    if let Some(result) = cache.get(&argument) {
        return result.clone();
    }
    let result = function(
        &mut |inner| call_cached(cache, function, inner),
        argument.clone(),
    );
    cache.insert(argument, result.clone());
    result
}

#[cfg(test)]
mod tests {
    use core::cell::Cell;

    use super::*;

    #[test]
    fn caches_one_result_per_argument() {
        let calls = Cell::new(0);
        let mut identity = Memoizer::new(|_, x: i32| {
            calls.set(calls.get() + 1);
            x
        });
        assert_eq!(identity.len(), 0);
        assert_eq!(identity.call(1), 1);
        assert_eq!(identity.len(), 1);
        assert_eq!(identity.call(1), 1);
        assert_eq!(identity.len(), 1);
        assert_eq!(identity.call(2), 2);
        assert_eq!(identity.len(), 2);
        assert_eq!(identity.call(2), 2);
        assert_eq!(identity.len(), 2);
        assert_eq!(calls.get(), 2);

        identity.clear();
        assert!(identity.is_empty());
        assert_eq!(identity.call(1), 1);
        assert_eq!(calls.get(), 3);
    }

    #[test]
    fn several_arguments_are_a_tuple() {
        let mut constant = Memoizer::new(|_, (): ()| 42);
        assert_eq!(constant.call(()), 42);

        // f64 is neither Eq nor Hash: key it by its bits
        let mut product = Memoizer::new(|_, (x, y): (i32, u64)| f64::from(x) * f64::from_bits(y));
        assert_eq!(product.call((2, 3.5_f64.to_bits())), 7.0);

        let mut sum = Memoizer::new(|_, (x, y, z): (i32, i32, i32)| x + y + z);
        assert_eq!(sum.call((1, 2, 3)), 6);
    }

    #[test]
    fn recursive_calls_go_through_the_cache() {
        let mut factorial = Memoizer::new(
            |factorial, n: u64| {
                if n == 0 { 1 } else { n * factorial(n - 1) }
            },
        );
        assert_eq!(factorial.call(5), 120);
        assert_eq!(factorial.len(), 6);
        assert_eq!(factorial.call(3), 6);
        assert_eq!(factorial.len(), 6);
        assert_eq!(factorial.call(10), 3_628_800);
        assert_eq!(factorial.len(), 11);

        let mut fibonacci = Memoizer::new(|fibonacci, n: u64| {
            if n < 2 {
                n
            } else {
                fibonacci(n - 1) + fibonacci(n - 2)
            }
        });
        assert_eq!(fibonacci.call(10), 55);
        assert_eq!(fibonacci.len(), 11);
        assert_eq!(fibonacci.call(0), 0);
        assert_eq!(fibonacci.call(1), 1);
        assert_eq!(fibonacci.len(), 11);
        assert_eq!(fibonacci.call(12), 144);
        assert_eq!(fibonacci.len(), 13);
        // without the cache, this would take about 2^63 calls
        assert_eq!(fibonacci.call(90), 2_880_067_194_370_816_120);
    }
}
