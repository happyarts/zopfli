use alloc::{vec, vec::Vec};

use crate::util::ZOPFLI_MIN_MATCH;

/// The matches the forward pass of `lz77_optimal` looks at in a block, found
/// once so that its iterations need no match search. For each position it
/// keeps the distance for every length up to the longest match, as runs of
/// `(last length << 16) | distance`, and where the forward pass skips through
/// a long repetition of one byte.
pub struct MatchCache {
    start: Vec<usize>,
    entries: Vec<u32>,
    run: Vec<bool>,
}

impl MatchCache {
    pub fn new(blocksize: usize) -> Self {
        Self {
            start: Vec::with_capacity(blocksize),
            entries: Vec::new(),
            run: vec![false; blocksize],
        }
    }

    pub fn is_filled(&self) -> bool {
        self.start.len() == self.run.len()
    }

    /// Marks the block position where the forward pass skips ahead through a
    /// long repetition.
    pub fn mark_run(&mut self, pos: usize) {
        self.run[pos] = true;
    }

    pub fn is_run(&self, pos: usize) -> bool {
        self.run[pos]
    }

    /// Appends the matches of the next position: `sublen[k]` is the distance
    /// for length `k`, up to `length`.
    pub fn push(&mut self, sublen: &[u16], length: usize) {
        self.start.push(self.entries.len());
        if length < ZOPFLI_MIN_MATCH {
            return;
        }
        for k in ZOPFLI_MIN_MATCH..=length {
            if k == length || sublen[k] != sublen[k + 1] {
                self.entries.push((k as u32) << 16 | u32::from(sublen[k]));
            }
        }
    }

    /// Appends a position with a single match of `length` at `dist`.
    pub fn push_one(&mut self, dist: u16, length: usize) {
        self.start.push(self.entries.len());
        self.entries.push((length as u32) << 16 | u32::from(dist));
    }

    /// The runs of the block position `pos`, see the type.
    pub fn matches(&self, pos: usize) -> &[u32] {
        let end = self.start.get(pos + 1).map_or(self.entries.len(), |&e| e);
        &self.entries[self.start[pos]..end]
    }

    /// The longest match at block position `pos` and its distance, or `(0, 0)`.
    pub fn longest(&self, pos: usize) -> (u16, u16) {
        self.matches(pos)
            .last()
            .map_or((0, 0), |&e| ((e >> 16) as u16, e as u16))
    }

    /// The distance the match search returns for `length` at block position `pos`.
    pub fn dist(&self, pos: usize, length: u16) -> u16 {
        let length = u32::from(length);
        for &e in self.matches(pos) {
            if e >> 16 >= length {
                return e as u16;
            }
        }
        unreachable!("no cached match of length {length}");
    }
}
