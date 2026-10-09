// Added in happyarts/zopfli.

use alloc::vec::Vec;

use crate::util::{ZOPFLI_MAX_MATCH, ZOPFLI_MIN_MATCH};

/// Every match of a block, found once in a pass of its own, so that the
/// passes over the block need no hashing and no match search. For each
/// position it keeps the distance for every length up to the longest match,
/// as runs of `(last length << 16) | distance`, and where the forward pass
/// skips ahead through a long repetition of one byte.
pub struct MatchCache {
    start: Vec<u32>,
    entries: Vec<u32>,
    run: Vec<bool>,
}

/// The matches of a block as the passes over it read them: block positions
/// from 0, runs as described at `MatchCache`.
pub trait Matches {
    /// Whether the forward pass skips through a long repetition from here.
    fn is_run(&self, pos: usize) -> bool;
    /// The runs at `pos`; lengths may reach past the block's end.
    fn matches(&self, pos: usize) -> &[u32];
    /// The longest match at `pos` within the block and its distance, or `(0, 0)`.
    fn longest(&self, pos: usize) -> (u16, u16);
    /// The distance the match search returns for `length` at `pos`.
    fn dist(&self, pos: usize, length: u16) -> u16 {
        let length = u32::from(length);
        for &e in self.matches(pos) {
            if e >> 16 >= length {
                return e as u16;
            }
        }
        unreachable!("no cached match of length {length}");
    }
}

impl Matches for MatchCache {
    fn is_run(&self, pos: usize) -> bool {
        self.run[pos]
    }

    fn matches(&self, pos: usize) -> &[u32] {
        let end = self
            .start
            .get(pos + 1)
            .map_or(self.entries.len(), |&e| e as usize);
        &self.entries[self.start[pos] as usize..end]
    }

    fn longest(&self, pos: usize) -> (u16, u16) {
        self.matches(pos)
            .last()
            .map_or((0, 0), |&e| ((e >> 16) as u16, e as u16))
    }
}

impl MatchCache {
    /// Whether a range of `size` positions fits: every position has at most one
    /// run per match length (3 to 258), and runs are counted in `u32`.
    pub const fn fits(size: usize) -> bool {
        size <= u32::MAX as usize / (ZOPFLI_MAX_MATCH - ZOPFLI_MIN_MATCH + 1)
    }

    pub fn new(blocksize: usize) -> Self {
        Self {
            start: Vec::with_capacity(blocksize),
            entries: Vec::new(),
            run: vec![false; blocksize],
        }
    }

    /// A cache for `blocksize` positions in the vectors of `self`.
    pub fn reuse(mut self, blocksize: usize) -> Self {
        self.start.clear();
        self.start.reserve(blocksize);
        self.entries.clear();
        self.run.clear();
        self.run.resize(blocksize, false);
        self
    }

    pub fn is_filled(&self) -> bool {
        self.start.len() == self.run.len()
    }

    /// Marks the block position where the forward pass skipped ahead through a
    /// long repetition.
    pub fn mark_run(&mut self, pos: usize) {
        self.run[pos] = true;
    }

    /// Appends the matches of the next position: `sublen[k]` is the distance
    /// for length `k`, up to `length`.
    pub fn push(&mut self, sublen: &[u16], length: usize) {
        self.start
            .push(u32::try_from(self.entries.len()).expect("checked by fits"));
        if length < ZOPFLI_MIN_MATCH {
            return;
        }
        for k in ZOPFLI_MIN_MATCH..=length {
            if k == length || sublen[k] != sublen[k + 1] {
                self.entries.push((k as u32) << 16 | u32::from(sublen[k]));
            }
        }
    }

    /// Appends the next position with matches already in the form of the runs.
    pub fn push_runs(&mut self, runs: &[u32]) {
        self.start
            .push(u32::try_from(self.entries.len()).expect("checked by fits"));
        self.entries.extend_from_slice(runs);
    }

    /// Appends a position with a single match of `length` at `dist`.
    pub fn push_one(&mut self, dist: u16, length: usize) {
        self.start
            .push(u32::try_from(self.entries.len()).expect("checked by fits"));
        self.entries.push((length as u32) << 16 | u32::from(dist));
    }
}
