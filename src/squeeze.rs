// Modified in happyarts/zopfli (see its commit history).

//! The squeeze functions do enhanced LZ77 compression by optimal parsing with a
//! cost model, rather than greedily choosing the longest length or using a single
//! step of lazy matching like regular implementations.
//!
//! Since the cost model is based on the Huffman tree that can only be calculated
//! after the LZ77 data is generated, there is a chicken and egg problem, and
//! multiple runs are done with updated cost models to converge to a better
//! solution.

use alloc::vec::Vec;
use core::cmp;

#[cfg(feature = "std")]
use log::{debug, trace};

use crate::{
    bintree::BinaryTree,
    cache::{Cache, NoCache},
    deflate::{calculate_block_size, optimize_huffman_for_rle, BlockType},
    hash::ZopfliHash,
    katajainen::length_limited_code_lengths,
    lz77::{
        find_longest_match, find_longest_match_loop, get_length_score, verify_len_dist, LitLen,
        Lz77Store,
    },
    matches::{MatchCache, Matches},
    symbols::{get_dist_extra_bits, get_dist_symbol, get_length_extra_bits, get_length_symbol},
    util::{
        ZOPFLI_MAX_MATCH, ZOPFLI_MIN_MATCH, ZOPFLI_NUM_D, ZOPFLI_NUM_LL, ZOPFLI_WINDOW_MASK,
        ZOPFLI_WINDOW_SIZE,
    },
    Options,
};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)] // False-positive
use crate::math::F64MathExt;

/// Cost model which should exactly match fixed tree.
fn get_cost_fixed(litlen: usize, dist: u16) -> f64 {
    let result = if dist == 0 {
        if litlen <= 143 {
            8
        } else {
            9
        }
    } else {
        let dbits = get_dist_extra_bits(dist);
        let lbits = get_length_extra_bits(litlen);
        let lsym = get_length_symbol(litlen);
        // Every dist symbol has length 5.
        7 + u32::from(lsym > 279) + 5 + dbits + lbits
    };
    f64::from(result)
}

/// Cost model based on symbol statistics.
fn get_cost_stat(litlen: usize, dist: u16, stats: &SymbolStats) -> f64 {
    assert!(litlen < ZOPFLI_NUM_LL); // Eases inlining and gets rid of index bound checks below
    if dist == 0 {
        stats.ll_symbols[litlen]
    } else {
        let lsym = get_length_symbol(litlen);
        let lbits = f64::from(get_length_extra_bits(litlen));
        let dsym = get_dist_symbol(dist) as usize;
        let dbits = f64::from(get_dist_extra_bits(dist));
        lbits + dbits + stats.ll_symbols[lsym] + stats.d_symbols[dsym]
    }
}

#[derive(Default)]
struct RanState {
    m_w: u32,
    m_z: u32,
}

impl RanState {
    const fn new() -> Self {
        Self { m_w: 1, m_z: 2 }
    }

    /// Get random number: "Multiply-With-Carry" generator of G. Marsaglia
    fn random_marsaglia(&mut self) -> u32 {
        self.m_z = 36969 * (self.m_z & 65535) + (self.m_z >> 16);
        self.m_w = 18000 * (self.m_w & 65535) + (self.m_w >> 16);
        (self.m_z << 16).wrapping_add(self.m_w) // 32-bit result.
    }
}

#[derive(Copy, Clone)]
struct SymbolStats {
    /* The literal and length symbols. */
    litlens: [usize; ZOPFLI_NUM_LL],
    /* The 32 unique dist symbols, not the 32768 possible dists. */
    dists: [usize; ZOPFLI_NUM_D],

    /* Length of each lit/len symbol in bits. */
    ll_symbols: [f64; ZOPFLI_NUM_LL],
    /* Length of each dist symbol in bits. */
    d_symbols: [f64; ZOPFLI_NUM_D],
}

impl Default for SymbolStats {
    fn default() -> Self {
        Self {
            litlens: [0; ZOPFLI_NUM_LL],
            dists: [0; ZOPFLI_NUM_D],
            ll_symbols: [0.0; ZOPFLI_NUM_LL],
            d_symbols: [0.0; ZOPFLI_NUM_D],
        }
    }
}

impl SymbolStats {
    fn randomize_stat_freqs(&mut self, state: &mut RanState) {
        fn randomize_freqs(freqs: &mut [usize], state: &mut RanState) {
            let n = freqs.len();
            let mut i = 0;
            let end = n;

            while i < end {
                if (state.random_marsaglia() >> 4).is_multiple_of(3) {
                    let index = state.random_marsaglia() as usize % n;
                    freqs[i] = freqs[index];
                }
                i += 1;
            }
        }
        randomize_freqs(&mut self.litlens, state);
        randomize_freqs(&mut self.dists, state);
        self.litlens[256] = 1; // End symbol.
    }

    /// Calculates the entropy of each symbol, based on the counts of each symbol. The
    /// result is similar to the result of `length_limited_code_lengths`, but with the
    /// actual theoretical bit lengths according to the entropy. Since the resulting
    /// values are fractional, they cannot be used to encode the tree specified by
    /// DEFLATE.
    fn calculate_entropy(&mut self) {
        fn calculate_and_store_entropy(count: &[usize], bitlengths: &mut [f64]) {
            let n = count.len();

            let sum = count.iter().sum();

            let log2sum = (if sum == 0 { n } else { sum } as f64).log2();

            for i in 0..n {
                // When the count of the symbol is 0, but its cost is requested anyway, it
                // means the symbol will appear at least once anyway, so give it the cost as if
                // its count is 1.
                if count[i] == 0 {
                    bitlengths[i] = log2sum;
                } else {
                    bitlengths[i] = log2sum - (count[i] as f64).log2();
                }
            }
        }

        calculate_and_store_entropy(&self.litlens, &mut self.ll_symbols);
        calculate_and_store_entropy(&self.dists, &mut self.d_symbols);
    }

    /// Appends the symbol statistics from the store.
    fn get_statistics(&mut self, store: &Lz77Store) {
        for &litlen in &store.litlens {
            match litlen {
                LitLen::Literal(lit) => self.litlens[lit as usize] += 1,
                LitLen::LengthDist(len, dist) => {
                    self.litlens[get_length_symbol(len as usize)] += 1;
                    self.dists[get_dist_symbol(dist) as usize] += 1;
                }
            }
        }
        self.litlens[256] = 1; /* End symbol. */

        self.calculate_entropy();
    }

    /// Sets the costs to the code lengths a dynamic block would use for these
    /// counts (after the RLE adjustment of the counts). A symbol without a code
    /// gets the longest code length, 15 bits: using it would need a new code.
    fn use_code_lengths(&mut self, from: &Self) {
        let mut litlens = from.litlens;
        let mut dists = from.dists;
        optimize_huffman_for_rle(&mut dists);
        optimize_huffman_for_rle(&mut litlens);
        for (cost, bits) in self
            .ll_symbols
            .iter_mut()
            .zip(length_limited_code_lengths(&litlens, 15))
        {
            *cost = if bits == 0 { 15.0 } else { f64::from(bits) };
        }
        for (cost, bits) in self
            .d_symbols
            .iter_mut()
            .zip(length_limited_code_lengths(&dists, 15))
        {
            *cost = if bits == 0 { 15.0 } else { f64::from(bits) };
        }
    }

    fn clear_freqs(&mut self) {
        self.litlens = [0; ZOPFLI_NUM_LL];
        self.dists = [0; ZOPFLI_NUM_D];
    }
}

fn add_weighed_stat_freqs(
    stats1: &SymbolStats,
    w1: f64,
    stats2: &SymbolStats,
    w2: f64,
) -> SymbolStats {
    let mut result = SymbolStats::default();

    for i in 0..ZOPFLI_NUM_LL {
        result.litlens[i] =
            (stats1.litlens[i] as f64 * w1 + stats2.litlens[i] as f64 * w2) as usize;
    }
    for i in 0..ZOPFLI_NUM_D {
        result.dists[i] = (stats1.dists[i] as f64 * w1 + stats2.dists[i] as f64 * w2) as usize;
    }
    result.litlens[256] = 1; // End symbol.
    result
}

// Table of distances that have a different distance symbol in the deflate
// specification. Each value is the first distance that has a new symbol. Only
// different symbols affect the cost model so only these need to be checked.
// See RFC 1951 section 3.2.5. Compressed blocks (length and distance codes).
const DSYMBOLS: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];

/// Finds the minimum possible cost this cost model can return for valid length and
/// distance symbols.
fn get_cost_model_min_cost<F: Fn(usize, u16) -> f64>(costmodel: F) -> f64 {
    let mut bestlength = 0; // length that has lowest cost in the cost model
    let mut bestdist = 0; // distance that has lowest cost in the cost model

    // Table of distances that have a different distance symbol in the deflate
    // specification. Each value is the first distance that has a new symbol. Only
    // different symbols affect the cost model so only these need to be checked.
    // See RFC 1951 section 3.2.5. Compressed blocks (length and distance codes).

    let mut mincost = f64::INFINITY;
    for i in 3..259 {
        let c = costmodel(i, 1);
        if c < mincost {
            bestlength = i;
            mincost = c;
        }
    }

    mincost = f64::INFINITY;
    for dsym in DSYMBOLS {
        let c = costmodel(3, dsym);
        if c < mincost {
            bestdist = dsym;
            mincost = c;
        }
    }
    costmodel(bestlength, bestdist)
}

/// Performs the forward pass for "squeeze". Gets the most optimal length to reach
/// every byte from a previous byte, using cost calculations.
/// `s`: the `ZopfliBlockState`
/// `in_data`: the input data array
/// `instart`: where to start
/// `inend`: where to stop (not inclusive)
/// `costmodel`: function to calculate the cost of some lit/len/dist pair.
/// `length_array`: output array of size `(inend - instart)` which will receive the best
///     length to reach this byte from a previous byte.
/// returns the cost that was, according to the `costmodel`, needed to get to the end.
fn get_best_lengths<F: Fn(usize, u16) -> f64, C: Cache>(
    lmc: &mut C,
    in_data: &[u8],
    instart: usize,
    inend: usize,
    costmodel: F,
    h: &mut ZopfliHash,
    costs: &mut Vec<f32>,
) -> (f64, Vec<u16>) {
    // Best cost to get here so far.
    let blocksize = inend - instart;
    let mut length_array = vec![0; blocksize + 1];
    if instart == inend {
        return (0.0, length_array);
    }
    let windowstart = instart.saturating_sub(ZOPFLI_WINDOW_SIZE);

    h.reset();
    let arr = &in_data[..inend];
    h.warmup(arr, windowstart, inend);
    for i in windowstart..instart {
        h.update(arr, i);
    }

    costs.resize(blocksize + 1, 0.0);
    for cost in costs.iter_mut().take(blocksize + 1).skip(1) {
        *cost = f32::INFINITY;
    }
    costs[0] = 0.0; /* Because it's the start. */

    let mut i = instart;
    let mut leng;
    let mut longest_match;
    let mut sublen = vec![0; ZOPFLI_MAX_MATCH + 1];
    let mincost = get_cost_model_min_cost(&costmodel);
    while i < inend {
        let mut j = i - instart; // Index in the costs array and length_array.
        h.update(arr, i);

        // If we're in a long repetition of the same character and have more than
        // ZOPFLI_MAX_MATCH characters before and after our position.
        if h.same[i & ZOPFLI_WINDOW_MASK] > ZOPFLI_MAX_MATCH as u16 * 2
            && i > instart + ZOPFLI_MAX_MATCH + 1
            && i + ZOPFLI_MAX_MATCH * 2 + 1 < inend
            && h.same[(i - ZOPFLI_MAX_MATCH) & ZOPFLI_WINDOW_MASK] > ZOPFLI_MAX_MATCH as u16
        {
            let symbolcost = costmodel(ZOPFLI_MAX_MATCH, 1);
            // Set the length to reach each one to ZOPFLI_MAX_MATCH, and the cost to
            // the cost corresponding to that length. Doing this, we skip
            // ZOPFLI_MAX_MATCH values to avoid calling ZopfliFindLongestMatch.

            for _ in 0..ZOPFLI_MAX_MATCH {
                costs[j + ZOPFLI_MAX_MATCH] = costs[j] + symbolcost as f32;
                length_array[j + ZOPFLI_MAX_MATCH] = ZOPFLI_MAX_MATCH as u16;
                i += 1;
                j += 1;
                h.update(arr, i);
            }
        }

        longest_match = find_longest_match(
            lmc,
            h,
            arr,
            i,
            inend,
            instart,
            ZOPFLI_MAX_MATCH,
            &mut Some(&mut sublen),
        );
        leng = longest_match.length;

        // Literal.
        if i < inend {
            let new_cost = costmodel(arr[i] as usize, 0) + f64::from(costs[j]);
            debug_assert!(new_cost >= 0.0);
            if new_cost < f64::from(costs[j + 1]) {
                costs[j + 1] = new_cost as f32;
                length_array[j + 1] = 1;
            }
        }
        // Lengths.
        let kend = cmp::min(leng as usize, inend - i);
        let mincostaddcostj = mincost + f64::from(costs[j]);

        for (k, &sublength) in sublen.iter().enumerate().take(kend + 1).skip(3) {
            // Calling the cost model is expensive, avoid this if we are already at
            // the minimum possible cost that it can return.
            if f64::from(costs[j + k]) <= mincostaddcostj {
                continue;
            }

            let new_cost = costmodel(k, sublength) + f64::from(costs[j]);
            debug_assert!(new_cost >= 0.0);
            if new_cost < f64::from(costs[j + k]) {
                debug_assert!(k <= ZOPFLI_MAX_MATCH);
                costs[j + k] = new_cost as f32;
                length_array[j + k] = k as u16;
            }
        }
        i += 1;
    }

    debug_assert!(costs[blocksize] >= 0.0);
    (f64::from(costs[blocksize]), length_array)
}

/// Finds the matches `get_best_lengths` would look at in the block, with the
/// same search, and keeps them.
fn fill_match_cache(
    in_data: &[u8],
    instart: usize,
    inend: usize,
    h: &mut ZopfliHash,
) -> MatchCache {
    let mut m = MatchCache::new(inend - instart);
    if instart == inend {
        return m;
    }
    let windowstart = instart.saturating_sub(ZOPFLI_WINDOW_SIZE);
    h.reset();
    let arr = &in_data[..inend];
    h.warmup(arr, windowstart, inend);
    for i in windowstart..instart {
        h.update(arr, i);
    }

    let mut i = instart;
    let mut sublen = vec![0; ZOPFLI_MAX_MATCH + 1];
    while i < inend {
        h.update(arr, i);

        // The same test for a long repetition of one byte as in `get_best_lengths`.
        if h.same[i & ZOPFLI_WINDOW_MASK] > ZOPFLI_MAX_MATCH as u16 * 2
            && i > instart + ZOPFLI_MAX_MATCH + 1
            && i + ZOPFLI_MAX_MATCH * 2 + 1 < inend
            && h.same[(i - ZOPFLI_MAX_MATCH) & ZOPFLI_WINDOW_MASK] > ZOPFLI_MAX_MATCH as u16
        {
            m.mark_run(i - instart);
            for _ in 0..ZOPFLI_MAX_MATCH {
                let (dist, length) =
                    find_longest_match_loop(h, arr, i, inend, ZOPFLI_MAX_MATCH, &mut None);
                debug_assert_eq!(length as usize, ZOPFLI_MAX_MATCH);
                m.push_one(dist, length as usize);
                i += 1;
                h.update(arr, i);
            }
        }

        let longest_match = find_longest_match(
            &mut NoCache,
            h,
            arr,
            i,
            inend,
            instart,
            ZOPFLI_MAX_MATCH,
            &mut Some(&mut sublen),
        );
        m.push(&sublen, longest_match.length as usize);
        i += 1;
    }
    debug_assert!(m.is_filled());
    m
}

/// Where `get_best_lengths` skips through a long repetition of one byte, found
/// from the data alone.
struct LongRuns {
    instart: usize,
    inend: usize,
    runstart: usize,
    /// How many bytes after each position repeat its byte, from
    /// `ZOPFLI_MAX_MATCH` before the block on.
    same: Vec<u16>,
}

impl LongRuns {
    /// The runs of `arr[instart..inend]`, in the vector `same`.
    fn new_in(mut same: Vec<u16>, arr: &[u8], instart: usize, inend: usize) -> Self {
        let runstart = instart.saturating_sub(ZOPFLI_MAX_MATCH);
        same.clear();
        same.resize(inend - runstart, 0);
        for i in (runstart..inend.saturating_sub(1)).rev() {
            if arr[i] == arr[i + 1] {
                same[i - runstart] = same[i + 1 - runstart].saturating_add(1);
            }
        }
        Self {
            instart,
            inend,
            runstart,
            same,
        }
    }

    /// Whether the byte before `i` and the bytes from `i` up to the longest
    /// match there are all the same (see `BinaryTree::insert`).
    fn repeats_before(&self, i: usize) -> bool {
        let limit = cmp::min(ZOPFLI_MAX_MATCH, self.inend - i);
        i > self.runstart && usize::from(self.same[i - 1 - self.runstart]) >= limit
    }

    /// The same test as in `get_best_lengths`; then the `ZOPFLI_MAX_MATCH`
    /// positions from `i` on all have a match of that length at distance 1.
    fn starts_at(&self, i: usize) -> bool {
        self.starts_within(i, self.instart, self.inend)
    }

    /// `starts_at` for the block `instart..inend` within these runs' range.
    fn starts_within(&self, i: usize, instart: usize, inend: usize) -> bool {
        self.same[i - self.runstart] > ZOPFLI_MAX_MATCH as u16 * 2
            && i > instart + ZOPFLI_MAX_MATCH + 1
            && i + ZOPFLI_MAX_MATCH * 2 + 1 < inend
            && self.same[i - ZOPFLI_MAX_MATCH - self.runstart] > ZOPFLI_MAX_MATCH as u16
    }
}

/// The matches of a chunk, from `fill_match_cache_tree`, with the long
/// repetitions in it.
pub struct ChunkMatches {
    cache: MatchCache,
    runs: LongRuns,
    tree: BinaryTree,
}

#[cfg(feature = "std")]
std::thread_local! {
    /// The vectors of the last chunk's matches on this thread, for the next chunk.
    static SPARE: core::cell::RefCell<Option<ChunkMatches>> = const { core::cell::RefCell::new(None) };
}

impl ChunkMatches {
    /// Keeps the vectors for the next `fill_match_cache_tree` on this thread.
    pub fn recycle(self) {
        #[cfg(feature = "std")]
        SPARE.with(|spare| *spare.borrow_mut() = Some(self));
    }

    /// Where the chunk starts in the input.
    pub fn start(&self) -> usize {
        self.runs.instart
    }

    /// The chunk's matches, chunk positions from 0.
    pub fn cache(&self) -> &MatchCache {
        &self.cache
    }

    /// The block `instart..inend` of the chunk, as `lz77_optimal` reads it:
    /// long repetitions as for the block alone, matches cut at its end.
    pub fn block(&self, instart: usize, inend: usize) -> ChunkView<'_> {
        ChunkView {
            chunk: self,
            instart,
            inend,
        }
    }
}

/// A block of a `ChunkMatches`, see `ChunkMatches::block`.
pub struct ChunkView<'a> {
    chunk: &'a ChunkMatches,
    instart: usize,
    inend: usize,
}

impl Matches for ChunkView<'_> {
    fn is_run(&self, pos: usize) -> bool {
        // The run lengths of the chunk give the block's test: its thresholds
        // lie below where the block's end would cut them.
        self.chunk
            .runs
            .starts_within(self.instart + pos, self.instart, self.inend)
    }

    fn matches(&self, pos: usize) -> &[u32] {
        self.chunk
            .cache
            .matches(self.instart - self.chunk.start() + pos)
    }

    fn longest(&self, pos: usize) -> (u16, u16) {
        let room = (self.inend - self.instart - pos) as u32;
        if room < ZOPFLI_MIN_MATCH as u32 {
            return (0, 0);
        }
        let mut longest = (0, 0);
        for &e in self.matches(pos) {
            longest = (cmp::min(e >> 16, room) as u16, e as u16);
            if e >> 16 >= room {
                break;
            }
        }
        longest
    }
}

/// Fills a match cache with a binary tree match finder; skips through long
/// repetitions of one byte like `fill_match_cache`, with matches at distance 1.
pub fn fill_match_cache_tree(in_data: &[u8], instart: usize, inend: usize) -> ChunkMatches {
    #[cfg(feature = "std")]
    let spare = SPARE.with(|spare| spare.borrow_mut().take());
    #[cfg(not(feature = "std"))]
    let spare: Option<ChunkMatches> = None;
    let windowstart = instart.saturating_sub(ZOPFLI_WINDOW_SIZE);
    let (mut m, same, mut tree) = match spare {
        Some(c) => (
            c.cache.reuse(inend - instart),
            c.runs.same,
            c.tree.reuse(windowstart),
        ),
        None => (
            MatchCache::new(inend - instart),
            Vec::new(),
            BinaryTree::new(windowstart),
        ),
    };
    if instart == inend {
        return ChunkMatches {
            cache: m,
            runs: LongRuns::new_in(same, in_data, instart, inend),
            tree,
        };
    }
    let arr = &in_data[..inend];
    for i in windowstart..instart {
        tree.insert(arr, i, inend, false, None);
    }

    let runs_at = LongRuns::new_in(same, arr, instart, inend);
    let mut runs = Vec::with_capacity(ZOPFLI_MAX_MATCH);
    let mut i = instart;
    while i < inend {
        if runs_at.starts_at(i) {
            m.mark_run(i - instart);
            for _ in 0..ZOPFLI_MAX_MATCH {
                tree.insert(arr, i, inend, runs_at.repeats_before(i), None);
                m.push_one(1, ZOPFLI_MAX_MATCH);
                i += 1;
            }
        }
        runs.clear();
        tree.insert(arr, i, inend, runs_at.repeats_before(i), Some(&mut runs));
        m.push_runs(&runs);
        i += 1;
    }
    debug_assert!(m.is_filled());
    ChunkMatches {
        cache: m,
        runs: runs_at,
        tree,
    }
}

/// Does the same as `Lz77Store::greedy` for a block whose matches are all in `m`.
pub fn greedy_cached<M: Matches + ?Sized>(
    store: &mut Lz77Store,
    m: &M,
    in_data: &[u8],
    instart: usize,
    inend: usize,
) {
    let mut i = instart;
    let mut prev_length = 0;
    let mut prev_match = 0;
    let mut match_available = false;
    while i < inend {
        let (mut leng, mut dist) = m.longest(i - instart);
        let lengthscore = get_length_score(i32::from(leng), i32::from(dist));

        /* Lazy matching. */
        let prevlengthscore = get_length_score(i32::from(prev_length), i32::from(prev_match));
        if match_available {
            match_available = false;
            if lengthscore > prevlengthscore + 1 {
                store.lit_len_dist(u16::from(in_data[i - 1]), 0, i - 1);
                if (lengthscore as usize) >= ZOPFLI_MIN_MATCH && (leng as usize) < ZOPFLI_MAX_MATCH
                {
                    match_available = true;
                    prev_length = leng;
                    prev_match = dist;
                    i += 1;
                    continue;
                }
            } else {
                /* Add previous to output. */
                verify_len_dist(in_data, i - 1, prev_match, prev_length);
                store.lit_len_dist(prev_length, prev_match, i - 1);
                i += prev_length as usize - 1;
                continue;
            }
        } else if (lengthscore as usize) >= ZOPFLI_MIN_MATCH && (leng as usize) < ZOPFLI_MAX_MATCH {
            match_available = true;
            prev_length = leng;
            prev_match = dist;
            i += 1;
            continue;
        }
        /* End of lazy matching. */

        /* Add to output. */
        if (lengthscore as usize) >= ZOPFLI_MIN_MATCH {
            verify_len_dist(in_data, i, dist, leng);
            store.lit_len_dist(leng, dist, i);
        } else {
            leng = 1;
            dist = 0;
            store.lit_len_dist(u16::from(in_data[i]), dist, i);
        }
        i += leng as usize;
    }
}

/// Does the same as `get_best_lengths` for a block whose matches are all in `m`.
fn get_best_lengths_cached<F: Fn(usize, u16) -> f64, M: Matches + ?Sized>(
    m: &M,
    in_data: &[u8],
    instart: usize,
    inend: usize,
    costmodel: F,
    buffers: &mut PassBuffers,
) -> f64 {
    let PassBuffers {
        costs,
        length_array,
        match_costs,
        ..
    } = buffers;
    let blocksize = inend - instart;
    length_array.clear();
    length_array.resize(blocksize + 1, 0);
    if instart == inend {
        return 0.0;
    }

    costs.resize(blocksize + 1, 0.0);
    for cost in costs.iter_mut().take(blocksize + 1).skip(1) {
        *cost = f32::INFINITY;
    }
    costs[0] = 0.0;

    // The cost of a match depends only on its length and distance symbol, so it
    // is looked up instead of computed at every position.
    match_costs.clear();
    match_costs.resize(DSYMBOLS.len() * (ZOPFLI_MAX_MATCH + 1), 0.0);
    for (row, &dist) in match_costs
        .as_chunks_mut::<{ ZOPFLI_MAX_MATCH + 1 }>()
        .0
        .iter_mut()
        .zip(&DSYMBOLS)
    {
        for (k, cost) in row.iter_mut().enumerate().skip(ZOPFLI_MIN_MATCH) {
            *cost = costmodel(k, dist);
        }
    }
    let mut literal_costs = [0.0; 256];
    for (c, cost) in literal_costs.iter_mut().enumerate() {
        *cost = costmodel(c, 0);
    }

    let mincost = get_cost_model_min_cost(&costmodel);
    let mut j = 0;
    while j < blocksize {
        if m.is_run(j) {
            let symbolcost = costmodel(ZOPFLI_MAX_MATCH, 1);
            for _ in 0..ZOPFLI_MAX_MATCH {
                costs[j + ZOPFLI_MAX_MATCH] = costs[j] + symbolcost as f32;
                length_array[j + ZOPFLI_MAX_MATCH] = ZOPFLI_MAX_MATCH as u16;
                j += 1;
            }
        }
        let i = instart + j;
        let costj = f64::from(costs[j]);

        // Literal.
        let new_cost = literal_costs[in_data[i] as usize] + costj;
        debug_assert!(new_cost >= 0.0);
        if new_cost < f64::from(costs[j + 1]) {
            costs[j + 1] = new_cost as f32;
            length_array[j + 1] = 1;
        }
        // Lengths. Like `get_best_lengths`, leaves alone every length whose cost
        // is already at most that of the cheapest match; both tests are
        // combined without branches.
        let mincostaddcostj = mincost + costj;
        let kend = inend - i;
        let mut k = ZOPFLI_MIN_MATCH;
        for &e in m.matches(j) {
            let last = cmp::min((e >> 16) as usize, kend);
            if last < k {
                break;
            }
            let dsym = get_dist_symbol(e as u16) as usize;
            let row = &match_costs[dsym * (ZOPFLI_MAX_MATCH + 1) + k..][..=last - k];
            let reach = &mut costs[j + k..=j + last];
            let lengths = &mut length_array[j + k..=j + last];
            for (n, ((&cost, old), length)) in row.iter().zip(reach).zip(lengths).enumerate() {
                let new_cost = cost + costj;
                let better = (f64::from(*old) > mincostaddcostj) & (new_cost < f64::from(*old));
                *old = if better { new_cost as f32 } else { *old };
                *length = if better { (k + n) as u16 } else { *length };
            }
            k = last + 1;
        }
        j += 1;
    }

    debug_assert!(costs[blocksize] >= 0.0);
    f64::from(costs[blocksize])
}

/// The buffers of the passes over a block, made once for all of them.
#[derive(Default)]
struct PassBuffers {
    costs: Vec<f32>,
    length_array: Vec<u16>,
    path: Vec<u16>,
    match_costs: Vec<f64>,
}

impl PassBuffers {
    /// One pass: the shortest path through the block with `costmodel`, into `store`.
    fn parse<F: Fn(usize, u16) -> f64, M: Matches + ?Sized>(
        &mut self,
        store: &mut Lz77Store,
        m: &M,
        in_data: &[u8],
        instart: usize,
        inend: usize,
        costmodel: F,
    ) {
        get_best_lengths_cached(m, in_data, instart, inend, costmodel, self);
        trace(inend - instart, &self.length_array, &mut self.path);
        follow_path_cached(store, m, in_data, instart, &self.path);
    }
}

/// Does the same as `Lz77Store::follow_path` for a block whose matches are all in `m`.
fn follow_path_cached<M: Matches + ?Sized>(
    store: &mut Lz77Store,
    m: &M,
    in_data: &[u8],
    instart: usize,
    path: &[u16],
) {
    let mut pos = instart;
    for &length in path.iter().rev() {
        if length >= ZOPFLI_MIN_MATCH as u16 {
            let dist = m.dist(pos - instart, length);
            verify_len_dist(in_data, pos, dist, length);
            store.lit_len_dist(length, dist, pos);
            pos += length as usize;
        } else {
            store.lit_len_dist(u16::from(in_data[pos]), 0, pos);
            pos += 1;
        }
    }
}

/// Calculates the optimal path of lz77 lengths to use, from the calculated
/// `length_array`. The `length_array` must contain the optimal length to reach that
/// byte. The path will be filled with the lengths to use, so its data size will be
/// the amount of lz77 symbols.
fn trace(size: usize, length_array: &[u16], path: &mut Vec<u16>) {
    let mut index = size;
    path.clear();

    while index > 0 {
        let lai = length_array[index];
        let laiu = lai as usize;
        path.push(lai);
        debug_assert!(laiu <= index);
        debug_assert!(laiu <= ZOPFLI_MAX_MATCH);
        debug_assert_ne!(lai, 0);
        index -= laiu;
    }
}

/// Does a single run for `lz77_optimal`. For good compression, repeated runs
/// with updated statistics should be performed.
/// `s`: the block state
/// `in_data`: the input data array
/// `instart`: where to start
/// `inend`: where to stop (not inclusive)
/// `length_array`: array of size `(inend - instart)` used to store lengths
/// `costmodel`: function to use as the cost model for this squeeze run
/// `store`: place to output the LZ77 data
/// returns the cost that was, according to the `costmodel`, needed to get to the end.
///     This is not the actual cost.
#[allow(clippy::too_many_arguments)] // Not feasible to refactor in a more readable way
fn lz77_optimal_run<F: Fn(usize, u16) -> f64, C: Cache>(
    lmc: &mut C,
    in_data: &[u8],
    instart: usize,
    inend: usize,
    costmodel: F,
    store: &mut Lz77Store,
    h: &mut ZopfliHash,
    costs: &mut Vec<f32>,
) {
    let (cost, length_array) = get_best_lengths(lmc, in_data, instart, inend, costmodel, h, costs);
    let mut path = Vec::new();
    trace(inend - instart, &length_array, &mut path);
    store.follow_path(in_data, instart, inend, path, lmc);
    debug_assert!(cost < f64::INFINITY);
}

/// Does the same as `lz77_optimal`, but optimized for the fixed tree of the
/// deflate standard.
/// The fixed tree never gives the best compression. But this gives the best
/// possible LZ77 encoding possible with the fixed tree.
/// This does not create or output any fixed tree, only LZ77 data optimized for
/// using with a fixed tree.
/// If `instart` is larger than `0`, it uses values before `instart` as starting
/// dictionary.
pub fn lz77_optimal_fixed<C: Cache>(
    lmc: &mut C,
    in_data: &[u8],
    instart: usize,
    inend: usize,
    store: &mut Lz77Store,
) {
    let mut costs = Vec::with_capacity(inend - instart);
    lz77_optimal_run(
        lmc,
        in_data,
        instart,
        inend,
        get_cost_fixed,
        store,
        &mut ZopfliHash::new(),
        &mut costs,
    );
}

/// Whether `lz77_optimal` keeps the matches of a block of `size` positions in a
/// `MatchCache`.
pub fn uses_match_cache(options: &Options, size: usize) -> bool {
    options.match_cache && MatchCache::fits(size)
}

/// Calculates lit/len and dist pairs for given data.
/// If `instart` is larger than 0, it uses values before `instart` as starting
/// dictionary.
/// `chunk`: the block's matches, taken from those of the chunk around it.
pub fn lz77_optimal<C: Cache>(
    lmc: &mut C,
    in_data: &[u8],
    instart: usize,
    inend: usize,
    options: &Options,
    chunk: Option<ChunkView<'_>>,
) -> Lz77Store {
    if let Some(m) = chunk {
        return lz77_optimal_with(lmc, in_data, instart, inend, options, Some(&m));
    }
    if !uses_match_cache(options, inend - instart) {
        return lz77_optimal_with::<C, MatchCache>(lmc, in_data, instart, inend, options, None);
    }
    if options.tree_match_finder {
        let c = fill_match_cache_tree(in_data, instart, inend);
        let m = c.block(instart, inend);
        lz77_optimal_with(lmc, in_data, instart, inend, options, Some(&m))
    } else {
        let m = fill_match_cache(in_data, instart, inend, &mut ZopfliHash::new());
        lz77_optimal_with(lmc, in_data, instart, inend, options, Some(&m))
    }
}

/// `lz77_optimal` with the block's matches in `matches`, or searched in every
/// pass if `None`.
fn lz77_optimal_with<C: Cache, M: Matches>(
    lmc: &mut C,
    in_data: &[u8],
    instart: usize,
    inend: usize,
    options: &Options,
    matches: Option<&M>,
) -> Lz77Store {
    let max_iterations = options.iteration_count.get();
    let max_iterations_without_improvement = options.iterations_without_improvement.get();
    // Only the passes without the match cache search, with this hash.
    let mut h = matches.is_none().then(ZopfliHash::new);
    /* Dist to get to here with smallest cost. */
    let mut currentstore = Lz77Store::new();
    let mut outputstore = currentstore.clone();

    /* Initial run. */
    match matches {
        None => currentstore.greedy(lmc, in_data, instart, inend),
        Some(m) => greedy_cached(&mut currentstore, m, in_data, instart, inend),
    }
    let mut stats = SymbolStats::default();
    stats.get_statistics(&currentstore);

    let mut buffers = PassBuffers::default();

    let mut beststats = SymbolStats::default();

    let mut bestcost = f64::INFINITY;
    let mut lastcost = 0.0;
    /* Try randomizing the costs a bit once the size stabilizes. */
    let mut ran_state = RanState::new();
    let mut lastrandomstep = u64::MAX;

    /* Do regular deflate, then loop multiple shortest path runs, each time using
    the statistics of the previous run. */
    /* Repeat statistics with each time the cost model from the previous stat
    run. */
    let mut current_iteration: u64 = 0;
    let mut iterations_without_improvement: u64 = 0;
    loop {
        currentstore.reset();
        let costmodel = |a, b| get_cost_stat(a, b, &stats);
        match matches {
            None => lz77_optimal_run(
                lmc,
                in_data,
                instart,
                inend,
                costmodel,
                &mut currentstore,
                h.as_mut().unwrap(),
                &mut buffers.costs,
            ),
            Some(m) => buffers.parse(&mut currentstore, m, in_data, instart, inend, costmodel),
        }
        let cost = calculate_block_size(&currentstore, 0, currentstore.size(), BlockType::Dynamic);

        if cost < bestcost {
            iterations_without_improvement = 0;
            /* Copy to the output store. */
            outputstore.clone_from(&currentstore);
            beststats = stats;
            bestcost = cost;

            debug!("Iteration {current_iteration}: {cost} bit");
        } else {
            iterations_without_improvement += 1;
            trace!("Iteration {current_iteration}: {cost} bit");
            if iterations_without_improvement >= max_iterations_without_improvement {
                break;
            }
        }
        let iteration = current_iteration;
        current_iteration += 1;
        if current_iteration >= max_iterations {
            break;
        }
        let laststats = stats;
        stats.clear_freqs();
        stats.get_statistics(&currentstore);
        if lastrandomstep != u64::MAX {
            /* This makes it converge slower but better. Do it only once the
            randomness kicks in so that if the user does few iterations, it gives a
            better result sooner. */
            stats = add_weighed_stat_freqs(&stats, 1.0, &laststats, 0.5);
            stats.calculate_entropy();
        }
        if iteration > 5 && (cost - lastcost).abs() < f64::EPSILON {
            stats = beststats;
            stats.randomize_stat_freqs(&mut ran_state);
            stats.calculate_entropy();
            lastrandomstep = iteration;
        }
        lastcost = cost;
    }

    if let (true, Some(m)) = (options.code_length_passes, matches) {
        loop {
            let mut counts = SymbolStats::default();
            counts.get_statistics(&outputstore);
            stats.use_code_lengths(&counts);
            currentstore.reset();
            buffers.parse(&mut currentstore, m, in_data, instart, inend, |a, b| {
                get_cost_stat(a, b, &stats)
            });
            let cost =
                calculate_block_size(&currentstore, 0, currentstore.size(), BlockType::Dynamic);
            if cost >= bestcost {
                break;
            }
            debug!("Code length pass: {cost} bit");
            bestcost = cost;
            outputstore.clone_from(&currentstore);
        }
    }
    outputstore
}

#[cfg(all(test, feature = "std"))]
mod test {
    use std::num::NonZeroU64;

    use proptest::prelude::*;

    use super::*;

    /// The block's own match cache made from the chunk's: long repetitions found
    /// for the block alone, matches cut at its end.
    fn block_copy(
        chunk: &ChunkMatches,
        in_data: &[u8],
        instart: usize,
        inend: usize,
    ) -> MatchCache {
        let mut m = MatchCache::new(inend - instart);
        let runs_at = LongRuns::new_in(Vec::new(), &in_data[..inend], instart, inend);
        let mut runs = Vec::with_capacity(ZOPFLI_MAX_MATCH);
        let mut i = instart;
        while i < inend {
            if runs_at.starts_at(i) {
                m.mark_run(i - instart);
                for _ in 0..ZOPFLI_MAX_MATCH {
                    m.push_one(1, ZOPFLI_MAX_MATCH);
                    i += 1;
                }
            }
            let room = (inend - i) as u32;
            runs.clear();
            if room >= ZOPFLI_MIN_MATCH as u32 {
                for &e in chunk.cache().matches(i - chunk.start()) {
                    if e >> 16 >= room {
                        runs.push(room << 16 | (e & 0xFFFF));
                        break;
                    }
                    runs.push(e);
                }
            }
            m.push_runs(&runs);
            i += 1;
        }
        m
    }

    fn symbols(store: &Lz77Store) -> Vec<(u16, u16)> {
        store
            .litlens
            .iter()
            .map(|l| match *l {
                LitLen::Literal(c) => (c, 0),
                LitLen::LengthDist(len, dist) => (len, dist),
            })
            .collect()
    }

    proptest! {
        #[test]
        fn a_block_reads_the_chunk_matches_like_its_own_copy(
            runs in prop::collection::vec((0u8..4, 1usize..800), 1..60),
            cuts in prop::collection::vec(0.0..1.0f64, 3),
        ) {
            let data: Vec<u8> = runs.iter().flat_map(|&(b, n)| core::iter::repeat_n(b, n)).collect();
            let mut at: Vec<usize> = cuts.iter().map(|&c| (c * data.len() as f64) as usize).collect();
            at.sort_unstable();
            let (chunkstart, instart, inend) = (at[0], at[1], at[2]);
            let chunk = fill_match_cache_tree(&data, chunkstart, data.len());
            let copy = block_copy(&chunk, &data, instart, inend);
            let view = chunk.block(instart, inend);
            let options = Options {
                iteration_count: NonZeroU64::new(3).unwrap(),
                code_length_passes: true,
                ..Options::default()
            };
            let a = lz77_optimal_with(&mut NoCache, &data, instart, inend, &options, Some(&copy));
            let b = lz77_optimal_with(&mut NoCache, &data, instart, inend, &options, Some(&view));
            prop_assert_eq!(symbols(&a), symbols(&b));
            let (mut a, mut b) = (Lz77Store::new(), Lz77Store::new());
            greedy_cached(&mut a, &copy, &data, instart, inend);
            greedy_cached(&mut b, &view, &data, instart, inend);
            prop_assert_eq!(symbols(&a), symbols(&b));
        }
    }
}
