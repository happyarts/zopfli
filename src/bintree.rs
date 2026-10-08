use alloc::{vec, vec::Vec};
use core::cmp;

use crate::util::{ZOPFLI_MAX_MATCH, ZOPFLI_MIN_MATCH, ZOPFLI_WINDOW_SIZE};

const HASH_BITS: u32 = 15;
const MAX_DEPTH: usize = 8192;

/// A binary tree match finder (as in the LZMA SDK's `Bt3`): for every 3-byte
/// hash, the positions of the window form a binary search tree ordered by the
/// bytes that follow them. Inserting a position walks its tree once and finds,
/// for each length, a match at the smallest distance on that walk.
pub struct BinaryTree {
    head: Vec<u32>,
    son: Vec<u32>,
    base: usize,
}

impl BinaryTree {
    /// A tree for positions from `base` on.
    pub fn new(base: usize) -> Self {
        Self {
            head: vec![0; 1 << HASH_BITS],
            son: vec![0; 2 * ZOPFLI_WINDOW_SIZE],
            base,
        }
    }

    /// An empty tree from `base` in the vectors of `self`.
    pub fn reuse(mut self, base: usize) -> Self {
        self.head.fill(0);
        self.son.fill(0);
        self.base = base;
        self
    }

    /// Inserts `pos` and appends its matches to `out` as `(length << 16) | distance`,
    /// lengths increasing, each length at least 3 and at most `ZOPFLI_MAX_MATCH`
    /// and `end - pos`. Every position from `base` on must be inserted in order.
    /// `repeats` tells that the byte before `pos` and the `limit` bytes from it
    /// are all the same, so the walk stops at distance 1 with a match of the
    /// full length; it is then taken without comparing the bytes.
    pub fn insert(
        &mut self,
        data: &[u8],
        pos: usize,
        end: usize,
        repeats: bool,
        out: Option<&mut Vec<u32>>,
    ) {
        let limit = cmp::min(ZOPFLI_MAX_MATCH, end - pos);
        if limit < ZOPFLI_MIN_MATCH {
            return;
        }
        let mut out = out;
        let id = pos - self.base + ZOPFLI_WINDOW_SIZE;
        let v =
            u32::from(data[pos]) | u32::from(data[pos + 1]) << 8 | u32::from(data[pos + 2]) << 16;
        let h = (v.wrapping_mul(0x9E37_79B1) >> (32 - HASH_BITS)) as usize;
        let mut cur_match = self.head[h] as usize;
        self.head[h] = id as u32;

        let cyc = id & (ZOPFLI_WINDOW_SIZE - 1);
        let mut ptr0 = 2 * cyc + 1;
        let mut ptr1 = 2 * cyc;
        let (mut len0, mut len1) = (0, 0);
        if repeats && cur_match + 1 == id {
            debug_assert!(data[pos - 1..=pos - 1 + limit]
                .iter()
                .all(|&b| b == data[pos]));
            // What the first step of the walk below does for this match.
            let pair = 2 * ((cyc + ZOPFLI_WINDOW_SIZE - 1) & (ZOPFLI_WINDOW_SIZE - 1));
            if let Some(out) = out {
                out.push((limit as u32) << 16 | 1);
            }
            self.son[ptr1] = self.son[pair];
            self.son[ptr0] = self.son[pair + 1];
            return;
        }
        let mut max_len = ZOPFLI_MIN_MATCH - 1;
        let cur = &data[pos..pos + limit];
        for _ in 0..MAX_DEPTH {
            let delta = id - cur_match;
            if delta >= ZOPFLI_WINDOW_SIZE {
                break;
            }
            let pair = 2 * ((cyc + ZOPFLI_WINDOW_SIZE - delta) & (ZOPFLI_WINDOW_SIZE - 1));
            let prev = &data[pos - delta..pos - delta + limit];
            let mut len = cmp::min(len0, len1);
            if prev[len] == cur[len] {
                len += 1;
                while len < limit && prev[len] == cur[len] {
                    len += 1;
                }
                if max_len < len {
                    max_len = len;
                    if let Some(out) = out.as_deref_mut() {
                        out.push((len as u32) << 16 | delta as u32);
                    }
                    if len == limit {
                        self.son[ptr1] = self.son[pair];
                        self.son[ptr0] = self.son[pair + 1];
                        return;
                    }
                }
            }
            if prev[len] < cur[len] {
                self.son[ptr1] = cur_match as u32;
                ptr1 = pair + 1;
                cur_match = self.son[ptr1] as usize;
                len1 = len;
            } else {
                self.son[ptr0] = cur_match as u32;
                ptr0 = pair;
                cur_match = self.son[ptr0] as usize;
                len0 = len;
            }
        }
        self.son[ptr0] = 0;
        self.son[ptr1] = 0;
    }
}

#[cfg(all(test, feature = "std"))]
mod test {
    use proptest::prelude::*;

    use super::*;

    proptest! {
        #[test]
        fn taking_repeats_without_comparing_changes_nothing(
            runs in prop::collection::vec((0..3u8, 1..700usize), 1..200)
        ) {
            let data: Vec<u8> = runs
                .iter()
                .flat_map(|&(b, n)| std::iter::repeat_n(b, n))
                .collect();
            let mut fast = BinaryTree::new(0);
            let mut slow = BinaryTree::new(0);
            for pos in 0..data.len() {
                let limit = cmp::min(ZOPFLI_MAX_MATCH, data.len() - pos);
                let repeats = pos > 0 && data[pos - 1..=pos - 1 + limit].iter().all(|&b| b == data[pos]);
                let (mut a, mut b) = (Vec::new(), Vec::new());
                fast.insert(&data, pos, data.len(), repeats, Some(&mut a));
                slow.insert(&data, pos, data.len(), false, Some(&mut b));
                prop_assert_eq!(a, b);
                if pos % 64 == 0 {
                    prop_assert!(fast.head == slow.head && fast.son == slow.son);
                }
            }
            prop_assert!(fast.head == slow.head && fast.son == slow.son);
        }
    }
}
