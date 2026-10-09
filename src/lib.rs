// Modified in happyarts/zopfli (see its commit history).

#![deny(trivial_casts, trivial_numeric_casts, missing_docs)]

//! A reimplementation of the [Zopfli](https://github.com/google/zopfli) compression library in Rust.
//!
//! Zopfli is a state of the art DEFLATE compressor that heavily prioritizes compression over speed.
//! It usually compresses much better than other DEFLATE compressors, generating standard DEFLATE
//! streams that can be decompressed with any DEFLATE decompressor, at the cost of being
//! significantly slower.
//!
//! # Features
//!
//! This crate exposes the following features. You can enable or disable them in your `Cargo.toml`
//! as needed.
//!
//! - `gzip` (enabled by default): enables support for compression in the gzip format.
//! - `zlib` (enabled by default): enables support for compression in the Zlib format.
//! - `std` (enabled by default): enables linking against the Rust standard library. When not enabled,
//!   the crate is built with the `#![no_std]` attribute and can be used in any environment where
//!   [`alloc`](https://doc.rust-lang.org/alloc/) (i.e., a memory allocator) is available. In addition,
//!   the crate exposes minimalist versions of the `std` I/O traits it needs to function, allowing users
//!   to implement them.
//! - `nightly`: enables code constructs and features specific to the nightly Rust toolchain. Currently,
//!   this feature improves rustdoc generation and enables the namesake feature on `crc32fast`, but this
//!   may change in the future. This feature also used to enable `simd-adler32`'s namesake feature, but
//!   it no longer does as the latest `simd-adler32` release does not build with the latest nightlies
//!   (as of 2024-05-18) when that feature is enabled.

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(feature = "nightly", feature(doc_cfg))]

// No-op log implementation for no-std targets
#[cfg(not(feature = "std"))]
macro_rules! debug {
    ( $( $_:expr ),* ) => {};
}
#[cfg(not(feature = "std"))]
macro_rules! trace {
    ( $( $_:expr ),* ) => {};
}
#[cfg(not(feature = "std"))]
macro_rules! log_enabled {
    ( $( $_:expr ),* ) => {
        false
    };
}

#[cfg_attr(not(feature = "std"), macro_use)]
extern crate alloc;

pub use deflate::{BlockType, DeflateEncoder};
#[cfg(feature = "gzip")]
pub use gzip::GzipEncoder;
#[cfg(all(test, feature = "std"))]
use proptest::prelude::*;
#[cfg(feature = "zlib")]
pub use zlib::ZlibEncoder;

mod bintree;
mod blocksplitter;
mod cache;
mod deflate;
#[cfg(feature = "gzip")]
mod gzip;
mod hash;
#[cfg(any(doc, not(feature = "std")))]
mod io;
mod iter;
mod katajainen;
mod lz77;
mod matches;
#[cfg(not(feature = "std"))]
mod math;
mod squeeze;
mod symbols;
mod tree;
mod util;
#[cfg(feature = "zlib")]
mod zlib;

use core::num::NonZeroU64;
#[cfg(all(not(doc), feature = "std"))]
use std::io::{Error, Write};

#[cfg(any(doc, not(feature = "std")))]
pub use io::{Error, ErrorKind, Write};

/// Options for the Zopfli compression algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(all(test, feature = "std"), derive(proptest_derive::Arbitrary))]
pub struct Options {
    /// Maximum amount of times to rerun forward and backward pass to optimize LZ77
    /// compression cost.
    /// Good values: 10, 15 for small files, 5 for files over several MB in size or
    /// it will be too slow.
    ///
    /// Default value: 15.
    #[cfg_attr(
        all(test, feature = "std"),
        proptest(
            strategy = "(1..=10u64).prop_map(|iteration_count| NonZeroU64::new(iteration_count).unwrap())"
        )
    )]
    pub iteration_count: NonZeroU64,
    /// Stop after rerunning forward and backward pass this many times without finding
    /// a smaller representation of the block.
    ///
    /// Default value: practically infinite (maximum `u64` value)
    pub iterations_without_improvement: NonZeroU64,
    /// Maximum amount of blocks to split into (0 for unlimited, but this can give
    /// extreme results that hurt compression on some files).
    ///
    /// Default value: 15.
    pub maximum_block_splits: u16,
    /// Finds the matches of each block once and keeps them, so that the
    /// iterations need no match search. Same output. Replaces the longest match
    /// cache in this path: 4 bytes per position and 4 per run of equal
    /// distance, instead of its 28 bytes per position.
    ///
    /// Default value: true.
    pub match_cache: bool,
    /// When writing a dynamic block, also tries other Huffman code lengths
    /// (another smoothing of the counts, lower length limits) and replacing short
    /// matches that cost more than their literals; keeps what is smallest.
    ///
    /// Default value: false.
    pub final_block_trials: bool,
    /// After the iterations, repeats the optimal parse with the real Huffman code
    /// lengths of the best result as long as that makes the block smaller.
    /// Needs `match_cache`.
    ///
    /// Default value: false.
    pub code_length_passes: bool,
    /// Finds matches with a binary tree instead of hash chains, once for each
    /// chunk of input (block splitting and all blocks). It can find other
    /// distances than the hash chains, so the output can differ. Needs
    /// `match_cache`.
    ///
    /// Default value: false.
    pub tree_match_finder: bool,
    /// Compresses the chunks the input arrives in (1 MB with `compress`) on
    /// several threads at once; same output. A chunk goes to a worker thread
    /// with the window before it once the next one arrives; threads start as
    /// chunks come, up to the available parallelism, and at most twice as many
    /// chunks as threads are held before a write waits. An error of the sink
    /// can surface on a later call; a write that returns an error has taken
    /// none of its data. Only with the `std` feature and dynamic blocks.
    ///
    /// Default value: false.
    pub parallel_chunks: bool,
    /// Joins neighbouring blocks, also across the chunks the input arrives in,
    /// wherever one block is smaller than the two (at most 16 blocks into one;
    /// blocks stored uncompressed are not joined). Never makes the output
    /// larger. Blocks that are not joined are written as without this option,
    /// except that empty chunks give no empty blocks. The last block is held
    /// back, with its input and the 32 KiB before it, until the next one is
    /// known: with the 1 MiB chunks of `compress`, up to about 16 MiB of input.
    /// Dynamic blocks only.
    ///
    /// Default value: false.
    pub merge_blocks: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            iteration_count: NonZeroU64::new(15).unwrap(),
            iterations_without_improvement: NonZeroU64::new(u64::MAX).unwrap(),
            maximum_block_splits: 15,
            match_cache: true,
            final_block_trials: false,
            code_length_passes: false,
            tree_match_finder: false,
            parallel_chunks: false,
            merge_blocks: false,
        }
    }
}

/// The output file format to use to store data compressed with Zopfli.
#[derive(Debug, Copy, Clone)]
#[cfg(feature = "std")]
pub enum Format {
    /// The gzip file format, as defined in
    /// [RFC 1952](https://datatracker.ietf.org/doc/html/rfc1952).
    ///
    /// This file format can be easily decompressed with the gzip
    /// program.
    #[cfg(feature = "gzip")]
    Gzip,
    /// The zlib file format, as defined in
    /// [RFC 1950](https://datatracker.ietf.org/doc/html/rfc1950).
    ///
    /// The zlib format has less header overhead than gzip, but it
    /// stores less metadata.
    #[cfg(feature = "zlib")]
    Zlib,
    /// The raw DEFLATE stream format, as defined in
    /// [RFC 1951](https://datatracker.ietf.org/doc/html/rfc1951).
    ///
    /// Raw DEFLATE streams are not meant to be stored as-is because
    /// they lack error detection and correction metadata. They
    /// are usually embedded in other file formats, such as gzip
    /// and zlib.
    Deflate,
}

/// Compresses data from a source with the Zopfli algorithm, using the specified
/// options, and writes the result to a sink in the defined output format.
#[cfg(feature = "std")]
pub fn compress<R: std::io::Read, W: Write>(
    options: Options,
    output_format: Format,
    mut in_data: R,
    out: W,
) -> Result<(), Error> {
    match output_format {
        #[cfg(feature = "gzip")]
        Format::Gzip => {
            let mut gzip_encoder = GzipEncoder::new_buffered(options, BlockType::Dynamic, out)?;
            std::io::copy(&mut in_data, &mut gzip_encoder)?;
            gzip_encoder.into_inner()?.finish().map(|_| ())
        }
        #[cfg(feature = "zlib")]
        Format::Zlib => {
            let mut zlib_encoder = ZlibEncoder::new_buffered(options, BlockType::Dynamic, out)?;
            std::io::copy(&mut in_data, &mut zlib_encoder)?;
            zlib_encoder.into_inner()?.finish().map(|_| ())
        }
        Format::Deflate => {
            let mut deflate_encoder =
                DeflateEncoder::new_buffered(options, BlockType::Dynamic, out);
            std::io::copy(&mut in_data, &mut deflate_encoder)?;
            deflate_encoder.into_inner()?.finish().map(|_| ())
        }
    }
}

#[doc(hidden)]
#[deprecated(
    since = "0.8.2",
    note = "Object pools are no longer used. This function is now a no-op and will be removed in version 0.9.0."
)]
#[cfg(feature = "std")] // TODO remove for 0.9.0
pub fn prewarm_object_pools() {}

#[cfg(all(test, feature = "std"))]
mod test {
    use std::io;

    use miniz_oxide::inflate;
    use proptest::proptest;

    use super::*;

    proptest! {
        #[test]
        fn deflating_is_reversible(
            options: Options,
            btype: BlockType,
            data in prop::collection::vec(any::<u8>(), 0..64 * 1024)
        ) {
            let mut compressed_data = Vec::with_capacity(data.len());

            let mut encoder = DeflateEncoder::new(options, btype, &mut compressed_data);
            io::copy(&mut &*data, &mut encoder).unwrap();
            encoder.finish().unwrap();

            let decompressed_data = inflate::decompress_to_vec(&compressed_data).expect("Could not inflate compressed stream");
            prop_assert_eq!(data, decompressed_data, "Decompressed data should match input data");
        }

        #[test]
        fn match_cache_does_not_change_the_output(
            iterations in 1..6u64,
            cuts in cuts(),
            runs in runs(4, 1500, 200)
        ) {
            let data = from_runs(&runs);
            let options = |match_cache| Options {
                iteration_count: NonZeroU64::new(iterations).unwrap(),
                match_cache,
                ..Options::default()
            };
            prop_assert_eq!(
                compress_in_pieces(options(true), &data, &cuts),
                compress_in_pieces(options(false), &data, &cuts)
            );
        }

        #[test]
        fn parallel_chunks_do_not_change_the_output(
            others: bool,
            merge_blocks: bool,
            cuts in cuts(),
            runs in runs(6, 3000, 400)
        ) {
            let data = from_runs(&runs);
            let options = |parallel_chunks| Options {
                iteration_count: NonZeroU64::new(2).unwrap(),
                final_block_trials: others,
                code_length_passes: others,
                tree_match_finder: others,
                parallel_chunks,
                merge_blocks,
                ..Options::default()
            };
            let parallel = compress_in_pieces(options(true), &data, &cuts);
            prop_assert_eq!(&parallel, &compress_in_pieces(options(false), &data, &cuts));
            let decompressed = inflate::decompress_to_vec(&parallel).expect("Could not inflate compressed stream");
            prop_assert_eq!(data, decompressed);
        }

        #[test]
        fn merging_blocks_never_makes_the_output_larger(
            others: bool,
            cuts in cuts(),
            runs in runs(6, 3000, 400)
        ) {
            let data = from_runs(&runs);
            let options = |merge_blocks| Options {
                iteration_count: NonZeroU64::new(2).unwrap(),
                final_block_trials: others,
                code_length_passes: others,
                tree_match_finder: others,
                merge_blocks,
                ..Options::default()
            };
            let merged = compress_in_pieces(options(true), &data, &cuts);
            prop_assert!(merged.len() <= compress_in_pieces(options(false), &data, &cuts).len());
            let decompressed = inflate::decompress_to_vec(&merged).expect("Could not inflate compressed stream");
            prop_assert_eq!(data, decompressed);
        }

        #[test]
        fn all_options_are_reversible(
            iterations in 1..5u64,
            cuts in cuts(),
            runs in runs(6, 1500, 300)
        ) {
            let data = from_runs(&runs);
            let options = Options {
                iteration_count: NonZeroU64::new(iterations).unwrap(),
                final_block_trials: true,
                code_length_passes: true,
                tree_match_finder: true,
                merge_blocks: true,
                ..Options::default()
            };
            let compressed = compress_in_pieces(options, &data, &cuts);
            let decompressed = inflate::decompress_to_vec(&compressed).expect("Could not inflate compressed stream");
            prop_assert_eq!(data, decompressed);
        }
    }

    /// A sink that fails once `room` bytes are written.
    struct Failing {
        room: usize,
    }

    impl io::Write for Failing {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            if buf.len() > self.room {
                return Err(io::Error::other("full"));
            }
            self.room -= buf.len();
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn parallel_chunks_report_a_failing_sink() {
        let data = from_runs(
            &(0..3000)
                .map(|i| ((i % 7) as u8, 1 + i % 900))
                .collect::<Vec<_>>(),
        );
        for merge_blocks in [false, true] {
            let options = Options {
                iteration_count: NonZeroU64::new(1).unwrap(),
                parallel_chunks: true,
                merge_blocks,
                ..Options::default()
            };
            let mut encoder =
                DeflateEncoder::new(options, BlockType::Dynamic, Failing { room: 100 });
            // The error surfaces on a later write or at the end.
            let written = data
                .chunks(64 * 1024)
                .try_for_each(|piece| encoder.write_all(piece));
            assert!(written.is_err() || encoder.finish().is_err());
        }
    }

    #[test]
    fn encoders_are_send_sync_and_unwind_safe() {
        fn check<T: Send + Sync + std::panic::UnwindSafe + std::panic::RefUnwindSafe>() {}
        check::<DeflateEncoder<Vec<u8>>>();
        #[cfg(feature = "gzip")]
        check::<GzipEncoder<Vec<u8>>>();
        #[cfg(feature = "zlib")]
        check::<ZlibEncoder<Vec<u8>>>();
    }

    /// Runs `f` on a thread of its own; fails if it panics or takes more than
    /// a minute.
    fn within_a_minute(f: impl FnOnce() + Send + 'static) {
        let (done, wait) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
            let _ = done.send(result.is_ok());
        });
        let finished = wait.recv_timeout(std::time::Duration::from_secs(60));
        assert_eq!(finished, Ok(true), "failed or still running after a minute");
    }

    #[test]
    fn a_panic_in_a_worker_reaches_the_caller_and_nothing_waits_for_it() {
        within_a_minute(|| {
            let mut first = deflate::TEST_PANIC.to_vec();
            first.resize(30_000, 7);
            let other = vec![3; 30_000];
            // More chunks than the threads hold: a write waits for the first
            // chunk and meets the panic.
            let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
            let pieces =
                || core::iter::once(&first).chain(core::iter::repeat_n(&other, 2 * threads + 4));
            let is_test_panic = |payload: Box<dyn core::any::Any + Send>| {
                payload.downcast_ref::<&str>() == Some(&"test panic in a worker")
            };
            for merge_blocks in [false, true] {
                let options = Options {
                    iteration_count: NonZeroU64::new(1).unwrap(),
                    parallel_chunks: true,
                    merge_blocks,
                    ..Options::default()
                };
                // Caught, then finished: an error.
                let mut encoder = DeflateEncoder::new(options, BlockType::Dynamic, Vec::new());
                let written = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    for piece in pieces() {
                        encoder.write_all(piece).unwrap();
                    }
                }));
                assert!(is_test_panic(written.unwrap_err()));
                assert!(encoder.write_all(&other).is_err());
                assert!(encoder.finish().is_err());

                // Not caught: the encoder is dropped while the panic unwinds.
                let dropped = std::panic::catch_unwind(|| {
                    let mut encoder = DeflateEncoder::new(options, BlockType::Dynamic, Vec::new());
                    for piece in pieces() {
                        encoder.write_all(piece).unwrap();
                    }
                });
                assert!(is_test_panic(dropped.unwrap_err()));
            }
        });
    }

    /// Data from runs of one of `symbols` byte values, each up to `longest`
    /// long: many matches, and long repetitions of one byte.
    fn runs(symbols: u8, longest: usize, count: usize) -> impl Strategy<Value = Vec<(u8, usize)>> {
        prop::collection::vec((0..symbols, 1..longest), 0..count)
    }

    fn from_runs(runs: &[(u8, usize)]) -> Vec<u8> {
        runs.iter()
            .flat_map(|&(b, n)| std::iter::repeat_n(b, n))
            .collect()
    }

    /// Where to cut the input into separate writes, as fractions of its length
    /// (in 1/1024); equal cuts give empty writes.
    fn cuts() -> impl Strategy<Value = Vec<usize>> {
        prop::collection::vec(0..=1024usize, 0..6)
    }

    /// Compresses `data` with one `write` call per piece between the `cuts`,
    /// including empty ones, so that the encoder sees several chunks.
    fn compress_in_pieces(options: Options, data: &[u8], cuts: &[usize]) -> Vec<u8> {
        let mut cuts: Vec<usize> = cuts.iter().map(|&c| c * data.len() / 1024).collect();
        cuts.sort_unstable();
        let mut out = Vec::new();
        let mut encoder = DeflateEncoder::new(options, BlockType::Dynamic, &mut out);
        let mut last = 0;
        for &cut in cuts.iter().chain(core::iter::once(&data.len())) {
            let mut piece = &data[last..cut];
            // `write_all` skips empty pieces; `write` hands them over.
            if piece.is_empty() {
                assert_eq!(encoder.write(piece).unwrap(), 0);
            }
            while !piece.is_empty() {
                let n = encoder.write(piece).unwrap();
                piece = &piece[n..];
            }
            last = cut;
        }
        encoder.finish().unwrap();
        out
    }
}
