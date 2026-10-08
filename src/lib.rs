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

mod blocksplitter;
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
}

impl Default for Options {
    fn default() -> Self {
        Self {
            iteration_count: NonZeroU64::new(15).unwrap(),
            iterations_without_improvement: NonZeroU64::new(u64::MAX).unwrap(),
            maximum_block_splits: 15,
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
///
/// The input is compressed in master blocks of 1,000,000 bytes, as
/// `ZopfliCompress` of the original Zopfli does, however many bytes each
/// `read` of `in_data` returns.
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
            let mut gzip_encoder = GzipEncoder::new(options, BlockType::Dynamic, out)?;
            copy_in_master_blocks(&mut in_data, &mut gzip_encoder)?;
            gzip_encoder.finish().map(|_| ())
        }
        #[cfg(feature = "zlib")]
        Format::Zlib => {
            let mut zlib_encoder = ZlibEncoder::new(options, BlockType::Dynamic, out)?;
            copy_in_master_blocks(&mut in_data, &mut zlib_encoder)?;
            zlib_encoder.finish().map(|_| ())
        }
        Format::Deflate => {
            let mut deflate_encoder = DeflateEncoder::new(options, BlockType::Dynamic, out);
            copy_in_master_blocks(&mut in_data, &mut deflate_encoder)?;
            deflate_encoder.finish().map(|_| ())
        }
    }
}

/// Copies `reader` to `writer` in pieces of `ZOPFLI_MASTER_BLOCK_SIZE` bytes
/// (the last one may be shorter), whatever amounts `reader` hands out at a
/// time. The encoders compress each write as a chunk of its own, so this makes
/// the output independent of the reader and splits the input into the same
/// master blocks as the original Zopfli.
#[cfg(feature = "std")]
fn copy_in_master_blocks<R: std::io::Read, W: std::io::Write>(
    reader: &mut R,
    writer: &mut W,
) -> std::io::Result<()> {
    let mut buffer = vec![0; util::ZOPFLI_MASTER_BLOCK_SIZE];
    loop {
        let mut filled = 0;
        while filled < buffer.len() {
            match reader.read(&mut buffer[filled..]) {
                Ok(0) => break,
                Ok(n) => filled += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        writer.write_all(&buffer[..filled])?;
        if filled < buffer.len() {
            return Ok(());
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
    }

    /// Hands out at most a few bytes per `read`.
    struct Trickle<'a>(&'a [u8]);

    impl io::Read for Trickle<'_> {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let n = buf.len().min(self.0.len()).min(7);
            buf[..n].copy_from_slice(&self.0[..n]);
            self.0 = &self.0[n..];
            Ok(n)
        }
    }

    #[test]
    fn compress_uses_master_blocks_for_every_reader() {
        // A bit more than one master block: long runs of a few byte values.
        let data: Vec<u8> = (0..1_100_000u32).map(|i| (i / 4999 % 5) as u8).collect();
        let options = Options {
            iteration_count: NonZeroU64::new(1).unwrap(),
            ..Options::default()
        };
        let mut master_blocks = Vec::new();
        let mut encoder = DeflateEncoder::new(options, BlockType::Dynamic, &mut master_blocks);
        for block in data.chunks(1_000_000) {
            io::Write::write_all(&mut encoder, block).unwrap();
        }
        encoder.finish().unwrap();

        // A `&[u8]`, which `io::copy` used to hand over in one piece.
        let mut from_slice = Vec::new();
        compress(options, Format::Deflate, &data[..], &mut from_slice).unwrap();
        assert!(from_slice == master_blocks, "a slice gives other chunks");
        let mut from_trickle = Vec::new();
        compress(options, Format::Deflate, Trickle(&data), &mut from_trickle).unwrap();
        assert!(
            from_trickle == master_blocks,
            "a slow reader gives other chunks"
        );
    }

    /// Our output and that of the published 0.8.3 for `data`, written in the
    /// pieces that end at `cuts`.
    fn ours_and_reference(
        iteration_count: NonZeroU64,
        maximum_block_splits: u16,
        btype: BlockType,
        data: &[u8],
        cuts: &[usize],
    ) -> (Vec<u8>, Vec<u8>) {
        let mut ours = Vec::new();
        let mut encoder = DeflateEncoder::new(
            Options {
                iteration_count,
                maximum_block_splits,
                ..Options::default()
            },
            btype,
            &mut ours,
        );
        let reference_btype = match btype {
            BlockType::Uncompressed => zopfli_reference::BlockType::Uncompressed,
            BlockType::Fixed => zopfli_reference::BlockType::Fixed,
            BlockType::Dynamic => zopfli_reference::BlockType::Dynamic,
        };
        let mut theirs = Vec::new();
        let mut reference = zopfli_reference::DeflateEncoder::new(
            zopfli_reference::Options {
                iteration_count,
                maximum_block_splits,
                ..zopfli_reference::Options::default()
            },
            reference_btype,
            &mut theirs,
        );
        // The same pieces to both, empty ones included.
        let mut last = 0;
        for &cut in cuts.iter().chain(core::iter::once(&data.len())) {
            assert_eq!(
                io::Write::write(&mut encoder, &data[last..cut]).unwrap(),
                cut - last
            );
            assert_eq!(
                io::Write::write(&mut reference, &data[last..cut]).unwrap(),
                cut - last
            );
            last = cut;
        }
        encoder.finish().unwrap();
        reference.finish().unwrap();
        (ours, theirs)
    }

    proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(32))]
        #[test]
        fn output_is_that_of_zopfli_0_8_3(
            iterations in 1..6u64,
            maximum_block_splits in 0..20u16,
            btype: BlockType,
            cuts in prop::collection::vec(0..=1024usize, 0..6),
            runs in prop::collection::vec((0..6u8, 1..3000usize), 0..60),
            noise in prop::collection::vec(any::<u8>(), 0..2000),
        ) {
            // Runs of a few byte values: many matches and long repetitions of one
            // byte (the forward pass's shortcut); some random bytes in between.
            let mut data: Vec<u8> = runs
                .iter()
                .flat_map(|&(b, n)| core::iter::repeat_n(b, n))
                .collect();
            let at = noise.len() * 7 % (data.len() + 1);
            data.splice(at..at, noise);
            let mut cuts: Vec<usize> = cuts.iter().map(|&c| c * data.len() / 1024).collect();
            cuts.sort_unstable();

            let iteration_count = NonZeroU64::new(iterations).unwrap();
            let (ours, theirs) =
                ours_and_reference(iteration_count, maximum_block_splits, btype, &data, &cuts);
            prop_assert!(ours == theirs);
        }
    }

    #[test]
    fn output_is_that_of_zopfli_0_8_3_on_fixed_inputs() {
        // The start of each test file, text, data of two and of four byte values
        // (many distances for each length), and random bytes.
        let mut inputs: Vec<Vec<u8>> =
            std::fs::read_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/test/data"))
                .unwrap()
                .map(|entry| {
                    let mut data = std::fs::read(entry.unwrap().path()).unwrap();
                    data.truncate(100_000);
                    data
                })
                .collect();
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        let mut random = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let words = [
            "zopfli ", "deflate ", "huffman ", "the ", "a ", "block, ", "match.\n",
        ];
        inputs.push(
            (0..8_000)
                .flat_map(|_| words[random() as usize % words.len()].bytes())
                .collect(),
        );
        inputs.push((0..40_000).map(|_| (random() % 2) as u8).collect());
        inputs.push(
            (0..40_000)
                .map(|_| b"ACGT"[random() as usize % 4])
                .collect(),
        );
        inputs.push((0..40_000).map(|_| random() as u8).collect());

        let iterations = |n| NonZeroU64::new(n).unwrap();
        for data in &inputs {
            let (ours, theirs) =
                ours_and_reference(iterations(2), 15, BlockType::Dynamic, data, &[]);
            assert!(ours == theirs, "dynamic blocks of {} bytes", data.len());
            let (ours, theirs) = ours_and_reference(iterations(1), 15, BlockType::Fixed, data, &[]);
            assert!(ours == theirs, "fixed blocks of {} bytes", data.len());
        }
    }

    /// The output of a `DeflateEncoder` with `threads` for `data` written in
    /// the pieces that end at `cuts`.
    fn deflate_in_pieces(threads: usize, iterations: u64, data: &[u8], cuts: &[usize]) -> Vec<u8> {
        let options = Options {
            iteration_count: NonZeroU64::new(iterations).unwrap(),
            ..Options::default()
        };
        let mut out = Vec::new();
        let mut encoder = DeflateEncoder::new(options, BlockType::Dynamic, &mut out)
            .with_threads(core::num::NonZeroUsize::new(threads).unwrap());
        // The same pieces each time, empty ones included.
        let mut last = 0;
        for &cut in cuts.iter().chain(core::iter::once(&data.len())) {
            assert_eq!(
                io::Write::write(&mut encoder, &data[last..cut]).unwrap(),
                cut - last
            );
            last = cut;
        }
        encoder.finish().unwrap();
        out
    }

    proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(24))]
        #[test]
        fn threads_do_not_change_the_output(
            threads in 2..5usize,
            iterations in 1..3u64,
            cuts in prop::collection::vec(0..=1024usize, 0..10),
            runs in prop::collection::vec((0..6u8, 1..3000usize), 0..100),
        ) {
            let data: Vec<u8> = runs
                .iter()
                .flat_map(|&(b, n)| core::iter::repeat_n(b, n))
                .collect();
            let mut cuts: Vec<usize> = cuts.iter().map(|&c| c * data.len() / 1024).collect();
            cuts.sort_unstable();
            prop_assert!(
                deflate_in_pieces(threads, iterations, &data, &cuts)
                    == deflate_in_pieces(1, iterations, &data, &cuts)
            );
        }
    }

    #[test]
    fn threads_in_gzip_and_zlib_do_not_change_the_output() {
        let data: Vec<u8> = (0..120_000u32).map(|i| (i / 700 % 7) as u8 * 3).collect();
        let options = Options {
            iteration_count: NonZeroU64::new(1).unwrap(),
            ..Options::default()
        };
        let threads = core::num::NonZeroUsize::new(3).unwrap();
        #[cfg(feature = "gzip")]
        {
            let gzip = |threads| {
                let mut out = Vec::new();
                let mut encoder = GzipEncoder::new(options, BlockType::Dynamic, &mut out)
                    .unwrap()
                    .with_threads(threads);
                for piece in data.chunks(10_000) {
                    io::Write::write_all(&mut encoder, piece).unwrap();
                }
                encoder.finish().unwrap();
                out
            };
            assert!(gzip(threads) == gzip(core::num::NonZeroUsize::MIN));
        }
        #[cfg(feature = "zlib")]
        {
            let zlib = |threads| {
                let mut out = Vec::new();
                let mut encoder = ZlibEncoder::new(options, BlockType::Dynamic, &mut out)
                    .unwrap()
                    .with_threads(threads);
                for piece in data.chunks(10_000) {
                    io::Write::write_all(&mut encoder, piece).unwrap();
                }
                encoder.finish().unwrap();
                out
            };
            assert!(zlib(threads) == zlib(core::num::NonZeroUsize::MIN));
        }
        let _ = (options, threads, &data);
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
    fn threads_report_a_failing_sink() {
        let data: Vec<u8> = (0..300_000u32).map(|i| (i / 300 % 5) as u8).collect();
        let options = Options {
            iteration_count: NonZeroU64::new(1).unwrap(),
            ..Options::default()
        };
        let threads = core::num::NonZeroUsize::new(2).unwrap();
        let full_size = deflate_in_pieces(
            1,
            1,
            &data,
            &(1..30).map(|i| i * 10_000).collect::<Vec<_>>(),
        )
        .len();
        for room in [0, 50, 500, full_size / 2, full_size - 1, full_size] {
            let mut encoder = DeflateEncoder::new(options, BlockType::Dynamic, Failing { room })
                .with_threads(threads);
            let mut failed = false;
            for piece in data.chunks(10_000) {
                // A write that fails has not taken its data: the same piece again.
                if io::Write::write(&mut encoder, piece).is_err() {
                    failed = true;
                    assert!(io::Write::write(&mut encoder, piece).is_err());
                    break;
                }
            }
            if failed {
                // Dropped after an error: neither a panic nor waiting for the
                // queued chunks.
                drop(encoder);
            } else {
                failed = encoder.finish().is_err();
            }
            assert_eq!(failed, room < full_size, "room {room} of {full_size}");
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

    #[test]
    fn changing_threads_between_writes_does_not_change_the_output() {
        let data: Vec<u8> = (0..400_000u32).map(|i| (i / 900 % 6) as u8 * 5).collect();
        let options = Options {
            iteration_count: NonZeroU64::new(1).unwrap(),
            ..Options::default()
        };
        let compress = |threads: [usize; 3]| {
            let mut out = Vec::new();
            let mut encoder = DeflateEncoder::new(options, BlockType::Dynamic, &mut out);
            for (k, piece) in data.chunks(40_000).enumerate() {
                if k % 4 == 0 {
                    let n = core::num::NonZeroUsize::new(threads[k / 4 % 3]).unwrap();
                    encoder = encoder.with_threads(n);
                }
                io::Write::write_all(&mut encoder, piece).unwrap();
            }
            encoder.finish().unwrap();
            out
        };
        let expected = compress([1, 1, 1]);
        for threads in [[1, 4, 1], [4, 1, 4], [2, usize::MAX, 1]] {
            assert!(compress(threads) == expected, "threads {threads:?}");
        }
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
            let options = Options {
                iteration_count: NonZeroU64::new(1).unwrap(),
                ..Options::default()
            };
            let mut first = deflate::TEST_PANIC.to_vec();
            first.resize(30_000, 7);
            let other = vec![3; 30_000];
            // More chunks than two threads hold: a write waits for the first
            // chunk and meets the panic.
            let pieces = || core::iter::once(&first).chain(core::iter::repeat_n(&other, 8));
            let threads = core::num::NonZeroUsize::new(2).unwrap();
            let is_test_panic = |payload: Box<dyn core::any::Any + Send>| {
                payload.downcast_ref::<&str>() == Some(&"test panic in a worker")
            };

            // Caught, then finished: an error.
            let mut encoder =
                DeflateEncoder::new(options, BlockType::Dynamic, Vec::new()).with_threads(threads);
            let written = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                for piece in pieces() {
                    io::Write::write_all(&mut encoder, piece).unwrap();
                }
            }));
            assert!(is_test_panic(written.unwrap_err()));
            assert!(io::Write::write_all(&mut encoder, &other).is_err());
            assert!(encoder.finish().is_err());

            // Not caught: the encoder is dropped while the panic unwinds.
            let dropped = std::panic::catch_unwind(|| {
                let mut encoder = DeflateEncoder::new(options, BlockType::Dynamic, Vec::new())
                    .with_threads(threads);
                for piece in pieces() {
                    io::Write::write_all(&mut encoder, piece).unwrap();
                }
            });
            assert!(is_test_panic(dropped.unwrap_err()));
        });
    }
}
