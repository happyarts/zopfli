use alloc::vec::Vec;
#[cfg(feature = "std")]
use core::num::NonZeroUsize;
use core::{cmp, iter};
#[cfg(feature = "std")]
use std::sync::{Mutex, PoisonError};

#[cfg(feature = "std")]
use log::{debug, log_enabled};

use crate::{
    blocksplitter::{blocksplit, blocksplit_lz77},
    iter::ToFlagLastIterator,
    katajainen::length_limited_code_lengths,
    lz77::{LitLen, Lz77Store},
    squeeze::{lz77_optimal, lz77_optimal_fixed},
    symbols::{
        get_dist_extra_bits, get_dist_extra_bits_value, get_dist_symbol,
        get_dist_symbol_extra_bits, get_length_extra_bits, get_length_extra_bits_value,
        get_length_symbol, get_length_symbol_extra_bits,
    },
    tree::lengths_to_symbols,
    util::{ZOPFLI_NUM_D, ZOPFLI_NUM_LL, ZOPFLI_WINDOW_SIZE},
    Error, Options, Write,
};

/// A DEFLATE encoder powered by the Zopfli algorithm that compresses data written
/// to it to the specified sink. Most users will find using [`compress`](crate::compress)
/// easier and more performant.
///
/// The data will be compressed as soon as possible, without trying to fill a
/// backreference window. As a consequence, frequent short writes may cause more
/// DEFLATE blocks to be emitted with less optimal Huffman trees, which can hurt
/// compression and runtime. If they are a concern, short writes can be conveniently
/// dealt with by wrapping this encoder with a [`BufWriter`](std::io::BufWriter), as done
/// by the [`new_buffered`](DeflateEncoder::new_buffered) method. An adequate write size
/// would be >32 KiB, which allows the second complete chunk to leverage a full-sized
/// backreference window.
pub struct DeflateEncoder<W: Write> {
    options: Options,
    btype: BlockType,
    have_chunk: bool,
    chunk_start: usize,
    window_and_chunk: Vec<u8>,
    bitwise_writer: Option<BitwiseWriter<W>>,
    /// See [`with_threads`](Self::with_threads).
    #[cfg(feature = "std")]
    threads: NonZeroUsize,
    /// With more than one thread: the chunks before the last one, worked out
    /// on other threads. Only used through `&mut self` (`get_mut`); the mutex
    /// keeps the encoder `Sync` and unwind safe.
    #[cfg(feature = "std")]
    pool: Option<Mutex<chunk_pool::ChunkPool>>,
}

impl<W: Write> DeflateEncoder<W> {
    /// Creates a new Zopfli DEFLATE encoder that will operate according to the
    /// specified options.
    pub fn new(options: Options, btype: BlockType, sink: W) -> Self {
        Self {
            options,
            btype,
            have_chunk: false,
            chunk_start: 0,
            window_and_chunk: Vec::with_capacity(ZOPFLI_WINDOW_SIZE),
            bitwise_writer: Some(BitwiseWriter::new(sink)),
            #[cfg(feature = "std")]
            threads: NonZeroUsize::MIN,
            #[cfg(feature = "std")]
            pool: None,
        }
    }

    /// Works out the chunks on up to `threads` threads at the same time. The
    /// output is the same as with one thread, which is the default.
    ///
    /// Each [`write`](Write::write) makes a chunk (1,000,000 bytes with
    /// [`compress`](crate::compress)); a chunk can only be written once the
    /// next one shows that it is not the last. With more than one thread, such
    /// a chunk goes to a worker thread together with the window before it,
    /// and is written to the sink, in order, by a later `write`,
    /// [`flush`](Write::flush) or [`finish`](Self::finish) once it is done;
    /// the last chunk is worked out by `finish` itself. So:
    ///
    /// - Only inputs of several chunks gain, and only dynamic blocks are
    ///   affected ([`BlockType::Dynamic`]).
    /// - Worker threads are started as chunks come, up to `threads`; each one
    ///   holds the data and the work of one chunk (a few tens of MB for a
    ///   1 MB chunk). At most twice as many chunks as threads are held before
    ///   a `write` waits for the oldest one. While `finish` works out the last
    ///   chunk, the workers may still be busy, so up to `threads + 1` threads
    ///   compute at the same time.
    /// - Called again after threads have started, this changes how many
    ///   chunks are held and how many more threads may start; threads already
    ///   started keep working until the encoder is finished.
    /// - An error of the sink can surface on a later call than the one whose
    ///   data caused it. A `write` that returns an error has not taken any of
    ///   its data.
    /// - `finish`, and dropping the encoder, wait until all chunks are done.
    ///   Dropped after a write to the sink failed, the encoder returns at
    ///   once: the threads finish the chunk they are working on, leave the
    ///   others and stop.
    /// - A panic of a worker thread is resumed on the thread that writes the
    ///   chunk; after that, the encoder only returns errors.
    ///
    /// Only available with the `std` feature.
    #[cfg(feature = "std")]
    pub fn with_threads(mut self, threads: NonZeroUsize) -> Self {
        self.set_threads(threads);
        self
    }

    #[cfg(feature = "std")]
    pub(crate) fn set_threads(&mut self, threads: NonZeroUsize) {
        self.threads = threads;
        if let Some(pool) = self.pool_mut() {
            pool.set_threads(threads);
        }
    }

    #[cfg(feature = "std")]
    fn pool_mut(&mut self) -> Option<&mut chunk_pool::ChunkPool> {
        self.pool
            .as_mut()
            .map(|pool| pool.get_mut().unwrap_or_else(PoisonError::into_inner))
    }

    /// Whether the chunks go through the thread pool: once it is in use, all
    /// further ones do, so that they are written in order.
    #[cfg(feature = "std")]
    fn uses_threads(&self) -> bool {
        self.btype == BlockType::Dynamic && (self.threads.get() > 1 || self.pool.is_some())
    }

    /// Creates a new Zopfli DEFLATE encoder that operates according to the
    /// specified options and is wrapped with a buffer to guarantee that
    /// data is compressed in large chunks, which is necessary for decent
    /// performance and good compression ratio.
    #[cfg(feature = "std")]
    pub fn new_buffered(options: Options, btype: BlockType, sink: W) -> std::io::BufWriter<Self> {
        std::io::BufWriter::with_capacity(
            crate::util::ZOPFLI_MASTER_BLOCK_SIZE,
            Self::new(options, btype, sink),
        )
    }

    /// Encodes any pending chunks of data and writes them to the sink,
    /// consuming the encoder and returning the wrapped sink. The sink
    /// will have received a complete DEFLATE stream when this method
    /// returns.
    ///
    /// The encoder is automatically [`finish`](Self::finish)ed when
    /// dropped, but explicitly finishing it with this method allows
    /// handling I/O errors.
    pub fn finish(mut self) -> Result<W, Error> {
        self.__finish().map(|sink| sink.unwrap())
    }

    /// Compresses the chunk stored at `window_and_chunk`. This includes
    /// a rolling window of the last `ZOPFLI_WINDOW_SIZE` data bytes, if
    /// available.
    #[inline]
    fn compress_chunk(&mut self, is_last: bool) -> Result<(), Error> {
        deflate_part(
            &self.options,
            self.btype,
            is_last,
            &self.window_and_chunk,
            self.chunk_start,
            self.window_and_chunk.len(),
            self.bitwise_writer.as_mut().unwrap(),
        )
    }

    /// Sets the next chunk that will be compressed by the next
    /// call to `compress_chunk` and updates the rolling data window
    /// accordingly.
    fn set_chunk(&mut self, chunk: &[u8]) {
        // Remove bytes exceeding the window size. Start with the
        // oldest bytes, which are at the beginning of the buffer.
        // The buffer length is then the position where the chunk
        // we've just received starts
        self.window_and_chunk.drain(
            ..self
                .window_and_chunk
                .len()
                .saturating_sub(ZOPFLI_WINDOW_SIZE),
        );
        self.chunk_start = self.window_and_chunk.len();

        self.window_and_chunk.extend_from_slice(chunk);

        self.have_chunk = true;
    }

    /// Encodes the last chunk and finishes any partial bits.
    /// The encoder will be unusable for further compression
    /// after this method returns. This is intended to be an
    /// implementation detail of the `Drop` trait and
    /// [`finish`](Self::finish) method.
    fn __finish(&mut self) -> Result<Option<W>, Error> {
        if self.bitwise_writer.is_none() {
            return Ok(None);
        }

        #[cfg(feature = "std")]
        let pooled = self.pool.is_some();
        #[cfg(not(feature = "std"))]
        let pooled = false;
        if pooled {
            #[cfg(feature = "std")]
            self.finish_pool()?;
        } else {
            self.compress_chunk(true)?;
        }

        let mut bitwise_writer = self.bitwise_writer.take().unwrap();
        bitwise_writer.finish_partial_bits()?;

        Ok(Some(bitwise_writer.out))
    }

    /// Writes all chunks of the thread pool, and the last chunk, which is
    /// worked out here meanwhile, as the final one.
    #[cfg(feature = "std")]
    fn finish_pool(&mut self) -> Result<(), Error> {
        if let Some(pool) = self.pool_mut() {
            pool.check()?;
        }
        // Without a chunk (after a write that failed), the final block is empty.
        let start = if self.have_chunk {
            self.chunk_start
        } else {
            self.window_and_chunk.len()
        };
        let (lz77, splitpoints) = blocksplit_plan(
            &self.options,
            &self.window_and_chunk,
            start,
            self.window_and_chunk.len(),
        );
        let bitwise_writer = self.bitwise_writer.as_mut().unwrap();
        let pool = self
            .pool
            .as_mut()
            .map(|pool| pool.get_mut().unwrap_or_else(PoisonError::into_inner));
        if let Some(pool) = pool {
            pool.write_all(bitwise_writer)?;
            let written = add_all_blocks(
                &splitpoints,
                &lz77,
                true,
                &self.window_and_chunk,
                bitwise_writer,
            );
            pool.set_failed(written.is_err());
            return written;
        }
        add_all_blocks(
            &splitpoints,
            &lz77,
            true,
            &self.window_and_chunk,
            bitwise_writer,
        )
    }

    /// Gets a reference to the underlying writer.
    pub fn get_ref(&self) -> &W {
        &self.bitwise_writer.as_ref().unwrap().out
    }

    /// Gets a mutable reference to the underlying writer.
    ///
    /// Note that mutating the output/input state of the stream may corrupt
    /// this object, so care must be taken when using this method.
    pub fn get_mut(&mut self) -> &mut W {
        &mut self.bitwise_writer.as_mut().unwrap().out
    }
}

impl<W: Write> Write for DeflateEncoder<W> {
    fn write(&mut self, buf: &[u8]) -> Result<usize, Error> {
        #[cfg(feature = "std")]
        if self.uses_threads() {
            if self.have_chunk {
                // The previous chunk is not the last: it goes to a worker, and its
                // end stays as the window of this one.
                let data = core::mem::take(&mut self.window_and_chunk);
                self.window_and_chunk
                    .extend_from_slice(&data[data.len().saturating_sub(ZOPFLI_WINDOW_SIZE)..]);
                self.have_chunk = false;
                let (options, threads) = (self.options, self.threads);
                self.pool
                    .get_or_insert_with(|| Mutex::new(chunk_pool::ChunkPool::new(options, threads)))
                    .get_mut()
                    .unwrap_or_else(PoisonError::into_inner)
                    .submit(data, self.chunk_start);
            }
            let bitwise_writer = self.bitwise_writer.as_mut().unwrap();
            if let Some(pool) = self.pool.as_mut() {
                pool.get_mut()
                    .unwrap_or_else(PoisonError::into_inner)
                    .write_finished(bitwise_writer)?;
            }
            self.set_chunk(buf);
            return Ok(buf.len());
        }

        // Any previous chunk is known to be non-last at this point,
        // so compress it now
        if self.have_chunk {
            self.compress_chunk(false)?;
        }

        // Set the chunk to be used for the next compression operation
        // to this chunk. We don't know whether it's last yet
        self.set_chunk(buf);

        Ok(buf.len())
    }

    fn flush(&mut self) -> Result<(), Error> {
        #[cfg(feature = "std")]
        {
            let bitwise_writer = self.bitwise_writer.as_mut().unwrap();
            if let Some(pool) = self.pool.as_mut() {
                pool.get_mut()
                    .unwrap_or_else(PoisonError::into_inner)
                    .write_finished(bitwise_writer)?;
            }
        }
        self.bitwise_writer.as_mut().unwrap().out.flush()
    }
}

impl<W: Write> Drop for DeflateEncoder<W> {
    fn drop(&mut self) {
        // After the sink failed or a thread panicked, the chunks still queued
        // for the threads are left out rather than worked out for nothing.
        #[cfg(feature = "std")]
        if self.pool_mut().is_some_and(|pool| pool.failed()) {
            return;
        }
        self.__finish().ok();
    }
}

// Boilerplate to make latest Rustdoc happy: https://github.com/rust-lang/rust/issues/117796
#[cfg(all(doc, feature = "std"))]
impl<W: crate::io::Write> std::io::Write for DeflateEncoder<W> {
    fn write(&mut self, _buf: &[u8]) -> std::io::Result<usize> {
        unimplemented!()
    }

    fn flush(&mut self) -> std::io::Result<()> {
        unimplemented!()
    }
}

/// Deflate a part, to allow for chunked, streaming compression with [`DeflateEncoder`].
/// It is possible to call this function multiple times in a row, shifting
/// instart and inend to next bytes of the data. If instart is larger than 0, then
/// previous bytes are used as the initial dictionary for LZ77.
/// This function will usually output multiple deflate blocks. If final is true, then
/// the final bit will be set on the last block.
/// Like deflate, but allows to specify start and end byte with instart and
/// inend. Only that part is compressed, but earlier bytes are still used for the
/// back window.
fn deflate_part<W: Write>(
    options: &Options,
    btype: BlockType,
    final_block: bool,
    in_data: &[u8],
    instart: usize,
    inend: usize,
    bitwise_writer: &mut BitwiseWriter<W>,
) -> Result<(), Error> {
    /* If btype=Dynamic is specified, it tries all block types. If a lesser btype is
    given, then however it forces that one. Neither of the lesser types needs
    block splitting as they have no dynamic huffman trees. */
    match btype {
        BlockType::Uncompressed => {
            add_non_compressed_block(final_block, in_data, instart, inend, bitwise_writer)
        }
        BlockType::Fixed => {
            let mut store = Lz77Store::new();

            lz77_optimal_fixed(in_data, instart, inend, &mut store);
            add_lz77_block(
                btype,
                final_block,
                in_data,
                &store,
                0,
                store.size(),
                0,
                bitwise_writer,
            )
        }
        BlockType::Dynamic => blocksplit_attempt(
            options,
            final_block,
            in_data,
            instart,
            inend,
            bitwise_writer,
        ),
    }
}

/// The type of data blocks to generate for a DEFLATE stream.
#[derive(PartialEq, Eq, Copy, Clone, Debug)]
#[cfg_attr(all(test, feature = "std"), derive(proptest_derive::Arbitrary))]
#[derive(Default)]
pub enum BlockType {
    /// Non-compressed blocks (BTYPE=00).
    ///
    /// The input data will be divided into chunks up to 64 KiB big and
    /// stored in the DEFLATE stream without compression. This is mainly
    /// useful for test and development purposes.
    Uncompressed,
    /// Compressed blocks with fixed Huffman codes (BTYPE=01).
    ///
    /// The input data will be compressed into DEFLATE blocks using a fixed
    /// Huffman tree defined in the DEFLATE specification. This provides fast
    /// but poor compression, as the Zopfli algorithm is not actually used.
    Fixed,
    /// Select the most space-efficient block types for the input data.
    /// This is the recommended type for the vast majority of Zopfli
    /// applications.
    ///
    /// This mode lets the Zopfli algorithm choose the combination of block
    /// types that minimizes data size. The emitted block types may be
    /// [`Uncompressed`](Self::Uncompressed) or [`Fixed`](Self::Fixed), in
    /// addition to compressed with dynamic Huffman codes (BTYPE=10).
    #[default]
    Dynamic,
}

fn fixed_tree() -> (Vec<u32>, Vec<u32>) {
    let mut ll = Vec::with_capacity(ZOPFLI_NUM_LL);
    ll.resize(144, 8);
    ll.resize(256, 9);
    ll.resize(280, 7);
    ll.resize(288, 8);
    let d = vec![5; ZOPFLI_NUM_D];
    (ll, d)
}

/// Changes the population counts in a way that the consequent Huffman tree
/// compression, especially its rle-part, will be more likely to compress this data
/// more efficiently. length contains the size of the histogram.
fn optimize_huffman_for_rle(counts: &mut [usize]) {
    let mut length = counts.len();
    // 1) We don't want to touch the trailing zeros. We may break the
    // rules of the format by adding more data in the distance codes.
    loop {
        if length == 0 {
            return;
        }
        if counts[length - 1] != 0 {
            // Now counts[0..length - 1] does not have trailing zeros.
            break;
        }
        length -= 1;
    }

    // 2) Let's mark all population counts that already can be encoded
    // with an rle code.
    let mut good_for_rle = vec![false; length];

    // Let's not spoil any of the existing good rle codes.
    // Mark any seq of 0's that is longer than 5 as a good_for_rle.
    // Mark any seq of non-0's that is longer than 7 as a good_for_rle.
    let mut symbol = counts[0];
    let mut stride = 0;
    for (i, &count) in counts.iter().enumerate().take(length) {
        if count == symbol {
            stride += 1;
        } else {
            if (symbol == 0 && stride >= 5) || (symbol != 0 && stride >= 7) {
                for k in 0..stride {
                    good_for_rle[i - k - 1] = true;
                }
            }
            stride = 1;
            symbol = count;
        }
    }

    // 3) Let's replace those population counts that lead to more rle codes.
    stride = 0;
    let mut limit = counts[0];
    let mut sum = 0;
    for i in 0..=length {
        // Heuristic for selecting the stride ranges to collapse.
        if i == length || good_for_rle[i] || (counts[i] as i32 - limit as i32).abs() >= 4 {
            if stride >= 4 || (stride >= 3 && sum == 0) {
                // The stride must end, collapse what we have, if we have enough (4).
                let count = if sum == 0 {
                    // Don't upgrade an all zeros stride to ones.
                    0
                } else {
                    cmp::max((sum + stride / 2) / stride, 1)
                };
                set_counts_to_count(counts, count, i, stride);
            }
            stride = 0;
            sum = 0;
            if length > 2 && i < length - 3 {
                // All interesting strides have a count of at least 4,
                // at least when non-zeros.
                limit = (counts[i] + counts[i + 1] + counts[i + 2] + counts[i + 3] + 2) / 4;
            } else if i < length {
                limit = counts[i];
            } else {
                limit = 0;
            }
        }
        stride += 1;
        if i != length {
            sum += counts[i];
        }
    }
}

// Ensures there are at least 2 distance codes to support buggy decoders.
// Zlib 1.2.1 and below have a bug where it fails if there isn't at least 1
// distance code (with length > 0), even though it's valid according to the
// deflate spec to have 0 distance codes. On top of that, some mobile phones
// require at least two distance codes. To support these decoders too (but
// potentially at the cost of a few bytes), add dummy code lengths of 1.
// References to this bug can be found in the changelog of
// Zlib 1.2.2 and here: http://www.jonof.id.au/forum/index.php?topic=515.0.
//
// d_lengths: the 32 lengths of the distance codes.
fn patch_distance_codes_for_buggy_decoders(d_lengths: &mut [u32]) {
    // Ignore the two unused codes from the spec
    let num_dist_codes = d_lengths
        .iter()
        .take(30)
        .filter(|&&d_length| d_length != 0)
        .count();

    match num_dist_codes {
        0 => {
            d_lengths[0] = 1;
            d_lengths[1] = 1;
        }
        1 => {
            let index = usize::from(d_lengths[0] != 0);
            d_lengths[index] = 1;
        }
        _ => {} // Two or more codes is fine.
    }
}

/// Same as `calculate_block_symbol_size`, but for block size smaller than histogram
/// size.
fn calculate_block_symbol_size_small(
    ll_lengths: &[u32],
    d_lengths: &[u32],
    lz77: &Lz77Store,
    lstart: usize,
    lend: usize,
) -> usize {
    let mut result = 0;

    debug_assert!(lend == lstart || lend - 1 < lz77.size());

    for &item in &lz77.litlens[lstart..lend] {
        match item {
            LitLen::Literal(litlens_i) => {
                debug_assert!(litlens_i < 259);
                result += ll_lengths[litlens_i as usize];
            }
            LitLen::LengthDist(litlens_i, dists_i) => {
                debug_assert!(litlens_i < 259);
                let ll_symbol = get_length_symbol(litlens_i as usize);
                let d_symbol = get_dist_symbol(dists_i) as usize;
                result += ll_lengths[ll_symbol];
                result += d_lengths[d_symbol];
                result += get_length_symbol_extra_bits(ll_symbol);
                result += get_dist_symbol_extra_bits(d_symbol);
            }
        }
    }
    result += ll_lengths[256]; // end symbol
    result as usize
}

/// Same as `calculate_block_symbol_size`, but with the histogram provided by the caller.
fn calculate_block_symbol_size_given_counts(
    ll_counts: &[usize],
    d_counts: &[usize],
    ll_lengths: &[u32],
    d_lengths: &[u32],
    lz77: &Lz77Store,
    lstart: usize,
    lend: usize,
) -> usize {
    if lstart + ZOPFLI_NUM_LL * 3 > lend {
        calculate_block_symbol_size_small(ll_lengths, d_lengths, lz77, lstart, lend)
    } else {
        let mut result = 0;
        for i in 0..256 {
            result += ll_lengths[i] * ll_counts[i] as u32;
        }
        for i in 257..286 {
            result += ll_lengths[i] * ll_counts[i] as u32;
            result += (get_length_symbol_extra_bits(i) as usize * ll_counts[i]) as u32;
        }
        for i in 0..30 {
            result += d_lengths[i] * d_counts[i] as u32;
            result += (get_dist_symbol_extra_bits(i) as usize * d_counts[i]) as u32;
        }
        result += ll_lengths[256]; // end symbol
        result as usize
    }
}

/// Calculates size of the part after the header and tree of an LZ77 block, in bits.
fn calculate_block_symbol_size(
    ll_lengths: &[u32],
    d_lengths: &[u32],
    lz77: &Lz77Store,
    lstart: usize,
    lend: usize,
) -> usize {
    if lstart + ZOPFLI_NUM_LL * 3 > lend {
        calculate_block_symbol_size_small(ll_lengths, d_lengths, lz77, lstart, lend)
    } else {
        let (ll_counts, d_counts) = lz77.get_histogram(lstart, lend);
        calculate_block_symbol_size_given_counts(
            &*ll_counts,
            &*d_counts,
            ll_lengths,
            d_lengths,
            lz77,
            lstart,
            lend,
        )
    }
}

/// Encodes the Huffman tree and returns how many bits its encoding takes; only returns the size
/// and runs faster.
fn encode_tree_no_output(
    ll_lengths: &[u32],
    d_lengths: &[u32],
    use_16: bool,
    use_17: bool,
    use_18: bool,
) -> usize {
    let mut hlit = 29; /* 286 - 257 */
    let mut hdist = 29; /* 32 - 1, but gzip does not like hdist > 29.*/

    let mut clcounts = [0; 19];
    /* The order in which code length code lengths are encoded as per deflate. */
    let order = [
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
    ];
    let mut result_size = 0;

    /* Trim zeros. */
    while hlit > 0 && ll_lengths[257 + hlit - 1] == 0 {
        hlit -= 1;
    }
    while hdist > 0 && d_lengths[1 + hdist - 1] == 0 {
        hdist -= 1;
    }
    let hlit2 = hlit + 257;

    let lld_total = hlit2 + hdist + 1; /* Total amount of literal, length, distance codes. */

    let mut i = 0;

    while i < lld_total {
        /* This is an encoding of a huffman tree, so now the length is a symbol */
        let symbol = if i < hlit2 {
            ll_lengths[i]
        } else {
            d_lengths[i - hlit2]
        } as u8;

        let mut count = 1;
        if use_16 || (symbol == 0 && (use_17 || use_18)) {
            let mut j = i + 1;
            let mut symbol_calc = if j < hlit2 {
                ll_lengths[j]
            } else {
                d_lengths[j - hlit2]
            } as u8;

            while j < lld_total && symbol == symbol_calc {
                count += 1;
                j += 1;
                symbol_calc = if j < hlit2 {
                    ll_lengths[j]
                } else {
                    d_lengths[j - hlit2]
                } as u8;
            }
        }

        i += count - 1;

        /* Repetitions of zeroes */
        if symbol == 0 && count >= 3 {
            if use_18 {
                while count >= 11 {
                    let count2 = if count > 138 { 138 } else { count };
                    clcounts[18] += 1;
                    count -= count2;
                }
            }
            if use_17 {
                while count >= 3 {
                    let count2 = if count > 10 { 10 } else { count };
                    clcounts[17] += 1;
                    count -= count2;
                }
            }
        }

        /* Repetitions of any symbol */
        if use_16 && count >= 4 {
            count -= 1; /* Since the first one is hardcoded. */
            clcounts[symbol as usize] += 1;
            while count >= 3 {
                let count2 = if count > 6 { 6 } else { count };
                clcounts[16] += 1;
                count -= count2;
            }
        }

        /* No or insufficient repetition */
        clcounts[symbol as usize] += count;
        while count > 0 {
            count -= 1;
        }
        i += 1;
    }

    let clcl = length_limited_code_lengths(&clcounts, 7);

    let mut hclen = 15;
    /* Trim zeros. */
    while hclen > 0 && clcounts[order[hclen + 4 - 1]] == 0 {
        hclen -= 1;
    }

    result_size += 14; /* hlit, hdist, hclen bits */
    result_size += (hclen + 4) * 3; /* clcl bits */
    for i in 0..19 {
        result_size += clcl[i] as usize * clcounts[i];
    }
    /* Extra bits. */
    result_size += clcounts[16] * 2;
    result_size += clcounts[17] * 3;
    result_size += clcounts[18] * 7;

    result_size
}

/// Gives the exact size of the tree, in bits, as it will be encoded in DEFLATE.
fn calculate_tree_size(ll_lengths: &[u32], d_lengths: &[u32]) -> usize {
    static TRUTH_TABLE: [(bool, bool, bool); 8] = [
        (false, false, false),
        (true, false, false),
        (false, true, false),
        (true, true, false),
        (false, false, true),
        (true, false, true),
        (false, true, true),
        (true, true, true),
    ];

    TRUTH_TABLE
        .iter()
        .map(|&(use_16, use_17, use_18)| {
            encode_tree_no_output(ll_lengths, d_lengths, use_16, use_17, use_18)
        })
        .min()
        .unwrap_or(0)
}

/// Encodes the Huffman tree and returns how many bits its encoding takes and returns output.
// TODO: This return value is unused.
fn encode_tree<W: Write>(
    ll_lengths: &[u32],
    d_lengths: &[u32],
    use_16: bool,
    use_17: bool,
    use_18: bool,
    bitwise_writer: &mut BitwiseWriter<W>,
) -> Result<usize, Error> {
    let mut hlit = 29; /* 286 - 257 */
    let mut hdist = 29; /* 32 - 1, but gzip does not like hdist > 29.*/

    let mut clcounts = [0; 19];
    /* The order in which code length code lengths are encoded as per deflate. */
    let order = [
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
    ];
    let mut result_size = 0;

    let mut rle = vec![];
    let mut rle_bits = vec![];

    /* Trim zeros. */
    while hlit > 0 && ll_lengths[257 + hlit - 1] == 0 {
        hlit -= 1;
    }
    while hdist > 0 && d_lengths[1 + hdist - 1] == 0 {
        hdist -= 1;
    }
    let hlit2 = hlit + 257;

    let lld_total = hlit2 + hdist + 1; /* Total amount of literal, length, distance codes. */

    let mut i = 0;

    while i < lld_total {
        /* This is an encoding of a huffman tree, so now the length is a symbol */
        let symbol = if i < hlit2 {
            ll_lengths[i]
        } else {
            d_lengths[i - hlit2]
        } as u8;

        let mut count = 1;
        if use_16 || (symbol == 0 && (use_17 || use_18)) {
            let mut j = i + 1;
            let mut symbol_calc = if j < hlit2 {
                ll_lengths[j]
            } else {
                d_lengths[j - hlit2]
            } as u8;

            while j < lld_total && symbol == symbol_calc {
                count += 1;
                j += 1;
                symbol_calc = if j < hlit2 {
                    ll_lengths[j]
                } else {
                    d_lengths[j - hlit2]
                } as u8;
            }
        }

        i += count - 1;

        /* Repetitions of zeroes */
        if symbol == 0 && count >= 3 {
            if use_18 {
                while count >= 11 {
                    let count2 = if count > 138 { 138 } else { count };
                    rle.push(18);
                    rle_bits.push(count2 - 11);
                    clcounts[18] += 1;
                    count -= count2;
                }
            }
            if use_17 {
                while count >= 3 {
                    let count2 = if count > 10 { 10 } else { count };
                    rle.push(17);
                    rle_bits.push(count2 - 3);
                    clcounts[17] += 1;
                    count -= count2;
                }
            }
        }

        /* Repetitions of any symbol */
        if use_16 && count >= 4 {
            count -= 1; /* Since the first one is hardcoded. */
            clcounts[symbol as usize] += 1;
            rle.push(symbol);
            rle_bits.push(0);

            while count >= 3 {
                let count2 = if count > 6 { 6 } else { count };
                rle.push(16);
                rle_bits.push(count2 - 3);
                clcounts[16] += 1;
                count -= count2;
            }
        }

        /* No or insufficient repetition */
        clcounts[symbol as usize] += count;
        while count > 0 {
            rle.push(symbol);
            rle_bits.push(0);
            count -= 1;
        }
        i += 1;
    }

    let clcl = length_limited_code_lengths(&clcounts, 7);
    let clsymbols = lengths_to_symbols(&clcl, 7);

    let mut hclen = 15;
    /* Trim zeros. */
    while hclen > 0 && clcounts[order[hclen + 4 - 1]] == 0 {
        hclen -= 1;
    }

    bitwise_writer.add_bits(hlit as u32, 5)?;
    bitwise_writer.add_bits(hdist as u32, 5)?;
    bitwise_writer.add_bits(hclen as u32, 4)?;

    for &item in order.iter().take(hclen + 4) {
        bitwise_writer.add_bits(clcl[item], 3)?;
    }

    for i in 0..rle.len() {
        let rle_i = rle[i] as usize;
        let rle_bits_i = rle_bits[i] as u32;
        let sym = clsymbols[rle_i];
        bitwise_writer.add_huffman_bits(sym, clcl[rle_i])?;
        /* Extra bits. */
        if rle_i == 16 {
            bitwise_writer.add_bits(rle_bits_i, 2)?;
        } else if rle_i == 17 {
            bitwise_writer.add_bits(rle_bits_i, 3)?;
        } else if rle_i == 18 {
            bitwise_writer.add_bits(rle_bits_i, 7)?;
        }
    }

    result_size += 14; /* hlit, hdist, hclen bits */
    result_size += (hclen + 4) * 3; /* clcl bits */
    for i in 0..19 {
        result_size += clcl[i] as usize * clcounts[i];
    }
    /* Extra bits. */
    result_size += clcounts[16] * 2;
    result_size += clcounts[17] * 3;
    result_size += clcounts[18] * 7;

    Ok(result_size)
}

fn add_dynamic_tree<W: Write>(
    ll_lengths: &[u32],
    d_lengths: &[u32],
    bitwise_writer: &mut BitwiseWriter<W>,
) -> Result<(), Error> {
    let mut best = 0;
    let mut bestsize = 0;

    for i in 0..8 {
        let size = encode_tree_no_output(ll_lengths, d_lengths, i & 1 > 0, i & 2 > 0, i & 4 > 0);
        if bestsize == 0 || size < bestsize {
            bestsize = size;
            best = i;
        }
    }

    encode_tree(
        ll_lengths,
        d_lengths,
        best & 1 > 0,
        best & 2 > 0,
        best & 4 > 0,
        bitwise_writer,
    )
    .map(|_| ())
}

/// Adds a deflate block with the given LZ77 data to the output.
/// `options`: global program options
/// `btype`: the block type, must be `Fixed` or `Dynamic`
/// `final`: whether to set the "final" bit on this block, must be the last block
/// `litlens`: literal/length array of the LZ77 data, in the same format as in
///     `Lz77Store`.
/// `dists`: distance array of the LZ77 data, in the same format as in
///     `Lz77Store`.
/// `lstart`: where to start in the LZ77 data
/// `lend`: where to end in the LZ77 data (not inclusive)
/// `expected_data_size`: the uncompressed block size, used for assert, but you can
///   set it to `0` to not do the assertion.
/// `bitwise_writer`: writer responsible for appending bits
#[allow(clippy::too_many_arguments)] // Not feasible to refactor in a more readable way
fn add_lz77_block<W: Write>(
    btype: BlockType,
    final_block: bool,
    in_data: &[u8],
    lz77: &Lz77Store,
    lstart: usize,
    lend: usize,
    expected_data_size: usize,
    bitwise_writer: &mut BitwiseWriter<W>,
) -> Result<(), Error> {
    if btype == BlockType::Uncompressed {
        let length = lz77.get_byte_range(lstart, lend);
        let pos = if lstart == lend { 0 } else { lz77.pos[lstart] };
        let end = pos + length;
        return add_non_compressed_block(final_block, in_data, pos, end, bitwise_writer);
    }

    bitwise_writer.add_bit(u8::from(final_block))?;

    let (ll_lengths, d_lengths) = match btype {
        BlockType::Uncompressed => unreachable!(),
        BlockType::Fixed => {
            bitwise_writer.add_bit(1)?;
            bitwise_writer.add_bit(0)?;
            fixed_tree()
        }
        BlockType::Dynamic => {
            bitwise_writer.add_bit(0)?;
            bitwise_writer.add_bit(1)?;
            let (_, ll_lengths, d_lengths) = get_dynamic_lengths(lz77, lstart, lend);

            let _detect_tree_size = bitwise_writer.bytes_written();
            add_dynamic_tree(&ll_lengths, &d_lengths, bitwise_writer)?;
            debug!(
                "treesize: {}",
                bitwise_writer.bytes_written() - _detect_tree_size
            );
            (ll_lengths, d_lengths)
        }
    };

    let ll_symbols = lengths_to_symbols(&ll_lengths, 15);
    let d_symbols = lengths_to_symbols(&d_lengths, 15);

    let detect_block_size = bitwise_writer.bytes_written();
    add_lz77_data(
        lz77,
        lstart,
        lend,
        expected_data_size,
        &ll_symbols,
        &ll_lengths,
        &d_symbols,
        &d_lengths,
        bitwise_writer,
    )?;

    /* End symbol. */
    bitwise_writer.add_huffman_bits(ll_symbols[256], ll_lengths[256])?;

    if log_enabled!(log::Level::Debug) {
        let _uncompressed_size = lz77.litlens[lstart..lend]
            .iter()
            .fold(0, |acc, &x| acc + x.size());
        let _compressed_size = bitwise_writer.bytes_written() - detect_block_size;
        debug!(
            "compressed block size: {} ({}k) (unc: {})",
            _compressed_size,
            _compressed_size / 1024,
            _uncompressed_size
        );
    }

    Ok(())
}

/// Calculates block size in bits.
/// litlens: lz77 lit/lengths
/// dists: ll77 distances
/// lstart: start of block
/// lend: end of block (not inclusive)
pub fn calculate_block_size(lz77: &Lz77Store, lstart: usize, lend: usize, btype: BlockType) -> f64 {
    match btype {
        BlockType::Uncompressed => {
            let length = lz77.get_byte_range(lstart, lend);
            let rem = length % 65535;
            let blocks = length / 65535 + usize::from(rem > 0);
            /* An uncompressed block must actually be split into multiple blocks if it's
            larger than 65535 bytes long. Eeach block header is 5 bytes: 3 bits,
            padding, LEN and NLEN (potential less padding for first one ignored). */
            (blocks * 5 * 8 + length * 8) as f64
        }
        BlockType::Fixed => {
            let fixed_tree = fixed_tree();
            let ll_lengths = fixed_tree.0;
            let d_lengths = fixed_tree.1;

            let mut result = 3.0; /* bfinal and btype bits */
            result +=
                calculate_block_symbol_size(&ll_lengths, &d_lengths, lz77, lstart, lend) as f64;
            result
        }
        BlockType::Dynamic => get_dynamic_lengths(lz77, lstart, lend).0 + 3.0,
    }
}

/// Tries out `OptimizeHuffmanForRle` for this block, if the result is smaller,
/// uses it, otherwise keeps the original. Returns size of encoded tree and data in
/// bits, not including the 3-bit block header.
fn try_optimize_huffman_for_rle(
    lz77: &Lz77Store,
    lstart: usize,
    lend: usize,
    ll_counts: &[usize],
    d_counts: &[usize],
    ll_lengths: Vec<u32>,
    d_lengths: Vec<u32>,
) -> (f64, Vec<u32>, Vec<u32>) {
    let mut ll_counts2 = Vec::from(ll_counts);
    let mut d_counts2 = Vec::from(d_counts);

    let treesize = calculate_tree_size(&ll_lengths, &d_lengths);
    let datasize = calculate_block_symbol_size_given_counts(
        ll_counts,
        d_counts,
        &ll_lengths,
        &d_lengths,
        lz77,
        lstart,
        lend,
    );

    optimize_huffman_for_rle(&mut ll_counts2);
    optimize_huffman_for_rle(&mut d_counts2);

    let ll_lengths2 = length_limited_code_lengths(&ll_counts2, 15);
    let mut d_lengths2 = length_limited_code_lengths(&d_counts2, 15);
    patch_distance_codes_for_buggy_decoders(&mut d_lengths2[..]);

    let treesize2 = calculate_tree_size(&ll_lengths2, &d_lengths2);
    let datasize2 = calculate_block_symbol_size_given_counts(
        ll_counts,
        d_counts,
        &ll_lengths2,
        &d_lengths2,
        lz77,
        lstart,
        lend,
    );

    if treesize2 + datasize2 < treesize + datasize {
        (((treesize2 + datasize2) as f64), ll_lengths2, d_lengths2)
    } else {
        ((treesize + datasize) as f64, ll_lengths, d_lengths)
    }
}

/// Calculates the bit lengths for the symbols for dynamic blocks. Chooses bit
/// lengths that give the smallest size of tree encoding + encoding of all the
/// symbols to have smallest output size. This are not necessarily the ideal Huffman
/// bit lengths. Returns size of encoded tree and data in bits, not including the
/// 3-bit block header.
fn get_dynamic_lengths(lz77: &Lz77Store, lstart: usize, lend: usize) -> (f64, Vec<u32>, Vec<u32>) {
    let (mut ll_counts, d_counts) = lz77.get_histogram(lstart, lend);
    ll_counts[256] = 1; /* End symbol. */

    let ll_lengths = length_limited_code_lengths(&*ll_counts, 15);
    let mut d_lengths = length_limited_code_lengths(&*d_counts, 15);

    patch_distance_codes_for_buggy_decoders(&mut d_lengths[..]);

    try_optimize_huffman_for_rle(
        lz77,
        lstart,
        lend,
        &*ll_counts,
        &*d_counts,
        ll_lengths,
        d_lengths,
    )
}

/// Adds all lit/len and dist codes from the lists as huffman symbols. Does not add
/// end code 256. `expected_data_size` is the uncompressed block size, used for
/// assert, but you can set it to `0` to not do the assertion.
#[allow(clippy::too_many_arguments)] // Not feasible to refactor in a more readable way
fn add_lz77_data<W: Write>(
    lz77: &Lz77Store,
    lstart: usize,
    lend: usize,
    expected_data_size: usize,
    ll_symbols: &[u32],
    ll_lengths: &[u32],
    d_symbols: &[u32],
    d_lengths: &[u32],
    bitwise_writer: &mut BitwiseWriter<W>,
) -> Result<(), Error> {
    let mut testlength = 0;

    for &item in &lz77.litlens[lstart..lend] {
        match item {
            LitLen::Literal(lit) => {
                let litlen = lit as usize;
                debug_assert!(litlen < 256);
                debug_assert!(ll_lengths[litlen] > 0);
                bitwise_writer.add_huffman_bits(ll_symbols[litlen], ll_lengths[litlen])?;
                testlength += 1;
            }
            LitLen::LengthDist(len, dist) => {
                let litlen = len as usize;
                assert!((3..=288).contains(&litlen)); // Eases inlining and gets rid of index bound checks below
                let lls = get_length_symbol(litlen);
                let ds = get_dist_symbol(dist) as usize;
                debug_assert!(ll_lengths[lls] > 0);
                debug_assert!(d_lengths[ds] > 0);
                bitwise_writer.add_huffman_bits(ll_symbols[lls], ll_lengths[lls])?;
                bitwise_writer.add_bits(
                    get_length_extra_bits_value(litlen),
                    get_length_extra_bits(litlen),
                )?;
                bitwise_writer.add_huffman_bits(d_symbols[ds], d_lengths[ds])?;
                bitwise_writer.add_bits(
                    u32::from(get_dist_extra_bits_value(dist)),
                    get_dist_extra_bits(dist),
                )?;
                testlength += litlen;
            }
        }
    }
    debug_assert!(expected_data_size == 0 || testlength == expected_data_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)] // Not feasible to refactor in a more readable way
fn add_lz77_block_auto_type<W: Write>(
    final_block: bool,
    in_data: &[u8],
    lz77: &Lz77Store,
    lstart: usize,
    lend: usize,
    expected_data_size: usize,
    bitwise_writer: &mut BitwiseWriter<W>,
) -> Result<(), Error> {
    let uncompressedcost = calculate_block_size(lz77, lstart, lend, BlockType::Uncompressed);
    let mut fixedcost = calculate_block_size(lz77, lstart, lend, BlockType::Fixed);
    let dyncost = calculate_block_size(lz77, lstart, lend, BlockType::Dynamic);

    /* Whether to perform the expensive calculation of creating an optimal block
    with fixed huffman tree to check if smaller. Only do this for small blocks or
    blocks which already are pretty good with fixed huffman tree. */
    let expensivefixed = (lz77.size() < 1000) || fixedcost <= dyncost * 1.1;

    let mut fixedstore = Lz77Store::new();
    if lstart == lend {
        /* Smallest empty block is represented by fixed block */
        bitwise_writer.add_bits(u32::from(final_block), 1)?;
        bitwise_writer.add_bits(1, 2)?; /* btype 01 */
        bitwise_writer.add_bits(0, 7)?; /* end symbol has code 0000000 */
        return Ok(());
    }
    if expensivefixed {
        /* Recalculate the LZ77 with lz77_optimal_fixed */
        let instart = lz77.pos[lstart];
        let inend = instart + lz77.get_byte_range(lstart, lend);

        lz77_optimal_fixed(in_data, instart, inend, &mut fixedstore);
        fixedcost = calculate_block_size(&fixedstore, 0, fixedstore.size(), BlockType::Fixed);
    }

    if uncompressedcost <= fixedcost && uncompressedcost <= dyncost {
        add_lz77_block(
            BlockType::Uncompressed,
            final_block,
            in_data,
            lz77,
            lstart,
            lend,
            expected_data_size,
            bitwise_writer,
        )
    } else if fixedcost <= dyncost {
        if expensivefixed {
            add_lz77_block(
                BlockType::Fixed,
                final_block,
                in_data,
                &fixedstore,
                0,
                fixedstore.size(),
                expected_data_size,
                bitwise_writer,
            )
        } else {
            add_lz77_block(
                BlockType::Fixed,
                final_block,
                in_data,
                lz77,
                lstart,
                lend,
                expected_data_size,
                bitwise_writer,
            )
        }
    } else {
        add_lz77_block(
            BlockType::Dynamic,
            final_block,
            in_data,
            lz77,
            lstart,
            lend,
            expected_data_size,
            bitwise_writer,
        )
    }
}

/// Calculates block size in bits, automatically using the best btype.
pub fn calculate_block_size_auto_type(lz77: &Lz77Store, lstart: usize, lend: usize) -> f64 {
    let uncompressedcost = calculate_block_size(lz77, lstart, lend, BlockType::Uncompressed);
    /* Don't do the expensive fixed cost calculation for larger blocks that are
    unlikely to use it. */
    let fixedcost = if lz77.size() > 1000 {
        uncompressedcost
    } else {
        calculate_block_size(lz77, lstart, lend, BlockType::Fixed)
    };
    let dyncost = calculate_block_size(lz77, lstart, lend, BlockType::Dynamic);
    uncompressedcost.min(fixedcost).min(dyncost)
}

fn add_all_blocks<W: Write>(
    splitpoints: &[usize],
    lz77: &Lz77Store,
    final_block: bool,
    in_data: &[u8],
    bitwise_writer: &mut BitwiseWriter<W>,
) -> Result<(), Error> {
    let mut last = 0;
    for &item in splitpoints {
        add_lz77_block_auto_type(false, in_data, lz77, last, item, 0, bitwise_writer)?;
        last = item;
    }
    add_lz77_block_auto_type(
        final_block,
        in_data,
        lz77,
        last,
        lz77.size(),
        0,
        bitwise_writer,
    )
}

fn blocksplit_attempt<W: Write>(
    options: &Options,
    final_block: bool,
    in_data: &[u8],
    instart: usize,
    inend: usize,
    bitwise_writer: &mut BitwiseWriter<W>,
) -> Result<(), Error> {
    let (lz77, splitpoints) = blocksplit_plan(options, in_data, instart, inend);
    add_all_blocks(&splitpoints, &lz77, final_block, in_data, bitwise_writer)
}

/// The LZ77 data of `instart..inend` and the LZ77 indices where
/// `blocksplit_attempt` splits it into blocks.
fn blocksplit_plan(
    options: &Options,
    in_data: &[u8],
    instart: usize,
    inend: usize,
) -> (Lz77Store, Vec<usize>) {
    let mut totalcost = 0.0;
    let mut lz77 = Lz77Store::new();

    /* byte coordinates rather than lz77 index */
    let mut splitpoints_uncompressed = Vec::with_capacity(options.maximum_block_splits as usize);

    blocksplit(
        in_data,
        instart,
        inend,
        options.maximum_block_splits,
        &mut splitpoints_uncompressed,
    );
    let npoints = splitpoints_uncompressed.len();
    let mut splitpoints = Vec::with_capacity(npoints);

    let mut last = instart;
    for &item in &splitpoints_uncompressed {
        let store = lz77_optimal(
            in_data,
            last,
            item,
            options.iteration_count.get(),
            options.iterations_without_improvement.get(),
        );
        totalcost += calculate_block_size_auto_type(&store, 0, store.size());

        // ZopfliAppendLZ77Store(&store, &lz77);
        debug_assert!(instart == inend || store.size() > 0);
        for (&litlens, &pos) in store.litlens.iter().zip(store.pos.iter()) {
            lz77.append_store_item(litlens, pos);
        }

        splitpoints.push(lz77.size());

        last = item;
    }

    let store = lz77_optimal(
        in_data,
        last,
        inend,
        options.iteration_count.get(),
        options.iterations_without_improvement.get(),
    );
    totalcost += calculate_block_size_auto_type(&store, 0, store.size());

    // ZopfliAppendLZ77Store(&store, &lz77);
    debug_assert!(instart == inend || store.size() > 0);
    for (&litlens, &pos) in store.litlens.iter().zip(store.pos.iter()) {
        lz77.append_store_item(litlens, pos);
    }

    /* Second block splitting attempt */
    if npoints > 1 {
        let mut splitpoints2 = Vec::with_capacity(splitpoints_uncompressed.len());
        let mut totalcost2 = 0.0;

        blocksplit_lz77(&lz77, options.maximum_block_splits, &mut splitpoints2);

        let mut last = 0;
        for &item in &splitpoints2 {
            totalcost2 += calculate_block_size_auto_type(&lz77, last, item);
            last = item;
        }
        totalcost2 += calculate_block_size_auto_type(&lz77, last, lz77.size());

        if totalcost2 < totalcost {
            splitpoints = splitpoints2;
        }
    }

    (lz77, splitpoints)
}

/// Since an uncompressed block can be max 65535 in size, it actually adds
/// multiple blocks if needed.
fn add_non_compressed_block<W: Write>(
    final_block: bool,
    in_data: &[u8],
    instart: usize,
    inend: usize,
    bitwise_writer: &mut BitwiseWriter<W>,
) -> Result<(), Error> {
    let in_data = &in_data[instart..inend];

    let in_data_chunks = in_data.chunks(65535).size_hint().0;

    for (chunk, is_final) in in_data
        .chunks(65535)
        .flag_last()
        // Make sure that we output at least one chunk if this is the final block
        .chain(iter::once((&[][..], true)))
        .take(if final_block {
            cmp::max(in_data_chunks, 1)
        } else {
            in_data_chunks
        })
    {
        let blocksize = chunk.len();
        let nlen = !blocksize;

        bitwise_writer.add_bit(u8::from(final_block && is_final))?;
        /* BTYPE 00 */
        bitwise_writer.add_bit(0)?;
        bitwise_writer.add_bit(0)?;

        bitwise_writer.finish_partial_bits()?;

        bitwise_writer.add_byte((blocksize % 256) as u8)?;
        bitwise_writer.add_byte(((blocksize / 256) % 256) as u8)?;
        bitwise_writer.add_byte((nlen % 256) as u8)?;
        bitwise_writer.add_byte(((nlen / 256) % 256) as u8)?;

        bitwise_writer.add_bytes(chunk)?;
    }

    Ok(())
}

struct BitwiseWriter<W> {
    bit: u8,
    bp: u8,
    len: usize,
    out: W,
}

impl<W: Write> BitwiseWriter<W> {
    const fn new(out: W) -> Self {
        Self {
            bit: 0,
            bp: 0,
            len: 0,
            out,
        }
    }

    const fn bytes_written(&self) -> usize {
        self.len + if self.bp > 0 { 1 } else { 0 }
    }

    /// For when you want to add a full byte.
    fn add_byte(&mut self, byte: u8) -> Result<(), Error> {
        self.add_bytes(&[byte])
    }

    /// For adding a slice of bytes.
    fn add_bytes(&mut self, bytes: &[u8]) -> Result<(), Error> {
        self.len += bytes.len();
        self.out.write_all(bytes)
    }

    fn add_bit(&mut self, bit: u8) -> Result<(), Error> {
        // Flush a full byte before buffering the new bit so that writer state stays constant
        // between failed writes
        if self.bp >= 8 {
            self.finish_partial_bits()?;
        }

        self.bit |= bit << self.bp;
        self.bp += 1;

        Ok(())
    }

    fn add_bits(&mut self, symbol: u32, length: u32) -> Result<(), Error> {
        // TODO: make more efficient (add more bits at once)
        for i in 0..length {
            let bit = ((symbol >> i) & 1) as u8;
            self.add_bit(bit)?;
        }

        Ok(())
    }

    /// Adds bits, like `add_bits`, but the order is inverted. The deflate specification
    /// uses both orders in one standard.
    fn add_huffman_bits(&mut self, symbol: u32, length: u32) -> Result<(), Error> {
        // TODO: make more efficient (add more bits at once)
        for i in 0..length {
            let bit = ((symbol >> (length - i - 1)) & 1) as u8;
            self.add_bit(bit)?;
        }

        Ok(())
    }

    fn finish_partial_bits(&mut self) -> Result<(), Error> {
        if self.bp != 0 {
            self.add_byte(self.bit)?;
            self.bit = 0;
            self.bp = 0;
        }

        Ok(())
    }
}

fn set_counts_to_count(counts: &mut [usize], count: usize, i: usize, stride: usize) {
    for c in &mut counts[(i - stride)..i] {
        *c = count;
    }
}

#[cfg(test)]
mod test {
    use miniz_oxide::inflate;

    use super::*;

    #[test]
    fn test_set_counts_to_count() {
        let mut counts = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let count = 100;
        let i = 8;
        let stride = 5;

        set_counts_to_count(&mut counts, count, i, stride);

        assert_eq!(counts, vec![0, 1, 2, 100, 100, 100, 100, 100, 8, 9]);
    }

    #[test]
    fn weird_encoder_write_size_combinations_works() {
        let mut compressed_data = vec![];

        let default_options = Options::default();
        let mut encoder =
            DeflateEncoder::new(default_options, BlockType::default(), &mut compressed_data);

        encoder.write_all(&[0]).unwrap();
        encoder.write_all(&[]).unwrap();
        encoder.write_all(&[1, 2]).unwrap();
        encoder.write_all(&[]).unwrap();
        encoder.write_all(&[]).unwrap();
        encoder.write_all(&[3]).unwrap();
        encoder.write_all(&[4]).unwrap();

        encoder.finish().unwrap();

        let decompressed_data = inflate::decompress_to_vec(&compressed_data)
            .expect("Could not inflate compressed stream");

        assert_eq!(
            &[0, 1, 2, 3, 4][..],
            decompressed_data,
            "Decompressed data should match input data"
        );
    }

    #[test]
    fn too_small_write_buffer_works() {
        let mut tiny_buf = [0; 4];
        let mut encoder =
            DeflateEncoder::new(Options::default(), BlockType::Fixed, &mut tiny_buf[..]);

        encoder
            .write_all(b"data which won't compress into a very tiny buffer")
            .expect("First small data write should buffer into a chunk");

        encoder
            .finish()
            .expect_err("Flushing the pending buffer to the stream should fail gracefully");
    }
}

/// A chunk that starts with these bytes makes the thread working on it panic.
#[cfg(all(test, feature = "std"))]
pub(crate) const TEST_PANIC: &[u8] = b"zopfli test: panic in the thread of this chunk";

/// Works out the LZ77 data and block splits of an encoder's chunks on worker
/// threads, for [`DeflateEncoder::with_threads`]. Each chunk is handed over
/// with the window before it, so the plans are the ones the encoder would make
/// itself; the encoder's thread writes them in order.
#[cfg(feature = "std")]
mod chunk_pool {
    use std::{
        collections::BTreeMap,
        num::NonZeroUsize,
        panic::{self, AssertUnwindSafe},
        sync::{
            atomic::{AtomicBool, Ordering},
            mpsc, Arc, Mutex,
        },
        thread,
    };

    use super::{add_all_blocks, blocksplit_plan, BitwiseWriter};
    use crate::{lz77::Lz77Store, Error, Options, Write};

    /// A chunk: the window before it and the chunk itself, and where the chunk
    /// starts in that data.
    struct Job {
        index: usize,
        data: Vec<u8>,
        instart: usize,
    }

    /// A chunk with its LZ77 data and split points, or the panic of the thread
    /// that worked on it.
    struct Done {
        data: Vec<u8>,
        plan: thread::Result<(Lz77Store, Vec<usize>)>,
    }

    fn work_out(options: &Options, job: Job) -> (usize, Done) {
        let plan = panic::catch_unwind(AssertUnwindSafe(|| {
            #[cfg(test)]
            if job.data[job.instart..].starts_with(super::TEST_PANIC) {
                panic!("test panic in a worker");
            }
            blocksplit_plan(options, &job.data, job.instart, job.data.len())
        }));
        (
            job.index,
            Done {
                data: job.data,
                plan,
            },
        )
    }

    pub struct ChunkPool {
        options: Options,
        threads: usize,
        jobs: mpsc::Sender<Job>,
        queue: Arc<Mutex<mpsc::Receiver<Job>>>,
        done_sender: mpsc::Sender<(usize, Done)>,
        done: mpsc::Receiver<(usize, Done)>,
        /// Set when the pool is dropped: the workers leave the chunks still
        /// queued and stop.
        cancelled: Arc<AtomicBool>,
        workers: usize,
        /// Chunks handed over, chunks written, and finished ones not written yet.
        submitted: usize,
        written: usize,
        finished: BTreeMap<usize, Done>,
        /// The last attempt to write a chunk failed.
        failed: bool,
        /// A worker panicked: nothing more is written.
        poisoned: bool,
    }

    impl ChunkPool {
        pub fn new(options: Options, threads: NonZeroUsize) -> Self {
            let (jobs, queue) = mpsc::channel();
            let (done_sender, done) = mpsc::channel();
            Self {
                options,
                threads: threads.get(),
                jobs,
                queue: Arc::new(Mutex::new(queue)),
                done_sender,
                done,
                cancelled: Arc::new(AtomicBool::new(false)),
                workers: 0,
                submitted: 0,
                written: 0,
                finished: BTreeMap::new(),
                failed: false,
                poisoned: false,
            }
        }

        pub fn set_threads(&mut self, threads: NonZeroUsize) {
            self.threads = threads.get();
        }

        /// Whether the last attempt to write to the sink failed, or a worker
        /// panicked.
        pub fn failed(&self) -> bool {
            self.failed || self.poisoned
        }

        pub fn set_failed(&mut self, failed: bool) {
            self.failed = failed;
        }

        /// An error once a worker has panicked.
        pub fn check(&self) -> Result<(), Error> {
            if self.poisoned {
                return Err(Error::other("a zopfli worker thread panicked"));
            }
            Ok(())
        }

        /// Hands a chunk over: `data` holds the window before it and the chunk
        /// from `instart` on. Starts a worker while there are fewer than threads
        /// and than chunks not written yet.
        pub fn submit(&mut self, data: Vec<u8>, instart: usize) {
            let job = Job {
                index: self.submitted,
                data,
                instart,
            };
            self.submitted += 1;
            if self.workers < self.threads
                && self.workers < self.submitted - self.written
                && self.start_worker()
            {
                self.workers += 1;
            }
            // Without a worker (none could be started), this thread works it out.
            let job = if self.workers > 0 {
                match self.jobs.send(job) {
                    Ok(()) => return,
                    Err(mpsc::SendError(job)) => job,
                }
            } else {
                job
            };
            let (index, done) = work_out(&self.options, job);
            self.finished.insert(index, done);
        }

        fn start_worker(&self) -> bool {
            let queue = Arc::clone(&self.queue);
            let done = self.done_sender.clone();
            let cancelled = Arc::clone(&self.cancelled);
            let options = self.options;
            thread::Builder::new()
                .name("zopfli".into())
                .spawn(move || loop {
                    let job = match queue.lock() {
                        Ok(queue) => queue.recv(),
                        Err(_) => return,
                    };
                    let Ok(job) = job else { return };
                    if cancelled.load(Ordering::Relaxed) {
                        continue;
                    }
                    if done.send(work_out(&options, job)).is_err() {
                        return;
                    }
                })
                .is_ok()
        }

        /// Takes the chunks that have finished meanwhile.
        fn collect(&mut self) {
            while let Ok((index, done)) = self.done.try_recv() {
                self.finished.insert(index, done);
            }
        }

        /// Waits until the chunk `index` is among the finished ones.
        fn wait_for(&mut self, index: usize) {
            while !self.finished.contains_key(&index) {
                // The pool holds a sender of `done` itself, so this only waits:
                // every chunk handed over is queued, with a worker or finished.
                if let Ok((i, done)) = self.done.recv() {
                    self.finished.insert(i, done);
                }
            }
        }

        /// Writes the chunks that have finished, in order; while more than
        /// twice as many chunks as threads are not written, waits for the
        /// oldest one.
        pub fn write_finished<W: Write>(
            &mut self,
            bitwise_writer: &mut BitwiseWriter<W>,
        ) -> Result<(), Error> {
            self.check()?;
            loop {
                self.collect();
                if self.written == self.submitted {
                    return Ok(());
                }
                if !self.finished.contains_key(&self.written) {
                    if self.submitted - self.written <= self.threads.saturating_mul(2) {
                        return Ok(());
                    }
                    self.wait_for(self.written);
                }
                self.write_next(bitwise_writer)?;
            }
        }

        /// Writes every chunk handed over, waiting for those not finished.
        pub fn write_all<W: Write>(
            &mut self,
            bitwise_writer: &mut BitwiseWriter<W>,
        ) -> Result<(), Error> {
            self.check()?;
            while self.written < self.submitted {
                self.wait_for(self.written);
                self.write_next(bitwise_writer)?;
            }
            Ok(())
        }

        /// Writes the chunk `written`, which has finished. It stays until it is
        /// written, so that a write that fails leaves it in place.
        fn write_next<W: Write>(
            &mut self,
            bitwise_writer: &mut BitwiseWriter<W>,
        ) -> Result<(), Error> {
            if self.finished[&self.written].plan.is_err() {
                // The panic of the worker goes on here, as it would have without
                // threads (also when this runs from `Drop`). Its chunk is gone, so
                // nothing after it can be written any more.
                self.poisoned = true;
                if let Some(Done {
                    plan: Err(payload), ..
                }) = self.finished.remove(&self.written)
                {
                    panic::resume_unwind(payload);
                }
            }
            let next = &self.finished[&self.written];
            if let Ok((lz77, splitpoints)) = &next.plan {
                let written = add_all_blocks(splitpoints, lz77, false, &next.data, bitwise_writer);
                self.failed = written.is_err();
                written?;
            }
            self.finished.remove(&self.written);
            self.written += 1;
            Ok(())
        }
    }

    impl Drop for ChunkPool {
        fn drop(&mut self) {
            self.cancelled.store(true, Ordering::Relaxed);
        }
    }
}
