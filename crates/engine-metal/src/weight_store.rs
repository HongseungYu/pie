use std::fs::File;

use crate::device::alloc::FileWriter;
use crate::device::{Buffer, Context, Handles};
use crate::error::{Fault, Result};
type Job = (u64, u64, u64);
type Grouped = Vec<(usize, Vec<Job>)>;
type Writes = crate::device::alloc::Writes;

#[derive(Clone, Debug)]
struct Chunk {
    start: u64,
    buffer: Buffer,
}

#[derive(Clone, Debug)]
pub struct Store {
    chunks: Vec<Chunk>,
    bytes: u64,
}

impl Store {
    pub fn zeroed(device: &Context, spans: &[(u64, u64)], ceiling: u64) -> Result<Store> {
        let mut chunks: Vec<Chunk> = Vec::new();
        let mut start = 0u64;
        let mut held = 0u64;
        let total = spans
            .last()
            .map_or(0, |&(offset, reserved)| offset + reserved);
        for &(offset, reserved) in spans {
            debug_assert_eq!(
                offset,
                start + held,
                "the spans are packed from zero, in order"
            );
            if reserved > ceiling {
                return Err(Fault::Ceiling {
                    what: "bytes of one plane in one buffer",
                    need: reserved,
                    have: ceiling,
                });
            }
            if held > 0 && held + reserved > ceiling {
                chunks.push(Chunk {
                    start,
                    buffer: Buffer::zeroed(device, held)?,
                });
                start += held;
                held = 0;
            }
            held += reserved;
        }
        chunks.push(Chunk {
            start,
            buffer: Buffer::zeroed(device, held)?,
        });
        Ok(Store {
            chunks,
            bytes: total,
        })
    }

    fn locate(&self, offset: u64, len: u64) -> Result<(usize, u64)> {
        let end = offset.checked_add(len).ok_or(Fault::Ceiling {
            what: "bytes of the weight store",
            need: u64::MAX,
            have: self.bytes,
        })?;
        let at = self
            .chunks
            .partition_point(|chunk| chunk.start <= offset)
            .saturating_sub(1);
        let chunk = &self.chunks[at];
        if offset < chunk.start || end > chunk.start + chunk.buffer.bytes() {
            return Err(Fault::Ceiling {
                what: "bytes of one chunk of the weight store",
                need: end,
                have: chunk.start + chunk.buffer.bytes(),
            });
        }
        Ok((at, offset - chunk.start))
    }

    pub fn bind(&self, handles: &Handles, offset: u64, len: u64) -> Result<u32> {
        let (at, local) = self.locate(offset, len)?;
        handles.bind(&self.chunks[at].buffer, local, len)
    }

    pub fn write(&mut self, offset: u64, bytes: &[u8]) -> Result<()> {
        let (at, local) = self.locate(offset, bytes.len() as u64)?;
        self.chunks[at].buffer.write(local, bytes)
    }

    pub fn zero_span(&mut self, offset: u64, len: u64) -> Result<()> {
        let (at, local) = self.locate(offset, len)?;
        self.chunks[at].buffer.zero_span(local, len)
    }

    pub fn read(&self, offset: u64, into: &mut [u8]) -> Result<()> {
        let (at, local) = self.locate(offset, into.len() as u64)?;
        self.chunks[at].buffer.read(local, into)
    }

    fn group(&self, jobs: &[Job]) -> Result<Grouped> {
        let mut groups: Grouped = Vec::new();
        for &(into, from, len) in jobs {
            let (at, local) = self.locate(into, len)?;
            match groups.iter_mut().find(|(chunk, _)| *chunk == at) {
                Some((_, list)) => list.push((local, from, len)),
                None => groups.push((at, vec![(local, from, len)])),
            }
        }
        Ok(groups)
    }

    pub fn write_from_file(
        &mut self,
        file: &File,
        jobs: &[(u64, u64, u64)],
        threads: usize,
    ) -> Result<()> {
        // One threaded pass over every chunk's jobs; `PIE_STORE_SERIAL_CHUNKS=1`
        // keeps the pass-a-chunk order for measuring what it cost.
        if serial_chunks() {
            for (at, local) in self.group(jobs)? {
                self.chunks[at]
                    .buffer
                    .write_from_file(file, &local, threads)?;
            }
            return Ok(());
        }
        let writers = self.file_writers(jobs)?;
        FileWriter::pread_many(file, &writers, threads)
    }

    /// Which chunk holds `offset`, and the chunk's `[start, end)`.
    #[must_use]
    pub fn chunk_of(&self, offset: u64) -> (usize, u64, u64) {
        let at = self
            .chunks
            .partition_point(|chunk| chunk.start <= offset)
            .saturating_sub(1);
        let chunk = &self.chunks[at];
        (at, chunk.start, chunk.start + chunk.buffer.bytes())
    }

    pub fn file_writers(&mut self, jobs: &[Job]) -> Result<Writes> {
        let mut out = Vec::new();
        for (at, local) in self.group(jobs)? {
            let writer = self.chunks[at].buffer.file_writer(&local)?;
            out.push((writer, local));
        }
        Ok(out)
    }

    /// Writers for seat-to-seat copies: each job's source and destination
    /// must lie in one chunk of the store.
    pub fn copiers(&mut self, jobs: &[Job]) -> Result<Writes> {
        let mut groups: Grouped = Vec::new();
        for &(into, from, len) in jobs {
            let (at, local_into) = self.locate(into, len)?;
            let (from_at, local_from) = self.locate(from, len)?;
            if from_at != at {
                return Err(Fault::Residency(format!(
                    "a seat copy from {from} to {into} crosses two chunks of the weight \
                     store; the pool's regions are not one buffer here"
                )));
            }
            match groups.iter_mut().find(|(chunk, _)| *chunk == at) {
                Some((_, list)) => list.push((local_into, local_from, len)),
                None => groups.push((at, vec![(local_into, local_from, len)])),
            }
        }
        let mut out = Vec::new();
        for (at, local) in groups {
            let writer = self.chunks[at].buffer.copier(&local)?;
            out.push((writer, local));
        }
        Ok(out)
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    #[must_use]
    pub fn chunks(&self) -> usize {
        self.chunks.len()
    }
}

fn serial_chunks() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("PIE_STORE_SERIAL_CHUNKS").as_deref().map(str::trim),
            Ok("1" | "on" | "true" | "yes")
        )
    })
}
