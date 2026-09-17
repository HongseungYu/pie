use crate::error::{Fault, Result};

#[cfg(target_vendor = "apple")]
use objc2::rc::Retained;
#[cfg(target_vendor = "apple")]
use objc2::runtime::ProtocolObject;
#[cfg(target_vendor = "apple")]
use objc2_metal::MTLBuffer;

#[cfg(target_vendor = "apple")]
pub(crate) type Slab = Retained<ProtocolObject<dyn MTLBuffer>>;
#[cfg(not(target_vendor = "apple"))]
pub(crate) type Slab = ();

#[cfg(target_vendor = "apple")]
pub(crate) fn slab_id(slab: &Slab) -> u64 {
    let object: &objc2::runtime::ProtocolObject<dyn MTLBuffer> = slab;
    std::ptr::from_ref(object).cast::<u8>() as u64
}

#[cfg(not(target_vendor = "apple"))]
pub(crate) fn slab_id(slab: &Slab) -> u64 {
    let _ = slab;
    0
}

#[cfg(target_vendor = "apple")]
pub(crate) fn slab_address(slab: &Slab) -> u64 {
    slab.gpuAddress()
}

#[derive(Clone)]
pub struct Buffer {
    slab: Slab,
    bytes: u64,
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("bytes", &self.bytes)
            .finish()
    }
}

// SAFETY: `MTLBuffer` is thread-safe for retain/release, `contents`, and
// encoder binding; `Send` only permits the one-time move onto the firing thread.
unsafe impl Send for Buffer {}

impl Buffer {
    pub fn zeroed(device: &super::Context, bytes: u64) -> Result<Buffer> {
        #[cfg(target_vendor = "apple")]
        {
            if bytes == 0 {
                return Ok(Buffer {
                    slab: device.empty(),
                    bytes: 0,
                });
            }
            let slab = device.reserve(bytes)?;
            let mut buffer = Buffer { slab, bytes };
            buffer.zero_span(0, bytes)?;
            Ok(buffer)
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (device, bytes);
            Err(Fault::Deviceless)
        }
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    pub(crate) fn slab(&self) -> &Slab {
        &self.slab
    }

    #[cfg(target_vendor = "apple")]
    pub(crate) fn raw(&self) -> &objc2::runtime::ProtocolObject<dyn MTLBuffer> {
        &self.slab
    }

    #[must_use]
    pub fn address_at(&self, offset: u64) -> Option<u64> {
        self.span(offset, 0).ok()?;
        #[cfg(target_vendor = "apple")]
        {
            slab_address(&self.slab).checked_add(offset)
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            None
        }
    }

    pub fn write_from_file(
        &mut self,
        file: &std::fs::File,
        jobs: &[(u64, u64, u64)],
        threads: usize,
    ) -> Result<()> {
        let writer = self.file_writer(jobs)?;
        writer.pread(file, jobs, threads)
    }

    /// A copier over this reservation: every job's source and destination
    /// span lies inside it. `FileWriter::copy` moves the bytes.
    pub fn copier(&mut self, jobs: &[(u64, u64, u64)]) -> Result<FileWriter> {
        for &(into, from, len) in jobs {
            self.span(into, len)?;
            self.span(from, len)?;
        }
        #[cfg(target_vendor = "apple")]
        {
            Ok(FileWriter {
                base: self.slab.contents().as_ptr() as usize,
                _keep: self.slab.clone(),
            })
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            Err(Fault::Deviceless)
        }
    }

    pub fn file_writer(&mut self, jobs: &[(u64, u64, u64)]) -> Result<FileWriter> {
        for &(into, _, len) in jobs {
            self.span(into, len)?;
        }
        #[cfg(target_vendor = "apple")]
        {
            Ok(FileWriter {
                base: self.slab.contents().as_ptr() as usize,
                _keep: self.slab.clone(),
            })
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            Err(Fault::Deviceless)
        }
    }
}

pub struct FileWriter {
    #[cfg(target_vendor = "apple")]
    base: usize,
    #[cfg(target_vendor = "apple")]
    _keep: Slab,
}

// SAFETY: `_keep` retains the mapping for the handle's life; the only
// access offered is a checked `pread` under the one-seat-one-band contract.
unsafe impl Send for FileWriter {}

impl FileWriter {
    /// Seat-to-seat copies inside the reservation, `threads` at a time.
    pub fn copy(&self, jobs: &[(u64, u64, u64)], threads: usize) -> Result<()> {
        if jobs.is_empty() {
            return Ok(());
        }
        #[cfg(target_vendor = "apple")]
        {
            let base = self.base;
            let threads = threads.clamp(1, jobs.len());
            let per = jobs.len().div_ceil(threads);
            std::thread::scope(|scope| {
                for chunk in jobs.chunks(per) {
                    scope.spawn(move || {
                        for &(into, from, len) in chunk {
                            let at = |offset: u64| {
                                base + usize::try_from(offset)
                                    .expect("an offset inside a live mapping")
                            };
                            // SAFETY: both spans were proved inside the live
                            // mapping by `copier`; `copy` tolerates overlap.
                            unsafe {
                                std::ptr::copy(
                                    at(from) as *const u8,
                                    at(into) as *mut u8,
                                    usize::try_from(len).expect("a length inside a live mapping"),
                                );
                            }
                        }
                    });
                }
            });
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = threads;
            Err(Fault::Deviceless)
        }
    }

    pub fn pread(
        &self,
        file: &std::fs::File,
        jobs: &[(u64, u64, u64)],
        threads: usize,
    ) -> Result<()> {
        if jobs.is_empty() {
            return Ok(());
        }
        #[cfg(target_vendor = "apple")]
        {
            use std::os::fd::AsRawFd;
            let base = self.base;
            let fd = file.as_raw_fd();
            let threads = threads.clamp(1, jobs.len());
            let per = jobs.len().div_ceil(threads);
            let failed: std::sync::Mutex<Option<Fault>> = std::sync::Mutex::new(None);
            std::thread::scope(|scope| {
                for chunk in jobs.chunks(per) {
                    let failed = &failed;
                    scope.spawn(move || {
                        for &(into, from, len) in chunk {
                            // SAFETY: destinations are disjoint and inside the
                            // live mapping, per `file_writer`.
                            let dst = (base
                                + usize::try_from(into).expect("an offset inside a live mapping"))
                                as *mut u8;
                            if let Err(why) = unsafe { pread_all(fd, dst, from, len) } {
                                *failed
                                    .lock()
                                    .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(why);
                                return;
                            }
                        }
                    });
                }
            });
            match failed
                .into_inner()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
            {
                Some(why) => Err(why),
                None => Ok(()),
            }
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (file, threads);
            Err(Fault::Deviceless)
        }
    }
}

impl Buffer {
    pub fn write(&mut self, offset: u64, bytes: &[u8]) -> Result<()> {
        self.span(offset, bytes.len() as u64)?;
        #[cfg(target_vendor = "apple")]
        {
            // SAFETY: `contents` is a live mapping of the whole reservation;
            // `span` proved the copy's span lies inside it, non-overlapping.
            unsafe {
                let base = self.slab.contents().as_ptr().cast::<u8>();
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    base.add(usize::try_from(offset).expect("an offset inside a live mapping")),
                    bytes.len(),
                );
            }
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = offset;
            Err(Fault::Deviceless)
        }
    }

    pub fn zero_span(&mut self, offset: u64, len: u64) -> Result<()> {
        self.span(offset, len)?;
        #[cfg(target_vendor = "apple")]
        {
            if len == 0 {
                return Ok(());
            }
            // SAFETY: as `write` — a live mapping and a span proved inside it.
            unsafe {
                let base = self.slab.contents().as_ptr().cast::<u8>();
                std::ptr::write_bytes(
                    base.add(usize::try_from(offset).expect("an offset inside a live mapping")),
                    0,
                    usize::try_from(len).expect("a length inside a live mapping"),
                );
            }
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = offset;
            Err(Fault::Deviceless)
        }
    }

    pub fn read(&self, offset: u64, into: &mut [u8]) -> Result<()> {
        self.span(offset, into.len() as u64)?;
        #[cfg(target_vendor = "apple")]
        {
            // SAFETY: as `write`, with the direction reversed.
            unsafe {
                let base = self.slab.contents().as_ptr().cast::<u8>();
                std::ptr::copy_nonoverlapping(
                    base.add(usize::try_from(offset).expect("an offset inside a live mapping")),
                    into.as_mut_ptr(),
                    into.len(),
                );
            }
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = offset;
            Err(Fault::Deviceless)
        }
    }

    pub fn span(&self, offset: u64, len: u64) -> Result<()> {
        let end = offset.checked_add(len).ok_or(Fault::Ceiling {
            what: "bytes of a device reservation",
            need: u64::MAX,
            have: self.bytes,
        })?;
        if end > self.bytes {
            return Err(Fault::Ceiling {
                what: "bytes of a device reservation",
                need: end,
                have: self.bytes,
            });
        }
        Ok(())
    }
}

#[cfg(target_vendor = "apple")]
unsafe fn pread_all(fd: std::os::fd::RawFd, dst: *mut u8, from: u64, len: u64) -> Result<()> {
    let mut done: u64 = 0;
    while done < len {
        let want = usize::try_from(len - done)
            .unwrap_or(usize::MAX)
            .min(1 << 30);
        let at = i64::try_from(from + done).map_err(|_| {
            Fault::Residency(format!(
                "a seat source offset {} does not fit `off_t`",
                from + done
            ))
        })?;
        // SAFETY: the caller's contract on `dst`; `want` bytes from `dst +
        // done` stay inside it.
        let got = unsafe { libc::pread(fd, dst.add(done as usize).cast(), want, at) };
        if got < 0 {
            return Err(Fault::Residency(format!(
                "a seat source read of {want} bytes at {} failed: {}",
                from + done,
                std::io::Error::last_os_error()
            )));
        }
        if got == 0 {
            return Err(Fault::Residency(format!(
                "a seat source read at {} met the end of the file {} bytes short",
                from + done,
                len - done
            )));
        }
        done += got as u64;
    }
    Ok(())
}
