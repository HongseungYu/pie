use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::error::{Fault, Result};

#[must_use]
pub fn page() -> usize {
    // SAFETY: `sysconf` of a defined name, reading no memory.
    let said = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if said > 0 { said as usize } else { 4096 }
}

pub struct Mapping {
    at: std::ptr::NonNull<u8>,
    span: usize,
    len: u64,
    file: std::fs::File,
    path: PathBuf,
}

// SAFETY: `MAP_PRIVATE` and `PROT_READ`, with no interior mutability — every
// reference reads bytes nothing in the process can change, which is `Sync`
// as well as `Send`.
unsafe impl Send for Mapping {}
// SAFETY: as `Send` — read-only pages, no interior mutability.
unsafe impl Sync for Mapping {}

impl Mapping {
    pub fn of(path: impl AsRef<Path>) -> Result<Arc<Mapping>> {
        let path = path.as_ref();
        let file = std::fs::File::open(path).map_err(|why| Fault::Mapped {
            step: "open",
            what: path.display().to_string(),
            why: why.to_string(),
        })?;
        Mapping::of_file(file, path.to_path_buf())
    }

    pub fn of_file(file: std::fs::File, named: PathBuf) -> Result<Arc<Mapping>> {
        let what = || named.display().to_string();
        let len = file
            .metadata()
            .map_err(|why| Fault::Mapped {
                step: "stat",
                what: what(),
                why: why.to_string(),
            })?
            .len();
        if len == 0 {
            return Err(Fault::Mapped {
                step: "size",
                what: what(),
                why: "the artifact holds no bytes".into(),
            });
        }
        let page = page();
        let span = usize::try_from(len)
            .ok()
            .and_then(|len| len.checked_next_multiple_of(page))
            .ok_or_else(|| Fault::Mapped {
                step: "size",
                what: what(),
                why: format!("{len} bytes does not fit this process's address space"),
            })?;
        // SAFETY: a fresh mapping of a file held open, at a page-rounded
        // span POSIX zero-fills past `len`.
        let at = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                span,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                std::os::fd::AsRawFd::as_raw_fd(&file),
                0,
            )
        };
        if at == libc::MAP_FAILED {
            return Err(Fault::Mapped {
                step: "map",
                what: what(),
                why: std::io::Error::last_os_error().to_string(),
            });
        }
        let at = std::ptr::NonNull::new(at.cast::<u8>()).ok_or_else(|| Fault::Mapped {
            step: "map",
            what: what(),
            why: "the kernel answered a null mapping".into(),
        })?;
        Ok(Arc::new(Mapping {
            at,
            span,
            len,
            file,
            path: named,
        }))
    }

    #[must_use]
    pub fn file(&self) -> &std::fs::File {
        &self.file
    }

    #[must_use]
    pub fn len(&self) -> u64 {
        self.len
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[must_use]
    pub fn span(&self) -> usize {
        self.span
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    #[must_use]
    pub fn backing(&self) -> Option<u64> {
        self.file.metadata().ok().map(|it| it.len())
    }

    #[must_use]
    pub fn links(&self) -> Option<u64> {
        use std::os::unix::fs::MetadataExt;
        self.file.metadata().ok().map(|it| it.nlink())
    }
}

impl std::ops::Deref for Mapping {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        // SAFETY: `len` readable bytes inside a mapping of `span >= len`
        // bytes this type owns for its whole life, and which nothing may
        // write through.
        unsafe {
            std::slice::from_raw_parts(
                self.at.as_ptr(),
                usize::try_from(self.len).expect("a length inside a live mapping"),
            )
        }
    }
}

impl std::fmt::Debug for Mapping {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mapping")
            .field("path", &self.path)
            .field("bytes", &self.len)
            .field("span", &self.span)
            .finish()
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: unmapping this type's own mapping; every `MTLBuffer`
        // minted over it holds an `Arc` to this, so the last was released first.
        unsafe {
            libc::munmap(self.at.as_ptr().cast(), self.span);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn scratch(name: &str, bytes: usize) -> PathBuf {
        let path = std::env::temp_dir().join(format!("pie-map-{}-{name}", std::process::id()));
        let mut file = std::fs::File::create(&path).expect("a scratch artifact");
        let pattern: Vec<u8> = (0..bytes).map(|at| (at % 251) as u8).collect();
        file.write_all(&pattern).expect("the pattern lands");
        path
    }

    #[test]
    fn mapping_every_case() {
        an_empty_artifact_is_refused_by_name();
    }

    fn an_empty_artifact_is_refused_by_name() {
        let path = scratch("empty", 0);
        let fault = Mapping::of(&path).expect_err("a zero-byte artifact does not map");
        let _ = std::fs::remove_file(&path);
        let said = fault.to_string();
        assert!(
            said.contains("holds no bytes"),
            "the refusal says why: {said}"
        );
    }
}
