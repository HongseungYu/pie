// Inherited from main unchanged: where an expert's bytes come from.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::host_source::HostSource;
use crate::mapping::Mapping;

use super::*;

#[derive(Debug)]
pub(super) enum Bytes {
    Landed(HostSource),
    Artifact(Arc<Mapping>),
}

#[derive(Debug)]
pub struct Source {
    pub(super) bytes: Bytes,
    pub(super) bands: BTreeMap<usize, u64>,
}

impl Source {
    #[must_use]
    pub fn landed(plan: &Plan, host: HostSource) -> Source {
        Source {
            bytes: Bytes::Landed(host),
            bands: plan.host_of.clone(),
        }
    }

    #[must_use]
    pub fn from_host(host: HostSource, bands: BTreeMap<usize, u64>) -> Source {
        Source {
            bytes: Bytes::Landed(host),
            bands,
        }
    }

    #[must_use]
    pub fn artifact(map: Arc<Mapping>, bands: BTreeMap<usize, u64>) -> Source {
        Source {
            bytes: Bytes::Artifact(map),
            bands,
        }
    }

    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self.bytes {
            Bytes::Landed(_) => "landed",
            Bytes::Artifact(_) => "artifact",
        }
    }

    #[must_use]
    pub fn backing(&self) -> Option<(u64, u64)> {
        match &self.bytes {
            Bytes::Landed(host) => host.backing(),
            Bytes::Artifact(map) => Some((map.backing()?, map.links()?)),
        }
    }

    pub(crate) fn at(&self, param: usize) -> Option<u64> {
        self.bands.get(&param).copied()
    }

    pub(crate) fn file(&self) -> Option<&std::fs::File> {
        match &self.bytes {
            Bytes::Landed(host) => host.file(),
            Bytes::Artifact(map) => Some(map.file()),
        }
    }

    pub(crate) fn get(&self, from: usize, len: usize) -> Option<&[u8]> {
        let all: &[u8] = match &self.bytes {
            Bytes::Landed(host) => host,
            Bytes::Artifact(map) => map,
        };
        all.get(from..from.checked_add(len)?)
    }

    pub(crate) fn len(&self) -> u64 {
        match &self.bytes {
            Bytes::Landed(host) => host.len() as u64,
            Bytes::Artifact(map) => map.len(),
        }
    }

    pub(crate) fn settle(&mut self) {
        match &mut self.bytes {
            Bytes::Landed(host) => host.settle(),
            Bytes::Artifact(_) => {}
        }
    }
}
