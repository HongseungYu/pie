// Replaced main's: the shared LRU pool, one seat to an expert of any layer.

#[derive(Debug)]
pub(super) struct Band {
    pub(super) name: String,
    pub(super) at: u64,
    pub(super) from: u64,
    pub(super) stride: u64,
}

/// One router's mixture: its bands and where each of its experts sits in
/// the shared pool, if anywhere.
#[derive(Debug)]
pub(super) struct Group {
    pub(super) experts: u32,
    pub(super) bands: Vec<Band>,
    pub(super) seat_of: Vec<Option<u32>>,
    pub(super) ever: Vec<bool>,
}

const NONE: u32 = u32::MAX;

/// The shared LRU pool: one seat holds one `(group, expert)` of any group.
/// The seats in use sit on a doubly linked list ordered by recency, least
/// recently used at the head, so a victim is the head rather than a scan
/// over every seat (the scan cost llama.cpp's expert store 4 ms a step at
/// 11k seats). Seats never used yet wait on `free` and go first.
/// What keeps a seat from being taken, and who lets it go.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Hold {
    /// Nothing: the LRU may hand it out.
    Free,
    /// A matmul of the segment just cut reads it; the next cut releases it.
    Segment,
    /// This fire borrowed it for its prefill ring; the fire's end returns it.
    Fire,
    /// A copy is landing in it; the join that takes the copy releases it.
    Inflight,
}

#[derive(Debug)]
pub(super) struct Pool {
    pub(super) slots: u32,
    pub(super) in_seat: Vec<Option<(u32, u32)>>,
    pub(super) holds: Vec<Hold>,
    pub(super) prev: Vec<u32>,
    pub(super) next: Vec<u32>,
    pub(super) linked: Vec<bool>,
    pub(super) head: u32,
    pub(super) tail: u32,
    pub(super) free: Vec<u32>,
}

impl Pool {
    pub(super) fn new(slots: u32) -> Pool {
        Pool {
            slots,
            in_seat: vec![None; slots as usize],
            holds: vec![Hold::Free; slots as usize],
            prev: vec![NONE; slots as usize],
            next: vec![NONE; slots as usize],
            linked: vec![false; slots as usize],
            head: NONE,
            tail: NONE,
            free: (0..slots).rev().collect(),
        }
    }

    pub(super) fn unlink(&mut self, seat: u32) {
        let s = seat as usize;
        if !self.linked[s] {
            return;
        }
        let (p, n) = (self.prev[s], self.next[s]);
        if p == NONE {
            self.head = n;
        } else {
            self.next[p as usize] = n;
        }
        if n == NONE {
            self.tail = p;
        } else {
            self.prev[n as usize] = p;
        }
        self.prev[s] = NONE;
        self.next[s] = NONE;
        self.linked[s] = false;
    }

    pub(super) fn push_back(&mut self, seat: u32) {
        let s = seat as usize;
        debug_assert!(!self.linked[s], "a seat is on the list once");
        self.prev[s] = self.tail;
        self.next[s] = NONE;
        if self.tail == NONE {
            self.head = seat;
        } else {
            self.next[self.tail as usize] = seat;
        }
        self.tail = seat;
        self.linked[s] = true;
    }

    /// Least recently used, now: the next miss takes it.
    pub(super) fn push_front(&mut self, seat: u32) {
        let s = seat as usize;
        debug_assert!(!self.linked[s], "a seat is on the list once");
        self.next[s] = self.head;
        self.prev[s] = NONE;
        if self.head == NONE {
            self.tail = seat;
        } else {
            self.prev[self.head as usize] = seat;
        }
        self.head = seat;
        self.linked[s] = true;
    }

    /// Most recently used, now.
    pub(super) fn bump(&mut self, seat: u32) {
        self.unlink(seat);
        self.push_back(seat);
    }

    /// A seat to take: one never used, else the least recently used seat
    /// nothing holds.
    pub(super) fn victim(&mut self) -> Option<u32> {
        if let Some(seat) = self.free.pop() {
            return Some(seat);
        }
        let mut at = self.head;
        while at != NONE {
            if self.holds[at as usize] == Hold::Free {
                return Some(at);
            }
            at = self.next[at as usize];
        }
        None
    }

    pub(super) fn hold(&mut self, seat: u32, hold: Hold) {
        self.holds[seat as usize] = hold;
    }

    /// Let go of every seat held this way, and only those: a segment's pins
    /// fall at the next cut without dropping the fire's borrowed ring.
    pub(super) fn release(&mut self, hold: Hold) {
        for held in &mut self.holds {
            if *held == hold {
                *held = Hold::Free;
            }
        }
    }

    /// Take a seat back for reuse: nothing sits in it and nothing holds it.
    pub(super) fn strip(&mut self, seat: u32) {
        self.in_seat[seat as usize] = None;
        self.holds[seat as usize] = Hold::Free;
        self.unlink(seat);
        self.free.push(seat);
    }

    /// Empty: every seat free and unheld, the list as it was at open.
    pub(super) fn clear(&mut self) {
        *self = Pool::new(self.slots);
    }

    pub(super) fn resident(&self) -> u64 {
        self.in_seat.iter().filter(|held| held.is_some()).count() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pool_evicts_least_recently_used_first() {
        let mut pool = Pool::new(3);
        // Never-used seats go first, in order.
        assert_eq!(pool.victim(), Some(0));
        pool.bump(0);
        assert_eq!(pool.victim(), Some(1));
        pool.bump(1);
        assert_eq!(pool.victim(), Some(2));
        pool.bump(2);
        // Now the list decides: 0 is the oldest.
        assert_eq!(pool.victim(), Some(0));
        pool.bump(0);
        assert_eq!(pool.victim(), Some(1));
        // A held head is skipped.
        pool.hold(1, Hold::Segment);
        assert_eq!(pool.victim(), Some(2));
        pool.hold(2, Hold::Segment);
        pool.hold(0, Hold::Segment);
        assert_eq!(pool.victim(), None);
        // Unlinking the tail and the middle keeps the list whole.
        pool.release(Hold::Segment);
        pool.unlink(0);
        pool.unlink(2);
        assert_eq!(pool.head, 1);
        assert_eq!(pool.tail, 1);
        pool.unlink(1);
        assert_eq!(pool.head, NONE);
        assert_eq!(pool.tail, NONE);
    }
}
