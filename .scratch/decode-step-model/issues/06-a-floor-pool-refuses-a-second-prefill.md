# 06 A pool at the floor refuses a second prefill

Status: resolved
Type: task

Found by configuration 1's protocol (a 1024-token warm-up request, then the measured one, at
1024 seats): the second request's prefill died with "one segment of this fire routes to
more than 1024 distinct experts ... and the shared pool seats 1024: every seat is held by a
matmul this same segment will run, by a copy landing in it, or by this fire's prefill
ring" (`sm-c1-s1024-a`, `out/sm-c1-s1024-a.log`). The earlier floor runs (`sm-h*-dmax`)
had one request against a fresh pool and never met it.

The cause: a segment's seats are held (`Hold::Segment`) until the next cut releases them,
and the last decode segment of a fire has no next cut — its holds survive into the next
fire's `begin_fire`, where a prefill's `borrow_ring` wants every one of the floor's 2E seats
and finds ten of them held. Any pool would carry those ten stale holds a fire long; only
the floor cannot spare them.

Fix: `begin_fire` releases `Hold::Segment` after joining the prefetch and returning the
ring. The fire before is over and its final frame has landed — the token came out of it —
so nothing on the device reads those seats. No effect on what is resident or counted.
