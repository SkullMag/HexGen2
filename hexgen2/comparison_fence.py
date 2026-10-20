"""CPU ownership handoff for globally ordered NCCL cache-launch epochs.

The transport owner waits until model computation reaches a completed step.
All ranks then enqueue the cache epoch before any model owner resumes. GPU
completion is separate, allowing NCCL 2.26 implicit launch ordering to overlap
the already-enqueued transfer with later model work.
"""
import threading


class LaunchFence:
    def __init__(self, timeout=120):
        self.condition=threading.Condition();self.requested=0;self.parked=0;self.launched=0;self.timeout=timeout

    def pending(self):
        with self.condition:return self.requested

    def request(self, epoch):
        with self.condition:
            if epoch!=self.requested+1:raise ValueError('Cache epoch skipped or repeated')
            self.requested=epoch;self.condition.notify_all()

    def wait_parked(self, epoch):
        with self.condition:
            if not self.condition.wait_for(lambda:self.parked>=epoch, self.timeout):
                raise TimeoutError('Model owner did not park for cache epoch')

    def park(self, epoch):
        with self.condition:
            if epoch!=self.parked+1:raise ValueError('Model epoch skipped or repeated')
            self.parked=epoch;self.condition.notify_all()
            if not self.condition.wait_for(lambda:self.launched>=epoch, self.timeout):
                raise TimeoutError('Cache epoch was not launched')

    def release(self, epoch):
        with self.condition:
            if epoch!=self.requested or self.parked!=epoch or self.launched!=epoch-1:
                raise ValueError('Cache launch released without model ownership')
            self.launched=epoch;self.condition.notify_all()
