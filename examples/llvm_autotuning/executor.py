# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import multiprocessing
from collections import deque
from contextlib import contextmanager
from enum import Enum
from itertools import islice
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, validator
from pydantic.class_validators import root_validator
from submitit import AutoExecutor

logger = logging.getLogger(__name__)


class Executor(BaseModel):
    """Defines the execution environment for jobs.

    E.g. a node on a cluster, the local machine, etc. To create jobs,
    instantiate this class and submit functions to using the executor API:

        >>> executor = Executor(executor="local", block=True)
        >>> with executor.get_executor() as executor:
        ...     executor.submit(my_job, arg1, arg2)
        ...     executor.submit(another_job)
    """

    class Type(str, Enum):
        """Types of execution environments."""

        SLURM = "slurm"
        """Submit jobs to a SLURM cluster scheduler."""
        LOCAL = "local"
        """Submit jobs to run on the current machine."""
        DEBUG = "debug"
        """Submit jobs to run synchronously on the current machine."""
        NOOP = "noop"
        """Submitted jobs return immediately without executing. This can be
        useful for debugging, where you want to validate the code and
        configuration without performing any computation.
        """

    type: Type = Field(allow_mutation=False)
    """The execution environment."""

    slurm_partition: Optional[str] = Field(default=None, allow_mutation=False)
    """The name of the SLURM partition to submit jobs to.

    Only used for :code:`Type.SLURM` executors.
    """

    cpus: int = Field(default=1, allow_mutation=False, ge=1)
    """The number of CPU threads to provision.

    If the type of executor is :code:`Type.SLURM`, this is the number of CPU
    threads to provision for each job. If the type of executor is
    :code:`Type.LOCAL`, this is the number of parallel jobs to process in a
    thread pool. Has no effect for :code:`Type.DEBUG` and :code:`Type.NOOP`.
    """

    timeout_hours: float = Field(default=12, allow_mutation=False, gt=0)

    block: bool = Field(default=False, allow_mutation=False)
    """If :code:`True`, the :code:`get_executor()` context manager will block
    until all jobs have completed when exiting scope. Jobs are still submitted
    asynchronously for parallel execution.
    """

    # === Start of public API. ===

    @contextmanager
    def get_executor(self, logs_dir: Path, cpus=None):
        if self.type == self.Type.SLURM:
            executor = AutoExecutor(folder=logs_dir)
            executor.update_parameters(
                timeout_min=int(round(self.timeout_hours * 60)),
                nodes=1,
                cpus_per_task=cpus or self.cpus,
                slurm_partition=self.slurm_partition,
            )
            name = self.slurm_partition
        elif self.type == self.Type.LOCAL:
            executor, name = (
                LocalParallelExecutor(
                    cpus=cpus or self.cpus,
                    timeout_seconds=int(round(self.timeout_hours * 3600)),
                ),
                "local",
            )
        elif self.type == self.Type.DEBUG:
            executor, name = LocalSynchronousExecutor(), "local"
        elif self.type == self.Type.NOOP:
            executor, name = DummyExecutor(), "noop"
        else:
            assert False, f"Unknown executor: {self.type} ({type(self.type).__name__})"
        executor = WrappedExecutor(executor, name=name)
        yield executor

        if self.type == self.Type.LOCAL:
            executor.executor.close()

        if self.block:
            wait_on_jobs(
                executor.jobs,
                executor_name=str(executor),
                cancel_on_error=self.type == self.Type.SLURM,
            )

    # === Start of implementation details. ===

    @validator("slurm_partition")
    def validate_slurm_partition(cls, value, *, values, **kwargs):
        del kwargs
        if values["type"] == cls.Type.SLURM:
            assert value, f"Must specify a partition for executor: {values['executor']}"
        return value

    @root_validator
    def local_always_blocks(cls, values):
        if values["type"] == cls.Type.LOCAL or values["type"] == cls.Type.NOOP:
            values["block"] = True

        return values

    class Config:
        validate_assignment = True


class WrappedExecutor:
    """An executor-like interface that records all jobs that are submitted."""

    def __init__(self, executor, name: str):
        self.executor = executor
        self.jobs = []
        self.name = name

    def submit(self, *args, **kwargs):
        job = self.executor.submit(*args, **kwargs)
        logger.info("Submitting job %s to %s ...", job.job_id, self)
        self.jobs.append(job)
        return job

    def __repr__(self) -> str:
        return self.name


def wait_on_jobs(jobs, executor_name: str = "executor", cancel_on_error: bool = True):
    njobs = len(jobs)
    jobs = deque(jobs)

    def cancel_all_jobs(jobs):
        print(f"Cancelling {len(jobs)} {executor_name} jobs")
        for job in jobs:
            try:
                job.cancel()
            except:  # noqa
                pass

    # Produce a list of the first few job IDs
    max_num_job_ids_to_show = 8
    job_ids = [j.job_id for j in islice(jobs, max_num_job_ids_to_show)]
    job_ids = ", ".join(str(x) for x in job_ids)
    job_ids = f"job ID: {job_ids}" if len(jobs) == 1 else f"job IDs: {job_ids}"
    if len(jobs) > max_num_job_ids_to_show:
        job_ids = f"{job_ids} ..."

    logger.info(
        f"Waiting for {len(jobs)} {executor_name} jobs to complete with {job_ids}"
    )
    completed = 0
    while jobs:
        job = jobs.popleft()
        if cancel_on_error:
            try:
                job.result()
                completed += 1
                logger.info(f"Jobs completed = {completed} of {njobs} ...")
            except Exception as e:  # noqa Intentionally broad.
                logger.error(f"Caught: {type(e).__name__}: {e}")
                jobs.append(job)
                return cancel_all_jobs(jobs)
        else:
            job.result()
            completed += 1
            logger.info(f"Jobs completed = {completed} of {njobs} ...")
    logger.info("All done.")


class LocalParallelExecutor:
    """An executor which uses a process pool to process jobs in parallel on the
    local machine.
    """

    class LocalJob:
        def __init__(self, job_id: int, async_result, timeout_seconds: int):
            self._async_result = async_result
            self.job_id = job_id
            self.timeout_seconds = timeout_seconds

        def result(self):
            return self._async_result.get(timeout=self.timeout_seconds)

        def cancel(self):
            pass

    def __init__(self, cpus: int, timeout_seconds: int):
        self.last_job_id = 0
        self.process_pool = multiprocessing.Pool(cpus)
        self.timeout_seconds = timeout_seconds

    def submit(self, fn, *args, **kwargs):
        self.last_job_id += 1
        return self.LocalJob(
            self.last_job_id,
            self.process_pool.apply_async(fn, args, kwargs),
            self.timeout_seconds,
        )

    def close(self):
        self.process_pool.close()


class LocalSynchronousExecutor:
    """An executor where each job is executed synchronously when result() is
    called."""

    class LocalJob:
        def __init__(self, job_id: int, fn, *args, **kwargs):
            self._callback = lambda: fn(*args, **kwargs)
            self.job_id = job_id

        def result(self):
            return self._callback()

        def cancel(self):
            pass

    def __init__(self):
        self.last_job_id = 0

    def submit(self, fn, *args, **kwargs):
        self.last_job_id += 1
        return self.LocalJob(self.last_job_id, fn, *args, **kwargs)


class DummyExecutor:
    class DummyJob:
        def __init__(self, job_id: int):
            self.job_id = job_id

        def result(self):
            return None

        def cancel(self):
            pass

    def __init__(self) -> None:
        self.last_job_id = 0

    def submit(self, fn, *args, **kwargs):
        del fn
        del args
        del kwargs
        self.last_job_id += 1
        return self.DummyJob(self.last_job_id)
