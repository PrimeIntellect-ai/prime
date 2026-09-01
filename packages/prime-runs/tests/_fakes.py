"""Stand-ins for verifiers ``Trace``/``Episode``.

The SDK never imports a producer package — records are duck-typed — so the
tests must not either, or they would be testing an import rather than the
protocol. These objects implement exactly the surface
:mod:`prime_runs.projection` and :mod:`prime_runs.metrics` touch, which makes
that surface explicit and makes an accidental widening of it fail here.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class Dumpable:
    """Anything the projection calls ``model_dump()`` on."""

    def __init__(self, **values: Any) -> None:
        self._values = values

    def model_dump(self, mode: str = "python", exclude_none: bool = False) -> Dict[str, Any]:
        if exclude_none:
            return {key: value for key, value in self._values.items() if value is not None}
        return dict(self._values)


@dataclass
class Reward:
    score: float


@dataclass
class Agent:
    name: str = "solver"
    trainable: bool = True


@dataclass
class Branch:
    messages: List[Dumpable] = field(default_factory=list)
    num_input_tokens: int = 10
    num_output_tokens: int = 5


class TaskData(Dumpable):
    def __init__(self, idx: int = 0, answer: str = "42", **extra: Any) -> None:
        super().__init__(idx=idx, answer=answer, **extra)
        self.idx = idx


@dataclass
class Task:
    data: TaskData = field(default_factory=TaskData)


@dataclass
class Trace:
    id: str = "trace-1"
    task: Task = field(default_factory=Task)
    agent: Agent = field(default_factory=Agent)
    branches: List[Branch] = field(default_factory=list)
    tools: Optional[List[Dumpable]] = None
    reward: float = 1.0
    timing: Dumpable = field(default_factory=lambda: Dumpable(total_ms=120))
    is_completed: bool = True
    is_truncated: bool = False
    metrics: Dict[str, float] = field(default_factory=dict)
    last_error: Optional[Dumpable] = None
    stop_condition: Optional[str] = "stop"
    usage: Optional[Dumpable] = None
    info: Dict[str, Any] = field(default_factory=dict)
    rewards: Dict[str, Optional[Reward]] = field(default_factory=dict)

    def to_record(self) -> Dict[str, Any]:
        return {"id": self.id, "reward": self.reward}


@dataclass
class EnvInfo:
    id: str = "gsm8k"
    name: Optional[str] = None


@dataclass
class TrainWorkInfo:
    step: int
    type: str = "train"


@dataclass
class EvalWorkInfo:
    step: int
    type: str = "eval"


@dataclass
class TrainRunInfo:
    """verifiers stamps this on every episode a training run dispatches."""

    id: str
    work: Any
    type: str = "train"
    name: Optional[str] = None


@dataclass
class Episode:
    id: str = "episode-1"
    traces: List[Trace] = field(default_factory=list)
    ok: bool = True
    env: EnvInfo = field(default_factory=EnvInfo)
    run: Optional[Any] = None

    def to_record(self) -> Dict[str, Any]:
        return {"id": self.id, "traces": [trace.to_record() for trace in self.traces]}


def make_trace(
    *,
    trace_id: str = "trace-1",
    idx: int = 0,
    trainable: bool = True,
    reward: float = 1.0,
    agent: str = "solver",
    rewards: Optional[Dict[str, Optional[Reward]]] = None,
    metrics: Optional[Dict[str, float]] = None,
    branches: int = 1,
) -> Trace:
    return Trace(
        id=trace_id,
        task=Task(data=TaskData(idx=idx)),
        agent=Agent(name=agent, trainable=trainable),
        branches=[
            Branch(messages=[Dumpable(role="assistant", content=f"branch {n}")])
            for n in range(branches)
        ],
        reward=reward,
        metrics=metrics or {},
        rewards=rewards or {},
    )


def make_episode(
    episode_id: str = "episode-1", traces: Optional[List[Trace]] = None, ok: bool = True
) -> Episode:
    return Episode(id=episode_id, traces=traces if traces is not None else [make_trace()], ok=ok)


def make_train_episode(
    episode_id: str = "episode-1",
    *,
    step: int = 10,
    work: str = "train",
    idx: int = 0,
    reward: float = 1.0,
    advantage: Optional[float] = None,
    env_id: str = "gsm8k",
    run_id: str = "run-1",
) -> Episode:
    """An episode the way prime-rl hands it over: stamped with the run and the
    training step it was dispatched at, ``advantage`` in the trace's info."""
    trace = make_trace(trace_id=f"{episode_id}-t", idx=idx, reward=reward)
    if advantage is not None:
        trace.info["advantage"] = advantage
    work_info = TrainWorkInfo(step=step) if work == "train" else EvalWorkInfo(step=step)
    return Episode(
        id=episode_id,
        traces=[trace],
        env=EnvInfo(id=env_id),
        run=TrainRunInfo(id=run_id, work=work_info),
    )
