
import datetime
import typing as T
from dataclasses import dataclass, field


class Processor(T.Protocol):
    name: str = 'default'

    def process(self, *args, **kwargs) -> T.Any:
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> T.Any:
        return self.process(*args, **kwargs)


@dataclass
class GlobalContext:

    current_datetime: datetime.datetime
    data: dict = field(default_factory=dict)

    def get(self, key: str, default: T.Any = None) -> T.Any:
        return self.data.get(key, default)

    def set(self, key: str, value: T.Any):
        if key in self.data:
            del self.data[key]
        self.data[key] = value

    def set_result(self, key: str, value: T.Any):
        if isinstance(value, dict):
            for k, v in value.items():
                self.set(k, v)
        elif value is not None:
            self.set(key, value)

    def set_datetime(self, datetime: datetime.datetime):
        self.current_datetime = datetime


class BatchProcessPipeline:
    def __init__(self, processors: list[Processor] = []):
        self.processors = processors
        self.context = GlobalContext(current_datetime=None)

    def register_processor(self, processor: Processor):
        self.processors.append(processor)

    def register_tasks(self, task_generator: T.Callable[[], T.Generator[tuple[datetime.datetime, T.Any], None, None]]):
        self.task_generator = task_generator

    def get_processor(self, name: str) -> Processor:
        for processor in self.processors:
            if processor.name == name:
                return processor
        raise ValueError(f"Processor {name} not found")

    def get_processor_by_type(self, type: T.Type[Processor]) -> Processor:
        for processor in self.processors:
            if isinstance(processor, type):
                return processor
        raise ValueError(f"Processor of type {type} not found")

    def run(self, *args, **kwargs) -> T.Any:
        for dt, task in self.task_generator():
            self.context.set_datetime(dt)
            if task is not None:
                self.context.set('task', task)
            for processor in self.processors:
                result = processor(self.context)
                self.context.set_result(processor.name, result)
