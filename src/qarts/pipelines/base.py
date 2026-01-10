
import datetime
import typing as T
from dataclasses import dataclass, field


class Processor(T.Protocol):
    _name: str = 'default'
    _input_fields: list[str] = None
    _output_fields: list[str] = None

    def process(self, *args, **kwargs) -> T.Any:
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> T.Any:
        return self.process(*args, **kwargs)
    
    @property
    def input_fields(self) -> list[str]:
        if self._input_fields is None:
            return []
        else:
            return self._input_fields

    @property
    def output_fields(self) -> list[str]:
        if self._output_fields is None:
            return [self.name]
        else:
            return self._output_fields

    @property
    def name(self) -> str:
        return self._name


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
    def __init__(self, ):
        self.processors = []
        self.context = GlobalContext(current_datetime=None)
        self.all_output_fields = set()
        self.all_input_fields = set()
        self.input_fields = {}
        self.output_fields = {}

    def register_processor(self, processor: Processor):
        for field in processor.input_fields:
            if field not in self.all_output_fields:
                raise ValueError(f"Input field {field} not found in any processor")
            self.all_input_fields.add(field)
        for field in processor.output_fields:
            if field in self.all_output_fields or field == 'task':
                raise ValueError(f"Output field {field} already registered")
            self.all_output_fields.add(field)
        self.input_fields[processor.name] = processor.input_fields
        self.output_fields[processor.name] = processor.output_fields
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
