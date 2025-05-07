from dataclasses import dataclass
from typing import Any


@dataclass
class PatternData:
    welcome_message: str
    pattern: Any
