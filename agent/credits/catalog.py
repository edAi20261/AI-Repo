from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping

from shared.config import Config
from shared.errors import NotFoundError


class Action(StrEnum):
    CHAT_MESSAGE = "chat_message"
    GRADE_ANSWERS = "grade_answers"
    STUDENT_REPORT = "student_report"
    CLASS_REPORT = "class_report"
    HOMEWORK_CAPTURE = "homework_capture"
    HOMEWORK_MARK = "homework_mark"
    PERSONA_BUILD = "persona_build"
    VOICE_TRANSCRIBE = "voice_transcribe"
    VOICE_SPEAK = "voice_speak"


@dataclass(frozen=True)
class Plan:
    code: str
    monthly_credits: int


@dataclass(frozen=True)
class Price:
    credits: int
    per: int

    def of(self, units: int) -> int:
        return self.credits * math.ceil(units / self.per)


@dataclass(frozen=True)
class CreditCatalog:
    plans: Mapping[str, Plan]
    prices: Mapping[Action, Price]
    default_plan: str

    @classmethod
    def from_config(cls, config: Config) -> CreditCatalog:
        plans = {code: Plan(code, spec["monthly_credits"])
                 for code, spec in config.get("credits.plans").items()}
        prices = {Action(action): Price(spec["credits"], spec["per"])
                  for action, spec in config.get("credits.prices").items()}
        unpriced = set(Action) - prices.keys()
        if unpriced:
            raise ValueError(f"credits.prices has no price for {sorted(unpriced)}")
        default_plan = config.get("credits.default_plan")
        if default_plan not in plans:
            raise ValueError(f"credits.default_plan {default_plan!r} is not in credits.plans")
        return cls(MappingProxyType(plans), MappingProxyType(prices), default_plan)

    def plan(self, code: str) -> Plan:
        if code not in self.plans:
            raise NotFoundError("There is no such plan.", details={"plan": code})
        return self.plans[code]

    def price(self, action: Action, units: int) -> int:
        return self.prices[action].of(units)
