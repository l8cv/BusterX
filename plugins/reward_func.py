import re
from typing import Any, List

from math_verify import parse, verify
from swift.rewards import ORM, orms


def judge_answer(answer: str, solution: str) -> bool:
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL)
    parsed_answer = parse(answer)
    return verify(parsed_answer, solution)


class ExternalFormat(ORM):
    def __call__(self, completions: List[Any], **kwargs: Any) -> List[float]:
        rewards = []
        for completion in completions:
            pattern = re.compile(r"^<think>(.*?)</think>\n\n([^\s].+)$", flags=re.DOTALL)
            if pattern.match(completion):
                answer = re.sub(r"<think>.*?</think>", "", completion, flags=re.DOTALL)
                if "<think>" in answer or "</think>" in answer:
                    rewards.append(-1.0)
                elif len(answer.strip()) < 50:
                    rewards.append(-1.0)
                else:
                    rewards.append(0.0)
            else:
                rewards.append(-1.0)
        return rewards


class ExternalFormatBasedAccuracy(ORM):
    def __call__(self, completions: List[Any], solution: List[Any], **kwargs: Any) -> List[float]:
        format_rewards = ExternalFormat()(completions)

        rewards = []
        for completion, sol, fmt_reward in zip(completions, solution, format_rewards, strict=False):
            fmt_reward_adjusted = fmt_reward + 1.0
            is_correct = judge_answer(completion, sol)
            accuracy_score = 1.0 if is_correct else 0.0
            rewards.append(fmt_reward_adjusted * accuracy_score)

        return rewards


class ExternalAccuracy(ORM):
    def __call__(self, completions: List[Any], solution: List[Any], **kwargs: Any) -> List[float]:
        rewards = []
        for completion, sol in zip(completions, solution, strict=False):
            is_correct = judge_answer(completion, sol)
            accuracy_score = 1.0 if is_correct else 0.0
            rewards.append(accuracy_score)

        return rewards


orms["external_format"] = ExternalFormat
orms["external_format_based_accuracy"] = ExternalFormatBasedAccuracy
orms["external_accuracy"] = ExternalAccuracy
