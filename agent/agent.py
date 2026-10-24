"""Conversational agent: a research scientist asks about core configurations, Claude runs them
through the prediction backend via tools and summarizes the results."""

from __future__ import annotations

from collections.abc import Callable

import anthropic

from agent.tools import Session, make_tools

MODEL = "claude-opus-5-5"

SYSTEM = """You are an assistant for reactor physicists studying a 2-D quarter-core PWR model. \
They ask how core outputs (k_eff, reactivity, power and 2-group flux distributions) respond to \
control-rod positions and assembly loading changes. You answer by running configurations through \
the tools and interpreting the results.

How to work:
- Call describe_core once per conversation before the first evaluation, so you know the banks, \
materials, coordinate convention and the backend's limitations.
- Translate the request into explicit configurations. Put all configurations to be compared in one \
evaluate_configurations call, with the scientist's baseline (usually all rods withdrawn) first so the \
delta fields are relative to it. If the request is ambiguous (which banks, what insertion), state the \
interpretation you chose rather than stalling, unless the choice would change the answer's meaning.
- Use get_map when the question is spatial (where flux is depressed, radial tilt, shadowing).

How to report:
- Every number comes from a tool result; never estimate or extrapolate values you did not compute. \
If the model cannot represent something asked for (boron, temperature, burnup, pin-level peaking), \
say so plainly.
- Give units and normalization: k_eff to 5 decimals, reactivity and rod worth in pcm, power and \
flux as relative values normalized to mean fuel power = 1. Refer to assemblies as (row, col).
- Lead with a compact comparison table, then a short physical interpretation (e.g. thermal flux \
depression around inserted rods, power shifting outward). Keep it concise.
- Name the backend that produced the numbers and note that results are a research surrogate, not \
safety-grade. When the backend is the stand-in reference solver, say the ML surrogate is not yet in use."""


class CoreAgent:
    """Multi-turn conversation; `ask` runs one user turn through the tool loop."""

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        session: Session | None = None,
        model: str = MODEL,
        effort: str = "medium",
    ):
        self.client = client or anthropic.Anthropic()
        self.session = session or Session()
        self.tools = make_tools(self.session)
        self.model = model
        self.effort = effort
        self.messages: list = []

    def ask(self, text: str, on_tool_call: Callable[[str, dict], None] | None = None) -> str:
        self.messages.append({"role": "user", "content": text})
        runner = self.client.beta.messages.tool_runner(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM,
            tools=self.tools,
            messages=list(self.messages),
            thread=False,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            max_iterations=10,
        )
        last = None
        for message in runner:
            last = message
            # Mirror the history: the runner keeps its own copy.
            self.messages.append({"role": "assistant", "content": message.content})
            if on_tool_call:
                for block in message.content:
                    if block.type == "tool_use":
                        on_tool_call(block.name, block.input)
            tool_response = runner.generate_tool_call_response()  # cached; tools run once
            if tool_response is not None:
                self.messages.append(tool_response)

        if last is None:
            return ""
        if last.stop_reason == "refusal":
            return "[The model declined this request.]"
        text_out = "".join(b.text for b in last.content if b.type == "text")
        if last.stop_reason == "max_tokens":
            text_out += "\n[Response truncated at max_tokens.]"
        return text_out
