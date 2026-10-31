"""LLM-played callers. Uses Gemini (a different family from the Groq agent) so the agent isn't graded by itself."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from shifa import llm
from shifa.config import get_settings

HANGUP = "[HANGUP]"

SIM_PROMPT = """You are role-playing a CALLER phoning Shifa Family Clinic in Lahore. You are NOT the receptionist.
Persona: {persona}
Your goal: {goal}
Rules:
- Write only your next line as the caller: 1-2 short sentences, in the persona's language style.
- Give details (name, preferred time) only when asked, like a real caller.
- Stay in character; never mention being an AI or a test.
- When your goal is done, clearly impossible, or the receptionist has wrapped up, reply exactly {hangup}"""


class SimulatedCaller:
    def __init__(self, persona: dict):
        self.persona = persona
        s = get_settings()
        self.model = llm.gemini(s.sim_model) if s.google_api_key else llm.chat("smart", 0.7)
        self.system = SIM_PROMPT.format(persona=persona["persona"], goal=persona["goal"], hangup=HANGUP)
        self.history: list = []

    def next_line(self, agent_reply: str | None) -> str:
        if agent_reply is None:
            line = self.persona["opening"]
        else:
            # Roles flip: from the simulator's point of view, the receptionist is the "human".
            self.history.append(HumanMessage(agent_reply))
            line = llm.invoke_with_backoff(self.model, [SystemMessage(self.system), *self.history]).text.strip()
        self.history.append(AIMessage(line))
        return line
