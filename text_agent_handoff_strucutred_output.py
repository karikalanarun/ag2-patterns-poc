from autogen import LLMConfig, ConversableAgent  # type: ignore
from autogen.agentchat.group.patterns.pattern import DefaultPattern  # type: ignore
from typing import Union, List, Literal, Annotated
from pydantic import BaseModel, Field
from autogen.agentchat.group.targets.transition_target import (
    RevertToUserTarget,
    TerminateTarget,
)

"""
this file will expose a pattern that will use only one structured output for both text and live agent
we will create a pattern with the aim of
1. a single agent model
2. doing both text and agent handoff using same agent
3. by structure output we are aiming to achieve this
4. the output will be very close to small talk implementation without Markdown
"""

"""
TODO: 
1. First create llm_config using api key - Done
    i. create structured output - lets do this finally
2. create conversable agent
3. wrap it around Default Pattern
4. wrap it with local Pattern class and expose it in a variable
"""


# --- Markdown Value Types ---


# Simple text block with a string value
class PlainText(BaseModel):
    type: Literal["text"]  # Discriminator for type-safe parsing
    value: str  # Raw string content


# Concatenation of two markdown values (recursive type)
class JoinMdValue(BaseModel):
    type: Literal["join"]
    lhs: "MDValue"  # Left-hand side markdown element
    rhs: "MDValue"  # Right-hand side markdown element


# Markdown bold formatting
class MdBoldValue(BaseModel):
    type: Literal["bold"]
    value: "MDValue"  # Any valid MDValue can be bolded


# Markdown italic formatting
class MdItalicValue(BaseModel):
    type: Literal["italic"]
    value: "MDValue"


# Markdown underline formatting
class MdUnderlineValue(BaseModel):
    type: Literal["underline"]
    value: "MDValue"


# Markdown strikethrough formatting
class MdStrikethroughValue(BaseModel):
    type: Literal["strikethrough"]
    value: "MDValue"


# Markdown hyperlink with href
class MdHyperLinkValue(BaseModel):
    type: Literal["hyperlink"]
    value: "MDValue"  # Display text
    href: str  # Link URL


# Ordered list with nested markdown elements
class MDOrderedList(BaseModel):
    type: Literal["order_list"]
    value: List["MDValue"]


# Unordered list with nested markdown elements
class MDUnOrderedList(BaseModel):
    type: Literal["unorder_list"]
    value: List["MDValue"]


# Newline for formatting
class MDNewLine(BaseModel):
    type: Literal["new_line"]


# Union of all markdown value types
# `Annotated + Field(discriminator="type")` tells Pydantic how to distinguish each variant
MDValue = Annotated[
    Union[
        PlainText,
        JoinMdValue,
        MdBoldValue,
        MdItalicValue,
        MdUnderlineValue,
        MdStrikethroughValue,
        MdHyperLinkValue,
        MDOrderedList,
        MDUnOrderedList,
        MDNewLine,
    ],
    Field(discriminator="type"),  # Enables automatic model selection at runtime
]

# --- Contract Actions ---


# Request to hand over to a human agent
class RequestAgentHandover(BaseModel):
    action: Literal["request_agent_handover"]
    meta: dict = Field(default_factory=dict)  # Meta field reserved for future data


# Feedback: user is satisfied
class SendSatisfiedFeedback(BaseModel):
    action: Literal["send_satisfied_feedback"]
    meta: dict = Field(default_factory=dict)


# Feedback: user is not satisfied
class SendNotSatisfiedFeedback(BaseModel):
    action: Literal["send_not_satisfied_feedback"]
    meta: dict = Field(default_factory=dict)


# Union of contract actions - easy to extend later
ContractAction = Union[
    RequestAgentHandover, SendSatisfiedFeedback, SendNotSatisfiedFeedback
]

# --- Button and Message Types ---


# Simple button that sends a reply string
class SendRlyAction(BaseModel):
    send_rly: str  # Text payload to send as a reply


# Button that triggers a contract-based action
class ContractActionWrapper(BaseModel):
    contract_action: ContractAction


# Union of possible button actions
BtnAction = SendRlyAction


# A UI button with label and one of the defined actions
class Button(BaseModel):
    label: str
    action: BtnAction


# Message that contains markdown content and interactive buttons
class TextWithBtns(BaseModel):
    type: Literal["text_with_btns"]
    text: str  # Main markdown body
    buttons: List[Button]  # Interactive choices


# Simple message with just markdown content
class TextMessage(BaseModel):
    type: Literal["text"]
    value: str


class DoHumanAgentHandoff(BaseModel):
    type: Literal["handoff"]
    team: Annotated[str, "team id that will be used to agent off to a team"]


# Union of message types
# Discriminator helps in parsing mixed message objects cleanly
Message = Annotated[
    Union[TextMessage, TextWithBtns, DoHumanAgentHandoff], Field(discriminator="type")
]


# class BotMessage(BaseModel):
#     message: Message


class BotMessage(BaseModel):
    type: Literal["text", "text_with_btns", "hand_off_to_live_agent"]
    text: str | None = None
    buttons: list[str] | None = None
    live_agent_team_id: str | None = None



# llm_config = LLMConfig(
#     api_type="openai",
#     model="gpt-4o-mini",
#     response_format=BotMessage,
#     api_key=OPEN_API_KEY,
# )

agent_prompt = """
## 🎯 Prompt: AI Customer Support Agent

### 🧠 Role
You are an **AI-powered Customer Support Agent** working on behalf of **[CompanyName]**, a company that sells [Product/Service]. Your goal is to **assist users, resolve issues, and escalate only when necessary**, all while maintaining a friendly, professional tone.

---

### 📋 Objectives
1. Answer customer queries based on the knowledge base provided.
2. Ask clarifying questions if more information is needed.
3. Escalate to a human agent team only if:
   - The issue involves confidential account access.
   - The problem is unresolved after 3 back-and-forth attempts.
   - The user asked explicitly
4. You have to handoff to only team with the teamid "awesome-123". Ask confirmation with buttons before handoff
5. After the handoff, ask the user to wait till the human from live agent responds
5. Log the conversation summary at the end.
    - the conversation will be end once the handoff happened
    - the conversation will be eneded when user say "exit"

---

### 🗣️ Tone & Style
- Friendly, empathetic, and concise.
- Avoid jargon unless the customer uses it first.
- Use contractions to sound more natural (e.g., "you're" instead of "you are").

---

### 📦 Constraints
- Do **not** make up policies or features that aren't in the knowledge base.
- If you’re unsure, say: “Let me check on that and get back to you.”
- Limit each response to **3 short paragraphs** or fewer.
- If user sends empty messages consider it as a conversation start and greet them.


"""
config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "tags": ["gpt-4o-mini"],
    },
)


admin_created_agent = ConversableAgent(
    name="alan_turing",
    # llm_config=llm_config,
    config_list={"config_list": config_list}
    system_message=agent_prompt,
    # human_input_mode="NEVER",
)


user = ConversableAgent(name="user", human_input_mode="ALWAYS")

pattern = DefaultPattern(
    initial_agent=admin_created_agent,
    agents=[admin_created_agent],
    group_after_work=TerminateTarget(),
    user_agent=user,
)
