import asyncio
import agent_repl

from text_agent_handoff_strucutred_output import pattern

# from text_agent_handoff_using_cv import pattern, context_variables
from autogen.agentchat import run_group_chat, initiate_group_chat  # type:ignore


messages: list[dict] = []


def send_to_swarm(user_msg: str):
    global messages
    res = run_group_chat(
        pattern=pattern,
        # messages="Hey...!",
        messages=[*messages, {"content": user_msg, "role": "user", "name": "user"}],
    )
    return res


def main():
    msg = ""
    global messages
    while True:
        res = send_to_swarm(msg)
        for i in res.events:
            print(i)
            i.print()
            # if i.type == "input_request":
            # print(i.model_dump_json)
        messages = res.messages
        print(f"here are the messages: {messages}")
        msg = input("User: ")
        # i.content.respond(msg)


# asyncio.run(main())
main()
