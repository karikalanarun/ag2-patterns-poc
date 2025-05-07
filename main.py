import asyncio
import agent_repl

# from text_agent_handoff_strucutred_output import pattern
from text_agent_handoff_using_cv import pattern, context_variables
from autogen.agentchat import run_group_chat


def main():
    res = run_group_chat(pattern=pattern, messages="Hey...!")
    res.process()
    # for i in res.events:
    #     print(i, res.context_variables)


# asyncio.run(main())
main()
