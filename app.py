from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
llm = OpenAI(temperature=0)
template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
Friend: {input}
AI:"""
PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
conversation = ConversationChain(
    prompt=PROMPT,
    llm=llm,
    verbose=False,
    memory=ConversationBufferMemory(human_prefix="Human"),
)

response_1 = conversation.predict(input="What's the weather like in London?")
response_2 = conversation.predict(input="Tell me a story about yourself")
response_3 = conversation.predict(input="Tell me what we were talking about few messages ago")
print(response_1)
print(response_2)
print(response_3)

while True:
    print("Human:")
    question = input()
    response = conversation.predict(input=question)
    print("AI:")
    print(response)