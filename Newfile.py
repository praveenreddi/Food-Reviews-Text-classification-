lass CustomAssistantAgent(autogen.AssistantAgent):
    def __init__(self, name, system_message="", **kwargs):
        super().__init__(name=name, system_message=system_message, **kwargs)
        self.llm = LlamaLLM(llm_model="openai")

    def _generate_reply(self, messages=None, sender=None, **kwargs):
        """Override the default _generate_reply method to use your custom LLM"""
        if messages is None:
            messages = self.chat_messages[sender]

        # Format messages for your LLM
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

        # Call your LLM
        response = self.llm.call(prompt)

        return response

# Create your custom assistant
assistant = CustomAssistantAgent(
    name="assistant",
    system_message="You are a helpful AI assistant."
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE"
)

# Start conversation
user_proxy.initiate_chat(
    assistant,
    message="Help me solve this problem: calculate the factorial of 5."
)
