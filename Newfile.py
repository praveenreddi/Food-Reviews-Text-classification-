import autogen
import os
import requests
import json

# Enhanced LlamaLLM with debugging
class LlamaLLM:
    def __init__(self, llm_model="openai"):
        self.llm_model = llm_model
        self.llm_url = "https://api.abacus.ai/api/v0/llm/completions"
        self.model = "gpt-4-32k" if llm_model == "openai" else "llama3-70b-instruct"
        print(f"Initialized LlamaLLM with model: {self.model}")

    def call(self, prompt):
        print(f"LlamaLLM received prompt: {prompt[:100]}...")  # Print first 100 chars

        try:
            # Your API call implementation
            access_token = os.environ.get("ABACUS_API_TOKEN", "your-default-token")
            print(f"Using access token: {access_token[:5]}...")  # Print first 5 chars for security

            messages = [{"role": "user", "content": prompt}]

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {access_token}"
            }

            data = {
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 800,
                "top_p": 1
            }

            print(f"Sending request to: {self.llm_url}")
            response = requests.post(self.llm_url, headers=headers, json=data)

            print(f"Response status code: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                print(f"API response: {json.dumps(result)[:200]}...")  # Print first 200 chars

                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

                if not content:
                    print("Warning: Received empty content from API")
                    # Fallback response if API returns empty
                    return "I apologize, but I couldn't generate a response. Please try again."

                print(f"Extracted content: {content[:100]}...")  # Print first 100 chars
                return content
            else:
                error_msg = f"API call failed with status code {response.status_code}: {response.text}"
                print(f"Error: {error_msg}")
                return f"Error calling LLM API: {error_msg}"

        except Exception as e:
            error_msg = f"Exception in LlamaLLM.call: {str(e)}"
            print(f"Error: {error_msg}")
            return f"Error processing request: {error_msg}"

# Create a custom agent class with enhanced debugging
class CustomAssistantAgent(autogen.AssistantAgent):
    def __init__(self, name, system_message="", **kwargs):
        super().__init__(name=name, system_message=system_message, **kwargs)
        self.llm = LlamaLLM(llm_model="openai")
        print(f"Initialized CustomAssistantAgent: {name}")

    def _generate_reply(self, messages=None, sender=None, **kwargs):
        """Override the default _generate_reply method with debugging"""
        print(f"_generate_reply called with sender: {sender}")

        if messages is None:
            print(f"Using chat_messages for sender: {sender}")
            messages = self.chat_messages[sender]

        print(f"Processing {len(messages)} messages")

        # Format messages for your LLM
        formatted_messages = []
        for m in messages:
            role = m.get('role', 'unknown')
            content = m.get('content', '')
            formatted_messages.append(f"{role}: {content}")

        prompt = "\n".join(formatted_messages)
        print(f"Formatted prompt length: {len(prompt)}")

        # Call your LLM
        print("Calling LlamaLLM...")
        response = self.llm.call(prompt)
        print(f"LlamaLLM response length: {len(response) if response else 0}")

        if not response:
            print("Warning: Empty response from LlamaLLM")
            response = "I apologize, but I couldn't generate a response. Please try again."

        return response

# Create your custom assistant
assistant = CustomAssistantAgent(
    name="assistant",
    system_message="You are a helpful AI assistant."
)

# Configure code execution to use local execution instead of Docker
code_execution_config = {
    "work_dir": "coding",
    "use_docker": False  # Disable Docker
}

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",
    code_execution_config=code_execution_config
)

# Start conversation with debugging
print("Starting conversation...")
user_proxy.initiate_chat(
    assistant,
    message="Help me solve this problem: calculate the factorial of 5."
)
