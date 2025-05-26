import asyncio
import aiohttp
from typing import List, Dict, Any
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_core.models import ChatCompletionClient

class CustomLLMClient(ChatCompletionClient):
    def __init__(self, base_url: str, access_token: str):
        self.base_url = base_url
        self.access_token = access_token

    async def create(self, messages, model, **kwargs):
        # Format messages for your LLM
        prompt = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
        prompt += "Assistant:"

        # Call your LLM
        headers = {"Authorization": f"Bearer {self.access_token}"}
        payload = {"prompt": prompt, "max_tokens": 500}

        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, headers=headers, json=payload) as response:
                result = await response.json()

        # Return response in correct format
        response_text = result.get("response", "No response")
        return {
            "choices": [{"message": {"role": "assistant", "content": response_text}}],
            "model": model
        }

    # ADD THESE MISSING METHODS:
    @property
    def capabilities(self):
        return {"chat_completion": True}

    def actual_usage(self):
        return {}

    def remaining_tokens(self):
        return 1000

    def total_usage(self):
        return {}

    def count_tokens(self, messages):
        return 100

    def create_stream(self, messages, model, **kwargs):
        pass

    def close(self):
        pass

# Main code
async def run_chat():
    # REPLACE WITH YOUR ACTUAL VALUES:
    LLM_URL = "YOUR_LLM_ENDPOINT"  
    TOKEN = "YOUR_TOKEN"

    # Create agents
    llm_client = CustomLLMClient(LLM_URL, TOKEN)
    assistant = AssistantAgent("assistant", model_client=llm_client)
    user = UserProxyAgent("user")

    # Start chat
    result = await user.a_initiate_chat(
        recipient=assistant,
        message="Calculate factorial of 5",
        max_turns=2
    )
    print("Done!")

# Run in Databricks:
await run_chat()




import asyncio
import aiohttp
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_core.models import ChatCompletionClient

class CustomLLMClient(ChatCompletionClient):
    def __init__(self, base_url: str, access_token: str):
        self.base_url = base_url
        self.access_token = access_token

    async def create(self, messages, model, **kwargs):
        # Format messages for your LLM
        prompt = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
        prompt += "Assistant:"

        # Call your LLM
        headers = {"Authorization": f"Bearer {self.access_token}"}
        payload = {"prompt": prompt, "max_tokens": 500}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, headers=headers, json=payload) as response:
                result = await response.json()
                
        # Return response in correct format
        response_text = result.get("response", "No response")
        return {
            "choices": [{"message": {"role": "assistant", "content": response_text}}],
            "model": model
        }

# Main code
async def run_chat():
    # Your LLM details
    LLM_URL = "YOUR_LLM_ENDPOINT"
    TOKEN = "YOUR_TOKEN"
    
    # Create agents
    llm_client = CustomLLMClient(LLM_URL, TOKEN)
    assistant = AssistantAgent("assistant", model_client=llm_client)
    user = UserProxyAgent("user")
    
    # Start chat
    result = await user.a_initiate_chat(
        recipient=assistant,
        message="Calculate factorial of 5",
        max_turns=2
    )
    print("Done!")

# For Databricks - use this:
await run_chat()


You're absolutely right! Let me give you a **simple, clean version** for Databricks:

## Simple AutoGen 0.4 Code for Databricks

```python
import asyncio
import aiohttp
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_core.models import ChatCompletionClient

class CustomLLMClient(ChatCompletionClient):
    def __init__(self, base_url: str, access_token: str):
        self.base_url = base_url
        self.access_token = access_token

    async def create(self, messages, model, **kwargs):
        # Format messages for your LLM
        prompt = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
        prompt += "Assistant:"

        # Call your LLM
        headers = {"Authorization": f"Bearer {self.access_token}"}
        payload = {"prompt": prompt, "max_tokens": 500}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, headers=headers, json=payload) as response:
                result = await response.json()
                
        # Return response in correct format
        response_text = result.get("response", "No response")
        return {
            "choices": [{"message": {"role": "assistant", "content": response_text}}],
            "model": model
        }

# Main code
async def run_chat():
    # Your LLM details
    LLM_URL = "YOUR_LLM_ENDPOINT"
    TOKEN = "YOUR_TOKEN"
    
    # Create agents
    llm_client = CustomLLMClient(LLM_URL, TOKEN)
    assistant = AssistantAgent("assistant", model_client=llm_client)
    user = UserProxyAgent("user")
    
    # Start chat
    result = await user.a_initiate_chat(
        recipient=assistant,
        message="Calculate factorial of 5",
        max_turns=2
    )
    print("Done!")

# For Databricks - use this:
await run_chat()
```

## What to Change:

1. **Replace these two lines:**
   ```python
   LLM_URL = "https://your-actual-endpoint.com/generate"
   TOKEN = "your_actual_access_token"
   ```

2. **Adjust the payload** based on your LLM's API:
   ```python
   payload = {"prompt": prompt, "max_tokens": 500}
   # Change to whatever your LLM expects
   ```

3. **Adjust response parsing**:
   ```python
   response_text = result.get("response", "No response")
   # Change "response" to whatever key your LLM uses
   ```

That's it! Much simpler and works in Databricks. Just update those 3 things with your actual LLM details.




import asyncio
import aiohttp
from typing import List, Dict, Any

# CORRECT IMPORTS FOR AUTOGEN 0.4
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_core.models import ChatCompletionClient
from autogen_core import CancellationToken

class CustomLLMClient(ChatCompletionClient):
    """
    Custom LLM Client that integrates your LLM directly with AutoGen 0.4
    """
    def __init__(self, base_url: str, access_token: str, model_name: str = "custom-model"):
        self.base_url = base_url
        self.access_token = access_token
        self.model_name = model_name

    async def create(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Main method that AutoGen calls to get LLM responses
        """
        try:
            print(f"🔄 Calling custom LLM with {len(messages)} messages...")
            
            # Format messages for your LLM
            formatted_prompt = self._format_messages(messages)
            print(f"📝 Formatted prompt: {formatted_prompt[:100]}...")

            # Call your LLM API
            response_content = await self._call_llm_async(formatted_prompt)
            print(f"✅ LLM responded with: {response_content[:100]}...")

            # Return in OpenAI-compatible format (required by AutoGen)
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": response_content
                    },
                    "finish_reason": "stop"
                }],
                "model": model,
                "usage": {
                    "prompt_tokens": len(formatted_prompt.split()),
                    "completion_tokens": len(response_content.split()),
                    "total_tokens": len(formatted_prompt.split()) + len(response_content.split())
                }
            }

        except Exception as e:
            print(f"❌ Error in LLM call: {e}")
            # Return error response in correct format
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": f"I apologize, but I encountered an error: {str(e)}"
                    },
                    "finish_reason": "stop"
                }],
                "model": model,
                "usage": {"total_tokens": 0}
            }

    def _format_messages(self, messages: List[Dict[str, Any]]) -> str:
        """
        Convert AutoGen messages to your LLM's expected format
        """
        formatted_parts = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if content and content.strip():
                if role == "system":
                    formatted_parts.append(f"System: {content}")
                elif role == "user":
                    formatted_parts.append(f"User: {content}")
                elif role == "assistant":
                    formatted_parts.append(f"Assistant: {content}")

        # Create the full prompt
        full_prompt = "\n".join(formatted_parts)
        if full_prompt:
            full_prompt += "\nAssistant:"
        
        return full_prompt

    async def _call_llm_async(self, formatted_prompt: str) -> str:
        """
        Make the actual HTTP call to your LLM endpoint
        """
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }

        # Adjust this payload based on your LLM's API requirements
        payload = {
            "prompt": formatted_prompt,
            "max_tokens": 1000,
            "temperature": 0.7,
            "stop": ["\nUser:", "\nSystem:"],  # Stop tokens to prevent infinite generation
            # Add other parameters your LLM expects:
            # "top_p": 0.9,
            # "frequency_penalty": 0.0,
            # "presence_penalty": 0.0,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, headers=headers, json=payload) as response:
                response.raise_for_status()
                result = await response.json()

                # Parse response based on your LLM's format
                # Adjust these based on how your LLM returns responses:
                if "response" in result:
                    return result["response"].strip()
                elif "text" in result:
                    return result["text"].strip()
                elif "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0].get("text", "").strip()
                elif "generated_text" in result:
                    return result["generated_text"].strip()
                else:
                    return "No response generated from LLM."

    @property
    def capabilities(self) -> Dict[str, Any]:
        """
        Define what your LLM can do
        """
        return {
            "chat_completion": True,
            "function_calling": False,  # Set to True if your LLM supports function calling
            "streaming": False,         # Set to True if your LLM supports streaming
            "vision": False,           # Set to True if your LLM supports images
        }

async def main():
    """
    Main function demonstrating Approach 1: Direct Custom LLM Integration
    """
    
    print("🚀 APPROACH 1: Direct Custom LLM Integration")
    print("=" * 60)
    
    # REPLACE THESE WITH YOUR ACTUAL VALUES
    LLM_ENDPOINT = "YOUR_LLM_ENDPOINT_URL"  # e.g., "https://api.your-llm.com/v1/generate"
    ACCESS_TOKEN = "YOUR_ACCESS_TOKEN"       # Your actual access token
    
    # Validate configuration
    if LLM_ENDPOINT == "YOUR_LLM_ENDPOINT_URL" or ACCESS_TOKEN == "YOUR_ACCESS_TOKEN":
        print("❌ Please update LLM_ENDPOINT and ACCESS_TOKEN with your actual values!")
        return

    try:
        # Step 1: Create your custom LLM client
        print("📡 Creating custom LLM client...")
        custom_llm = CustomLLMClient(
            base_url=LLM_ENDPOINT,
            access_token=ACCESS_TOKEN,
            model_name="your-custom-model"
        )

        # Step 2: Create AssistantAgent with your custom LLM
        print("🤖 Creating assistant agent...")
        assistant = AssistantAgent(
            name="math_assistant",
            model_client=custom_llm,
            system_message="You are a helpful AI assistant specialized in mathematics. "
                          "Solve problems step by step and provide clear explanations."
        )

        # Step 3: Create UserProxyAgent (represents the user)
        print("👤 Creating user proxy agent...")
        user_proxy = UserProxyAgent(
            name="user",
            human_input_mode="NEVER"  # Won't ask for human input
        )

        # Step 4: Start the conversation
        print("\n💬 Starting conversation...")
        print("-" * 40)

        # Create cancellation token for async control
        cancellation_token = CancellationToken()

        # Initiate chat between user and assistant
        result = await user_proxy.a_initiate_chat(
            recipient=assistant,
            message="Help me solve this problem: calculate the factorial of 5 step by step.",
            max_turns=3  # Limit conversation to 3 turns
        )

        print("\n" + "=" * 60)
        print("✅ APPROACH 1 COMPLETED SUCCESSFULLY!")
        print(f"📊 Conversation Summary:")
        print(f"   - Total messages: {len(result.messages) if hasattr(result, 'messages') else 'N/A'}")
        print(f"   - Final result: {result.summary if hasattr(result, 'summary') else 'Conversation completed'}")

    except Exception as e:
        print(f"\n❌ Error in Approach 1: {e}")
        print("🔧 Please check your LLM endpoint and access token.")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())





import asyncio
import aiohttp
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_core.components.models import LLMMessage, ChatCompletionContext
from autogen_agentchat.models import ChatCompletionClient
from typing import List, Dict, Any

class CustomLLMClient(ChatCompletionClient):
    def __init__(self, base_url: str, access_token: str):
        self.base_url = base_url
        self.access_token = access_token

    async def create_chat_completion(
        self,
        messages: List[LLMMessage],
        model: str,
        **kwargs: Any
    ) -> ChatCompletionContext:
        try:
            # Format messages for your LLM
            formatted_prompt = self._format_messages(messages)
            
            # Call your LLM
            response_content = await self._call_llm_async(formatted_prompt)

            # Create response
            response_message = LLMMessage(
                content=response_content,
                source="assistant"
            )

            return ChatCompletionContext(
                messages=messages + [response_message],
                model=model
            )

        except Exception as e:
            print(f"Error in LLM call: {e}")
            error_message = LLMMessage(
                content="I apologize, but I encountered an error processing your request.",
                source="assistant"
            )
            return ChatCompletionContext(
                messages=messages + [error_message],
                model=model
            )

    def _format_messages(self, messages: List[LLMMessage]) -> str:
        formatted_parts = []
        for msg in messages:
            if msg.content and msg.content.strip():
                if msg.source == "system":
                    formatted_parts.append(f"System: {msg.content}")
                elif msg.source == "user":
                    formatted_parts.append(f"User: {msg.content}")
                elif msg.source == "assistant":
                    formatted_parts.append(f"Assistant: {msg.content}")

        full_prompt = "\n".join(formatted_parts)
        if full_prompt:
            full_prompt += "\nAssistant:"
        return full_prompt

    async def _call_llm_async(self, formatted_prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }

        payload = {
            "prompt": formatted_prompt,
            "max_tokens": 1000,
            "temperature": 0.7,
            # Adjust these based on your LLM's API
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, headers=headers, json=payload) as response:
                response.raise_for_status()
                result = await response.json()

                # Adjust based on your LLM's response format
                if "response" in result:
                    return result["response"].strip()
                elif "text" in result:
                    return result["text"].strip()
                else:
                    return "No response generated."

    @property
    def capabilities(self) -> Dict[str, Any]:
        return {"chat_completion": True, "function_calling": False, "streaming": False}

async def main():
    # REPLACE WITH YOUR ACTUAL VALUES
    LLM_ENDPOINT = "YOUR_LLM_ENDPOINT_URL"
    ACCESS_TOKEN = "YOUR_ACCESS_TOKEN"

    # Create your custom LLM client
    custom_llm = CustomLLMClient(
        base_url=LLM_ENDPOINT,
        access_token=ACCESS_TOKEN
    )

    # Create agents
    assistant = AssistantAgent(
        name="assistant",
        model_client=custom_llm,
        system_message="You are a helpful AI assistant. Solve problems step by step."
    )

    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER"
    )

    # Start conversation
    print("Starting conversation...")
    result = await user_proxy.a_initiate_chat(
        recipient=assistant,
        message="Help me solve this problem: calculate the factorial of 5.",
        max_turns=3
    )

    print("Conversation completed!")

if __name__ == "__main__":
    asyncio.run(main())




import autogen
import os
import requests
import json

# Enhanced LlamaLLM with debugging
class LlamaLLM:
    def __i, llm_model="openai"):
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









import autogen
import os
import requests
import json

# Enhanced LlamaLLM with debugging
class LlamaLLM:
    def __i, llm_model="openai"):
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
