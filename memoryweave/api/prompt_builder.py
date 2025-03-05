class PromptBuilder:
    """Builds prompts for LLM interaction."""

    def __init__(self, system_template=None):
        self.system_template = system_template or (
            "You are a helpful assistant. Use the entire conversation context for your answer.\n"
            "Do not disclose that you have a memory system. If asked about user info, incorporate "
            "it naturally if available.\n"
        )

    def build_chat_prompt(
        self, user_message, memories=None, conversation_history=None, query_type=None
    ):
        """Build a prompt with memories and conversation history."""
        # Choose number of top memories based on query type
        max_memories = 5
        if query_type:
            if query_type in ["personal", "temporal"]:
                max_memories = 5
            else:
                max_memories = 3

        # Format memories section
        memory_text = ""
        if memories:
            top_memories = sorted(
                memories, key=lambda x: x.get("relevance_score", 0), reverse=True
            )[:max_memories]

            if top_memories:
                memory_text = "MEMORY HIGHLIGHTS:\n"
                for m in top_memories:
                    # Handle different content formats
                    if isinstance(m.get("content"), dict) and "text" in m["content"]:
                        content = m["content"]["text"]
                    elif isinstance(m.get("content"), str):
                        content = m["content"]
                    else:
                        # Try to get content from other possible fields
                        content = m.get("text", str(m))

                    memory_text += (
                        f"- {content[:150]}...\n" if len(content) > 150 else f"- {content}\n"
                    )
                memory_text += "\n"

        # Build system prompt with memories
        system_prompt = self.system_template + memory_text

        # Append conversation history
        history_text = ""
        max_history_turns = 10
        recent_history = conversation_history[-max_history_turns:] if conversation_history else []

        for turn in recent_history:
            role = "User" if turn["role"] == "user" else "Assistant"
            history_text += f"{role}: {turn['content']}\n"

        # Combine all parts
        prompt = f"{system_prompt}\n{history_text}User: {user_message}\nAssistant:"

        return prompt
