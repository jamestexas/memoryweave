#!/usr/bin/env python3

"""
RAM Benchmark - Retrieval Augmented Memory Benchmarking Tool

This script benchmarks MemoryWeave against other memory/retrieval systems:
1. Standard RAG (vector similarity only)
2. Recency-biased retrieval (like ChatGPT)
3. Context window stuffing (baseline)

It tests across multiple scenarios designed to highlight differences in approach:
- Document retrieval
- Conversational memory
- Temporal references
- Mixed knowledge
- Large context handling
"""

import json
import logging
import os
import time
import traceback
from datetime import datetime
from enum import Enum
from typing import Any

# Disable warning messages from dependencies
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import matplotlib.pyplot as plt
import psutil
import rich_click as click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

# Import MemoryWeave components
from memoryweave.api.chunked_memory_weave import ChunkedMemoryWeaveAPI
from memoryweave.api.memory_weave import MemoryWeaveAPI

# Configure logging
console = Console(highlight=True)
logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(show_path=False, rich_tracebacks=True)],
)

# Silence other loggers
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("faiss").setLevel(logging.ERROR)

logger = logging.getLogger("ram_benchmark")

DEFAULT_MODEL = "unsloth/Llama-3.2-3B-Instruct"


# Define system types for benchmarking
class SystemType(str, Enum):
    """Types of systems to benchmark."""

    MEMORYWEAVE_CHUNKED = "memoryweave_chunked"
    MEMORYWEAVE_STANDARD = "memoryweave_standard"
    STANDARD_RAG = "standard_rag"
    RECENCY_BIASED = "recency_biased"
    CONTEXT_WINDOW = "context_window"


# Test scenarios
class BenchmarkScenario:
    """Base class for benchmark scenarios."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.memories = []
        self.queries = []
        self.expected_answers = {}

    def get_memories(self) -> list[dict[str, Any]]:
        """Get memories for this scenario."""
        return self.memories

    def get_queries(self) -> list[str]:
        """Get test queries for this scenario."""
        return self.queries

    def get_expected_answers(self, query: str) -> list[str]:
        """Get expected answers for a query."""
        return self.expected_answers.get(query, [])

    def get_metrics(self) -> dict[str, Any]:
        """Get scenario-specific metrics."""
        return {
            "memory_count": len(self.memories),
            "query_count": len(self.queries),
            "avg_memory_length": sum(len(m.get("text", "")) for m in self.memories)
            / len(self.memories)
            if self.memories
            else 0,
        }


class DocumentRetrieval(BenchmarkScenario):
    """Document-focused retrieval scenario."""

    def __init__(self):
        super().__init__(
            "Document Retrieval", "Tests ability to retrieve information from longer documents"
        )

        # Article about neural networks, broken into chunks
        self.article = """
Neural networks are a series of algorithms that mimic the operations of a human brain to recognize relationships between vast amounts of data. They are used for a variety of applications in artificial intelligence, including image and speech recognition, natural language processing, and more.

The concept of neural networks was inspired by the biological neural networks that constitute animal brains. The basic unit of computation in a neural network is the neuron, also called a node or unit. It receives input from other nodes or from an external source and computes an output. Each input has an associated weight (w), which is assigned based on its relative importance to other inputs.

The most common type of neural network is the feedforward neural network, in which information moves in only one direction—forward—from the input nodes, through the hidden nodes (if any), and to the output nodes. There are no cycles or loops in the network. Feedforward neural networks are widely used for pattern recognition.

Convolutional Neural Networks (CNNs) are a specific type of neural network that has proven highly effective in areas such as image recognition and classification. CNNs are regularized versions of multilayer perceptrons designed to recognize patterns in images.

Recurrent Neural Networks (RNNs) are particularly useful for sequential data, like text, speech, or time series data. Unlike feedforward networks, RNNs have feedback connections, allowing them to maintain a 'memory' of previous inputs.

A more advanced version of RNNs is Long Short-Term Memory networks (LSTMs), which are particularly good at learning from experiences when there are very long time lags of unknown duration between important events.

The training process of a neural network involves the use of backpropagation, an algorithm that adjusts the weights of the connections between neurons to minimize the difference between the network's output and the desired output.

Deep learning refers to neural networks with more than three layers, including the input and output layers. The 'deep' in deep learning refers to this idea of having multiple layers of neurons.

Some challenges with neural networks include overfitting, where the network memorizes the training data too well and performs poorly on new, unseen data; and the vanishing gradient problem, which can make it difficult to train networks with many layers.

Despite these challenges, neural networks have been incredibly successful and are a key component of many state-of-the-art systems in fields like computer vision, natural language processing, and speech recognition. With continuing advances in computational power and algorithm design, the capabilities of neural networks are expected to grow even further in the coming years.
"""

        # Create memories from article
        chunks = self._chunk_text(self.article, 200, 50)
        for i, chunk in enumerate(chunks):
            self.memories.append({
                "text": chunk,
                "type": "article",
                "topic": "neural networks",
                "timestamp": time.time() - (len(chunks) - i) * 3600,  # Older to newer
            })

        # Add a few other documents for variety
        self.memories.append({
            "text": "Transformers are a type of neural network architecture that has revolutionized natural language processing. They rely on a mechanism called attention, which allows the model to focus on different parts of the input sequence when generating each part of the output.",
            "type": "article",
            "topic": "transformers",
            "timestamp": time.time() - 86400,
        })

        self.memories.append({
            "text": "Reinforcement learning is an area of machine learning where an agent learns to make decisions by taking actions in an environment to maximize some notion of cumulative reward. It differs from supervised learning in that the agent is not told which actions to take but must discover which actions yield the most reward by trying them.",
            "type": "article",
            "topic": "reinforcement learning",
            "timestamp": time.time() - 86400 * 2,
        })

        # Create test queries
        self.queries = [
            "What is the basic unit of computation in a neural network?",
            "What are CNNs used for?",
            "How is RNN different from a feedforward neural network?",
            "What is backpropagation used for?",
            "What are some challenges with neural networks?",
        ]

        # Define expected answers
        self.expected_answers = {
            "What is the basic unit of computation in a neural network?": [
                "neuron",
                "node",
                "unit",
            ],
            "What are CNNs used for?": [
                "image recognition",
                "image classification",
                "recognition",
                "classification",
            ],
            "How is RNN different from a feedforward neural network?": [
                "feedback connections",
                "memory",
                "previous inputs",
            ],
            "What is backpropagation used for?": ["adjusts", "weights", "minimize", "difference"],
            "What are some challenges with neural networks?": ["overfitting", "vanishing gradient"],
        }

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        """Split text into chunks with overlap."""
        chunks = []
        paragraphs = text.split("\n\n")
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding this paragraph would exceed the chunk size
            if len(current_chunk) > 0 and len(current_chunk) + len(para) > chunk_size:
                chunks.append(current_chunk)
                # Keep some overlap for context
                overlap_text = current_chunk.split(" ")[-overlap:]
                current_chunk = " ".join(overlap_text) + " " + para
            else:
                if current_chunk:
                    current_chunk += " " + para
                else:
                    current_chunk = para

        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)

        return chunks


class ConversationalMemory(BenchmarkScenario):
    """Tests ability to recall information from past conversation turns."""

    def __init__(self):
        super().__init__(
            "Conversational Memory",
            "Tests ability to recall information from previous conversation turns",
        )

        # Create a simulated conversation with embedded personal details
        conversation = [
            {
                "role": "user",
                "content": "Hi there! My name is Jordan and I'm planning a trip to Japan next month.",
            },
            {
                "role": "assistant",
                "content": "Hello Jordan! That sounds exciting. Japan is a wonderful destination. Do you have specific cities or regions you'd like to visit?",
            },
            {
                "role": "user",
                "content": "I want to visit Tokyo and Kyoto. I'm especially interested in seeing cherry blossoms and trying authentic ramen. I'll be traveling with my wife Emma.",
            },
            {
                "role": "assistant",
                "content": "Great choices! Tokyo and Kyoto offer very different experiences of Japan. For cherry blossoms (sakura), the timing can vary each year, but they typically bloom in late March to early April. Both cities have excellent viewing spots. Tokyo has Ueno Park and Shinjuku Gyoen, while Kyoto has the beautiful Philosopher's Path and Maruyama Park. For authentic ramen, Tokyo's Shinjuku district has some renowned shops. You and Emma should definitely try different regional styles!",
            },
            {
                "role": "user",
                "content": "That sounds perfect. We're planning to stay for 10 days. I was thinking 5 days in Tokyo and 5 in Kyoto. Do you think that's a good split?",
            },
            {
                "role": "assistant",
                "content": "A 5-5 split between Tokyo and Kyoto is actually quite balanced for a 10-day trip! Tokyo is vast and modern with countless neighborhoods to explore, while Kyoto is more traditional with numerous temples, shrines, and cultural sites. Both deserve ample time. This arrangement also gives you enough time to take day trips if you wish - from Tokyo you could visit Kamakura or Nikko, and from Kyoto you might consider Nara or Osaka.",
            },
            {
                "role": "user",
                "content": "Great. I should also mention that Emma is vegetarian. Will that be difficult to accommodate in Japan?",
            },
            {
                "role": "assistant",
                "content": "That's an important consideration. Japan can be challenging for vegetarians but is becoming more accommodating. In Tokyo, you'll find more vegetarian-friendly restaurants, especially in areas like Shibuya and Roppongi. In Kyoto, many Buddhist temple restaurants (shojin ryori) serve traditional vegetarian cuisine. Some tips: learn the phrase 'Watashi wa bejitarian desu' (I am vegetarian), look for Indian restaurants which often have vegetarian options, and consider using the Happy Cow app to find suitable places. Also, be aware that dashi (fish stock) is common in many dishes that might otherwise seem vegetarian.",
            },
            {
                "role": "user",
                "content": "That's very helpful. By the way, we're staying at the Century Tokyo Hotel and the Kyoto Gracery. Have you heard of them?",
            },
            {
                "role": "assistant",
                "content": "Both are good choices! The Century Tokyo Hotel is well-located with good access to transportation, making it convenient for exploring the city. The Kyoto Gracery is in a central location, close to shopping areas and within reasonable distance to many attractions. Both hotels are known for good service and comfortable accommodations. You and Emma should be well-situated for your adventures in both cities.",
            },
            {
                "role": "user",
                "content": "One more thing - we arrive on April 12th. Is that a good time for cherry blossoms?",
            },
            {
                "role": "assistant",
                "content": "April 12th could be good timing for cherry blossoms, though it depends on the specific year and weather conditions. Typically, early to mid-April is toward the end of the cherry blossom season in Tokyo, while Kyoto's season often runs a bit later. By April 12th, Tokyo's blossoms might be starting to fall, creating beautiful 'sakura snow,' while Kyoto might be at peak bloom or approaching it. I'd recommend checking a cherry blossom forecast closer to your trip date. Even if you miss peak bloom in some areas, Japan is beautiful in spring regardless!",
            },
        ]

        # Extract memories from conversation
        timestamp_base = time.time() - 86400  # Start from yesterday
        for i, turn in enumerate(conversation):
            turn_timestamp = timestamp_base + i * 600  # 10-minute intervals

            self.memories.append({
                "text": turn["content"],
                "type": "conversation",
                "role": turn["role"],
                "timestamp": turn_timestamp,
                "turn_index": i,
            })

        # Create test queries about the conversation
        self.queries = [
            "What is my name?",
            "Where am I planning to travel?",
            "When do I arrive in Japan?",
            "What hotels am I staying at?",
            "What dietary restriction does Emma have?",
            "How many days am I planning to stay in each city?",
        ]

        # Define expected answers
        self.expected_answers = {
            "What is my name?": ["Jordan"],
            "Where am I planning to travel?": ["Japan", "Tokyo", "Kyoto"],
            "When do I arrive in Japan?": ["April 12th", "April"],
            "What hotels am I staying at?": ["Century Tokyo Hotel", "Kyoto Gracery"],
            "What dietary restriction does Emma have?": ["vegetarian"],
            "How many days am I planning to stay in each city?": ["5", "five", "10", "ten"],
        }


class TemporalReferences(BenchmarkScenario):
    """Tests ability to understand and respond to time-based queries."""

    def __init__(self):
        super().__init__(
            "Temporal References", "Tests ability to understand and respond to time-based queries"
        )

        # Create memories with explicit timestamps spread across time
        now = time.time()

        # Yesterday's memories
        yesterday = now - 86400
        self.memories.append({
            "text": "I went for a run in Central Park this morning.",
            "type": "activity",
            "timestamp": yesterday + 28800,  # 8 AM yesterday
        })

        self.memories.append({
            "text": "I had sushi for lunch at Katsu restaurant.",
            "type": "meal",
            "timestamp": yesterday + 43200,  # Noon yesterday
        })

        self.memories.append({
            "text": "I watched the movie 'Inception' in the evening.",
            "type": "entertainment",
            "timestamp": yesterday + 72000,  # 8 PM yesterday
        })

        # Last week's memories
        last_week = now - 7 * 86400
        self.memories.append({
            "text": "I started reading a book called 'Project Hail Mary' by Andy Weir.",
            "type": "activity",
            "timestamp": last_week + 50000,  # During last week
        })

        self.memories.append({
            "text": "I met with my friend Alex for coffee at Starbucks.",
            "type": "social",
            "timestamp": last_week + 150000,  # Later last week
        })

        # This morning's memories
        this_morning = now - 28800  # 8 hours ago
        self.memories.append({
            "text": "I had oatmeal with berries for breakfast.",
            "type": "meal",
            "timestamp": this_morning,
        })

        # Create test queries with temporal references
        self.queries = [
            "What did I do yesterday?",
            "What movie did I watch yesterday?",
            "What did I eat for lunch yesterday?",
            "What book did I start reading last week?",
            "Who did I meet last week?",
            "What did I have for breakfast this morning?",
        ]

        # Define expected answers
        self.expected_answers = {
            "What did I do yesterday?": [
                "run",
                "Central Park",
                "sushi",
                "lunch",
                "Katsu",
                "Inception",
                "movie",
            ],
            "What movie did I watch yesterday?": ["Inception"],
            "What did I eat for lunch yesterday?": ["sushi", "Katsu"],
            "What book did I start reading last week?": ["Project Hail Mary", "Andy Weir"],
            "Who did I meet last week?": ["Alex", "friend", "coffee", "Starbucks"],
            "What did I have for breakfast this morning?": ["oatmeal", "berries"],
        }


class MixedKnowledge(BenchmarkScenario):
    """Tests ability to handle a mix of personal, factual, and conversational information."""

    def __init__(self):
        super().__init__(
            "Mixed Knowledge",
            "Tests ability to handle a mix of personal, factual, and conversational information",
        )

        # Personal facts
        self.memories.append({
            "text": "My name is Taylor Chen and I live in San Francisco.",
            "type": "personal",
            "timestamp": time.time() - 604800,  # 1 week ago
        })

        self.memories.append({
            "text": "I have a dog named Rusty who is a 3-year-old Golden Retriever.",
            "type": "personal",
            "timestamp": time.time() - 518400,  # 6 days ago
        })

        self.memories.append({
            "text": "I work as a software engineer at TechCorp, specializing in backend development.",
            "type": "personal",
            "timestamp": time.time() - 432000,  # 5 days ago
        })

        # Factual knowledge
        self.memories.append({
            "text": "The capital of France is Paris. Paris is known for the Eiffel Tower and the Louvre Museum.",
            "type": "factual",
            "timestamp": time.time() - 345600,  # 4 days ago
        })

        self.memories.append({
            "text": "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure.",
            "type": "factual",
            "timestamp": time.time() - 259200,  # 3 days ago
        })

        # Conversation bits
        self.memories.append({
            "text": "User: I'm thinking of learning Python or JavaScript. Which would you recommend?\nAssistant: Both are excellent choices. Python is often recommended for beginners due to its readability and simplicity. It's great for data science, AI, automation, and backend development. JavaScript is essential for web development and has expanded to server-side with Node.js. If you're interested in web development, JavaScript might be better. For data science or general programming, Python might be preferable. What's your primary interest?",
            "type": "conversation",
            "timestamp": time.time() - 172800,  # 2 days ago
        })

        self.memories.append({
            "text": "User: I think I'll go with Python since I'm interested in data analysis.\nAssistant: That's a great choice for data analysis! Python has excellent libraries like Pandas, NumPy, and Matplotlib that make data manipulation and visualization straightforward. You might also want to look into Jupyter Notebooks, which are perfect for data exploration. Would you like some resources to help you get started with Python for data analysis?",
            "type": "conversation",
            "timestamp": time.time() - 86400,  # 1 day ago
        })

        # More personal facts mixed in
        self.memories.append({
            "text": "My favorite food is Thai curry, especially Panang curry with tofu.",
            "type": "personal",
            "timestamp": time.time() - 43200,  # 12 hours ago
        })

        self.memories.append({
            "text": "I'm planning a vacation to Hawaii next month, specifically to the islands of Oahu and Maui.",
            "type": "personal",
            "timestamp": time.time() - 21600,  # 6 hours ago
        })

        # Create mixed test queries
        self.queries = [
            "What's my name?",
            "What kind of dog do I have?",
            "What's the capital of France?",
            "What programming language did I decide to learn?",
            "Where am I planning to go on vacation?",
            "What's my favorite food?",
            "What's my job?",
        ]

        # Define expected answers
        self.expected_answers = {
            "What's my name?": ["Taylor", "Chen"],
            "What kind of dog do I have?": ["Rusty", "Golden Retriever"],
            "What's the capital of France?": ["Paris"],
            "What programming language did I decide to learn?": ["Python", "data analysis"],
            "Where am I planning to go on vacation?": ["Hawaii", "Oahu", "Maui"],
            "What's my favorite food?": ["Thai", "curry", "Panang"],
            "What's my job?": ["software engineer", "TechCorp", "backend"],
        }


class LargeContextHandling(BenchmarkScenario):
    """Tests ability to handle and retrieve from large amounts of contextual data."""

    def __init__(self):
        super().__init__(
            "Large Context Handling",
            "Tests ability to handle and retrieve from large amounts of contextual data",
        )

        # Generate several academic article-like entries
        articles = [
            {
                "title": "Quantum Computing: Present and Future",
                "content": """
Quantum computing is an area of computing focused on developing computer technology based on the principles of quantum theory. Quantum computers make use of quantum bits, or qubits, which can represent multiple states simultaneously through superposition.

Unlike classical computers that use bits representing either 0 or 1, quantum computers use qubits that can represent 0, 1, or both simultaneously due to superposition. This allows quantum computers to process a vast number of possibilities at once. Additionally, the principle of quantum entanglement allows qubits that are entangled to be perfectly correlated regardless of distance.

The most common implementations of quantum computers currently use superconducting circuits. These circuits must be cooled to extremely low temperatures, close to absolute zero, to maintain quantum coherence. Other implementations include trapped ions, photonic quantum computers, and topological quantum computers.

One of the most promising applications of quantum computing is in cryptography. Quantum computers could potentially break many of the cryptographic systems in use today, while also enabling new, more secure forms of cryptography. Other applications include optimization problems, drug discovery, and materials science.

Despite significant progress, quantum computing still faces considerable challenges. Quantum decoherence, which causes qubits to lose their quantum properties, is a major obstacle. Error correction is also more difficult in quantum systems than in classical ones. Nevertheless, companies like IBM, Google, and Microsoft are making substantial investments in quantum computing research and development.
                """.strip(),
                "timestamp": time.time() - 86400 * 10,  # 10 days ago
            },
            {
                "title": "Advancements in Renewable Energy Storage",
                "content": """
Energy storage is a critical component for the widespread adoption of renewable energy sources like solar and wind, which are intermittent by nature. As renewable energy becomes more prevalent, the need for effective energy storage solutions grows.

Lithium-ion batteries currently dominate the market for electrical energy storage. They are widely used in electric vehicles and grid-scale storage due to their high energy density and efficiency. However, they face challenges including limited resource availability, high costs, and safety concerns.

Flow batteries are another promising technology, particularly for grid-scale storage. Unlike conventional batteries, flow batteries store energy in external tanks of liquid electrolyte, allowing for easy scaling of energy capacity. Vanadium redox flow batteries are the most developed, but newer chemistries are being explored to reduce costs and improve performance.

Pumped hydro storage remains the most widely deployed grid-scale energy storage technology globally. It works by pumping water to an elevated reservoir when energy is abundant, then releasing it through turbines to generate electricity when needed. While effective, it requires specific geographical features and has environmental impacts.

Compressed air energy storage (CAES) involves compressing air using excess electricity and storing it in underground caverns. When electricity is needed, the compressed air is heated and expanded through a turbine. Advanced adiabatic CAES systems aim to capture and store the heat generated during compression for later use, improving efficiency.

Thermal energy storage systems store excess energy as heat, which can later be converted back to electricity. Molten salt storage, used in concentrated solar power plants, is a prominent example. Other approaches include phase change materials and thermochemical storage.

Hydrogen can serve as a long-term energy storage medium. Excess electricity can produce hydrogen through water electrolysis, which can be stored and later used in fuel cells or combustion processes to generate electricity. While promising for seasonal storage, the overall efficiency of the hydrogen cycle is lower than batteries.

Emerging technologies include gravity-based storage (using excess electricity to lift heavy weights, then generating electricity as they descend), cryogenic energy storage (cooling air to a liquid state, then using its expansion to drive turbines), and advanced flywheel systems.

The future of renewable energy storage will likely involve multiple technologies working together in integrated systems. Local conditions, specific requirements, and available resources will determine the optimal mix of storage solutions for each application.
                """.strip(),
                "timestamp": time.time() - 86400 * 8,  # 8 days ago
            },
            {
                "title": "The Evolution of Artificial Neural Networks",
                "content": """
Artificial neural networks (ANNs) have evolved dramatically since their conceptual origins in the 1940s. The perceptron, developed by Frank Rosenblatt in 1958, was the first implemented neural network model, designed for pattern recognition despite its inability to solve non-linearly separable problems.

The discovery of backpropagation in the 1970s, though not widely recognized until the 1980s, enabled efficient training of multi-layer neural networks. This algorithm calculates gradients to adjust network weights, minimizing the difference between actual and desired outputs. By the 1990s, support vector machines often outperformed neural networks for many machine learning tasks.

The 2000s saw the rise of deep learning with networks containing many layers. Key innovations included better initialization methods, activation functions like ReLU that reduced the vanishing gradient problem, and regularization techniques such as dropout that prevented overfitting. These advances, coupled with increased computational power and data availability, fueled the deep learning revolution.

Convolutional Neural Networks (CNNs) revolutionized computer vision by incorporating the principles of local receptive fields, shared weights, and spatial or temporal subsampling. AlexNet's 2012 ImageNet victory marked a turning point, demonstrating CNNs' superior performance for image classification. Architectures like VGG, Inception, ResNet, and EfficientNet have since pushed performance boundaries while addressing efficiency concerns.

Recurrent Neural Networks (RNNs) process sequential data by maintaining a memory of previous inputs. Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) architectures address the vanishing gradient problem in standard RNNs, enabling learning of long-term dependencies. These networks excel in language modeling, speech recognition, and time series analysis.

Introduced in 2017, the Transformer architecture revolutionized natural language processing by replacing recurrence with attention mechanisms. This innovation led to models like BERT and GPT that achieve state-of-the-art results across numerous language tasks. Transformers efficiently process entire sequences at once, capturing dependencies regardless of distance.

Generative models like Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs) can create new data resembling training examples. These models have produced remarkably realistic images, videos, and text, finding applications in art, design, data augmentation, and simulation.

Recent innovations include graph neural networks for graph-structured data, self-supervised learning reducing dependence on labeled data, neural architecture search automating network design, and neuroevolution using evolutionary algorithms to optimize architectures. Hybrid approaches combining neural networks with symbolic AI are also gaining traction.

Challenges remain, including interpretability issues, vulnerability to adversarial examples, excessive energy consumption, and difficulties in incorporating prior knowledge. Despite these challenges, neural networks continue to advance rapidly, revolutionizing industries from healthcare to transportation and fundamentally changing human-computer interaction.
                """.strip(),
                "timestamp": time.time() - 86400 * 6,  # 6 days ago
            },
            {
                "title": "Climate Change Mitigation Strategies",
                "content": """
Climate change mitigation strategies aim to reduce greenhouse gas emissions and enhance carbon sinks to limit global warming. Transitioning to renewable energy sources is central to these efforts, as electricity and heat production account for approximately 25% of global emissions.

Solar and wind power have seen rapidly declining costs, making them economically competitive with fossil fuels in many regions. Hydroelectric power remains significant, while emerging technologies like tidal, geothermal, and advanced bioenergy offer additional renewable options. Nuclear power, though controversial, provides low-carbon electricity and is included in many countries' mitigation plans.

Energy efficiency improvements across buildings, industry, and transportation can substantially reduce emissions while often providing economic benefits. In buildings, better insulation, efficient lighting, and smart systems can cut energy use by 50-90%. Industrial efficiency measures include waste heat recovery, improved process design, and materials recycling. Transportation efficiency improvements involve vehicle electrification, aerodynamic design, lightweight materials, and modal shifts to public transit, cycling, and walking.

Carbon capture and storage (CCS) technologies prevent CO2 emissions from reaching the atmosphere. Post-combustion capture removes CO2 from flue gases, while pre-combustion capture converts fuel to hydrogen and CO2 before combustion. Oxy-fuel combustion uses pure oxygen for combustion, producing a more concentrated CO2 stream. Though technically viable, CCS remains costly and faces deployment challenges.

Changing agricultural practices can reduce emissions and increase carbon sequestration. Conservation agriculture preserves soil carbon, while precision agriculture minimizes fertilizer use. Improved livestock management, including alternative feeds and better manure management, can reduce methane emissions. Agroforestry integrates trees into agricultural landscapes, enhancing carbon storage.

Reforestation and afforestation increase carbon storage in biomass and soils. Protecting existing forests, especially primary forests, prevents carbon releases and preserves biodiversity. Urban forestry and green infrastructure offer additional carbon storage while providing cooling, flood mitigation, and recreational benefits.

Behavioral changes can significantly reduce individual carbon footprints. Dietary shifts toward plant-based foods lower agricultural emissions, while reducing waste cuts emissions across the supply chain. Transportation choices like using public transit, cycling, carpooling, and avoiding air travel can substantially reduce personal emissions.

Carbon dioxide removal (CDR) technologies may be necessary to achieve climate goals. Bioenergy with carbon capture and storage (BECCS) combines biomass energy with CCS. Direct air capture removes CO2 directly from the atmosphere. Enhanced weathering accelerates natural CO2-absorbing processes, while ocean fertilization enhances marine carbon uptake. Most CDR approaches remain speculative or unproven at scale.

Policy frameworks are essential for implementing mitigation strategies. Carbon pricing through taxes or cap-and-trade systems internalizes the costs of emissions. Regulations can enforce emission standards and technology adoption. Subsidies for clean energy and energy efficiency can accelerate transitions. International agreements like the Paris Agreement coordinate global efforts and provide support for developing countries.

Successful climate change mitigation requires an integrated approach combining technological, social, economic, and policy measures. The urgency of the climate crisis necessitates simultaneous deployment of multiple strategies to rapidly reduce emissions while developing negative emissions technologies for the future.
                """.strip(),
                "timestamp": time.time() - 86400 * 4,  # 4 days ago
            },
            {
                "title": "Advances in Genomic Medicine",
                "content": """
Genomic medicine applies genomic information to medical diagnosis and treatment, transforming healthcare through personalized approaches. The Human Genome Project, completed in 2003, laid the foundation by sequencing the first human genome at a cost of approximately $3 billion. Today, whole genome sequencing costs under $1,000 and can be completed in days rather than years.

Next-generation sequencing (NGS) technologies have revolutionized genomic data collection, enabling large-scale studies like the UK Biobank, which contains genetic information from 500,000 individuals. These technologies include short-read sequencing, which provides high accuracy but struggles with repetitive regions, and long-read sequencing, which better captures structural variations but typically has higher error rates.

Pharmacogenomics examines how genetic variations affect drug responses, enabling doctors to select medications and dosages based on a patient's genetic profile. For example, variations in the CYP2C19 gene affect metabolism of many common drugs, including some antidepressants and proton pump inhibitors. Genetic testing before prescribing certain medications can prevent adverse reactions and treatment failures.

Cancer genomics has transformed oncology by identifying driver mutations that promote tumor growth. Targeted therapies like trastuzumab for HER2-positive breast cancer and imatinib for chronic myeloid leukemia specifically address cancer's genetic underpinnings. Liquid biopsies detect circulating tumor DNA in blood samples, allowing non-invasive monitoring of cancer evolution and treatment response.

Rare disease diagnosis has been revolutionized by genomic techniques. While individual rare diseases affect small populations, collectively they affect approximately 300 million people worldwide. Whole exome or genome sequencing can identify causative mutations after years of diagnostic odysseys, ending uncertainty for patients and families and sometimes revealing treatment options.

Gene therapy, once considered futuristic, now offers approved treatments for several conditions. Approaches include gene addition (introducing functional copies of genes), gene silencing (reducing expression of harmful genes), and gene editing (modifying DNA directly). CRISPR-Cas9 technology has dramatically simplified gene editing, though challenges in delivery, specificity, and safety remain.

Predictive genomics assesses disease risk based on genetic profiles. Polygenic risk scores combine effects of many genetic variants to estimate risk for complex conditions like coronary artery disease, type 2 diabetes, and breast cancer. While not deterministic, these scores can identify individuals who might benefit from enhanced screening or preventive interventions.

Non-coding regions, which comprise about 98% of the genome, regulate gene expression and contribute significantly to disease risk. Initiatives like ENCODE (Encyclopedia of DNA Elements) are mapping these regulatory regions to understand their function. Studies increasingly show that variations in non-coding regions contribute to many common diseases.

Epigenetics studies heritable changes in gene expression that don't involve DNA sequence alterations. DNA methylation, histone modifications, and chromatin remodeling affect how genes are expressed. Environmental factors like diet, stress, and chemical exposures can influence epigenetic patterns, potentially affecting disease risk. Epigenetic therapies are being developed for cancer and other conditions.

Ethical and social considerations remain crucial as genomic medicine advances. Privacy concerns arise from the inherently identifiable nature of genomic data. Equity issues include ensuring diverse population representation in genomic research and equal access to genomic technologies. Incidental findings—potentially significant health information unrelated to the original reason for testing—raise complex disclosure questions. Addressing these challenges while advancing the field will be essential for realizing genomic medicine's full potential.
                """.strip(),
                "timestamp": time.time() - 86400 * 2,  # 2 days ago
            },
        ]

        # Create memories from these articles (will be many when chunked)
        for article in articles:
            # Create a memory for the title
            self.memories.append({
                "text": f"Article title: {article['title']}",
                "type": "article_title",
                "timestamp": article["timestamp"],
            })

            # Chunk the content into smaller pieces
            chunks = self._chunk_text(article["content"], 250, 50)
            for i, chunk in enumerate(chunks):
                self.memories.append({
                    "text": chunk,
                    "type": "article_content",
                    "article_title": article["title"],
                    "chunk_index": i,
                    "timestamp": article["timestamp"] + i * 60,  # Space out chunks by a minute
                })

        # Personal information interleaved with articles
        personal_info = [
            {
                "text": "My name is Riley Johnson and I'm a graduate student in Computer Science.",
                "type": "personal",
                "timestamp": time.time() - 86400 * 9,
            },
            {
                "text": "I'm interested in quantum computing and machine learning.",
                "type": "personal",
                "timestamp": time.time() - 86400 * 7,
            },
            {
                "text": "I'm working on a research project about neural network optimization.",
                "type": "personal",
                "timestamp": time.time() - 86400 * 5,
            },
            {
                "text": "I'm considering pursuing a PhD after I graduate next year.",
                "type": "personal",
                "timestamp": time.time() - 86400 * 3,
            },
            {
                "text": "I attended a conference on climate change last weekend in Boston.",
                "type": "personal",
                "timestamp": time.time() - 86400,
            },
        ]

        # Add personal info to memories
        self.memories.extend(personal_info)

        # Create test queries about specific pieces of information
        self.queries = [
            "What are the main applications of quantum computing?",
            "What is my name and what am I studying?",
            "What are some renewable energy storage technologies?",
            "What conference did I attend recently and where was it?",
            "How have neural networks evolved over time?",
            "What are some strategies for mitigating climate change?",
            "What is pharmacogenomics?",
            "What research topic am I working on?",
            "When did the Human Genome Project complete?",
        ]

        # Define expected answers
        self.expected_answers = {
            "What are the main applications of quantum computing?": [
                "cryptography",
                "optimization",
                "drug discovery",
                "materials science",
            ],
            "What is my name and what am I studying?": [
                "Riley Johnson",
                "Computer Science",
                "graduate student",
            ],
            "What are some renewable energy storage technologies?": [
                "batteries",
                "lithium-ion",
                "flow batteries",
                "pumped hydro",
                "compressed air",
                "thermal",
                "hydrogen",
            ],
            "What conference did I attend recently and where was it?": [
                "climate change",
                "Boston",
                "last weekend",
            ],
            "How have neural networks evolved over time?": [
                "perceptron",
                "backpropagation",
                "deep learning",
                "CNN",
                "RNN",
                "Transformer",
            ],
            "What are some strategies for mitigating climate change?": [
                "renewable energy",
                "energy efficiency",
                "carbon capture",
                "agricultural practices",
                "reforestation",
            ],
            "What is pharmacogenomics?": [
                "genetic variations",
                "drug responses",
                "medications",
                "dosages",
                "genetic profile",
            ],
            "What research topic am I working on?": ["neural network", "optimization"],
            "When did the Human Genome Project complete?": ["2003"],
        }

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        """Split text into chunks with overlap."""
        chunks = []
        paragraphs = text.split("\n\n")
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding this paragraph would exceed the chunk size
            if len(current_chunk) > 0 and len(current_chunk) + len(para) > chunk_size:
                chunks.append(current_chunk)
                # Keep some overlap for context
                overlap_text = " ".join(current_chunk.split(" ")[-overlap:])
                current_chunk = overlap_text + " " + para
            else:
                if current_chunk:
                    current_chunk += " " + para
                else:
                    current_chunk = para

        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)

        return chunks


class BenchmarkRunner:
    """Main class for running memory and retrieval benchmarks."""

    def _create_chunked_memoryweave(
        self,
        model_name: str | None = None,
        embedding_model_name: str | None = None,
        debug: bool = False,
    ) -> ChunkedMemoryWeaveAPI:
        """Create a MemoryWeave system with chunking support."""
        try:
            # Patch ComponentName if needed
            from memoryweave.interfaces.pipeline import ComponentType

            if not hasattr(ComponentType, "TEXT_CHUNKER"):
                # Add TEXT_CHUNKER to ComponentType enum dynamically
                ComponentType.TEXT_CHUNKER = "text_chunker"

            # Create the system
            system = ChunkedMemoryWeaveAPI(
                model_name=self.model_name,
                embedding_model_name=self.embedding_model
                or "sentence-transformers/all-MiniLM-L6-v2",
                debug=self.debug,
            )
            return system
        except Exception as e:
            console.print(f"Error initializing ChunkedMemoryWeaveAPI: {e}")
            if self.debug:
                import traceback

                console.print(traceback.format_exc())
            return None

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        embedding_model: str = None,
        systems_to_test: list[SystemType] = None,
        scenarios_to_run: list[str] = None,
        output_dir: str = "./benchmark_results",
        max_memories_per_scenario: int = 500,
        debug: bool = False,
    ):
        """Initialize the benchmark runner."""
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.output_dir = output_dir
        self.max_memories_per_scenario = max_memories_per_scenario
        self.debug = debug

        # set default systems to test if not provided
        if systems_to_test is None:
            self.systems_to_test = [
                SystemType.MEMORYWEAVE_CHUNKED,
                SystemType.MEMORYWEAVE_STANDARD,
                SystemType.STANDARD_RAG,
                SystemType.RECENCY_BIASED,
                # SystemType.CONTEXT_WINDOW,  # Optional, can be expensive/slow
            ]
        else:
            self.systems_to_test = systems_to_test

        # set up scenarios
        self.all_scenarios = {
            "document": DocumentRetrieval(),
            "conversation": ConversationalMemory(),
            "temporal": TemporalReferences(),
            "mixed": MixedKnowledge(),
            "large_context": LargeContextHandling(),
        }

        # Filter scenarios based on input
        if scenarios_to_run:
            self.scenarios = {k: v for k, v in self.all_scenarios.items() if k in scenarios_to_run}
        else:
            self.scenarios = self.all_scenarios

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize systems (lazy-loaded when needed)
        self.systems = {}

        # Results storage
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "systems_tested": [s.value for s in self.systems_to_test],
            "scenarios_run": list(self.scenarios.keys()),
            "scenario_results": {},
            "system_metrics": {},
        }

        # Configure logging based on debug setting
        if debug:
            logger.setLevel(logging.DEBUG)

        # Track memory usage
        self.memory_tracker = {"peak_usage": {}}

    def initialize_systems(self):
        """Initialize memory systems to be benchmarked."""
        console.print("[bold cyan]Initializing Memory Systems[/bold cyan]")

        # Make sure embedding_model has a default value
        if self.embedding_model is None:
            self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
            console.print(
                f"No embedding model specified, using default: {self.embedding_model}",
                style="bold yellow",
            )

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[cyan]{task.completed}/{task.total}"),
            TimeElapsedColumn(),
        ) as progress:
            init_task = progress.add_task(
                "Initializing systems...", total=len(self.systems_to_test)
            )

            for system_type in self.systems_to_test:
                try:
                    if system_type == SystemType.MEMORYWEAVE_CHUNKED:
                        self.systems[system_type] = self._create_chunked_memoryweave(
                            model_name=self.model_name,
                            embedding_model_name=self.embedding_model,
                            debug=self.debug,
                        )
                        console.print("  [green]✓[/green] MemoryWeave (Chunked) initialized")

                    elif system_type == SystemType.MEMORYWEAVE_STANDARD:
                        self.systems[system_type] = MemoryWeaveAPI(
                            model_name=self.model_name,
                            embedding_model_name=self.embedding_model,
                            debug=self.debug,
                        )
                        console.print("  [green]✓[/green] MemoryWeave (Standard) initialized")

                    elif system_type == SystemType.STANDARD_RAG:
                        # Initialize standard RAG system (simplified implementation)
                        self.systems[system_type] = self._create_standard_rag()
                        console.print("  [green]✓[/green] Standard RAG initialized")

                    elif system_type == SystemType.RECENCY_BIASED:
                        # Initialize recency-biased system (simplified implementation)
                        self.systems[system_type] = self._create_recency_biased()
                        console.print("  [green]✓[/green] Recency-biased system initialized")

                    elif system_type == SystemType.CONTEXT_WINDOW:
                        # Initialize context window system (simplified implementation)
                        self.systems[system_type] = self._create_context_window()
                        console.print("  [green]✓[/green] Context window system initialized")

                    # Measure baseline memory usage after system initialization
                    self.memory_tracker["peak_usage"][system_type] = self._get_memory_usage()

                except Exception as e:
                    console.print(
                        f"  [red]✗[/red] Failed to initialize {system_type.value}: {str(e)}"
                    )
                    if self.debug:
                        console.print(traceback.format_exc())

                progress.advance(init_task)

        console.print()

    def _create_standard_rag(self):
        """Create a standard RAG system for comparison."""
        # This is a simplified implementation that uses MemoryWeave's API but configures
        # it to behave like a standard RAG system (similarity only, no advanced features)
        system = MemoryWeaveAPI(
            model_name=self.model_name,
            embedding_model_name=self.embedding_model,
            enable_category_management=False,
            enable_personal_attributes=False,
            enable_semantic_coherence=False,
            enable_dynamic_thresholds=False,
            debug=self.debug,
        )

        # Configure the system to use only similarity retrieval
        system.strategy.initialize({
            "confidence_threshold": 0.1,
            "similarity_weight": 1.0,  # Only use similarity
            "associative_weight": 0.0,  # Disable associative retrieval
            "temporal_weight": 0.0,  # Disable temporal relevance
            "activation_weight": 0.0,  # Disable activation boosting
        })

        return system

    def _create_recency_biased(self):
        """Create a recency-biased system that prioritizes recent memories."""
        system = MemoryWeaveAPI(
            model_name=self.model_name,
            embedding_model_name=self.embedding_model,
            enable_category_management=False,
            enable_personal_attributes=False,
            enable_semantic_coherence=False,
            debug=self.debug,
        )

        # Configure the system to prioritize recency
        system.strategy.initialize({
            "confidence_threshold": 0.1,
            "similarity_weight": 0.5,  # Some weight on similarity
            "associative_weight": 0.0,  # Disable associative retrieval
            "temporal_weight": 0.5,  # Heavy weight on temporal relevance
            "activation_weight": 0.0,  # Disable activation patterns
        })

        return system

    def _create_context_window(self):
        """Create a simple context window system."""
        # This will just be a wrapper around the model with a simple context window
        # that doesn't use any fancy retrieval techniques
        return MemoryWeaveAPI(
            model_name=self.model_name,
            embedding_model_name=self.embedding_model,
            debug=self.debug,
        )

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)  # Convert to MB

    def run_benchmarks(self):
        """Run all benchmarks."""
        # Initialize all systems
        self.initialize_systems()

        # Track overall metrics
        system_metrics = {
            system_type.value: {"avg_time": 0, "avg_accuracy": 0, "memory_mb": 0}
            for system_type in self.systems_to_test
        }

        # Run benchmarks for each scenario
        for scenario_name, scenario in self.scenarios.items():
            console.print(f"\n[bold cyan]Running benchmark: {scenario.name}[/bold cyan]")
            console.print(f"[dim]{scenario.description}[/dim]\n")

            # Get memories for this scenario (with limit)
            memories = scenario.get_memories()[: self.max_memories_per_scenario]
            queries = scenario.get_queries()

            # Display scenario stats
            console.print(f"[bold]Memory count:[/bold] {len(memories)}")
            console.print(f"[bold]Query count:[/bold] {len(queries)}")

            # Run benchmark for each system
            scenario_results = {}

            for system_type in self.systems_to_test:
                if system_type not in self.systems:
                    console.print(
                        f"[yellow]Skipping {system_type.value} (not initialized)[/yellow]"
                    )
                    continue

                console.print(f"\n[bold]Testing: {system_type.value}[/bold]")

                # Reset the system for a fresh start
                system = self.systems[system_type]

                # Use memory batching for efficiency
                batch_size = 50
                memory_count = len(memories)
                batch_count = (memory_count + batch_size - 1) // batch_size

                # Add memories in batches with progress tracking
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(),
                    TimeElapsedColumn(),
                ) as progress:
                    memory_task = progress.add_task(
                        f"Adding {memory_count} memories to {system_type.value}...",
                        total=memory_count,
                    )

                    for i in range(0, memory_count, batch_size):
                        batch = memories[i : i + batch_size]
                        for memory in batch:
                            text = memory.get("text", "")
                            metadata = {k: v for k, v in memory.items() if k != "text"}
                            system.add_memory(text, metadata)
                            progress.advance(memory_task)

                # Run queries and measure performance
                console.print(f"\nRunning {len(queries)} queries on {system_type.value}...")

                query_results = []
                query_times = []
                accuracy_scores = []

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(),
                    TimeElapsedColumn(),
                ) as progress:
                    query_task = progress.add_task(
                        f"Processing queries on {system_type.value}...", total=len(queries)
                    )

                    for query in queries:
                        # Measure query time
                        start_time = time.time()

                        # Get response
                        response = system.chat(query, max_new_tokens=150)

                        # Calculate elapsed time
                        elapsed = time.time() - start_time
                        query_times.append(elapsed)

                        # Calculate accuracy based on expected answers
                        expected_answers = scenario.get_expected_answers(query)
                        accuracy = self._calculate_accuracy(response, expected_answers)
                        accuracy_scores.append(accuracy)

                        # Store result
                        query_results.append({
                            "query": query,
                            "response": response,
                            "time_seconds": elapsed,
                            "accuracy": accuracy,
                            "expected": expected_answers,
                        })

                        progress.advance(query_task)

                # Calculate peak memory usage after running this system
                peak_memory = self._get_memory_usage()
                memory_increase = peak_memory - self.memory_tracker["peak_usage"].get(
                    system_type, 0
                )

                # Calculate summary metrics
                avg_time = sum(query_times) / len(query_times) if query_times else 0
                avg_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0

                # Update system metrics
                system_metrics[system_type.value]["avg_time"] += avg_time / len(self.scenarios)
                system_metrics[system_type.value]["avg_accuracy"] += avg_accuracy / len(
                    self.scenarios
                )
                system_metrics[system_type.value]["memory_mb"] = max(
                    system_metrics[system_type.value]["memory_mb"], memory_increase
                )

                # Display summary for this system
                console.print(f"\n[bold]Results for {system_type.value}:[/bold]")
                console.print(f"  Average query time: [cyan]{avg_time:.3f}s[/cyan]")
                console.print(f"  Average accuracy: [cyan]{avg_accuracy:.2f}[/cyan]")
                console.print(f"  Memory usage: [cyan]{memory_increase:.2f} MB[/cyan]")

                # Store result
                scenario_results[system_type.value] = {
                    "queries": query_results,
                    "avg_time": avg_time,
                    "avg_accuracy": avg_accuracy,
                    "memory_usage_mb": memory_increase,
                }

            # Save scenario results
            self.results["scenario_results"][scenario_name] = scenario_results

        # Save overall system metrics
        self.results["system_metrics"] = system_metrics

        # Display overall results
        self.display_results()

        # Save results to file
        self.save_results()

    def _calculate_accuracy(self, response: str, expected_answers: list[str]) -> float:
        """Calculate accuracy score based on presence of expected answers in response."""
        if not expected_answers:
            return 0.0

        response_lower = response.lower()
        found_count = sum(1 for ans in expected_answers if ans.lower() in response_lower)

        # Partial credit: divide found by total expected
        return found_count / len(expected_answers)

    def display_results(self):
        """Display benchmark results in a nice table format."""
        console.print("\n[bold cyan]Overall Benchmark Results[/bold cyan]")

        # Create a table for overall system comparison
        table = Table(title="System Performance Comparison")

        # Add columns
        table.add_column("System", style="cyan")
        table.add_column("Avg Time (s)", style="yellow")
        table.add_column("Avg Accuracy", style="green")
        table.add_column("Memory Usage (MB)", style="magenta")

        # Add rows for each system
        for system_type in self.systems_to_test:
            system_name = system_type.value
            metrics = self.results["system_metrics"].get(system_name, {})

            if metrics:
                table.add_row(
                    system_name,
                    f"{metrics['avg_time']:.3f}",
                    f"{metrics['avg_accuracy']:.2f}",
                    f"{metrics['memory_mb']:.2f}",
                )

        console.print(table)

        # Create comparison charts
        self._create_comparative_charts()

    def _create_comparative_charts(self):
        """Create comparative visualizations of the results."""
        try:
            # Check if we have valid results to plot
            if not self.results["system_metrics"]:
                console.print("[yellow]No metrics available for visualization[/yellow]")
                return

            # Extract data for plotting
            systems = []
            times = []
            accuracies = []
            memories = []

            for system, metrics in self.results["system_metrics"].items():
                systems.append(system)
                times.append(metrics["avg_time"])
                accuracies.append(metrics["avg_accuracy"])
                memories.append(metrics["memory_mb"])

            # set up figure with 3 subplots
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

            # Plot response times
            ax1.bar(systems, times, color="skyblue")
            ax1.set_title("Average Response Time")
            ax1.set_ylabel("Time (seconds)")
            ax1.tick_params(axis="x", rotation=45)

            # Plot accuracy scores
            ax2.bar(systems, accuracies, color="lightgreen")
            ax2.set_title("Average Accuracy")
            ax2.set_ylabel("Accuracy Score")
            ax2.tick_params(axis="x", rotation=45)

            # Plot memory usage
            ax3.bar(systems, memories, color="salmon")
            ax3.set_title("Memory Usage")
            ax3.set_ylabel("Memory (MB)")
            ax3.tick_params(axis="x", rotation=45)

            # Adjust layout and save figure
            plt.tight_layout()

            # Save the chart
            chart_path = os.path.join(self.output_dir, "performance_comparison.png")
            plt.savefig(chart_path)

            console.print(
                f"\n[bold green]Performance comparison chart saved to:[/bold green] {chart_path}"
            )

            # Optionally display the plot (commented out for non-interactive runs)
            # plt.show()

        except Exception as e:
            console.print(f"[yellow]Failed to create charts: {str(e)}[/yellow]")
            if self.debug:
                console.print(traceback.format_exc())

    def save_results(self):
        """Save benchmark results to a JSON file."""
        # Create a filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"benchmark_results_{timestamp}.json")

        try:
            # Write results to file
            with open(filename, "w") as f:
                json.dump(self.results, f, indent=2)

            console.print(f"\n[bold green]Results saved to:[/bold green] {filename}")

        except Exception as e:
            console.print(f"[yellow]Failed to save results: {str(e)}[/yellow]")
            if self.debug:
                console.print(traceback.format_exc())


@click.command()
@click.option(
    "--model",
    default=DEFAULT_MODEL,
    show_default=True,
    help=f"Name of the Hugging Face model to load (default: {DEFAULT_MODEL})",
)
@click.option(
    "--embedding-model",
    default=None,
    help="Name of the embedding model to use (default: use MemoryWeave's default)",
)
@click.option(
    "--systems",
    multiple=True,
    type=click.Choice([s.value for s in SystemType]),
    default=["memoryweave_chunked"],
    show_default=True,
    help="Specific systems to test (can specify multiple). Default: test all systems.",
)
@click.option(
    "--scenarios",
    multiple=True,
    type=click.Choice(["document", "conversation", "temporal", "mixed", "large_context"]),
    default=["mixed"],
    show_default=True,
    help="Specific scenarios to run (can specify multiple). Default: run all scenarios.",
)
@click.option(
    "--output-dir",
    default="./benchmark_results",
    help="Directory to save benchmark results (default: ./benchmark_results)",
)
@click.option(
    "--max-memories",
    default=500,
    help="Maximum number of memories to use per scenario (default: 500)",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging for more detailed output.",
)
def main(model, embedding_model, systems, scenarios, output_dir, max_memories, debug):
    """
    RAM Benchmark: Retrieval Augmented Memory Benchmarking Tool

    This tool benchmarks MemoryWeave against other memory/retrieval systems across
    multiple scenarios to evaluate performance, accuracy, and resource usage.
    """
    # Configure logging based on debug flag
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        console.print("[yellow]Debug logging enabled[/yellow]")

    # Print banner
    console.print(
        Panel.fit(
            "[bold cyan]RAM Benchmark: Retrieval Augmented Memory[/bold cyan]\n\n"
            f"Model: [yellow]{model}[/yellow]\n"
            f"Output directory: [yellow]{output_dir}[/yellow]\n"
            "This benchmark compares different memory and retrieval approaches.",
            border_style="cyan",
        )
    )

    # Convert systems to enum types
    systems_to_test = [SystemType(s) for s in systems] if systems else None

    # Initialize and run benchmark
    try:
        benchmark = BenchmarkRunner(
            model_name=model,
            embedding_model=embedding_model,
            systems_to_test=systems_to_test,
            scenarios_to_run=scenarios,
            output_dir=output_dir,
            max_memories_per_scenario=max_memories,
            debug=debug,
        )

        benchmark.run_benchmarks()

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Benchmark interrupted by user[/bold yellow]")
    except Exception as e:
        console.print(f"[bold red]Error running benchmark: {str(e)}[/bold red]")
        if debug:
            console.print(traceback.format_exc())


if __name__ == "__main__":
    main()
