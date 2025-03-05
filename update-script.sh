#!/bin/bash

# Update Memory class imports
echo "Updating Memory class imports..."
find . -type f -name "*.py" -not -path "./.venv/*" -exec sed -i '' 's/from memoryweave.storage.memory_store import Memory/from memoryweave.interfaces.memory import Memory/g' {} \;

# Update MemoryStore imports
echo "Updating MemoryStore imports..."
find . -type f -name "*.py" -not -path "./.venv/*" -exec sed -i '' 's/from memoryweave.storage.memory_store import MemoryStore/from memoryweave.storage.refactored import StandardMemoryStore/g' {} \;

# Update both combined imports
echo "Updating combined imports..."
find . -type f -name "*.py" -not -path "./.venv/*" -exec sed -i '' 's/from memoryweave.storage.memory_store import Memory, MemoryStore/from memoryweave.interfaces.memory import Memory\nfrom memoryweave.storage.refactored import StandardMemoryStore/g' {} \;

# Update initialization code
echo "Updating initialization code..."
find . -type f -name "*.py" -not -path "./.venv/*" -exec sed -i '' 's/memory_store = MemoryStore()/memory_store = StandardMemoryStore()\nmemory_adapter = MemoryAdapter(memory_store)/g' {} \;

# Update chunked memory store imports
echo "Updating chunked memory store imports..."
find . -type f -name "*.py" -not -path "./.venv/*" -exec sed -i '' 's/from memoryweave.storage.chunked_memory_store import ChunkedMemoryStore/from memoryweave.storage.refactored import ChunkedMemoryStore/g' {} \;

# Update hybrid memory store imports
echo "Updating hybrid memory store imports..."
find . -type f -name "*.py" -not -path "./.venv/*" -exec sed -i '' 's/from memoryweave.storage.hybrid_memory_store import HybridMemoryStore/from memoryweave.storage.refactored import HybridMemoryStore/g' {} \;

# Add memory adapter import where it might be needed
echo "Adding memory adapter import where it might be needed..."
find . -type f -name "*.py" -not -path "./.venv/*" -exec grep -l "StandardMemoryStore" {} \; | xargs -I {} sed -i '' '/from memoryweave.storage.refactored import StandardMemoryStore/s/StandardMemoryStore/StandardMemoryStore, MemoryAdapter/g' {}

echo "Memory class imports updated!"
echo "Import updates completed. Please review the changes!"