<!-- classifier/README.md or README.md -->
# Hierarchical Text Classification System

A modular, extensible system for hierarchical text classification using large language models. Features a capability-based architecture that enables complex analysis workflows including classification, sentiment analysis, recommendation detection, and temporal trend analysis.

## Features

- **Hierarchical Classification**: BFS and bundled classification strategies
- **Modular Capabilities**: Plugin architecture for adding new analysis features
- **VLLM Server Integration**: Scalable inference with remote VLLM servers
- **Stem Analysis**: Topic-specific sentiment, recommendations, and trend detection
- **Dependency Management**: Automatic capability dependency resolution
- **Batch Processing**: Efficient processing with progress tracking

## Quick Start

### Installation
```bash
# Clone the repository
cd new_oss_classifier

# Install dependencies
uv pip install -e .
```

### Start VLLM Server
```bash
# Start server on default port (9005) with GPUs 7,6
python classifier/server/vllm_server.py start \
    --port 9005 \
    --gpu-ids 7,6

# Or with custom configuration
python classifier/server/vllm_server.py start \
    --instance-id 1 \
    --port 9006 \
    --gpu-ids 5,4 \
    --model openai/gpt-oss-120b

# Check server status
python classifier/server/vllm_server.py status --port 9005
```

### Run Classification
```bash
# Basic hierarchical classification
uv run classifier/cli/batch_classify.py \
    --config topics.json \
    --input-file data.csv \
    --input-column "text" \
    --save-path results.json \
    -v

# With stem analysis capabilities
uv run classifier/cli/batch_classify.py \
    --config topics.json \
    --input-file data.csv \
    --input-column "text" \
    --enable-stem-polarity \
    --enable-stem-recommendations \
    --enable-stem-trends \
    --save-path results.json \
    -v

# With bundled classification strategy
uv run classifier/cli/batch_classify.py \
    --config topics.json \
    --input-file data.csv \
    --input-column "text" \
    --classification-strategy bundled \
    --bundle-size 4 \
    --save-path results.json \
    -v
```

### Stop Server
```bash
# Stop server on specific port
python classifier/server/vllm_server.py stop --port 9005

# Or by instance ID
python classifier/server/vllm_server.py stop --instance-id 0
```

## Architecture

### Core Components
```
classifier/
├── capabilities/          # Capability implementations
│   ├── base.py           # Base capability interface
│   ├── classification/   # Classification strategies
│   ├── recommendations/  # Recommendation detection
│   ├── alerts/          # Alert detection
│   ├── trend/           # Global trend detection
│   ├── stem/            # Stem-level analysis
│   └── registry.py      # Capability registry
├── core/                # Core utilities
│   ├── hierarchy.py     # Hierarchy management
│   └── policies.py      # Acceptance policies
├── orchestration/       # Capability orchestration
├── server/             # VLLM server integration
└── cli/                # Command-line interface
```

### Capability System

Capabilities are self-contained analysis modules that:
- Declare dependencies on other capabilities
- Define input/output schemas with Pydantic
- Generate prompts for LLM processing
- Post-process and format results

## Adding a New Capability

### Example: Sentiment Capability

#### 1. Create Directory Structure
```bash
mkdir -p classifier/capabilities/sentiment
touch classifier/capabilities/sentiment/{__init__.py,capability.py,models.py,prompts.py}
```

#### 2. Define Pydantic Models
```python
# classifier/capabilities/sentiment/models.py
from typing import Literal
from pydantic import BaseModel

class SentimentOutput(BaseModel):
    """Output schema for sentiment analysis."""
    has_sentiment: bool
    sentiment: Literal["positive", "negative", "neutral", "mixed"] = None
    confidence: int = 0  # 1-5
    reasoning: str = ""
```

#### 3. Create Prompt Template
```python
# classifier/capabilities/sentiment/prompts.py
def sentiment_prompt(text: str) -> str:
    """Generate sentiment analysis prompt."""
    return f"""
Analyze the sentiment of the following text.

**Text**: {text}

Determine the overall sentiment as positive, negative, neutral, or mixed.

Return a JSON object with:
- **has_sentiment**: Boolean
- **sentiment**: One of "positive", "negative", "neutral", "mixed"
- **confidence**: Integer 1-5
- **reasoning**: One sentence explanation

Example:
{{
  "has_sentiment": true,
  "sentiment": "positive",
  "confidence": 5,
  "reasoning": "The text expresses clear satisfaction and praise."
}}
"""
```

#### 4. Implement Capability Class
```python
# classifier/capabilities/sentiment/capability.py
from typing import Any, Dict, Type
from pydantic import BaseModel
from ..base import Capability
from .models import SentimentOutput
from .prompts import sentiment_prompt

class SentimentCapability(Capability):
    """Detects sentiment in text."""
    
    @property
    def name(self) -> str:
        return "sentiment"
    
    @property
    def schema(self) -> Type[BaseModel]:
        return SentimentOutput
    
    def create_prompt(self, text: str, context: Dict[str, Any] = None) -> str:
        return sentiment_prompt(text)
    
    def format_for_export(self, result: Any) -> Any:
        """Format result for JSON export."""
        if result is None:
            return {"has_sentiment": False}
        
        if hasattr(result, "model_dump"):
            return result.model_dump()
        return result
```

#### 5. Export from Package
```python
# classifier/capabilities/sentiment/__init__.py
from .capability import SentimentCapability
from .models import SentimentOutput
from .prompts import sentiment_prompt

__all__ = [
    "SentimentCapability",
    "SentimentOutput",
    "sentiment_prompt",
]
```

#### 6. Register Capability
```python
# In classifier/capabilities/__init__.py, add:
from .sentiment import SentimentCapability, SentimentOutput

# Add to __all__:
__all__ = [
    # ... existing exports ...
    "SentimentCapability",
    "SentimentOutput",
]

# In classifier/capabilities/registry.py, update create_default_registry():
def create_default_registry() -> CapabilityRegistry:
    from .sentiment import SentimentCapability
    
    registry = CapabilityRegistry()
    # ... existing registrations ...
    registry.register(SentimentCapability())
    return registry
```

#### 7. Add CLI Flag
```python
# In classifier/cli/batch_classify.py, add:
@click.option(
    "--enable-sentiment",
    is_flag=True,
    help="Enable sentiment analysis",
)
def main(..., enable_sentiment: bool, ...):
    # In capability_names list:
    if enable_sentiment:
        capability_names.append("sentiment")
```

#### 8. Use the Capability
```bash
uv run classifier/cli/batch_classify.py \
    --config topics.json \
    --input-file data.csv \
    --input-column "text" \
    --enable-sentiment \
    --save-path results.json \
    -v
```

## Adding a Classification Strategy

### Example: DFS Classification

#### 1. Create Strategy File
```python
# classifier/capabilities/classification/dfs.py
from typing import Any, Callable, Dict, List, Type
from pydantic import BaseModel
from ...core import AcceptancePolicy, DefaultPolicy
from .base import ClassificationCapability
from .models import ClassificationOutput, SingleClassificationResult
from .prompts import standard_classification_prompt

class DFSClassificationCapability(ClassificationCapability):
    """Depth-first hierarchical classification."""
    
    def __init__(
        self,
        prompt_fn: Callable[[Dict[str, Any]], str] = None,
        policy: AcceptancePolicy = None,
        separator: str = ">",
    ):
        self.prompt_fn = prompt_fn or standard_classification_prompt
        self.policy = policy or DefaultPolicy()
        self.separator = separator
    
    @property
    def name(self) -> str:
        return "classification"
    
    @property
    def schema(self) -> Type[BaseModel]:
        return SingleClassificationResult
    
    def execute_classification(
        self,
        texts: List[str],
        hierarchy: Dict[str, Any],
        processor: Any,
    ) -> Dict[str, ClassificationOutput]:
        """
        Execute DFS hierarchical classification.
        
        Explores one branch completely before moving to the next.
        """
        root_name = hierarchy.get("name", "[ROOT]")
        
        # Initialize results
        final_results: Dict[str, ClassificationOutput] = {
            text: ClassificationOutput(
                text=text,
                classification_paths=[],
                node_results={}
            )
            for text in texts
        }
        
        # DFS traversal for each text
        for text in texts:
            self._dfs_traverse(
                text=text,
                node=hierarchy,
                parent_path=root_name,
                result=final_results[text],
                processor=processor,
            )
        
        return final_results
    
    def _dfs_traverse(
        self,
        text: str,
        node: Dict[str, Any],
        parent_path: str,
        result: ClassificationOutput,
        processor: Any,
    ):
        """Recursively traverse tree depth-first."""
        for child in node.get("children", []):
            # Generate prompt and classify
            prompt = f"{self.prompt_fn(child)}\n\nText:\n{text}"
            processor.process_with_schema(
                prompts=[prompt],
                schema=SingleClassificationResult
            )
            classifications = processor.parse_results_with_schema(
                schema=SingleClassificationResult
            )
            classification = classifications[0]
            
            # Store result
            node_name = child["name"]
            result.node_results[node_name] = classification
            
            # If relevant, add path and recurse
            if self.policy.accept(classification, child):
                current_path = f"{parent_path}{self.separator}{node_name}"
                result.classification_paths.append(current_path)
                
                # Recurse to children
                self._dfs_traverse(
                    text=text,
                    node=child,
                    parent_path=current_path,
                    result=result,
                    processor=processor,
                )
```

#### 2. Export Strategy
```python
# In classifier/capabilities/classification/__init__.py
from .dfs import DFSClassificationCapability

__all__ = [
    # ... existing exports ...
    "DFSClassificationCapability",
]
```

#### 3. Use in CLI
```python
# In batch_classify.py
from classifier.capabilities import DFSClassificationCapability

# In main function:
if classification_strategy == "dfs":
    dfs_cap = DFSClassificationCapability(
        prompt_fn=prompt_fn,
        policy=policy,
        separator=">"
    )
    registry.register(dfs_cap)
```
```bash
uv run classifier/cli/batch_classify.py \
    --config topics.json \
    --input-file data.csv \
    --classification-strategy dfs \
    --save-path results.json \
    -v
```

## Adding a Stem Capability

Stem capabilities analyze complete classification paths.

### Example: Stem Complexity Analyzer

#### 1. Create Models
```python
# classifier/capabilities/stem/complexity/models.py
from typing import Literal
from pydantic import BaseModel

class ComplexityOutput(BaseModel):
    has_complexity: bool
    complexity_level: Literal["simple", "moderate", "complex", "very_complex"] = None
    reasoning: str = ""
```

#### 2. Create Prompt
```python
# classifier/capabilities/stem/complexity/prompts.py
from typing import Dict, List

def stem_complexity_prompt(
    text: str,
    stem_path: str,
    stem_definitions: List[Dict[str, str]] = None
) -> str:
    definitions_section = ""
    if stem_definitions:
        definitions_section = "\n**Topic Path:**\n\n"
        for node_info in stem_definitions:
            definitions_section += f"**{node_info['name']}**: {node_info.get('description', '')}\n"
    
    return f"""
Analyze the complexity of discussion about this topic path.

**Comment**: {text}
**Topic Path**: {stem_path}
{definitions_section}

Rate the complexity of how the topic is discussed:
- simple: Basic, surface-level discussion
- moderate: Some depth, clear explanation
- complex: Detailed, nuanced discussion
- very_complex: Highly technical, advanced concepts

Return JSON:
{{
  "has_complexity": true,
  "complexity_level": "moderate",
  "reasoning": "The comment discusses the topic with moderate depth."
}}
"""
```

#### 3. Implement Capability
```python
# classifier/capabilities/stem/complexity/capability.py
from typing import Any, Dict, Type
from pydantic import BaseModel
from ..base import StemCapability
from .models import ComplexityOutput
from .prompts import stem_complexity_prompt

class StemComplexityCapability(StemCapability):
    """Analyzes complexity of stem discussions."""
    
    @property
    def name(self) -> str:
        return "stem_complexity"
    
    @property
    def schema(self) -> Type[BaseModel]:
        return ComplexityOutput
    
    def get_stem_prompt_fn(self):
        return stem_complexity_prompt
    
    def _extract_stem_result(self, result_dict: Dict[str, Any]) -> Any:
        """Extract complexity rating."""
        if result_dict and result_dict.get("has_complexity"):
            return {
                "complexity_level": result_dict.get("complexity_level"),
                "reasoning": result_dict.get("reasoning", "")
            }
        return None
```

#### 4. Export and Register
```python
# classifier/capabilities/stem/complexity/__init__.py
from .capability import StemComplexityCapability
from .models import ComplexityOutput
from .prompts import stem_complexity_prompt

__all__ = [
    "StemComplexityCapability",
    "ComplexityOutput",
    "stem_complexity_prompt",
]

# In classifier/capabilities/stem/__init__.py
from .complexity import StemComplexityCapability, ComplexityOutput

__all__ = [
    # ... existing ...
    "StemComplexityCapability",
    "ComplexityOutput",
]
```

#### 5. Register in CLI
```python
# In batch_classify.py
@click.option("--enable-stem-complexity", is_flag=True)
def main(..., enable_stem_complexity: bool, ...):
    if enable_stem_complexity:
        registry.register(
            StemComplexityCapability(max_stem_definitions=max_stem_definitions)
        )
        capability_names.append("stem_complexity")
```

## Capability Dependencies

Capabilities can declare dependencies:
```python
class MyCapability(Capability):
    @property
    def dependencies(self) -> List[str]:
        """This capability needs classification first."""
        return ["classification"]
```

The orchestrator automatically resolves dependencies and executes capabilities in order.

## CLI Options

### Server Management
```bash
# Start server
python classifier/server/vllm_server.py start \
    --port 9005 \
    --gpu-ids 7,6 \
    --model openai/gpt-oss-120b \
    --tensor-parallel-size 2

# Check status
python classifier/server/vllm_server.py status --port 9005

# Stop server
python classifier/server/vllm_server.py stop --port 9005

# Restart server
python classifier/server/vllm_server.py restart --port 9005
```

### Batch Classification
```bash
# Basic usage
uv run classifier/cli/batch_classify.py \
    --config hierarchy.json \
    --input-file data.csv \
    --input-column "text" \
    --save-path results.json

# With capabilities
uv run classifier/cli/batch_classify.py \
    --config hierarchy.json \
    --input-file data.csv \
    --enable-recommendations \
    --enable-alerts \
    --enable-stem-polarity \
    --enable-stem-recommendations \
    --save-path results.json \
    -v

# Custom server
uv run classifier/cli/batch_classify.py \
    --config hierarchy.json \
    --input-file data.csv \
    --server-url http://localhost:9006/v1 \
    --max-concurrent 10 \
    --save-path results.json

# Chunked output
uv run classifier/cli/batch_classify.py \
    --config hierarchy.json \
    --input-file data.csv \
    --output-dir ./results/ \
    --chunk-size 100 \
    -v

# With policies
uv run classifier/cli/batch_classify.py \
    --config hierarchy.json \
    --input-file data.csv \
    --min-confidence 4 \
    --require-excerpt \
    --save-path results.json
```

### Available Capabilities

**Classification:**
- `--enable-classification` (default: enabled)
- `--classification-strategy bfs|bundled` (default: bfs)
- `--bundle-size N` (for bundled strategy)

**Global Analysis:**
- `--enable-recommendations` - Detect actionable recommendations
- `--enable-alerts` - Detect serious workplace concerns
- `--enable-trends` - Detect temporal change patterns

**Stem Analysis (requires classification):**
- `--enable-stem-polarity` - Sentiment toward specific topics
- `--enable-stem-recommendations` - Recommendations per topic
- `--enable-stem-trends` - Trends per topic
- `--enable-sub-stem-polarity` - Sentiment at all path levels

**Standalone Modes:**
- `--recommendations-only` - Only recommendations (no classification)
- `--alerts-only` - Only alerts (no classification)

## Hierarchy JSON Format
```json
{
  "name": "ROOT",
  "description": "Root of the hierarchy",
  "children": [
    {
      "name": "Teaching Effectiveness",
      "description": "Quality and effectiveness of instruction",
      "keywords": ["teaching", "instruction", "pedagogy"],
      "scope": "All aspects of teaching quality",
      "children": [
        {
          "name": "Explanations",
          "description": "Clarity and quality of explanations",
          "keywords": ["explain", "clear", "understand"],
          "scope": "How concepts are explained"
        }
      ]
    }
  ]
}
```

## Output Format
```json
[
  {
    "text": "The instructor provided excellent explanations.",
    "classification_result": {
      "classification_paths": [
        "ROOT>Teaching Effectiveness>Explanations"
      ]
    },
    "recommendations": [],
    "alerts": [],
    "stem_polarity": {
      "Teaching Effectiveness>Explanations": {
        "polarity": "Positive",
        "confidence": 5,
        "reasoning": "Expresses clear praise for explanations",
        "excerpt": "excellent explanations"
      }
    }
  }
]
```

## Testing
```bash
# Test with sample data
echo "The training was excellent but too short." | \
uv run classifier/cli/batch_classify.py \
    --config examples/topics.json \
    --enable-recommendations \
    --save-path test_results.json \
    -v
```

## Troubleshooting

### Server won't start
```bash
# Check if port is in use
lsof -i :9005

# Check GPU availability
nvidia-smi

# View server logs
cat .vllm_servers/instance_0/stderr.log
```

### Import errors
```bash
# Ensure package is installed
uv pip install -e .

# Or use PYTHONPATH
PYTHONPATH=. python classifier/cli/batch_classify.py --help
```

### Low throughput
```bash
# Increase concurrent requests
--max-concurrent 20

# Use bundled classification
--classification-strategy bundled --bundle-size 5
```

## Contributing

1. Create capability following examples above
2. Add tests for new capability
3. Update registry and CLI
4. Document in README

