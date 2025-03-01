"""
Tests for the Pipeline and PipelineBuilder components.
"""

import pytest
from unittest.mock import MagicMock

from memoryweave.interfaces.pipeline import IPipelineStage
from memoryweave.pipeline.builder import PipelineBuilder, Pipeline


class TestPipelineBuilder:
    """Test suite for the PipelineBuilder class."""
    
    @pytest.fixture
    def mock_stages(self):
        """Create mock pipeline stages."""
        stage1 = MagicMock(spec=IPipelineStage)
        stage1.process.return_value = "Stage 1 Output"
        
        stage2 = MagicMock(spec=IPipelineStage)
        stage2.process.return_value = "Stage 2 Output"
        
        stage3 = MagicMock(spec=IPipelineStage)
        stage3.process.return_value = "Final Output"
        
        return [stage1, stage2, stage3]
    
    def test_add_stage(self, mock_stages):
        """Test adding stages to the builder."""
        builder = PipelineBuilder()
        
        # Add stages one by one
        builder.add_stage(mock_stages[0])
        builder.add_stage(mock_stages[1])
        builder.add_stage(mock_stages[2])
        
        # Build the pipeline
        pipeline = builder.build()
        
        # Verify pipeline has all stages
        assert len(pipeline.get_stages()) == 3
        assert pipeline.get_stages()[0] == mock_stages[0]
        assert pipeline.get_stages()[1] == mock_stages[1]
        assert pipeline.get_stages()[2] == mock_stages[2]
    
    def test_builder_fluent_interface(self, mock_stages):
        """Test the builder's fluent interface."""
        builder = PipelineBuilder()
        
        # Chain add_stage calls
        result = builder.add_stage(mock_stages[0]).add_stage(mock_stages[1])
        
        # Check that add_stage returns the builder
        assert result is builder
        
        # Chain set_name
        result = builder.set_name("test_pipeline")
        
        # Check that set_name returns the builder
        assert result is builder
        
        # Build the pipeline
        pipeline = builder.build()
        
        # Verify pipeline name
        assert pipeline._name == "test_pipeline"
    
    def test_clear(self, mock_stages):
        """Test clearing the builder."""
        builder = PipelineBuilder()
        
        # Add stages
        builder.add_stage(mock_stages[0]).add_stage(mock_stages[1])
        
        # Clear builder
        builder.clear()
        
        # Build the pipeline
        pipeline = builder.build()
        
        # Verify pipeline has no stages
        assert len(pipeline.get_stages()) == 0


class TestPipeline:
    """Test suite for the Pipeline class."""
    
    @pytest.fixture
    def mock_stages(self):
        """Create mock pipeline stages with specific behavior."""
        stage1 = MagicMock(spec=IPipelineStage)
        stage1.process.return_value = {"processed_by": "stage1", "data": "modified"}
        
        stage2 = MagicMock(spec=IPipelineStage)
        stage2.process.side_effect = lambda data: {
            **data, 
            "processed_by": data["processed_by"] + ",stage2", 
            "additional": "value"
        }
        
        stage3 = MagicMock(spec=IPipelineStage)
        stage3.process.side_effect = lambda data: {
            **data, 
            "processed_by": data["processed_by"] + ",stage3", 
            "final": True
        }
        
        return [stage1, stage2, stage3]
    
    def test_init(self, mock_stages):
        """Test pipeline initialization."""
        pipeline = Pipeline(mock_stages, "test_pipeline")
        
        assert pipeline._stages == mock_stages
        assert pipeline._name == "test_pipeline"
    
    def test_get_stages(self, mock_stages):
        """Test getting pipeline stages."""
        pipeline = Pipeline(mock_stages)
        
        # Get stages
        stages = pipeline.get_stages()
        
        # Verify stages
        assert len(stages) == 3
        assert stages == mock_stages
        
        # Check that returned list is a copy
        stages.append(MagicMock())
        assert len(pipeline._stages) == 3
    
    def test_execute_empty_pipeline(self):
        """Test executing an empty pipeline."""
        pipeline = Pipeline([])
        
        # Execute pipeline with input data
        input_data = {"test": "data"}
        output_data = pipeline.execute(input_data)
        
        # Output should be same as input for empty pipeline
        assert output_data == input_data
    
    def test_execute_single_stage(self, mock_stages):
        """Test executing a pipeline with a single stage."""
        pipeline = Pipeline([mock_stages[0]])
        
        # Execute pipeline
        input_data = {"test": "data"}
        output_data = pipeline.execute(input_data)
        
        # Verify stage was called
        mock_stages[0].process.assert_called_once_with(input_data)
        
        # Verify output
        assert output_data == {"processed_by": "stage1", "data": "modified"}
    
    def test_execute_multi_stage(self, mock_stages):
        """Test executing a pipeline with multiple stages."""
        pipeline = Pipeline(mock_stages)
        
        # Execute pipeline
        input_data = {"test": "data"}
        output_data = pipeline.execute(input_data)
        
        # Verify each stage was called with the output of the previous stage
        mock_stages[0].process.assert_called_once_with(input_data)
        mock_stages[1].process.assert_called_once_with(mock_stages[0].process.return_value)
        mock_stages[2].process.assert_called_once_with(mock_stages[1].process.return_value)
        
        # Verify final output
        assert output_data["processed_by"] == "stage1,stage2,stage3"
        assert output_data["additional"] == "value"
        assert output_data["final"] is True