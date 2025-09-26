"""
Test suite for volume agents error handling and fallback mechanisms

Tests various failure scenarios to ensure the volume agents system
handles errors gracefully and provides meaningful fallback results.
"""

import pytest
import pandas as pd
import numpy as np
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Import the volume agents system
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.volume import VolumeAgentsOrchestrator
    VolumeAgentsOrchestrator, 
    VolumeAgentIntegrationManager,
    VolumeAgentResult,
    AggregatedVolumeAnalysis,
    volume_agents_logger
)

class TestVolumeAgentsErrorHandling:
    """Test suite for volume agents error handling scenarios"""
    
    @pytest.fixture
    def sample_stock_data(self):
        """Create sample stock data for testing"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'open': np.random.uniform(100, 200, 100),
            'high': np.random.uniform(110, 220, 100),
            'low': np.random.uniform(90, 180, 100),
            'close': np.random.uniform(100, 200, 100),
            'volume': np.random.uniform(10000, 100000, 100)
        }, index=dates)
        return data
    
    @pytest.fixture
    def mock_gemini_client(self):
        """Create mock Gemini client"""
        mock_client = Mock()
        mock_client.analyze_volume_agent_specific = Mock(return_value="Mock analysis response")
        return mock_client
    
    @pytest.fixture
    def orchestrator(self, mock_gemini_client):
        """Create orchestrator instance with mock client"""
        return VolumeAgentsOrchestrator(mock_gemini_client)
    
    @pytest.fixture
    def integration_manager(self, mock_gemini_client):
        """Create integration manager instance with mock client"""
        return VolumeAgentIntegrationManager(mock_gemini_client)
    
    @pytest.mark.asyncio
    async def test_single_agent_failure(self, orchestrator, sample_stock_data):
        """Test scenario where one agent fails but others succeed"""
        
        # Mock one agent to fail
        with patch.object(orchestrator.volume_anomaly, 'process_volume_anomalies') as mock_anomaly:
            mock_anomaly.side_effect = Exception("Anomaly agent network timeout")
            
            result = await orchestrator.analyze_stock_volume_comprehensive(
                sample_stock_data, "TEST", {}
            )
            
            # Verify partial success handling
            assert isinstance(result, AggregatedVolumeAnalysis)
            assert result.successful_agents >= 4  # Other agents should succeed
            assert result.failed_agents == 1
            assert 'volume_anomaly' in result.individual_results
            assert not result.individual_results['volume_anomaly'].success
            assert "network timeout" in result.individual_results['volume_anomaly'].error_message
    
    @pytest.mark.asyncio
    async def test_multiple_agents_failure(self, orchestrator, sample_stock_data):
        """Test scenario where multiple agents fail but system continues"""
        
        # Mock multiple agents to fail
        with patch.object(orchestrator.volume_anomaly, 'process_volume_anomalies') as mock_anomaly, \
             patch.object(orchestrator.institutional_activity, 'process_institutional_activity') as mock_institutional:
            
            mock_anomaly.side_effect = Exception("Anomaly processing error")
            mock_institutional.side_effect = Exception("Institutional data unavailable")
            
            result = await orchestrator.analyze_stock_volume_comprehensive(
                sample_stock_data, "TEST", {}
            )
            
            # Verify partial results handling
            assert result.successful_agents >= 3
            assert result.failed_agents == 2
            assert result.overall_confidence > 0  # Should still provide some confidence
            assert 'partial_analysis_warning' in result.unified_analysis
    
    @pytest.mark.asyncio
    async def test_all_agents_failure(self, orchestrator, sample_stock_data):
        """Test scenario where all agents fail - fallback analysis should activate"""
        
        # Mock all agents to fail
        with patch.object(orchestrator.volume_anomaly, 'process_volume_anomalies') as mock_anomaly, \
             patch.object(orchestrator.institutional_activity, 'process_institutional_activity') as mock_institutional, \
             patch.object(orchestrator.volume_confirmation, 'process_volume_confirmation') as mock_confirmation, \
             patch.object(orchestrator.support_resistance, 'process_support_resistance_volume') as mock_sr, \
             patch.object(orchestrator.volume_momentum, 'process_volume_momentum') as mock_momentum:
            
            # All agents fail
            mock_anomaly.side_effect = Exception("Service unavailable")
            mock_institutional.side_effect = Exception("Service unavailable") 
            mock_confirmation.side_effect = Exception("Service unavailable")
            mock_sr.side_effect = Exception("Service unavailable")
            mock_momentum.side_effect = Exception("Service unavailable")
            
            result = await orchestrator.analyze_stock_volume_comprehensive(
                sample_stock_data, "TEST", {}
            )
            
            # Verify fallback behavior
            assert result.successful_agents == 0
            assert result.failed_agents == 5
            assert 'error' in result.unified_analysis or 'fallback_analysis' in result.unified_analysis
    
    @pytest.mark.asyncio 
    async def test_timeout_handling(self, orchestrator, sample_stock_data):
        """Test agent timeout handling"""
        
        # Mock an agent to hang (timeout)
        async def mock_slow_process(*args, **kwargs):
            await asyncio.sleep(35)  # Longer than timeout (30s)
            return {"test": "data"}
        
        with patch.object(orchestrator.volume_anomaly, 'process_volume_anomalies', 
                         side_effect=mock_slow_process):
            
            result = await orchestrator.analyze_stock_volume_comprehensive(
                sample_stock_data, "TEST", {}
            )
            
            # Verify timeout handling
            assert 'volume_anomaly' in result.individual_results
            anomaly_result = result.individual_results['volume_anomaly']
            assert not anomaly_result.success
            assert "timed out" in anomaly_result.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_invalid_data_handling(self, orchestrator):
        """Test handling of invalid input data"""
        
        # Test with empty dataframe
        empty_data = pd.DataFrame()
        result = await orchestrator.analyze_stock_volume_comprehensive(empty_data, "TEST", {})
        
        assert not result.successful_agents
        assert 'error' in result.unified_analysis or 'fallback_analysis' in result.unified_analysis
        
        # Test with None data
        result = await orchestrator.analyze_stock_volume_comprehensive(None, "TEST", {})
        assert not result.successful_agents
        
        # Test with missing required columns
        invalid_data = pd.DataFrame({'price': [1, 2, 3]})
        result = await orchestrator.analyze_stock_volume_comprehensive(invalid_data, "TEST", {})
        assert not result.successful_agents
    
    @pytest.mark.asyncio
    async def test_integration_manager_error_handling(self, integration_manager, sample_stock_data):
        """Test integration manager error handling and health checks"""
        
        # Test health check functionality
        should_use, reason = integration_manager.should_use_volume_agents()
        assert isinstance(should_use, bool)
        assert isinstance(reason, str)
        
        # Test with orchestrator failure
        with patch.object(integration_manager.orchestrator, 'analyze_stock_volume_comprehensive') as mock_orchestrator:
            mock_orchestrator.side_effect = Exception("Orchestrator failure")
            
            result = await integration_manager.get_comprehensive_volume_analysis(
                sample_stock_data, "TEST", {}
            )
            
            # Should return degraded analysis result
            assert not result.get('success', True)
            assert 'error' in result or 'degraded_mode' in result
            assert 'processing_time' in result
    
    def test_agent_performance_metrics(self, integration_manager):
        """Test agent performance metrics tracking"""
        
        # Initialize metrics
        metrics = integration_manager.get_agent_performance_metrics()
        assert isinstance(metrics, dict)
        assert len(metrics) == 5  # Five volume agents
        
        # Test metrics update
        integration_manager.update_agent_metrics('volume_anomaly', True, 5.0, 0.8)
        integration_manager.update_agent_metrics('volume_anomaly', False, 2.0, None)
        
        updated_metrics = integration_manager.get_agent_performance_metrics()
        anomaly_metrics = updated_metrics['volume_anomaly']
        
        assert anomaly_metrics['total_calls'] == 2
        assert anomaly_metrics['successful_calls'] == 1
        assert anomaly_metrics['failed_calls'] == 1
        assert anomaly_metrics['success_rate'] == 0.5
    
    def test_agent_disable_logic(self, integration_manager):
        """Test automatic agent disabling logic"""
        
        # Simulate multiple failures for an agent
        for _ in range(6):
            integration_manager.update_agent_metrics('volume_anomaly', False, 1.0, None)
        
        should_disable, reason = integration_manager.should_disable_agent('volume_anomaly')
        assert should_disable
        assert 'success rate' in reason.lower() or 'reliability' in reason.lower()
    
    def test_health_monitoring_system(self, integration_manager):
        """Test comprehensive health monitoring"""
        
        # Test system health summary
        health_summary = integration_manager.get_system_health_summary()
        assert 'system_status' in health_summary
        assert 'health_percentage' in health_summary
        assert 'recommendation' in health_summary
        assert 'agent_status' in health_summary
        
        # Test individual agent health
        agent_health = integration_manager.get_agent_health_status()
        assert len(agent_health) == 5
        for agent_name, health_data in agent_health.items():
            assert 'healthy' in health_data
            assert 'diagnostics' in health_data
            assert 'status' in health_data
    
    @pytest.mark.asyncio
    async def test_chart_generation_fallback(self, sample_stock_data):
        """Test chart generation fallback mechanisms"""
        
        # Import chart visualizer
        from patterns.visualization import ChartVisualizer
        
        # Test with no volume agents data
        fig = ChartVisualizer.plot_enhanced_volume_chart_with_agents(
            sample_stock_data, {}, None, None, "TEST"
        )
        assert fig is not None
        
        # Test with invalid volume agents data
        invalid_volume_data = {'invalid': 'data'}
        fig = ChartVisualizer.plot_enhanced_volume_chart_with_agents(
            sample_stock_data, {}, invalid_volume_data, None, "TEST"
        )
        assert fig is not None
        
        # Test with corrupted data that causes chart generation to fail
        with patch('patterns.visualization.ChartVisualizer._create_agent_enhanced_volume_chart') as mock_create:
            mock_create.side_effect = Exception("Chart creation failed")
            
            fig = ChartVisualizer.plot_enhanced_volume_chart_with_agents(
                sample_stock_data, {}, {'success': True}, None, "TEST"
            )
            assert fig is not None  # Should fall back to basic chart
    
    def test_logging_system(self):
        """Test comprehensive logging system"""
        
        # Test operation logging
        operation_id = volume_agents_logger.log_operation_start(
            'test_operation', 'TEST', ['agent1', 'agent2']
        )
        assert operation_id.startswith('VA_')
        
        # Test agent execution logging
        volume_agents_logger.log_agent_execution(
            operation_id, 'test_agent', True, 5.0, confidence=0.8
        )
        
        # Test error logging
        try:
            raise ValueError("Test error")
        except Exception as e:
            volume_agents_logger.log_error_with_context(
                operation_id, e, {'symbol': 'TEST', 'stage': 'testing'}
            )
        
        # Test partial success logging
        volume_agents_logger.log_partial_success(
            operation_id, ['agent1'], ['agent2'], fallback_activated=True
        )
        
        # Test fallback activation logging
        volume_agents_logger.log_fallback_activation(
            operation_id, 'Testing fallback', 'test_fallback'
        )
        
        # Test operation completion
        volume_agents_logger.log_operation_complete(
            operation_id, True, 10.0, {'agents': 2, 'success_rate': 0.5}
        )
        
        # All logging calls should execute without errors
        assert True
    
    @pytest.mark.asyncio
    async def test_network_failure_simulation(self, orchestrator, sample_stock_data):
        """Test network failure scenarios"""
        
        # Simulate network timeout for LLM calls
        with patch.object(orchestrator.gemini_client, 'analyze_volume_agent_specific') as mock_llm:
            mock_llm.side_effect = Exception("Network timeout")
            
            result = await orchestrator.analyze_stock_volume_comprehensive(
                sample_stock_data, "TEST", {}
            )
            
            # Agents should still succeed with data processing, just no LLM analysis
            assert result.successful_agents > 0
            for agent_result in result.individual_results.values():
                if agent_result.success:
                    assert agent_result.analysis_data is not None
                    # LLM response should be None due to failure
                    assert agent_result.llm_response is None
    
    @pytest.mark.asyncio
    async def test_data_corruption_handling(self, orchestrator):
        """Test handling of corrupted or malformed data"""
        
        # Create corrupted data with NaN values
        corrupted_data = pd.DataFrame({
            'open': [100, np.nan, 120],
            'high': [110, 130, np.nan],
            'low': [90, 100, 110],
            'close': [105, np.nan, 115],
            'volume': [1000, 0, np.nan]
        })
        
        result = await orchestrator.analyze_stock_volume_comprehensive(
            corrupted_data, "TEST", {}
        )
        
        # System should handle corrupted data gracefully
        # May provide reduced functionality but shouldn't crash
        assert isinstance(result, AggregatedVolumeAnalysis)
        # At least some basic analysis should be possible
        assert result.total_processing_time > 0


class TestVolumeAgentsStressTest:
    """Stress tests for volume agents system"""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling multiple concurrent volume analysis requests"""
        
        # Create multiple orchestrators
        orchestrators = [VolumeAgentsOrchestrator() for _ in range(3)]
        
        # Create test data
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        test_data = pd.DataFrame({
            'open': np.random.uniform(100, 200, 50),
            'high': np.random.uniform(110, 220, 50),
            'low': np.random.uniform(90, 180, 50),
            'close': np.random.uniform(100, 200, 50),
            'volume': np.random.uniform(10000, 100000, 50)
        }, index=dates)
        
        # Run concurrent analysis
        tasks = []
        for i, orch in enumerate(orchestrators):
            task = orch.analyze_stock_volume_comprehensive(
                test_data, f"TEST{i}", {}
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All requests should complete (success or controlled failure)
        assert len(results) == 3
        for result in results:
            if isinstance(result, Exception):
                # Acceptable if it's a controlled failure
                assert "timeout" in str(result).lower() or "unavailable" in str(result).lower()
            else:
                assert isinstance(result, AggregatedVolumeAnalysis)
    
    def test_memory_usage(self):
        """Test memory usage with large datasets"""
        
        # Create large dataset
        dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
        large_data = pd.DataFrame({
            'open': np.random.uniform(100, 200, 1000),
            'high': np.random.uniform(110, 220, 1000),
            'low': np.random.uniform(90, 180, 1000),
            'close': np.random.uniform(100, 200, 1000),
            'volume': np.random.uniform(10000, 100000, 1000)
        }, index=dates)
        
        # Create orchestrator
        orchestrator = VolumeAgentsOrchestrator()
        
        # Test input validation with large dataset
        validation_error = orchestrator._validate_inputs(large_data, "TEST", {})
        
        # Should handle large dataset without memory errors
        assert validation_error is None or "insufficient data" not in validation_error.lower()

if __name__ == "__main__":
    # Run the tests
    import subprocess
    import sys
    
    # Install pytest if not available
    try:
        import pytest
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytest", "pytest-asyncio"])
        import pytest
    
    # Run specific test classes
    pytest.main([
        __file__, 
        "-v",
        "--tb=short",
        "-k", "test_single_agent_failure or test_multiple_agents_failure or test_integration_manager_error_handling"
    ])