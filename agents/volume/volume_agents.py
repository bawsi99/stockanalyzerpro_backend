"""
Volume Agents Orchestrator and Integration Manager

This module provides the central orchestration and integration layer for all volume analysis agents.
It manages simultaneous execution, result aggregation, and provides a unified interface to the main system.
"""

import logging
import asyncio
import time
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
import pandas as pd
import numpy as np
import traceback
import matplotlib.pyplot as plt

# Import all volume agents
from .volume_anomaly import VolumeAnomalyProcessor, VolumeAnomalyCharts
from .institutional_activity import InstitutionalActivityProcessor, InstitutionalActivityCharts
from .volume_confirmation import VolumeConfirmationProcessor, VolumeConfirmationCharts
from .support_resistance import SupportResistanceProcessor, SupportResistanceCharts
from .volume_momentum import VolumeTrendMomentumProcessor, VolumeTrendMomentumCharts

# Set up logging
logger = logging.getLogger(__name__)

class VolumeAgentsLogger:
    """
    Comprehensive logging system for volume agents with structured logging and metrics
    """
    
    def __init__(self, base_logger=None):
        self.base_logger = base_logger or logger
        self.operation_id = 0
        
    def get_next_operation_id(self) -> str:
        """Generate unique operation ID for tracking"""
        self.operation_id += 1
        return f"VA_{int(time.time())}_{self.operation_id:04d}"
    
    def log_operation_start(self, operation_type: str, symbol: str, agent_names: List[str] = None, **kwargs) -> str:
        """Log the start of a volume agents operation"""
        operation_id = self.get_next_operation_id()
        
        log_data = {
            'operation_id': operation_id,
            'operation_type': operation_type,
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'agents': agent_names or [],
            'status': 'started'
        }
        log_data.update(kwargs)
        
        self.base_logger.info(f"[VOLUME_AGENTS] Operation started: {json.dumps(log_data)}")
        return operation_id
    
    def log_operation_complete(self, operation_id: str, success: bool, processing_time: float, 
                             result_summary: Dict[str, Any] = None, **kwargs):
        """Log the completion of a volume agents operation"""
        log_data = {
            'operation_id': operation_id,
            'status': 'completed',
            'success': success,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat(),
            'result_summary': result_summary or {}
        }
        log_data.update(kwargs)
        
        level = logging.INFO if success else logging.WARNING
        self.base_logger.log(level, f"[VOLUME_AGENTS] Operation completed: {json.dumps(log_data)}")
    
    def log_agent_execution(self, operation_id: str, agent_name: str, success: bool, 
                          processing_time: float, error_message: str = None, 
                          confidence: float = None, **kwargs):
        """Log individual agent execution results"""
        log_data = {
            'operation_id': operation_id,
            'agent_name': agent_name,
            'success': success,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat(),
            'confidence': confidence,
            'error_message': error_message
        }
        log_data.update(kwargs)
        
        level = logging.INFO if success else logging.ERROR
        self.base_logger.log(level, f"[VOLUME_AGENTS] Agent execution: {json.dumps(log_data)}")
    
    def log_partial_success(self, operation_id: str, successful_agents: List[str], 
                          failed_agents: List[str], fallback_activated: bool = False):
        """Log partial success scenarios"""
        log_data = {
            'operation_id': operation_id,
            'event_type': 'partial_success',
            'successful_agents': successful_agents,
            'failed_agents': failed_agents,
            'success_rate': len(successful_agents) / (len(successful_agents) + len(failed_agents)),
            'fallback_activated': fallback_activated,
            'timestamp': datetime.now().isoformat()
        }
        
        self.base_logger.warning(f"[VOLUME_AGENTS] Partial success: {json.dumps(log_data)}")
    
    def log_fallback_activation(self, operation_id: str, reason: str, fallback_type: str):
        """Log fallback mechanism activation"""
        log_data = {
            'operation_id': operation_id,
            'event_type': 'fallback_activation',
            'reason': reason,
            'fallback_type': fallback_type,
            'timestamp': datetime.now().isoformat()
        }
        
        self.base_logger.warning(f"[VOLUME_AGENTS] Fallback activated: {json.dumps(log_data)}")
    
    def log_error_with_context(self, operation_id: str, error: Exception, context: Dict[str, Any]):
        """Log errors with full context for debugging"""
        log_data = {
            'operation_id': operation_id,
            'event_type': 'error',
            'error_type': type(error).__name__,
            'error_message': str(error),
            'error_traceback': traceback.format_exc(),
            'context': context,
            'timestamp': datetime.now().isoformat()
        }
        
        self.base_logger.error(f"[VOLUME_AGENTS] Error with context: {json.dumps(log_data)}")

# Global logger instance
volume_agents_logger = VolumeAgentsLogger(logger)

@dataclass
class VolumeAgentResult:
    """Individual volume agent analysis result"""
    agent_name: str
    success: bool
    processing_time: float
    chart_image: Optional[bytes] = None
    analysis_data: Dict[str, Any] = field(default_factory=dict)
    prompt_text: Optional[str] = None
    llm_response: Optional[str] = None
    error_message: Optional[str] = None
    confidence_score: Optional[float] = None

@dataclass
class AggregatedVolumeAnalysis:
    """Aggregated results from all volume agents"""
    individual_results: Dict[str, VolumeAgentResult] = field(default_factory=dict)
    unified_analysis: Dict[str, Any] = field(default_factory=dict)
    total_processing_time: float = 0.0
    successful_agents: int = 0
    failed_agents: int = 0
    overall_confidence: float = 0.0
    consensus_signals: Dict[str, Any] = field(default_factory=dict)
    conflicting_signals: List[Dict[str, Any]] = field(default_factory=list)

class VolumeAgentsOrchestrator:
    """
    Central orchestrator for all volume analysis agents
    
    Manages simultaneous execution of all 5 volume agents:
    - volume_anomaly: Statistical volume spike detection
    - institutional_activity: Large order flow and institutional patterns
    - volume_confirmation: Price-volume relationship validation
    - support_resistance: Volume-based level identification
    - volume_momentum: Volume trend and momentum analysis
    """
    
    def __init__(self, gemini_client=None):
        # Support for either single gemini_client or None (will create per-agent clients)
        self.gemini_client = gemini_client
        self.use_distributed_keys = (gemini_client is None)
        
        # Initialize distributed agents - all agents now fully distributed
        try:
            from agents.volume.volume_confirmation.llm_agent import create_volume_confirmation_llm_agent
            from agents.volume.support_resistance.llm_agent import SupportResistanceLLMAgent
            from agents.volume.institutional_activity.agent import InstitutionalActivityAgent
            from agents.volume.volume_momentum.agent import VolumeMomentumAgent
            from agents.volume.volume_anomaly.agent import VolumeAnomalyAgent
            
            # Initialize all distributed agents
            self.llm_agents = {
                'volume_confirmation': create_volume_confirmation_llm_agent(),
                'support_resistance': SupportResistanceLLMAgent(),
                'institutional_activity': InstitutionalActivityAgent(),
                'volume_momentum': VolumeMomentumAgent(), 
                'volume_anomaly': VolumeAnomalyAgent()
            }
            
            print("✅ All volume agents initialized with distributed architecture")
            print(f"   - volume_confirmation: fully distributed agent")
            print(f"   - support_resistance: fully distributed agent")
            print(f"   - institutional_activity: fully distributed agent")
            print(f"   - volume_momentum: fully distributed agent")
            print(f"   - volume_anomaly: fully distributed agent")
            
        except Exception as e:
            print(f"⚠️ Failed to initialize distributed agents: {e}")
            self.llm_agents = {}
            raise RuntimeError(f"Volume agents initialization failed: {e}")
        
        # Agent configuration - distributed agents handle everything internally
        self.agent_config = {
            'volume_anomaly': {
                'enabled': True,
                'weight': 0.20,
                'timeout': None
            },
            'institutional_activity': {
                'enabled': True,
                'weight': 0.25,
                'timeout': None
            },
            'volume_confirmation': {
                'enabled': True,
                'weight': 0.20,
                'timeout': None
            },
            'support_resistance': {
                'enabled': True,
                'weight': 0.20,
                'timeout': None
            },
            'volume_momentum': {
                'enabled': True,
                'weight': 0.15,
                'timeout': None
            }
        }
        
        logger.info(f"VolumeAgentsOrchestrator initialized with {len(self.agent_config)} agents")

    async def analyze_stock_volume_comprehensive(self, 
                                               stock_data: pd.DataFrame, 
                                               symbol: str,
                                               indicators: Dict[str, Any] = None) -> AggregatedVolumeAnalysis:
        """
        Perform comprehensive volume analysis using all agents simultaneously
        
        Args:
            stock_data: DataFrame with OHLCV data
            symbol: Stock symbol
            indicators: Pre-calculated technical indicators
            
        Returns:
            AggregatedVolumeAnalysis containing all agent results
        """
        start_time = time.time()
        
        # Start comprehensive logging
        enabled_agent_names = [name for name, config in self.agent_config.items() if config['enabled']]
        operation_id = volume_agents_logger.log_operation_start(
            operation_type='comprehensive_volume_analysis',
            symbol=symbol,
            agent_names=enabled_agent_names,
            total_agents=len(enabled_agent_names),
            data_points=len(stock_data) if stock_data is not None else 0
        )
        
        # ENHANCED: Input validation with detailed error messages
        validation_error = self._validate_inputs(stock_data, symbol, indicators)
        if validation_error:
            logger.error(f"Input validation failed for {symbol}: {validation_error}")
            
            # Log validation failure
            volume_agents_logger.log_fallback_activation(
                operation_id, validation_error, 'input_validation_failure'
            )
            volume_agents_logger.log_operation_complete(
                operation_id, success=False, processing_time=time.time() - start_time,
                result_summary={'error': validation_error, 'stage': 'input_validation'}
            )
            
            return self._create_fallback_result(symbol, validation_error, time.time() - start_time)
        
        # Create tasks for all enabled agents
        tasks = []
        enabled_agents = []
        
        for agent_name, config in self.agent_config.items():
            if config['enabled']:
                task = asyncio.create_task(
                    self._run_single_agent(agent_name, config, stock_data, symbol, indicators)
                )
                tasks.append(task)
                enabled_agents.append(agent_name)
        
        # Execute all agents simultaneously with partial result collection
        logger.info(f"Executing {len(tasks)} volume agents simultaneously...")
        
        # Use asyncio.wait to collect partial results even if some agents timeout
        try:
            done, pending = await asyncio.wait(tasks, timeout=150, return_when=asyncio.ALL_COMPLETED)
            
            # Collect results from completed tasks
            agent_results = []
            for i, task in enumerate(tasks):
                if task in done:
                    try:
                        result = await task
                        agent_results.append(result)
                    except Exception as e:
                        logger.error(f"Agent {enabled_agents[i]} failed: {e}")
                        agent_results.append(e)
                else:
                    # Task didn't complete in time
                    logger.warning(f"Agent {enabled_agents[i]} timed out after 150 seconds")
                    task.cancel()  # Cancel the pending task
                    agent_results.append(TimeoutError(f"Agent {enabled_agents[i]} timed out"))
            
            # Log partial results collection
            completed_count = len([r for r in agent_results if not isinstance(r, Exception)])
            logger.info(f"Volume agents execution completed: {completed_count}/{len(tasks)} agents finished, {len(pending)} timed out after 150s")
            
        except Exception as gather_error:
            logger.error(f"Error in agent execution: {gather_error}")
            # Fallback to original gather method
            agent_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results with metrics tracking
        individual_results = {}
        successful_agents = 0
        failed_agents = 0
        
        for i, result in enumerate(agent_results):
            agent_name = enabled_agents[i]
            
            if isinstance(result, Exception):
                logger.error(f"Agent {agent_name} failed with exception: {result}")
                
                # Log individual agent failure with context
                volume_agents_logger.log_agent_execution(
                    operation_id, agent_name, success=False, processing_time=0.0,
                    error_message=str(result), exception_type=type(result).__name__
                )
                
                individual_results[agent_name] = VolumeAgentResult(
                    agent_name=agent_name,
                    success=False,
                    processing_time=0.0,
                    error_message=str(result)
                )
                failed_agents += 1
                
                # Track failed agent metrics - will be handled by integration manager
                    
            else:
                individual_results[agent_name] = result
                
                # Log individual agent execution
                volume_agents_logger.log_agent_execution(
                    operation_id, agent_name, 
                    success=result.success, 
                    processing_time=result.processing_time,
                    error_message=result.error_message if not result.success else None,
                    confidence=result.confidence_score,
                    has_chart=result.chart_image is not None,
                    has_llm_response=result.llm_response is not None
                )
                
                # Track agent metrics - will be handled by integration manager
                
                if result.success:
                    successful_agents += 1
                else:
                    failed_agents += 1
        
        total_processing_time = time.time() - start_time
        
        # Aggregate results
        aggregated_analysis = await self._aggregate_agent_results(
            individual_results, stock_data, symbol
        )
        
        # Create final result
        final_result = AggregatedVolumeAnalysis(
            individual_results=individual_results,
            unified_analysis=aggregated_analysis,
            total_processing_time=total_processing_time,
            successful_agents=successful_agents,
            failed_agents=failed_agents,
            overall_confidence=self._calculate_overall_confidence(individual_results),
            consensus_signals=self._identify_consensus_signals(individual_results),
            conflicting_signals=self._identify_conflicting_signals(individual_results)
        )
        
        # Log completion with detailed results
        total_agents = successful_agents + failed_agents
        operation_success = successful_agents > 0
        
        # Detect and log partial success scenarios
        if failed_agents > 0 and successful_agents > 0:
            successful_agent_names = [name for name, result in individual_results.items() if result.success]
            failed_agent_names = [name for name, result in individual_results.items() if not result.success]
            
            volume_agents_logger.log_partial_success(
                operation_id, successful_agent_names, failed_agent_names, fallback_activated=False
            )
        
        # Log final operation completion
        result_summary = {
            'successful_agents': successful_agents,
            'failed_agents': failed_agents,
            'total_agents': total_agents,
            'success_rate': successful_agents / total_agents if total_agents > 0 else 0.0,
            'overall_confidence': final_result.overall_confidence,
            'has_conflicts': len(final_result.conflicting_signals) > 0,
            'conflict_count': len(final_result.conflicting_signals)
        }
        
        volume_agents_logger.log_operation_complete(
            operation_id, success=operation_success, processing_time=total_processing_time,
            result_summary=result_summary
        )
        
        logger.info(f"Volume analysis completed in {total_processing_time:.2f}s - "
                   f"{successful_agents} successful, {failed_agents} failed agents")
        
        return final_result

    async def _run_single_agent(self, 
                               agent_name: str, 
                               config: Dict[str, Any],
                               stock_data: pd.DataFrame, 
                               symbol: str,
                               indicators: Dict[str, Any] = None) -> VolumeAgentResult:
        """
        Run a single volume agent with timeout protection
        """
        start_time = time.time()
        
        try:
            # Wait for agent completion with optional timeout
            if config['timeout'] is not None:
                result = await asyncio.wait_for(
                    self._execute_agent(agent_name, config, stock_data, symbol, indicators),
                    timeout=config['timeout']
                )
            else:
                # No timeout - let agent run until completion
                result = await self._execute_agent(agent_name, config, stock_data, symbol, indicators)
            
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            return result
            
        except asyncio.TimeoutError:
            processing_time = time.time() - start_time
            timeout_msg = f"Agent {agent_name} timed out after {config['timeout']}s" if config['timeout'] is not None else f"Agent {agent_name} timed out"
            logger.warning(timeout_msg)
            return VolumeAgentResult(
                agent_name=agent_name,
                success=False,
                processing_time=processing_time,
                error_message=f"Agent timed out after {config['timeout']} seconds" if config['timeout'] is not None else "Agent execution timed out"
            )
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Agent {agent_name} failed: {e}")
            return VolumeAgentResult(
                agent_name=agent_name,
                success=False,
                processing_time=processing_time,
                error_message=str(e)
            )

    async def _execute_agent(self, 
                           agent_name: str, 
                           config: Dict[str, Any],
                           stock_data: pd.DataFrame, 
                           symbol: str,
                           indicators: Dict[str, Any] = None) -> VolumeAgentResult:
        """
        Execute a specific volume agent
        """
        # Local import to avoid global dependency and allow patching without top-level edits
        import asyncio
        
        # Initialize start_time for timing calculations
        start_time = time.time()
        
        try:
            # All agents are now fully migrated to distributed architecture
            if agent_name not in self.llm_agents:
                raise ValueError(f"Agent {agent_name} not found in distributed agents")
                
            # Use fully distributed agent that handles everything internally
            print(f"[DISTRIBUTED_AGENT] {agent_name} comprehensive analysis starting for {symbol}")
            
            # Call the comprehensive analysis method (method name varies by agent)
            if hasattr(self.llm_agents[agent_name], 'analyze_complete'):
                # Use analyze_complete for agents that have it
                llm_agent_result = await self.llm_agents[agent_name].analyze_complete(
                    stock_data=stock_data,
                    symbol=symbol,
                    context=""
                )
            elif hasattr(self.llm_agents[agent_name], 'analyze_with_llm'):
                # Fallback to analyze_with_llm for agents using that method
                # Generate chart image first for multi-modal analysis
                chart_image = None
                try:
                    if agent_name == 'support_resistance':
                        # Support resistance chart generation is handled internally by the agent
                        # This is just a placeholder - the agent handles its own chart generation
                        chart_image = None
                    elif agent_name == 'volume_confirmation':
                        # Placeholder for volume_confirmation chart generation if needed
                        pass
                    
                    print(f"[DISTRIBUTED_AGENT] {agent_name} chart generation completed for {symbol} - has_image={chart_image is not None}")
                except Exception as chart_error:
                    logger.warning(f"Chart generation failed for distributed agent {agent_name}: {chart_error}")
                    print(f"[DISTRIBUTED_AGENT] {agent_name} chart generation failed for {symbol}: {chart_error}")
                
                llm_agent_result = await self.llm_agents[agent_name].analyze_with_llm(
                    stock_data=stock_data, 
                    symbol=symbol, 
                    chart_image=chart_image,
                    context=""
                )
            else:
                raise ValueError(f"Agent {agent_name} does not have required analysis method")
            
            if llm_agent_result.get('success', False):
                print(f"[DISTRIBUTED_AGENT] {agent_name} comprehensive analysis completed for {symbol}")
                
                # Extract components for compatibility with orchestrator interface
                technical_analysis = llm_agent_result.get('technical_analysis', {})
                llm_analysis = llm_agent_result.get('llm_analysis')
                
                # Build a simple prompt summary for logging
                prompt_text = f"Comprehensive {agent_name} analysis for {symbol} with integrated LLM processing"
                
                # Extract confidence score from technical analysis or use default
                confidence_score = self._extract_confidence_score(technical_analysis, llm_analysis)
                
                # Extract chart image from result if available
                chart_image = llm_agent_result.get('chart_image')
                
                return VolumeAgentResult(
                    agent_name=agent_name,
                    success=True,
                    processing_time=0.0,  # Will be set by caller
                    chart_image=chart_image,
                    analysis_data=technical_analysis,
                    prompt_text=prompt_text,
                    llm_response=llm_analysis,
                    confidence_score=confidence_score
                )
            else:
                # Distributed agent failed
                error_msg = llm_agent_result.get('error', 'Unknown distributed agent error')
                print(f"[DISTRIBUTED_AGENT] {agent_name} analysis failed for {symbol}: {error_msg}")
                return VolumeAgentResult(
                    agent_name=agent_name,
                    success=False,
                    processing_time=0.0,  # Will be set by caller
                    error_message=error_msg
                )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Agent {agent_name} failed: {e}")
            return VolumeAgentResult(
                agent_name=agent_name,
                success=False,
                processing_time=processing_time,
                error_message=str(e)
            )
    def _build_agent_prompt(self, agent_name: str, analysis_data: Dict[str, Any], symbol: str) -> str:
        """
        Build agent-specific prompt for LLM analysis
        """
        base_context = f"## Analysis Context for {symbol}\n\nAgent: {agent_name}\n\n"
        
        if agent_name == 'volume_anomaly':
            return base_context + f"""
Volume Anomaly Analysis Data:
{json.dumps(analysis_data, indent=2, default=str)}

Please analyze this volume anomaly data and provide insights on:
1. Statistical significance of volume spikes
2. Pattern classification and causes
3. Risk assessment for retail-driven anomalies
4. Trading implications of identified anomalies
"""
        
        elif agent_name == 'institutional_activity':
            return base_context + f"""
Institutional Activity Analysis Data:
{json.dumps(analysis_data, indent=2, default=str)}

Please analyze this institutional activity data and provide insights on:
1. Large order flow patterns
2. Block trade identification
3. Institutional accumulation/distribution signals
4. Impact on price discovery and market structure
"""
        
        elif agent_name == 'volume_confirmation':
            return base_context + f"""
Volume Confirmation Analysis Data:
{json.dumps(analysis_data, indent=2, default=str)}

Please analyze this volume confirmation data and provide insights on:
1. Price-volume relationship strength
2. Confirmation or divergence signals
3. Trend validation through volume
4. Signal reliability and false positive risks
"""
        
        # Note: support_resistance is now handled by dedicated LLM agent
        
        elif agent_name == 'volume_momentum':
            return base_context + f"""
Volume Momentum Analysis Data:
{json.dumps(analysis_data, indent=2, default=str)}

Please analyze this volume momentum data and provide insights on:
1. Volume trend strength and sustainability
2. Momentum acceleration/deceleration signals
3. Volume-based momentum indicators
4. Trend continuation probabilities
"""
        
        return base_context + f"Analysis Data: {json.dumps(analysis_data, indent=2, default=str)}"
    
    def _build_agent_prompt_with_template(self, agent_name: str, analysis_data: Dict[str, Any], symbol: str) -> str:
        """
        Build agent-specific prompt using template system for backend/llm agents
        """
        # Build context for the template
        context = self._build_agent_context(agent_name, analysis_data, symbol)
        
        # Load and format template
        template_content = self._load_agent_template(agent_name)
        
        # Replace context placeholder
        final_prompt = template_content.replace('{context}', context)
        
        return final_prompt
    
    def _build_agent_context(self, agent_name: str, analysis_data: Dict[str, Any], symbol: str) -> str:
        """
        Build context string for agent template
        """
        if agent_name == 'institutional_activity':
            return f"""Stock: {symbol}
Analysis Timestamp: {datetime.now().isoformat()}

INSTITUTIONAL ACTIVITY ANALYSIS DATA:
{json.dumps(analysis_data, indent=2, default=str)}

Key Metrics Summary:
- Volume Profile Analysis: {len(analysis_data.get('volume_profile', {}).get('volume_at_price', []))} price levels analyzed
- Large Block Transactions: {analysis_data.get('large_block_analysis', {}).get('total_large_blocks', 0)} detected
- Institutional Blocks: {analysis_data.get('large_block_analysis', {}).get('institutional_block_count', 0)} detected
- Activity Level: {analysis_data.get('institutional_activity_level', 'unknown')}
- Primary Activity: {analysis_data.get('primary_activity', 'unknown')}

Please analyze this data to identify institutional trading patterns and smart money flow."""
        
        # Note: support_resistance is now handled by dedicated LLM agent
        
        # Default context for other agents
        return f"Stock: {symbol}\nAnalysis Data:\n{json.dumps(analysis_data, indent=2, default=str)}"
    
    def _load_agent_template(self, agent_name: str) -> str:
        """
        Load prompt template for agent
        """
        template_map = {
            'institutional_activity': 'institutional_activity_analysis.txt'
            # Note: support_resistance is now handled by dedicated LLM agent
        }
        
        template_file = template_map.get(agent_name)
        if not template_file:
            # Fallback template
            return "You are a {agent_name} specialist. Analyze the following data:\n\n{context}\n\nProvide detailed analysis."
        
        # Load template from prompts directory
        import os
        template_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'prompts', template_file)
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Failed to load template {template_file}: {e}")
            # Fallback template
            return f"You are a {agent_name.replace('_', ' ')} specialist. Analyze the following data:\n\n{{context}}\n\nProvide detailed analysis."

    async def _aggregate_agent_results(self, 
                                     individual_results: Dict[str, VolumeAgentResult],
                                     stock_data: pd.DataFrame, 
                                     symbol: str) -> Dict[str, Any]:
        """
        Aggregate results from all volume agents into a unified analysis with robust partial results handling
        """
        aggregated = {
            'symbol': symbol,
            'analysis_timestamp': datetime.now().isoformat(),
            'volume_summary': {},
            'key_findings': [],
            'consensus_signals': {},
            'risk_assessment': {},
            'trading_implications': {},
            'agent_status': {},
            'partial_analysis_warning': None
        }
        
        # Collect successful and failed results for comprehensive analysis
        successful_results = {k: v for k, v in individual_results.items() if v.success}
        failed_results = {k: v for k, v in individual_results.items() if not v.success}
        
        total_agents = len(individual_results)
        successful_count = len(successful_results)
        
        # Add agent status summary
        aggregated['agent_status'] = {
            'total_agents': total_agents,
            'successful_agents': successful_count,
            'failed_agents': len(failed_results),
            'success_rate': successful_count / total_agents if total_agents > 0 else 0.0,
            'agent_details': {
                agent_name: {
                    'success': result.success,
                    'processing_time': result.processing_time,
                    'error_message': result.error_message if not result.success else None
                }
                for agent_name, result in individual_results.items()
            }
        }
        
        # Handle complete failure scenario
        if not successful_results:
            aggregated['error'] = "No agents completed successfully"
            # Provide orchestrator-level fallback when all agents fail
            try:
                aggregated['fallback_analysis'] = self._create_fallback_volume_analysis(stock_data, symbol)
            except Exception as _fb_ex:
                logger.warning(f"Fallback volume analysis failed: {_fb_ex}")
                aggregated['fallback_analysis'] = {}
            aggregated['partial_analysis_warning'] = "Analysis based on fallback methods only due to agent failures"
            return aggregated
        
        # Handle partial failure scenario
        if len(failed_results) > 0:
            failure_rate = len(failed_results) / total_agents
            
            if failure_rate > 0.6:  # More than 60% failed
                aggregated['partial_analysis_warning'] = f"High agent failure rate ({failure_rate:.1%}) - results may be incomplete"
            elif failure_rate > 0.3:  # More than 30% failed
                aggregated['partial_analysis_warning'] = f"Some agents failed ({failure_rate:.1%}) - consider results with caution"
            
            # Log failed agents for debugging
            failed_agent_list = list(failed_results.keys())
            logger.warning(f"Volume analysis for {symbol} had {len(failed_results)} failed agents: {failed_agent_list}")
        
        # Volume summary from multiple agents
        current_volume = stock_data['volume'].iloc[-1]
        volume_stats = {
            'current_volume': int(current_volume),
            'volume_percentile': self._calculate_volume_percentile(stock_data),
            'volume_trend': self._determine_volume_trend(stock_data)
        }
        aggregated['volume_summary'] = volume_stats
        
        # Extract key findings from each agent
        key_findings = []
        consensus_signals = {}
        
        for agent_name, result in successful_results.items():
            if result.analysis_data:
                # Extract key insights based on agent type
                findings = self._extract_agent_key_findings(agent_name, result.analysis_data)
                key_findings.extend(findings)
                
                # Extract signals for consensus analysis
                signals = self._extract_agent_signals(agent_name, result.analysis_data)
                consensus_signals[agent_name] = signals
        
        aggregated['key_findings'] = key_findings
        aggregated['consensus_signals'] = consensus_signals
        
        # Risk assessment
        aggregated['risk_assessment'] = self._aggregate_risk_assessment(successful_results)
        
        # Trading implications
        aggregated['trading_implications'] = self._aggregate_trading_implications(successful_results)
        
        return aggregated

    def _create_fallback_volume_analysis(self, stock_data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Create basic volume analysis when all agents fail (orchestrator-level)."""
        try:
            current_volume = stock_data['volume'].iloc[-1]
            volume_ma_20 = stock_data['volume'].rolling(window=20).mean().iloc[-1]

            # Basic volume metrics
            volume_ratio = current_volume / volume_ma_20 if volume_ma_20 > 0 else 1.0
            volume_percentile = self._calculate_volume_percentile(stock_data)
            volume_trend = self._determine_volume_trend(stock_data)

            return {
                'analysis_method': 'fallback',
                'current_volume': int(current_volume),
                'volume_ma_20': int(volume_ma_20) if not pd.isna(volume_ma_20) else 0,
                'volume_ratio': round(volume_ratio, 2),
                'volume_percentile': round(volume_percentile, 1),
                'volume_trend': volume_trend,
                'basic_signals': {
                    'high_volume': volume_ratio > 2.0,
                    'above_average': volume_ratio > 1.5,
                    'low_volume': volume_ratio < 0.5
                },
                'confidence': 0.3,  # Low confidence for fallback analysis
                'limitations': [
                    'Basic volume metrics only',
                    'No advanced pattern recognition',
                    'No institutional analysis',
                    'Limited signal reliability'
                ]
            }
        except Exception as e:
            logger.error(f"Fallback volume analysis failed: {e}")
            return {
                'analysis_method': 'fallback',
                'error': str(e)
            }

    def _calculate_overall_confidence(self, results: Dict[str, VolumeAgentResult]) -> float:
        """Enhanced confidence calculation with dynamic weight adjustments and performance optimization"""
        if not results:
            return 0.0
        
        # Calculate dynamic weights based on current performance and conditions
        dynamic_weights = self._calculate_dynamic_weights(results)
        
        total_weight = 0.0
        weighted_confidence = 0.0
        successful_agents = 0
        
        # Calculate advanced weighted confidence with multiple factors
        for agent_name, result in results.items():
            if result.success and result.confidence_score is not None:
                # Get dynamic weight instead of static config weight
                weight = dynamic_weights.get(agent_name, 0.2)
                confidence = result.confidence_score
                
                # Apply sophisticated adjustments
                adjusted_confidence = self._apply_confidence_adjustments(
                    agent_name, result, confidence, dynamic_weights
                )
                
                total_weight += weight
                weighted_confidence += adjusted_confidence * weight
                successful_agents += 1
        
        if total_weight == 0:
            return 0.0
            
        base_confidence = weighted_confidence / total_weight
        
        # Apply enhanced quality adjustments
        quality_adjustments = self._calculate_quality_adjustments(results, successful_agents)
        
        # Apply time-decay and market condition factors
        temporal_factors = self._calculate_temporal_confidence_factors(results)
        
        # Combine all factors
        final_confidence = base_confidence * quality_adjustments['success_factor'] * \
                          quality_adjustments['conflict_factor'] * temporal_factors['decay_factor'] * \
                          temporal_factors['market_factor']
        
        # Ensure confidence stays within bounds
        return max(0.0, min(1.0, final_confidence))

    def _identify_consensus_signals(self, results: Dict[str, VolumeAgentResult]) -> Dict[str, Any]:
        """Enhanced consensus signal identification with weighted analysis"""
        signals = {
            'consensus_strength': 0.0,
            'primary_signal': 'neutral',
            'confidence_level': 'low',
            'supporting_agents': [],
            'conflicting_agents': [],
            'signal_details': {}
        }
        
        # Extract signals from each successful agent with weights
        agent_signals = {}
        total_weight = 0.0
        
        for agent_name, result in results.items():
            if result.success and result.analysis_data:
                agent_config = self.agent_config.get(agent_name, {})
                weight = agent_config.get('weight', 0.2)
                
                extracted_signals = self._extract_agent_signals(agent_name, result.analysis_data)
                agent_signals[agent_name] = {
                    'signals': extracted_signals,
                    'weight': weight,
                    'confidence': result.confidence_score or 0.5
                }
                total_weight += weight
        
        if not agent_signals:
            return signals
        
        # Calculate weighted consensus
        weighted_scores = {'bullish': 0.0, 'bearish': 0.0, 'neutral': 0.0}
        
        for agent_name, agent_data in agent_signals.items():
            primary_signal = agent_data['signals'].get('primary_signal', 'neutral')
            signal_strength = agent_data['signals'].get('strength', 0.5)
            agent_weight = agent_data['weight']
            agent_confidence = agent_data['confidence']
            
            # Weight by both agent importance and confidence
            effective_weight = agent_weight * agent_confidence
            weighted_scores[primary_signal] += effective_weight * signal_strength
        
        # Determine primary consensus signal
        max_signal = max(weighted_scores, key=weighted_scores.get)
        max_score = weighted_scores[max_signal]
        
        signals['primary_signal'] = max_signal
        signals['consensus_strength'] = max_score / total_weight if total_weight > 0 else 0.0
        
        # Determine confidence level based on consensus strength
        if signals['consensus_strength'] > 0.7:
            signals['confidence_level'] = 'high'
        elif signals['consensus_strength'] > 0.5:
            signals['confidence_level'] = 'medium'
        else:
            signals['confidence_level'] = 'low'
        
        # Identify supporting and conflicting agents
        for agent_name, agent_data in agent_signals.items():
            agent_signal = agent_data['signals'].get('primary_signal', 'neutral')
            agent_strength = agent_data['signals'].get('strength', 0.5)
            
            if agent_signal == max_signal and agent_strength > 0.6:
                signals['supporting_agents'].append({
                    'agent': agent_name,
                    'signal': agent_signal,
                    'strength': agent_strength,
                    'weight': agent_data['weight']
                })
            elif agent_signal != max_signal and agent_signal != 'neutral':
                signals['conflicting_agents'].append({
                    'agent': agent_name,
                    'signal': agent_signal,
                    'strength': agent_strength,
                    'conflict_severity': abs(agent_strength - signals['consensus_strength'])
                })
        
        # Add detailed signal breakdown
        signals['signal_details'] = {
            'bullish_score': weighted_scores['bullish'],
            'bearish_score': weighted_scores['bearish'], 
            'neutral_score': weighted_scores['neutral'],
            'total_agents': len(agent_signals),
            'supporting_count': len(signals['supporting_agents']),
            'conflicting_count': len(signals['conflicting_agents'])
        }
        
        return signals

    def _identify_conflicting_signals(self, results: Dict[str, VolumeAgentResult]) -> List[Dict[str, Any]]:
        """Enhanced conflict identification with resolution strategies"""
        conflicts = []
        
        # Compare signals between agents
        successful_results = {k: v for k, v in results.items() if v.success}
        
        for agent1_name, result1 in successful_results.items():
            for agent2_name, result2 in successful_results.items():
                if agent1_name >= agent2_name:  # Avoid duplicates
                    continue
                
                conflict = self._detect_enhanced_signal_conflict(
                    agent1_name, result1,
                    agent2_name, result2
                )
                
                if conflict:
                    conflicts.append(conflict)
        
        # Sort conflicts by severity (most severe first)
        conflicts.sort(key=lambda x: x.get('severity_score', 0), reverse=True)
        
        # Add conflict resolution recommendations
        for conflict in conflicts:
            conflict['resolution_strategy'] = self._suggest_conflict_resolution(conflict)
        
        return conflicts

    # Helper methods for agent-specific processing
    def _extract_confidence_score(self, analysis_data: Dict[str, Any], llm_response: str = None) -> float:
        """Extract confidence score from analysis data or LLM response"""
        # Try to extract from analysis data first
        if analysis_data:
            if 'confidence_score' in analysis_data:
                return float(analysis_data['confidence_score'])
            
            # Look for confidence in nested structures
            for key, value in analysis_data.items():
                if isinstance(value, dict) and 'confidence' in value:
                    return float(value['confidence'])
        
        # Try to extract from LLM response
        if llm_response:
            try:
                # Look for confidence patterns in response
                import re
                confidence_patterns = [
                    r'confidence["\s]*:?\s*(\d+\.?\d*)%?',
                    r'confidence_score["\s]*:?\s*(\d+\.?\d*)',
                    r'confidence["\s]*:?\s*(\d+\.?\d*)'
                ]
                
                for pattern in confidence_patterns:
                    match = re.search(pattern, llm_response.lower())
                    if match:
                        score = float(match.group(1))
                        return score if score <= 1.0 else score / 100.0
            except:
                pass
        
        return 0.5  # Default moderate confidence

    def _calculate_volume_percentile(self, data: pd.DataFrame, window: int = 252) -> float:
        """Calculate current volume percentile"""
        recent_data = data.tail(window)
        current_volume = data['volume'].iloc[-1]
        return (recent_data['volume'] <= current_volume).mean() * 100

    def _determine_volume_trend(self, data: pd.DataFrame, window: int = 20) -> str:
        """Determine overall volume trend"""
        recent_volume = data['volume'].tail(window)
        trend_slope = np.polyfit(range(len(recent_volume)), recent_volume, 1)[0]
        
        if trend_slope > recent_volume.mean() * 0.01:  # 1% of mean
            return "increasing"
        elif trend_slope < -recent_volume.mean() * 0.01:
            return "decreasing"
        else:
            return "stable"

    def _extract_agent_key_findings(self, agent_name: str, analysis_data: Dict[str, Any]) -> List[str]:
        """Extract key findings from agent analysis data"""
        findings = []
        
        try:
            if agent_name == 'volume_anomaly':
                anomalies = analysis_data.get('anomalies', [])
                if anomalies:
                    high_significance = [a for a in anomalies if a.get('significance') == 'high']
                    if high_significance:
                        findings.append(f"Detected {len(high_significance)} high-significance volume anomalies")
            
            elif agent_name == 'institutional_activity':
                activity_level = analysis_data.get('activity_level', 'unknown')
                if activity_level in ['high', 'very_high']:
                    findings.append(f"High institutional activity detected: {activity_level}")
            
            elif agent_name == 'volume_confirmation':
                confirmation = analysis_data.get('price_volume_confirmation', {})
                if confirmation.get('status') == 'confirmed':
                    findings.append("Price movements confirmed by volume analysis")
            
            elif agent_name == 'support_resistance':
                levels = analysis_data.get('key_levels', [])
                strong_levels = [l for l in levels if l.get('strength', 0) > 0.7]
                if strong_levels:
                    findings.append(f"Identified {len(strong_levels)} strong volume-based S/R levels")
            
            elif agent_name == 'volume_momentum':
                momentum = analysis_data.get('volume_momentum', {})
                if momentum.get('strength', 0) > 0.7:
                    findings.append(f"Strong volume momentum detected: {momentum.get('direction', 'unknown')}")
        
        except Exception as e:
            logger.warning(f"Failed to extract findings from {agent_name}: {e}")
        
        return findings

    def _extract_agent_signals(self, agent_name: str, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract trading signals from agent analysis data with sophisticated strength calculation"""
        signals = {'primary_signal': 'neutral', 'strength': 0.5, 'details': {}, 'confidence_factors': {}}
        
        try:
            # Calculate sophisticated signal strength based on multiple factors
            strength_calculation = self._calculate_sophisticated_signal_strength(
                agent_name, analysis_data
            )
            
            signals.update(strength_calculation)
            
            # Agent-specific signal extraction with enhanced logic
            if agent_name == 'volume_anomaly':
                anomaly_signals = self._extract_volume_anomaly_signals(analysis_data)
                signals.update(anomaly_signals)
            
            elif agent_name == 'institutional_activity':
                institutional_signals = self._extract_institutional_signals(analysis_data)
                signals.update(institutional_signals)
            
            elif agent_name == 'volume_confirmation':
                confirmation_signals = self._extract_volume_confirmation_signals(analysis_data)
                signals.update(confirmation_signals)
            
            elif agent_name == 'support_resistance':
                sr_signals = self._extract_support_resistance_signals(analysis_data)
                signals.update(sr_signals)
            
            elif agent_name == 'volume_momentum':
                momentum_signals = self._extract_momentum_signals(analysis_data)
                signals.update(momentum_signals)
            
            # Apply cross-validation and quality adjustments
            signals = self._apply_signal_quality_adjustments(agent_name, signals, analysis_data)
        
        except Exception as e:
            logger.warning(f"Failed to extract signals from {agent_name}: {e}")
            signals = {'primary_signal': 'neutral', 'strength': 0.3, 'details': {'error': str(e)}, 'confidence_factors': {}}
        
        return signals
    
    def _calculate_sophisticated_signal_strength(self, agent_name: str, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate sophisticated signal strength based on multiple factors"""
        strength_factors = {
            'data_quality_factor': 0.0,
            'historical_accuracy_factor': 0.0,
            'market_condition_factor': 0.0,
            'cross_validation_factor': 0.0,
            'volatility_adjustment': 0.0
        }
        
        # Data quality assessment (25% weight)
        data_quality = self._assess_data_quality_for_strength(agent_name, analysis_data)
        strength_factors['data_quality_factor'] = data_quality * 0.25
        
        # Historical accuracy factor (20% weight)
        historical_accuracy = self._get_agent_historical_accuracy(agent_name)
        strength_factors['historical_accuracy_factor'] = historical_accuracy * 0.20
        
        # Market condition alignment (20% weight)
        market_alignment = self._assess_market_condition_alignment(agent_name, analysis_data)
        strength_factors['market_condition_factor'] = market_alignment * 0.20
        
        # Cross-agent validation factor (20% weight)
        cross_validation = self._calculate_cross_agent_validation(agent_name, analysis_data)
        strength_factors['cross_validation_factor'] = cross_validation * 0.20
        
        # Volatility adjustment (15% weight)
        volatility_adjustment = self._calculate_volatility_adjustment(agent_name, analysis_data)
        strength_factors['volatility_adjustment'] = volatility_adjustment * 0.15
        
        # Calculate final strength
        base_strength = sum(strength_factors.values())
        adjusted_strength = max(0.0, min(1.0, base_strength))
        
        return {
            'strength': adjusted_strength,
            'strength_factors': strength_factors,
            'strength_quality': self._determine_strength_quality(adjusted_strength)
        }
    
    def _extract_volume_anomaly_signals(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract enhanced volume anomaly signals"""
        signals = {'primary_signal': 'neutral', 'strength': 0.5, 'details': {}}
        
        try:
            anomalies = analysis_data.get('anomalies', [])
            if not anomalies:
                return signals
            
            # Analyze anomaly patterns
            recent_anomalies = anomalies[:5]  # Most recent 5
            high_significance = [a for a in recent_anomalies if a.get('significance') == 'high']
            
            bullish_contexts = ['breakout', 'accumulation', 'buying_pressure']
            bearish_contexts = ['breakdown', 'distribution', 'selling_pressure']
            
            bullish_count = sum(1 for a in recent_anomalies if a.get('price_context') in bullish_contexts)
            bearish_count = sum(1 for a in recent_anomalies if a.get('price_context') in bearish_contexts)
            
            if bullish_count > bearish_count and bullish_count >= 2:
                signals['primary_signal'] = 'bullish'
                signals['strength'] = min(0.9, 0.6 + (bullish_count / len(recent_anomalies)) * 0.3)
            elif bearish_count > bullish_count and bearish_count >= 2:
                signals['primary_signal'] = 'bearish'
                signals['strength'] = min(0.9, 0.6 + (bearish_count / len(recent_anomalies)) * 0.3)
            
            # Enhance strength based on significance
            if high_significance:
                significance_boost = min(0.15, len(high_significance) * 0.05)
                signals['strength'] = min(1.0, signals['strength'] + significance_boost)
            
            signals['details'] = {
                'total_anomalies': len(anomalies),
                'recent_anomalies': len(recent_anomalies),
                'high_significance_count': len(high_significance),
                'bullish_contexts': bullish_count,
                'bearish_contexts': bearish_count
            }
            
        except Exception as e:
            logger.warning(f"Error extracting volume anomaly signals: {e}")
        
        return signals
    
    def _extract_institutional_signals(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract enhanced institutional activity signals"""
        signals = {'primary_signal': 'neutral', 'strength': 0.5, 'details': {}}
        
        try:
            activity_level = analysis_data.get('activity_level', 'normal')
            activity_pattern = analysis_data.get('activity_pattern', 'neutral')
            block_trades = analysis_data.get('block_trades', [])
            
            # Pattern-based signal determination
            if activity_pattern == 'accumulation':
                signals['primary_signal'] = 'bullish'
                base_strength = 0.75
            elif activity_pattern == 'distribution':
                signals['primary_signal'] = 'bearish'
                base_strength = 0.75
            else:
                base_strength = 0.5
            
            # Activity level adjustment
            level_multipliers = {
                'very_high': 1.2,
                'high': 1.1,
                'normal': 1.0,
                'low': 0.9,
                'very_low': 0.8
            }
            
            base_strength *= level_multipliers.get(activity_level, 1.0)
            
            # Block trade influence
            if block_trades:
                recent_blocks = [bt for bt in block_trades if bt.get('recent', False)]
                if recent_blocks:
                    block_boost = min(0.1, len(recent_blocks) * 0.02)
                    base_strength += block_boost
            
            signals['strength'] = max(0.0, min(1.0, base_strength))
            signals['details'] = {
                'activity_level': activity_level,
                'activity_pattern': activity_pattern,
                'block_trade_count': len(block_trades),
                'recent_block_trades': len([bt for bt in block_trades if bt.get('recent', False)])
            }
            
        except Exception as e:
            logger.warning(f"Error extracting institutional signals: {e}")
        
        return signals
    
    def _extract_volume_confirmation_signals(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract enhanced volume confirmation signals"""
        signals = {'primary_signal': 'neutral', 'strength': 0.5, 'details': {}}
        
        try:
            confirmation = analysis_data.get('price_volume_confirmation', {})
            status = confirmation.get('status', 'unconfirmed')
            trend_direction = confirmation.get('trend_direction', 'neutral')
            confirmation_strength = confirmation.get('strength', 0.5)
            
            if status == 'confirmed' and trend_direction != 'neutral':
                signals['primary_signal'] = 'bullish' if trend_direction == 'uptrend' else 'bearish'
                
                # Base strength from confirmation strength
                base_strength = 0.6 + (confirmation_strength * 0.3)
                
                # Volume trend consistency boost
                volume_trend = analysis_data.get('volume_trend_consistency', 0.5)
                consistency_boost = (volume_trend - 0.5) * 0.2
                
                signals['strength'] = max(0.0, min(1.0, base_strength + consistency_boost))
            
            elif status == 'divergence':
                # Negative signal strength for divergence
                signals['strength'] = max(0.2, 0.7 - confirmation_strength * 0.3)
            
            signals['details'] = {
                'confirmation_status': status,
                'trend_direction': trend_direction,
                'confirmation_strength': confirmation_strength,
                'volume_trend_consistency': analysis_data.get('volume_trend_consistency', 0.5)
            }
            
        except Exception as e:
            logger.warning(f"Error extracting volume confirmation signals: {e}")
        
        return signals
    
    def _extract_support_resistance_signals(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract enhanced support/resistance signals"""
        signals = {'primary_signal': 'neutral', 'strength': 0.5, 'details': {}}
        
        try:
            key_levels = analysis_data.get('key_levels', [])
            level_tests = analysis_data.get('level_tests', [])
            breakout_potential = analysis_data.get('breakout_potential', {})
            
            strong_levels = [l for l in key_levels if l.get('strength', 0) > 0.7]
            recent_tests = [t for t in level_tests if t.get('recent', False)]
            
            if breakout_potential:
                direction = breakout_potential.get('direction', 'neutral')
                probability = breakout_potential.get('probability', 0.5)
                
                if direction in ['upward', 'downward'] and probability > 0.6:
                    signals['primary_signal'] = 'bullish' if direction == 'upward' else 'bearish'
                    
                    # Base strength from breakout probability
                    base_strength = 0.5 + (probability - 0.5) * 0.8
                    
                    # Strong level penalty/bonus
                    if strong_levels:
                        level_adjustment = len(strong_levels) * 0.05
                        if direction == 'upward':
                            base_strength += level_adjustment  # Support helps bullish breakout
                        else:
                            base_strength += level_adjustment  # Resistance helps bearish breakdown
                    
                    signals['strength'] = max(0.0, min(1.0, base_strength))
            
            signals['details'] = {
                'total_levels': len(key_levels),
                'strong_levels': len(strong_levels),
                'recent_tests': len(recent_tests),
                'breakout_direction': breakout_potential.get('direction', 'neutral'),
                'breakout_probability': breakout_potential.get('probability', 0.5)
            }
            
        except Exception as e:
            logger.warning(f"Error extracting support/resistance signals: {e}")
        
        return signals
    
    def _extract_momentum_signals(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract enhanced volume momentum signals"""
        signals = {'primary_signal': 'neutral', 'strength': 0.5, 'details': {}}
        
        try:
            momentum = analysis_data.get('volume_momentum', {})
            direction = momentum.get('direction', 'neutral')
            strength = momentum.get('strength', 0.5)
            acceleration = momentum.get('acceleration', 0.0)
            sustainability = momentum.get('sustainability', 0.5)
            
            if direction in ['increasing', 'decreasing'] and strength > 0.6:
                signals['primary_signal'] = 'bullish' if direction == 'increasing' else 'bearish'
                
                # Base strength from momentum strength
                base_strength = 0.5 + (strength - 0.5) * 0.8
                
                # Acceleration adjustment
                accel_adjustment = acceleration * 0.1  # Can be negative for deceleration
                
                # Sustainability bonus
                sustainability_bonus = (sustainability - 0.5) * 0.2
                
                final_strength = base_strength + accel_adjustment + sustainability_bonus
                signals['strength'] = max(0.0, min(1.0, final_strength))
            
            signals['details'] = {
                'momentum_direction': direction,
                'momentum_strength': strength,
                'acceleration': acceleration,
                'sustainability': sustainability
            }
            
        except Exception as e:
            logger.warning(f"Error extracting momentum signals: {e}")
        
        return signals
    
    def _assess_data_quality_for_strength(self, agent_name: str, analysis_data: Dict[str, Any]) -> float:
        """Assess data quality for signal strength calculation"""
        quality_score = 0.8  # Base quality
        
        try:
            # Check data completeness
            if not analysis_data or len(analysis_data) == 0:
                return 0.2
            
            # Agent-specific quality checks
            if agent_name == 'volume_anomaly':
                anomalies = analysis_data.get('anomalies', [])
                if len(anomalies) == 0:
                    quality_score -= 0.3
                elif len(anomalies) > 20:  # Too many may indicate noise
                    quality_score -= 0.1
            
            elif agent_name == 'institutional_activity':
                if 'activity_level' not in analysis_data:
                    quality_score -= 0.2
                if 'activity_pattern' not in analysis_data:
                    quality_score -= 0.2
            
            elif agent_name == 'volume_confirmation':
                confirmation = analysis_data.get('price_volume_confirmation', {})
                if not confirmation:
                    quality_score -= 0.3
            
            # Check for error indicators
            if analysis_data.get('error') or analysis_data.get('processing_error'):
                quality_score -= 0.4
            
            # Check data recency (if timestamp available)
            timestamp = analysis_data.get('timestamp')
            if timestamp:
                try:
                    from datetime import datetime, timedelta
                    data_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    age_hours = (datetime.now() - data_time).total_seconds() / 3600
                    if age_hours > 24:  # Data older than 24 hours
                        quality_score -= min(0.2, age_hours / 240)  # Gradual penalty up to 10 days
                except:
                    pass
            
        except Exception as e:
            logger.warning(f"Error assessing data quality for {agent_name}: {e}")
            quality_score = 0.5
        
        return max(0.0, min(1.0, quality_score))
    
    def _get_agent_historical_accuracy(self, agent_name: str) -> float:
        """Get historical accuracy factor for agent (placeholder)"""
        # In production, this would analyze historical prediction accuracy
        historical_accuracy_map = {
            'volume_anomaly': 0.72,
            'institutional_activity': 0.85,
            'volume_confirmation': 0.78,
            'support_resistance': 0.81,
            'volume_momentum': 0.75
        }
        
        return historical_accuracy_map.get(agent_name, 0.7)
    
    def _assess_market_condition_alignment(self, agent_name: str, analysis_data: Dict[str, Any]) -> float:
        """Assess how well agent's analysis aligns with current market conditions"""
        alignment_score = 0.7  # Base alignment
        
        try:
            # Market condition factors (simplified)
            # In production, this would use real market data
            
            # Volume-based agents perform better in high volatility
            if agent_name in ['volume_anomaly', 'institutional_activity']:
                # Assume higher alignment during volatile periods
                volatility_factor = 0.8  # Would be calculated from actual data
                alignment_score = 0.6 + (volatility_factor * 0.3)
            
            # Trend-based agents perform better in trending markets
            elif agent_name in ['volume_confirmation', 'volume_momentum']:
                trend_strength = 0.6  # Would be calculated from actual data
                alignment_score = 0.6 + (trend_strength * 0.3)
            
            # Support/resistance agents perform consistently
            elif agent_name == 'support_resistance':
                alignment_score = 0.75  # Generally stable performance
            
            # Check for specific market condition indicators in data
            market_stress = analysis_data.get('market_stress_indicator', 0.3)
            if market_stress > 0.7:
                # High stress may reduce reliability for some agents
                if agent_name in ['volume_confirmation', 'volume_momentum']:
                    alignment_score *= 0.9
                else:
                    alignment_score *= 1.1  # Volume agents may be more reliable in stress
            
        except Exception as e:
            logger.warning(f"Error assessing market alignment for {agent_name}: {e}")
            alignment_score = 0.7
        
        return max(0.0, min(1.0, alignment_score))
    
    def _calculate_cross_agent_validation(self, agent_name: str, analysis_data: Dict[str, Any]) -> float:
        """Calculate cross-agent validation factor"""
        validation_score = 0.6  # Base validation
        
        try:
            # This would compare current agent's signal with other agents' recent signals
            # For now, use simplified logic based on signal consistency patterns
            
            signal_strength = analysis_data.get('signal_strength', 0.5)
            signal_clarity = analysis_data.get('signal_clarity', 0.5)
            
            # Higher strength and clarity suggest better cross-validation potential
            validation_score = 0.4 + (signal_strength * 0.3) + (signal_clarity * 0.3)
            
            # Agent-specific adjustments based on typical cross-validation performance
            cross_validation_factors = {
                'institutional_activity': 1.2,  # Usually correlated well with others
                'volume_confirmation': 1.1,     # Good confirmation agent
                'volume_anomaly': 0.9,          # Sometimes gives false signals
                'support_resistance': 1.0,       # Neutral
                'volume_momentum': 1.0           # Neutral
            }
            
            factor = cross_validation_factors.get(agent_name, 1.0)
            validation_score *= factor
            
        except Exception as e:
            logger.warning(f"Error calculating cross-validation for {agent_name}: {e}")
            validation_score = 0.6
        
        return max(0.0, min(1.0, validation_score))
    
    def _calculate_volatility_adjustment(self, agent_name: str, analysis_data: Dict[str, Any]) -> float:
        """Calculate volatility-based adjustment factor"""
        adjustment = 0.0
        
        try:
            # Get volatility indicators from analysis data
            volatility_indicator = analysis_data.get('volatility_regime', 'normal')
            
            # Agent-specific volatility responses
            if volatility_indicator == 'high':
                if agent_name == 'volume_anomaly':
                    adjustment = 0.1  # Volume anomalies more reliable in high volatility
                elif agent_name == 'institutional_activity':
                    adjustment = 0.15  # Institutional activity clearer in volatile markets
                elif agent_name == 'support_resistance':
                    adjustment = -0.05  # S/R levels may be less reliable
                else:
                    adjustment = 0.0  # Neutral for others
            
            elif volatility_indicator == 'low':
                if agent_name == 'volume_momentum':
                    adjustment = -0.1  # Momentum less clear in low volatility
                elif agent_name == 'volume_confirmation':
                    adjustment = -0.05  # Confirmation signals weaker
                else:
                    adjustment = 0.0
            
            # Volume regime adjustments
            volume_regime = analysis_data.get('volume_regime', 'normal')
            if volume_regime == 'high' and 'volume' in agent_name:
                adjustment += 0.05  # Volume-based agents benefit from high volume
            elif volume_regime == 'low' and 'volume' in agent_name:
                adjustment -= 0.05  # Volume-based agents suffer from low volume
            
        except Exception as e:
            logger.warning(f"Error calculating volatility adjustment for {agent_name}: {e}")
        
        return max(-0.2, min(0.2, adjustment))  # Cap adjustments
    
    def _determine_strength_quality(self, strength: float) -> str:
        """Determine qualitative strength rating"""
        if strength >= 0.9:
            return 'excellent'
        elif strength >= 0.8:
            return 'high'
        elif strength >= 0.6:
            return 'good'
        elif strength >= 0.4:
            return 'moderate'
        elif strength >= 0.2:
            return 'weak'
        else:
            return 'very_weak'
    
    def _apply_signal_quality_adjustments(self, agent_name: str, signals: Dict[str, Any], 
                                        analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply final quality adjustments to signals"""
        try:
            # Get current strength
            current_strength = signals.get('strength', 0.5)
            
            # Apply confidence factor adjustments
            confidence_factors = signals.get('confidence_factors', {})
            
            # Data quality impact
            data_quality = confidence_factors.get('data_quality_factor', 0.0)
            if data_quality < 0.1:  # Very poor data quality
                current_strength *= 0.7  # Significant penalty
            elif data_quality > 0.2:  # Good data quality
                current_strength *= 1.05  # Small bonus
            
            # Historical accuracy impact
            historical_accuracy = confidence_factors.get('historical_accuracy_factor', 0.0)
            if historical_accuracy < 0.1:  # Poor historical performance
                current_strength *= 0.8
            elif historical_accuracy > 0.16:  # Good historical performance (20% * 0.8 threshold)
                current_strength *= 1.03
            
            # Final strength bounds check
            signals['strength'] = max(0.0, min(1.0, current_strength))
            
            # Add quality metadata
            signals['quality_adjusted'] = True
            signals['original_strength'] = signals.get('strength', 0.5)
            signals['quality_rating'] = self._determine_strength_quality(signals['strength'])
            
        except Exception as e:
            logger.warning(f"Error applying quality adjustments for {agent_name}: {e}")
        
        return signals
    
    def _calculate_dynamic_weights(self, results: Dict[str, VolumeAgentResult]) -> Dict[str, float]:
        """Calculate dynamic weights based on current performance and market conditions"""
        dynamic_weights = {}
        
        try:
            # Get base weights from configuration
            base_weights = {name: config.get('weight', 0.2) for name, config in self.agent_config.items()}
            
            # Calculate performance-based adjustments
            performance_adjustments = self._calculate_performance_based_adjustments(results)
            
            # Calculate market condition adjustments
            market_adjustments = self._calculate_market_condition_weight_adjustments(results)
            
            # Calculate volatility regime adjustments
            volatility_adjustments = self._calculate_volatility_weight_adjustments(results)
            
            # Calculate historical accuracy adjustments
            accuracy_adjustments = self._calculate_historical_accuracy_adjustments()
            
            # Combine all adjustments
            for agent_name in base_weights.keys():
                base_weight = base_weights[agent_name]
                
                # Apply multiplicative adjustments
                performance_factor = performance_adjustments.get(agent_name, 1.0)
                market_factor = market_adjustments.get(agent_name, 1.0)
                volatility_factor = volatility_adjustments.get(agent_name, 1.0)
                accuracy_factor = accuracy_adjustments.get(agent_name, 1.0)
                
                # Calculate adjusted weight
                adjusted_weight = base_weight * performance_factor * market_factor * \
                                volatility_factor * accuracy_factor
                
                # Apply bounds (weights should be between 0.05 and 0.4)
                adjusted_weight = max(0.05, min(0.4, adjusted_weight))
                
                dynamic_weights[agent_name] = adjusted_weight
            
            # Normalize weights to sum to 1.0
            total_weight = sum(dynamic_weights.values())
            if total_weight > 0:
                for agent_name in dynamic_weights:
                    dynamic_weights[agent_name] /= total_weight
            
        except Exception as e:
            logger.warning(f"Error calculating dynamic weights: {e}")
            # Fallback to base weights
            dynamic_weights = {name: config.get('weight', 0.2) for name, config in self.agent_config.items()}
        
        return dynamic_weights
    
    def _calculate_performance_based_adjustments(self, results: Dict[str, VolumeAgentResult]) -> Dict[str, float]:
        """Calculate performance-based weight adjustments"""
        adjustments = {}
        
        try:
            # Analyze processing times
            processing_times = {name: result.processing_time for name, result in results.items() if result.success}
            avg_processing_time = sum(processing_times.values()) / len(processing_times) if processing_times else 30.0
            
            # Analyze confidence scores
            confidence_scores = {name: result.confidence_score or 0.5 for name, result in results.items() if result.success}
            avg_confidence = sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0.5
            
            for agent_name in self.agent_config.keys():
                adjustment_factor = 1.0
                
                if agent_name in results and results[agent_name].success:
                    result = results[agent_name]
                    
                    # Processing time adjustment (faster = better)
                    if result.processing_time < avg_processing_time * 0.5:
                        adjustment_factor *= 1.1  # 10% bonus for fast processing
                    elif result.processing_time > avg_processing_time * 2.0:
                        adjustment_factor *= 0.9  # 10% penalty for slow processing
                    
                    # Confidence adjustment
                    confidence = result.confidence_score or 0.5
                    if confidence > avg_confidence * 1.2:
                        adjustment_factor *= 1.15  # 15% bonus for high confidence
                    elif confidence < avg_confidence * 0.8:
                        adjustment_factor *= 0.85  # 15% penalty for low confidence
                    
                    # Data quality assessment
                    if hasattr(result, 'analysis_data') and result.analysis_data:
                        data_quality = self._assess_agent_data_quality(agent_name, result)
                        if data_quality > 0.8:
                            adjustment_factor *= 1.05  # 5% bonus for high data quality
                        elif data_quality < 0.6:
                            adjustment_factor *= 0.95  # 5% penalty for low data quality
                
                else:
                    # Failed agents get significant penalty
                    adjustment_factor = 0.3
                
                adjustments[agent_name] = adjustment_factor
            
        except Exception as e:
            logger.warning(f"Error calculating performance adjustments: {e}")
            adjustments = {name: 1.0 for name in self.agent_config.keys()}
        
        return adjustments
    
    def _calculate_market_condition_weight_adjustments(self, results: Dict[str, VolumeAgentResult]) -> Dict[str, float]:
        """Calculate market condition-based weight adjustments"""
        adjustments = {}
        
        try:
            # Determine current market regime (simplified)
            market_regime = self._determine_current_market_regime(results)
            
            # Agent performance in different market regimes
            regime_performance = {
                'high_volatility': {
                    'volume_anomaly': 1.2,
                    'institutional_activity': 1.3,
                    'volume_confirmation': 0.9,
                    'support_resistance': 0.8,
                    'volume_momentum': 1.0
                },
                'low_volatility': {
                    'volume_anomaly': 0.8,
                    'institutional_activity': 0.9,
                    'volume_confirmation': 1.1,
                    'support_resistance': 1.2,
                    'volume_momentum': 0.9
                },
                'trending': {
                    'volume_anomaly': 1.0,
                    'institutional_activity': 1.1,
                    'volume_confirmation': 1.3,
                    'support_resistance': 0.9,
                    'volume_momentum': 1.2
                },
                'sideways': {
                    'volume_anomaly': 0.9,
                    'institutional_activity': 1.0,
                    'volume_confirmation': 0.8,
                    'support_resistance': 1.3,
                    'volume_momentum': 0.7
                },
                'normal': {
                    'volume_anomaly': 1.0,
                    'institutional_activity': 1.0,
                    'volume_confirmation': 1.0,
                    'support_resistance': 1.0,
                    'volume_momentum': 1.0
                }
            }
            
            # Get adjustments for current regime
            regime_adjustments = regime_performance.get(market_regime, regime_performance['normal'])
            
            for agent_name in self.agent_config.keys():
                adjustments[agent_name] = regime_adjustments.get(agent_name, 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating market condition adjustments: {e}")
            adjustments = {name: 1.0 for name in self.agent_config.keys()}
        
        return adjustments
    
    def _calculate_volatility_weight_adjustments(self, results: Dict[str, VolumeAgentResult]) -> Dict[str, float]:
        """Calculate volatility-based weight adjustments"""
        adjustments = {}
        
        try:
            # Estimate volatility from results
            volatility_level = self._estimate_volatility_level(results)
            
            # Volatility-based adjustments
            if volatility_level == 'high':
                adjustments = {
                    'volume_anomaly': 1.15,      # Volume spikes more significant
                    'institutional_activity': 1.25,  # Institutional activity clearer
                    'volume_confirmation': 0.85,     # Confirmations less reliable
                    'support_resistance': 0.75,      # S/R levels break more often
                    'volume_momentum': 1.05          # Momentum more pronounced
                }
            elif volatility_level == 'low':
                adjustments = {
                    'volume_anomaly': 0.85,          # Fewer meaningful anomalies
                    'institutional_activity': 0.9,   # Less institutional activity
                    'volume_confirmation': 1.15,     # Confirmations more reliable
                    'support_resistance': 1.2,       # S/R levels hold better
                    'volume_momentum': 0.8           # Momentum less clear
                }
            else:  # normal volatility
                adjustments = {name: 1.0 for name in self.agent_config.keys()}
            
        except Exception as e:
            logger.warning(f"Error calculating volatility adjustments: {e}")
            adjustments = {name: 1.0 for name in self.agent_config.keys()}
        
        return adjustments
    
    def _calculate_historical_accuracy_adjustments(self) -> Dict[str, float]:
        """Calculate historical accuracy-based weight adjustments"""
        adjustments = {}
        
        try:
            # Get historical accuracy data (placeholder)
            historical_accuracies = {
                'volume_anomaly': 0.72,
                'institutional_activity': 0.85,
                'volume_confirmation': 0.78,
                'support_resistance': 0.81,
                'volume_momentum': 0.75
            }
            
            # Convert to adjustment factors
            base_accuracy = 0.75  # Baseline accuracy
            
            for agent_name, accuracy in historical_accuracies.items():
                # Higher accuracy gets higher weight
                accuracy_ratio = accuracy / base_accuracy
                adjustment_factor = 0.8 + (accuracy_ratio - 1.0) * 0.4  # Range roughly 0.6 to 1.2
                adjustments[agent_name] = max(0.6, min(1.3, adjustment_factor))
            
        except Exception as e:
            logger.warning(f"Error calculating historical accuracy adjustments: {e}")
            adjustments = {name: 1.0 for name in self.agent_config.keys()}
        
        return adjustments
    
    def _apply_confidence_adjustments(self, agent_name: str, result: VolumeAgentResult, 
                                    confidence: float, dynamic_weights: Dict[str, float]) -> float:
        """Apply sophisticated confidence adjustments"""
        try:
            adjusted_confidence = confidence
            
            # Processing time penalty (more sophisticated)
            processing_time = result.processing_time
            if processing_time > 0:
                # Logarithmic penalty for very long processing times
                time_penalty = min(0.15, np.log(processing_time / 10.0) * 0.05) if processing_time > 10 else 0
                adjusted_confidence = max(0.0, adjusted_confidence - time_penalty)
            
            # Error-based penalty
            if result.error_message:
                adjusted_confidence *= 0.7  # 30% penalty for errors
            
            # Chart availability bonus (indicates successful processing)
            if result.chart_image:
                adjusted_confidence *= 1.05  # 5% bonus for chart generation
            
            # LLM response quality factor
            if result.llm_response:
                response_quality = self._assess_llm_response_quality(result.llm_response)
                adjusted_confidence *= (0.95 + response_quality * 0.1)  # 5% penalty to 5% bonus
            
            # Weight-based adjustment (higher weight agents should have higher confidence requirements)
            agent_weight = dynamic_weights.get(agent_name, 0.2)
            if agent_weight > 0.25:  # High-weight agent
                adjusted_confidence *= 1.02  # Small bonus for important agents
            elif agent_weight < 0.15:  # Low-weight agent
                adjusted_confidence *= 0.98  # Small penalty for less important agents
            
            return max(0.0, min(1.0, adjusted_confidence))
            
        except Exception as e:
            logger.warning(f"Error applying confidence adjustments for {agent_name}: {e}")
            return confidence
    
    def _calculate_quality_adjustments(self, results: Dict[str, VolumeAgentResult], 
                                     successful_agents: int) -> Dict[str, float]:
        """Calculate enhanced quality adjustment factors"""
        adjustments = {
            'success_factor': 1.0,
            'conflict_factor': 1.0,
            'consistency_factor': 1.0
        }
        
        try:
            total_agents = len(results)
            success_rate = successful_agents / total_agents if total_agents > 0 else 0.0
            
            # Enhanced success rate factor
            if success_rate >= 0.8:
                adjustments['success_factor'] = 1.05  # Bonus for high success
            elif success_rate >= 0.6:
                adjustments['success_factor'] = 1.0   # Neutral
            elif success_rate >= 0.4:
                adjustments['success_factor'] = 0.9   # Moderate penalty
            else:
                adjustments['success_factor'] = 0.7   # High penalty for low success
            
            # Enhanced conflict factor
            conflicts = self._identify_conflicting_signals(results)
            if conflicts:
                conflict_severity = sum(c.get('severity_score', 0) for c in conflicts) / len(conflicts)
                conflict_count_factor = min(0.4, len(conflicts) * 0.1)
                severity_factor = conflict_severity * 0.3
                total_conflict_penalty = conflict_count_factor + severity_factor
                adjustments['conflict_factor'] = max(0.5, 1.0 - total_conflict_penalty)
            
            # Consistency factor (how consistent are the successful agents)
            if successful_agents >= 2:
                consistency_score = self._calculate_agent_consistency(results)
                adjustments['consistency_factor'] = 0.9 + (consistency_score * 0.2)  # 0.9 to 1.1 range
            
        except Exception as e:
            logger.warning(f"Error calculating quality adjustments: {e}")
        
        return adjustments
    
    def _calculate_temporal_confidence_factors(self, results: Dict[str, VolumeAgentResult]) -> Dict[str, float]:
        """Calculate temporal factors including time decay and market conditions"""
        factors = {
            'decay_factor': 1.0,
            'market_factor': 1.0,
            'recency_factor': 1.0
        }
        
        try:
            # Time decay factor (if we have timestamp information)
            current_time = time.time()
            time_penalties = []
            
            for result in results.values():
                if result.success and hasattr(result, 'timestamp'):
                    try:
                        result_time = getattr(result, 'timestamp', current_time)
                        age_minutes = (current_time - result_time) / 60.0
                        # Gradual decay over time (5% penalty per hour)
                        decay_penalty = min(0.3, age_minutes * 0.0008)  # Max 30% penalty
                        time_penalties.append(decay_penalty)
                    except:
                        pass
            
            if time_penalties:
                avg_decay = sum(time_penalties) / len(time_penalties)
                factors['decay_factor'] = max(0.7, 1.0 - avg_decay)
            
            # Market condition factor
            market_regime = self._determine_current_market_regime(results)
            market_stress = self._estimate_market_stress_level(results)
            
            if market_regime in ['high_volatility', 'trending']:
                factors['market_factor'] = 1.05  # Slight bonus for clear market conditions
            elif market_regime == 'sideways':
                factors['market_factor'] = 0.95  # Slight penalty for unclear conditions
            
            # Market stress adjustment
            if market_stress > 0.7:
                factors['market_factor'] *= 0.9  # Additional penalty for high stress
            elif market_stress < 0.3:
                factors['market_factor'] *= 1.05  # Bonus for low stress
            
        except Exception as e:
            logger.warning(f"Error calculating temporal factors: {e}")
        
        return factors
    
    def _determine_current_market_regime(self, results: Dict[str, VolumeAgentResult]) -> str:
        """Determine current market regime from agent results"""
        try:
            # Simplified market regime detection based on agent signals
            successful_results = [r for r in results.values() if r.success]
            
            if not successful_results:
                return 'normal'
            
            # Look for regime indicators in agent data
            volatility_indicators = []
            trend_indicators = []
            
            for result in successful_results:
                if result.analysis_data:
                    # Volume anomaly agent can indicate volatility
                    if hasattr(result, 'agent_name') and 'anomaly' in getattr(result, 'agent_name', ''):
                        anomalies = result.analysis_data.get('anomalies', [])
                        high_sig_anomalies = [a for a in anomalies if a.get('significance') == 'high']
                        if len(high_sig_anomalies) > 3:
                            volatility_indicators.append('high')
                        elif len(anomalies) == 0:
                            volatility_indicators.append('low')
                    
                    # Look for trend indicators
                    if 'trend_direction' in result.analysis_data:
                        trend_dir = result.analysis_data['trend_direction']
                        if trend_dir in ['uptrend', 'downtrend']:
                            trend_indicators.append('trending')
                        elif trend_dir == 'sideways':
                            trend_indicators.append('sideways')
            
            # Determine regime
            if volatility_indicators.count('high') > volatility_indicators.count('low'):
                return 'high_volatility'
            elif volatility_indicators.count('low') > volatility_indicators.count('high'):
                return 'low_volatility'
            elif trend_indicators.count('trending') > trend_indicators.count('sideways'):
                return 'trending'
            elif trend_indicators.count('sideways') > trend_indicators.count('trending'):
                return 'sideways'
            else:
                return 'normal'
            
        except Exception as e:
            logger.warning(f"Error determining market regime: {e}")
            return 'normal'
    
    def _estimate_volatility_level(self, results: Dict[str, VolumeAgentResult]) -> str:
        """Estimate current volatility level from results"""
        try:
            volatility_score = 0.0
            score_count = 0
            
            for result in results.values():
                if result.success and result.analysis_data:
                    # Volume anomaly agent
                    anomalies = result.analysis_data.get('anomalies', [])
                    if anomalies:
                        high_sig = len([a for a in anomalies if a.get('significance') == 'high'])
                        volatility_score += min(1.0, high_sig / 5.0)  # Normalize to 0-1
                        score_count += 1
                    
                    # Institutional activity
                    activity_level = result.analysis_data.get('activity_level', 'normal')
                    if activity_level in ['very_high', 'high']:
                        volatility_score += 0.8
                    elif activity_level in ['very_low', 'low']:
                        volatility_score += 0.2
                    else:
                        volatility_score += 0.5
                    score_count += 1
            
            if score_count > 0:
                avg_volatility = volatility_score / score_count
                if avg_volatility > 0.7:
                    return 'high'
                elif avg_volatility < 0.4:
                    return 'low'
            
            return 'normal'
            
        except Exception as e:
            logger.warning(f"Error estimating volatility level: {e}")
            return 'normal'
    
    def _estimate_market_stress_level(self, results: Dict[str, VolumeAgentResult]) -> float:
        """Estimate market stress level (0.0 to 1.0)"""
        try:
            stress_indicators = []
            
            for result in results.values():
                if result.success and result.analysis_data:
                    # High processing times may indicate market stress
                    if result.processing_time > 45:
                        stress_indicators.append(0.7)
                    
                    # Multiple conflicts indicate uncertainty/stress
                    if result.error_message:
                        stress_indicators.append(0.8)
                    
                    # Volume anomalies may indicate stress
                    anomalies = result.analysis_data.get('anomalies', [])
                    extreme_anomalies = [a for a in anomalies if a.get('significance') == 'extreme']
                    if extreme_anomalies:
                        stress_indicators.append(0.6)
            
            if stress_indicators:
                return sum(stress_indicators) / len(stress_indicators)
            
            return 0.3  # Default moderate stress
            
        except Exception as e:
            logger.warning(f"Error estimating market stress: {e}")
            return 0.5
    
    def _calculate_agent_consistency(self, results: Dict[str, VolumeAgentResult]) -> float:
        """Calculate consistency score across successful agents"""
        try:
            successful_results = {k: v for k, v in results.items() if v.success}
            
            if len(successful_results) < 2:
                return 0.5  # Cannot assess consistency with < 2 agents
            
            # Extract signals from all successful agents
            agent_signals = {}
            for agent_name, result in successful_results.items():
                signals = self._extract_agent_signals(agent_name, result.analysis_data)
                agent_signals[agent_name] = signals
            
            # Calculate signal agreement
            signal_types = ['bullish', 'bearish', 'neutral']
            signal_counts = {sig_type: 0 for sig_type in signal_types}
            
            for signals in agent_signals.values():
                primary_signal = signals.get('primary_signal', 'neutral')
                signal_counts[primary_signal] += 1
            
            # Calculate consistency as the maximum agreement percentage
            total_agents = len(agent_signals)
            max_agreement = max(signal_counts.values())
            consistency_score = max_agreement / total_agents
            
            # Bonus for strength consistency
            strengths = [signals.get('strength', 0.5) for signals in agent_signals.values()]
            if strengths:
                strength_variance = np.var(strengths) if len(strengths) > 1 else 0
                strength_consistency = max(0.0, 1.0 - strength_variance * 2)  # Lower variance = higher consistency
                consistency_score = (consistency_score + strength_consistency) / 2
            
            return max(0.0, min(1.0, consistency_score))
            
        except Exception as e:
            logger.warning(f"Error calculating agent consistency: {e}")
            return 0.5
    
    def _assess_llm_response_quality(self, llm_response: str) -> float:
        """Assess the quality of LLM response (0.0 to 1.0)"""
        try:
            if not llm_response or len(llm_response.strip()) == 0:
                return 0.0
            
            quality_score = 0.5  # Base quality
            
            # Length factor (not too short, not too long)
            response_length = len(llm_response.strip())
            if 100 <= response_length <= 1000:
                quality_score += 0.1
            elif response_length < 50 or response_length > 2000:
                quality_score -= 0.1
            
            # Keyword presence (indicates structured analysis)
            analysis_keywords = ['analysis', 'trend', 'volume', 'signal', 'confidence', 'risk']
            keyword_count = sum(1 for keyword in analysis_keywords if keyword.lower() in llm_response.lower())
            quality_score += min(0.2, keyword_count * 0.05)
            
            # Structure indicators
            if any(marker in llm_response for marker in ['1.', '2.', '•', '-']):
                quality_score += 0.1  # Bonus for structured format
            
            # Avoid obvious error patterns
            error_patterns = ['error', 'failed', 'unable to', 'cannot analyze']
            if any(pattern in llm_response.lower() for pattern in error_patterns):
                quality_score -= 0.2
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.warning(f"Error assessing LLM response quality: {e}")
            return 0.5

    def _count_signal_consensus(self, agent_signals: Dict[str, Dict[str, Any]], signal_type: str) -> int:
        """Count how many agents agree on a signal type"""
        count = 0
        for signals in agent_signals.values():
            if signals.get('primary_signal') == signal_type:
                count += 1
        return count

    def _detect_enhanced_signal_conflict(self, agent1: str, result1: VolumeAgentResult,
                                        agent2: str, result2: VolumeAgentResult) -> Optional[Dict[str, Any]]:
        """Advanced conflict detection with multi-dimensional analysis and temporal patterns"""
        signals1 = self._extract_agent_signals(agent1, result1.analysis_data)
        signals2 = self._extract_agent_signals(agent2, result2.analysis_data)
        
        sig1 = signals1.get('primary_signal', 'neutral')
        sig2 = signals2.get('primary_signal', 'neutral')
        strength1 = signals1.get('strength', 0.5)
        strength2 = signals2.get('strength', 0.5)
        
        # Enhanced conflict detection with multiple conflict types
        conflict_type = self._determine_conflict_type(sig1, sig2, strength1, strength2)
        
        if conflict_type == 'none':
            return None
        
        # Multi-dimensional conflict analysis
        conflict_analysis = self._analyze_multi_dimensional_conflict(
            agent1, result1, signals1,
            agent2, result2, signals2
        )
        
        # Temporal pattern analysis
        temporal_factors = self._analyze_temporal_conflict_patterns(
            agent1, agent2, sig1, sig2, strength1, strength2
        )
        
        # Calculate enhanced severity score with multiple factors
        severity_score = self._calculate_enhanced_conflict_severity(
            conflict_analysis, temporal_factors, agent1, agent2
        )
        
        # Market context analysis
        market_context = self._analyze_conflict_market_context(
            conflict_analysis, temporal_factors
        )
        
        return {
            'type': conflict_type,
            'conflict_category': self._categorize_enhanced_conflict(agent1, agent2, conflict_analysis),
            'agents': {
                agent1: {
                    'signal': sig1,
                    'strength': strength1,
                    'confidence': result1.confidence_score or 0.5,
                    'weight': self.agent_config.get(agent1, {}).get('weight', 0.2),
                    'processing_time': result1.processing_time,
                    'data_quality': conflict_analysis['agent1_data_quality'],
                    'signal_stability': conflict_analysis['agent1_stability']
                },
                agent2: {
                    'signal': sig2,
                    'strength': strength2,
                    'confidence': result2.confidence_score or 0.5,
                    'weight': self.agent_config.get(agent2, {}).get('weight', 0.2),
                    'processing_time': result2.processing_time,
                    'data_quality': conflict_analysis['agent2_data_quality'],
                    'signal_stability': conflict_analysis['agent2_stability']
                }
            },
            'severity_score': severity_score,
            'severity_level': self._get_enhanced_severity_level(severity_score),
            'conflict_dimensions': conflict_analysis['dimensions'],
            'temporal_patterns': temporal_factors,
            'market_context': market_context,
            'impact_assessment': self._assess_enhanced_conflict_impact(
                agent1, agent2, severity_score, conflict_analysis, market_context
            ),
            'resolution_priority': self._calculate_resolution_priority(
                severity_score, conflict_analysis, market_context
            )
        }
    
    def _categorize_conflict(self, agent1: str, agent2: str) -> str:
        """Categorize the type of conflict based on agent types"""
        # Define agent categories
        categories = {
            'volume_anomaly': 'statistical',
            'institutional_activity': 'institutional', 
            'volume_confirmation': 'trend',
            'support_resistance': 'levels',
            'volume_momentum': 'momentum'
        }
        
        cat1 = categories.get(agent1, 'unknown')
        cat2 = categories.get(agent2, 'unknown')
        
        # Define conflict category combinations
        if cat1 == cat2:
            return f'{cat1}_internal'
        elif {cat1, cat2} == {'statistical', 'institutional'}:
            return 'retail_vs_institutional'
        elif {cat1, cat2} == {'trend', 'momentum'}:
            return 'trend_momentum_divergence'
        elif {cat1, cat2} == {'levels', 'trend'}:
            return 'levels_vs_trend'
        else:
            return f'{cat1}_vs_{cat2}'
    
    def _get_severity_level(self, severity_score: float) -> str:
        """Determine severity level from score"""
        if severity_score > 0.7:
            return 'critical'
        elif severity_score > 0.5:
            return 'high'
        elif severity_score > 0.3:
            return 'medium'
        else:
            return 'low'
    
    def _assess_conflict_impact(self, agent1: str, agent2: str, severity_score: float) -> Dict[str, Any]:
        """Assess the impact of the conflict on overall analysis"""
        weight1 = self.agent_config.get(agent1, {}).get('weight', 0.2)
        weight2 = self.agent_config.get(agent2, {}).get('weight', 0.2)
        combined_weight = weight1 + weight2
        
        return {
            'overall_impact': 'high' if combined_weight > 0.4 else 'medium' if combined_weight > 0.25 else 'low',
            'affected_weight_percentage': combined_weight * 100,
            'confidence_impact': severity_score * combined_weight,
            'recommendation_impact': 'significant' if severity_score > 0.6 and combined_weight > 0.3 else 'moderate'
        }
    
    def _suggest_conflict_resolution(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest resolution strategies for conflicts"""
        severity_level = conflict.get('severity_level', 'low')
        conflict_category = conflict.get('conflict_category', 'unknown')
        agents = conflict.get('agents', {})
        
        resolution = {
            'strategy': 'default',
            'primary_recommendation': 'monitor_closely',
            'confidence_adjustment': 0.0,
            'explanation': 'Unknown conflict type'
        }
        
        if severity_level == 'critical':
            resolution.update({
                'strategy': 'conservative_approach',
                'primary_recommendation': 'avoid_trading',
                'confidence_adjustment': -0.3,
                'explanation': 'Critical conflict detected - recommend avoiding trades until resolved'
            })
        
        elif conflict_category == 'retail_vs_institutional':
            # Institutional typically more reliable
            inst_agent = 'institutional_activity' if 'institutional_activity' in agents else None
            if inst_agent and agents[inst_agent]['confidence'] > 0.7:
                resolution.update({
                    'strategy': 'favor_institutional',
                    'primary_recommendation': 'follow_institutional_signal',
                    'confidence_adjustment': -0.1,
                    'explanation': 'Institutional activity typically more reliable than retail anomalies'
                })
        
        elif conflict_category == 'trend_momentum_divergence':
            resolution.update({
                'strategy': 'wait_for_confirmation',
                'primary_recommendation': 'wait_for_trend_confirmation',
                'confidence_adjustment': -0.2,
                'explanation': 'Trend-momentum divergence suggests potential reversal - wait for confirmation'
            })
        
        elif conflict_category == 'levels_vs_trend':
            resolution.update({
                'strategy': 'prioritize_levels',
                'primary_recommendation': 'respect_key_levels',
                'confidence_adjustment': -0.15,
                'explanation': 'Support/resistance levels often override short-term trend signals'
            })
        
        return resolution
    
    def _determine_conflict_type(self, sig1: str, sig2: str, strength1: float, strength2: float) -> str:
        """Determine the type of conflict between signals"""
        # Direct opposition conflicts
        if (sig1 == 'bullish' and sig2 == 'bearish') or (sig1 == 'bearish' and sig2 == 'bullish'):
            return 'direct_opposition'
        
        # Strength-based conflicts (same signal, very different strengths)
        if sig1 == sig2 and sig1 != 'neutral':
            strength_diff = abs(strength1 - strength2)
            if strength_diff > 0.4:  # Significant strength difference
                return 'strength_divergence'
        
        # Confidence-strength conflicts (one strong neutral vs weak directional)
        if sig1 == 'neutral' and sig2 != 'neutral' and strength2 < 0.3:
            return 'weak_signal_conflict'
        elif sig2 == 'neutral' and sig1 != 'neutral' and strength1 < 0.3:
            return 'weak_signal_conflict'
        
        return 'none'
    
    def _analyze_multi_dimensional_conflict(self, agent1: str, result1: VolumeAgentResult, signals1: Dict[str, Any],
                                          agent2: str, result2: VolumeAgentResult, signals2: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze conflict across multiple dimensions"""
        analysis = {
            'dimensions': [],
            'agent1_data_quality': 0.0,
            'agent2_data_quality': 0.0,
            'agent1_stability': 0.0,
            'agent2_stability': 0.0,
            'cross_correlation': 0.0,
            'data_overlap': 0.0
        }
        
        # Data quality assessment
        analysis['agent1_data_quality'] = self._assess_agent_data_quality(agent1, result1)
        analysis['agent2_data_quality'] = self._assess_agent_data_quality(agent2, result2)
        
        # Signal stability assessment
        analysis['agent1_stability'] = self._assess_signal_stability(agent1, signals1, result1)
        analysis['agent2_stability'] = self._assess_signal_stability(agent2, signals2, result2)
        
        # Cross-agent correlation analysis
        analysis['cross_correlation'] = self._calculate_agent_correlation(agent1, agent2)
        
        # Data source overlap analysis
        analysis['data_overlap'] = self._calculate_data_source_overlap(agent1, agent2)
        
        # Identify conflicting dimensions
        if abs(signals1.get('strength', 0.5) - signals2.get('strength', 0.5)) > 0.3:
            analysis['dimensions'].append('signal_strength')
        
        if abs(analysis['agent1_data_quality'] - analysis['agent2_data_quality']) > 0.2:
            analysis['dimensions'].append('data_quality')
        
        if abs(analysis['agent1_stability'] - analysis['agent2_stability']) > 0.2:
            analysis['dimensions'].append('signal_stability')
        
        if analysis['cross_correlation'] < -0.3:  # Negative correlation
            analysis['dimensions'].append('methodological_divergence')
        
        return analysis
    
    def _analyze_temporal_conflict_patterns(self, agent1: str, agent2: str, sig1: str, sig2: str, 
                                          strength1: float, strength2: float) -> Dict[str, Any]:
        """Analyze temporal patterns in conflicts between agents"""
        patterns = {
            'historical_agreement_rate': 0.0,
            'recent_conflict_frequency': 0.0,
            'conflict_duration_trend': 'stable',
            'resolution_success_rate': 0.0,
            'seasonal_patterns': [],
            'volatility_correlation': 0.0
        }
        
        # Get historical conflict data (would be from database in production)
        historical_data = self._get_historical_agent_conflicts(agent1, agent2)
        
        if historical_data:
            patterns['historical_agreement_rate'] = historical_data.get('agreement_rate', 0.5)
            patterns['recent_conflict_frequency'] = historical_data.get('recent_frequency', 0.1)
            patterns['resolution_success_rate'] = historical_data.get('resolution_rate', 0.7)
            
            # Analyze trends
            recent_conflicts = historical_data.get('recent_conflicts', [])
            if len(recent_conflicts) >= 5:
                # Simple trend analysis
                recent_count = sum(1 for c in recent_conflicts[-5:] if c.get('resolved', False))
                if recent_count > 3:
                    patterns['conflict_duration_trend'] = 'improving'
                elif recent_count < 2:
                    patterns['conflict_duration_trend'] = 'worsening'
        
        return patterns
    
    def _calculate_enhanced_conflict_severity(self, conflict_analysis: Dict[str, Any], 
                                            temporal_factors: Dict[str, Any],
                                            agent1: str, agent2: str) -> float:
        """Calculate enhanced conflict severity score using multiple factors"""
        base_severity = 0.0
        
        # Data quality factor (20%)
        quality_diff = abs(conflict_analysis['agent1_data_quality'] - conflict_analysis['agent2_data_quality'])
        quality_factor = min(1.0, quality_diff * 2.0) * 0.2
        
        # Stability factor (20%)
        stability_diff = abs(conflict_analysis['agent1_stability'] - conflict_analysis['agent2_stability'])
        stability_factor = min(1.0, stability_diff * 2.0) * 0.2
        
        # Agent importance factor (25%)
        weight1 = self.agent_config.get(agent1, {}).get('weight', 0.2)
        weight2 = self.agent_config.get(agent2, {}).get('weight', 0.2)
        importance_factor = (weight1 + weight2) / 0.8 * 0.25  # Normalized to max possible weight
        
        # Historical pattern factor (15%)
        history_factor = (1.0 - temporal_factors['historical_agreement_rate']) * 0.15
        
        # Cross-correlation factor (10%)
        correlation = conflict_analysis['cross_correlation']
        correlation_factor = abs(correlation) * 0.1 if correlation < 0 else 0
        
        # Dimension count factor (10%)
        dimension_count = len(conflict_analysis['dimensions'])
        dimension_factor = min(1.0, dimension_count / 4.0) * 0.1
        
        severity = (quality_factor + stability_factor + importance_factor + 
                   history_factor + correlation_factor + dimension_factor)
        
        return min(1.0, max(0.0, severity))
    
    def _analyze_conflict_market_context(self, conflict_analysis: Dict[str, Any], 
                                       temporal_factors: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market context affecting conflict resolution"""
        context = {
            'volatility_regime': 'normal',
            'trend_strength': 0.5,
            'volume_regime': 'normal',
            'market_stress_level': 0.3,
            'conflict_relevance': 0.7,
            'resolution_urgency': 'medium'
        }
        
        # Assess based on conflict dimensions and patterns
        if 'signal_strength' in conflict_analysis['dimensions']:
            context['conflict_relevance'] += 0.2
        
        if temporal_factors['recent_conflict_frequency'] > 0.3:
            context['resolution_urgency'] = 'high'
            context['market_stress_level'] += 0.2
        
        # Simple market regime detection (would use more sophisticated methods in production)
        dimension_count = len(conflict_analysis['dimensions'])
        if dimension_count >= 3:
            context['volatility_regime'] = 'high'
            context['market_stress_level'] += 0.3
        
        return context
    
    def _categorize_enhanced_conflict(self, agent1: str, agent2: str, conflict_analysis: Dict[str, Any]) -> str:
        """Enhanced conflict categorization with dimensional analysis"""
        # Get base category
        base_category = self._categorize_conflict(agent1, agent2)
        
        # Add dimensional modifiers
        dimensions = conflict_analysis['dimensions']
        
        if 'data_quality' in dimensions:
            return f"{base_category}_data_quality"
        elif 'signal_stability' in dimensions:
            return f"{base_category}_stability"
        elif 'methodological_divergence' in dimensions:
            return f"{base_category}_methodological"
        else:
            return base_category
    
    def _get_enhanced_severity_level(self, severity_score: float) -> str:
        """Enhanced severity level determination"""
        if severity_score > 0.8:
            return 'critical'
        elif severity_score > 0.6:
            return 'high'
        elif severity_score > 0.4:
            return 'medium'
        elif severity_score > 0.2:
            return 'low'
        else:
            return 'minimal'
    
    def _assess_enhanced_conflict_impact(self, agent1: str, agent2: str, severity_score: float,
                                       conflict_analysis: Dict[str, Any], 
                                       market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced conflict impact assessment"""
        base_impact = self._assess_conflict_impact(agent1, agent2, severity_score)
        
        # Add enhanced factors
        enhanced_impact = base_impact.copy()
        
        # Market context adjustments
        if market_context['volatility_regime'] == 'high':
            enhanced_impact['confidence_impact'] *= 1.3
            enhanced_impact['recommendation_impact'] = 'critical'
        
        # Data quality impact
        avg_quality = (conflict_analysis['agent1_data_quality'] + conflict_analysis['agent2_data_quality']) / 2
        if avg_quality < 0.6:
            enhanced_impact['data_reliability_impact'] = 'high'
        else:
            enhanced_impact['data_reliability_impact'] = 'low'
        
        # Stability impact
        avg_stability = (conflict_analysis['agent1_stability'] + conflict_analysis['agent2_stability']) / 2
        enhanced_impact['signal_stability_impact'] = 'high' if avg_stability < 0.5 else 'low'
        
        return enhanced_impact
    
    def _calculate_resolution_priority(self, severity_score: float, conflict_analysis: Dict[str, Any],
                                     market_context: Dict[str, Any]) -> str:
        """Calculate resolution priority based on multiple factors"""
        priority_score = severity_score
        
        # Adjust based on market context
        if market_context['resolution_urgency'] == 'high':
            priority_score += 0.2
        
        # Adjust based on agent importance
        dimension_count = len(conflict_analysis['dimensions'])
        priority_score += min(0.2, dimension_count * 0.05)
        
        # Determine priority level
        if priority_score > 0.8:
            return 'critical'
        elif priority_score > 0.6:
            return 'high'
        elif priority_score > 0.4:
            return 'medium'
        else:
            return 'low'
    
    # Helper methods for data quality and stability assessment
    def _assess_agent_data_quality(self, agent_name: str, result: VolumeAgentResult) -> float:
        """Assess data quality for an agent's result"""
        quality_score = 0.8  # Base quality
        
        if result.processing_time > 30:  # Long processing time may indicate issues
            quality_score -= 0.1
        
        if result.error_message:
            quality_score -= 0.3
        
        # Agent-specific quality assessments
        if agent_name == 'volume_anomaly' and result.analysis_data:
            anomalies = result.analysis_data.get('anomalies', [])
            if len(anomalies) > 10:  # Too many anomalies may indicate noise
                quality_score -= 0.1
        
        return max(0.0, min(1.0, quality_score))
    
    def _assess_signal_stability(self, agent_name: str, signals: Dict[str, Any], result: VolumeAgentResult) -> float:
        """Assess signal stability for an agent"""
        stability = 0.7  # Base stability
        
        strength = signals.get('strength', 0.5)
        if strength < 0.3:  # Weak signals are less stable
            stability -= 0.2
        elif strength > 0.8:  # Very strong signals are more stable
            stability += 0.1
        
        # Confidence affects stability
        confidence = result.confidence_score or 0.5
        if confidence < 0.4:
            stability -= 0.2
        elif confidence > 0.8:
            stability += 0.1
        
        return max(0.0, min(1.0, stability))
    
    def _calculate_agent_correlation(self, agent1: str, agent2: str) -> float:
        """Calculate historical correlation between agents (placeholder)"""
        # In production, this would analyze historical signal correlations
        correlation_map = {
            ('volume_anomaly', 'institutional_activity'): 0.3,
            ('volume_confirmation', 'volume_momentum'): 0.6,
            ('support_resistance', 'volume_confirmation'): 0.4,
            ('institutional_activity', 'volume_momentum'): 0.2,
            ('volume_anomaly', 'support_resistance'): 0.1
        }
        
        key = tuple(sorted([agent1, agent2]))
        return correlation_map.get(key, 0.0)
    
    def _calculate_data_source_overlap(self, agent1: str, agent2: str) -> float:
        """Calculate overlap in data sources between agents"""
        # All agents use OHLCV data, but different aspects
        overlap_map = {
            ('volume_anomaly', 'institutional_activity'): 0.7,  # Both focus on volume
            ('volume_confirmation', 'volume_momentum'): 0.8,     # Both use price-volume
            ('support_resistance', 'volume_confirmation'): 0.6,  # Both use price levels
            ('institutional_activity', 'volume_momentum'): 0.5,  # Moderate overlap
            ('volume_anomaly', 'support_resistance'): 0.4       # Lower overlap
        }
        
        key = tuple(sorted([agent1, agent2]))
        return overlap_map.get(key, 0.5)
    
    def _get_historical_agent_conflicts(self, agent1: str, agent2: str) -> Dict[str, Any]:
        """Get historical conflict data between agents (placeholder)"""
        # In production, this would query a database of historical conflicts
        return {
            'agreement_rate': 0.7,
            'recent_frequency': 0.2,
            'resolution_rate': 0.8,
            'recent_conflicts': [
                {'resolved': True, 'duration': 300},
                {'resolved': True, 'duration': 150},
                {'resolved': False, 'duration': 600},
                {'resolved': True, 'duration': 200}
            ]
        }

    def _aggregate_risk_assessment(self, results: Dict[str, VolumeAgentResult]) -> Dict[str, Any]:
        """Aggregate risk assessment from all agents"""
        risk_factors = []
        overall_risk = 'medium'
        
        for agent_name, result in results.items():
            if result.success and result.analysis_data:
                agent_risks = self._extract_agent_risks(agent_name, result.analysis_data)
                risk_factors.extend(agent_risks)
        
        # Determine overall risk level
        high_risk_count = sum(1 for risk in risk_factors if risk.get('level') == 'high')
        if high_risk_count > len(risk_factors) * 0.3:  # >30% high risk factors
            overall_risk = 'high'
        elif high_risk_count == 0:
            overall_risk = 'low'
        
        return {
            'overall_risk': overall_risk,
            'risk_factors': risk_factors,
            'risk_count': len(risk_factors)
        }

    def _extract_agent_risks(self, agent_name: str, analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract risk factors from agent analysis"""
        risks = []
        
        try:
            if agent_name == 'volume_anomaly':
                anomalies = analysis_data.get('anomalies', [])
                extreme_anomalies = [a for a in anomalies if a.get('significance') == 'high']
                if extreme_anomalies:
                    risks.append({
                        'type': 'volume_anomaly',
                        'level': 'high',
                        'description': f"{len(extreme_anomalies)} extreme volume spikes detected"
                    })
            
            elif agent_name == 'institutional_activity':
                activity = analysis_data.get('activity_level', 'unknown')
                if activity in ['very_high', 'extreme']:
                    risks.append({
                        'type': 'institutional_pressure',
                        'level': 'high',
                        'description': f"Extreme institutional activity: {activity}"
                    })
            
            # Add more risk extraction logic for other agents...
        
        except Exception as e:
            logger.warning(f"Failed to extract risks from {agent_name}: {e}")
        
        return risks

    def _aggregate_trading_implications(self, results: Dict[str, VolumeAgentResult]) -> Dict[str, Any]:
        """Aggregate trading implications from all agents"""
        implications = {
            'primary_strategy': 'hold',
            'confidence': 0.5,
            'entry_signals': [],
            'exit_signals': [],
            'risk_management': []
        }
        
        # Collect signals from all agents
        all_signals = []
        for agent_name, result in results.items():
            if result.success and result.analysis_data:
                signals = self._extract_agent_signals(agent_name, result.analysis_data)
                all_signals.append((agent_name, signals))
        
        # Determine primary strategy based on consensus
        bullish_count = sum(1 for _, sig in all_signals if sig.get('primary_signal') == 'bullish')
        bearish_count = sum(1 for _, sig in all_signals if sig.get('primary_signal') == 'bearish')
        
        if bullish_count > bearish_count and bullish_count >= len(all_signals) * 0.6:
            implications['primary_strategy'] = 'buy'
            implications['confidence'] = min(0.9, 0.5 + (bullish_count / len(all_signals)) * 0.4)
        elif bearish_count > bullish_count and bearish_count >= len(all_signals) * 0.6:
            implications['primary_strategy'] = 'sell'
            implications['confidence'] = min(0.9, 0.5 + (bearish_count / len(all_signals)) * 0.4)
        
        return implications

    def _validate_inputs(self, stock_data: pd.DataFrame, symbol: str, indicators: Dict[str, Any] = None) -> Optional[str]:
        """
        Validate inputs for volume agents analysis
        
        Returns:
            Error message if validation fails, None if validation passes
        """
        try:
            # Validate symbol
            if not symbol or not isinstance(symbol, str) or len(symbol.strip()) == 0:
                return "Invalid symbol: symbol must be a non-empty string"
            
            # Validate stock data
            if stock_data is None:
                return "Stock data is None"
            
            if stock_data.empty:
                return "Stock data is empty"
            
            # Check required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in stock_data.columns]
            if missing_columns:
                return f"Missing required columns in stock data: {missing_columns}"
            
            # Check minimum data points for meaningful analysis
            if len(stock_data) < 10:
                return f"Insufficient data points: {len(stock_data)} rows (minimum 10 required)"
            
            # Check for valid volume data (critical for volume analysis)
            if stock_data['volume'].isna().all():
                return "All volume data is NaN"
            
            if (stock_data['volume'] <= 0).all():
                return "All volume data is zero or negative"
            
            # Validate indicators format if provided
            if indicators is not None and not isinstance(indicators, dict):
                return "Indicators must be a dictionary or None"
            
            return None  # Validation passed
            
        except Exception as e:
            return f"Validation error: {str(e)}"
    
    def _create_fallback_result(self, symbol: str, error_message: str, processing_time: float) -> AggregatedVolumeAnalysis:
        """
        Create a fallback result when volume agents analysis cannot proceed
        """
        fallback_analysis = {
            'symbol': symbol,
            'analysis_timestamp': datetime.now().isoformat(),
            'error': error_message,
            'volume_summary': {
                'current_volume': 0,
                'volume_percentile': 0,
                'volume_trend': 'unknown'
            },
            'key_findings': [f"Volume analysis failed: {error_message}"],
            'consensus_signals': {},
            'risk_assessment': {
                'overall_risk': 'high',
                'risk_factors': [{'type': 'analysis_failure', 'level': 'high', 'description': error_message}],
                'risk_count': 1
            },
            'trading_implications': {
                'primary_strategy': 'avoid',
                'confidence': 0.0,
                'entry_signals': [],
                'exit_signals': [],
                'risk_management': ['Avoid trading due to analysis failure']
            }
        }
        
        return AggregatedVolumeAnalysis(
            individual_results={},
            unified_analysis=fallback_analysis,
            total_processing_time=processing_time,
            successful_agents=0,
            failed_agents=len(self.agent_config),
            overall_confidence=0.0,
            consensus_signals={},
            conflicting_signals=[]
        )


class VolumeAgentIntegrationManager:
    """
    Integration manager that provides a standardized interface between
    the main analysis system and the volume agents orchestrator
    """
    _global_instance = None  # Track global instance for metrics
    
    def __init__(self, gemini_client=None):
        self.orchestrator = VolumeAgentsOrchestrator(gemini_client)
        # Set global instance for metrics tracking
        VolumeAgentIntegrationManager._global_instance = self
        logger.info("VolumeAgentIntegrationManager initialized")

    async def get_comprehensive_volume_analysis(self,
                                              stock_data: pd.DataFrame,
                                              symbol: str,
                                              indicators: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get comprehensive volume analysis in a format compatible with the main system
        
        This method provides the main interface for the existing system to use
        the distributed volume agents with robust error handling and fallback mechanisms.
        """
        analysis_start_time = time.time()
        
        try:
            print(f"[VOLUME_AGENT_DEBUG] Integration start for {symbol}: data_len={(len(stock_data) if stock_data is not None else 'None')}")
            # ENHANCED: Comprehensive pre-flight health check
            should_use_agents, health_reason = self.should_use_volume_agents()
            print(f"[VOLUME_AGENT_DEBUG] Health check for {symbol}: should_use={should_use_agents} reason='{health_reason}'")
            
            if not should_use_agents:
                logger.warning(f"Volume agents unavailable: {health_reason}")
                print(f"[VOLUME_AGENT_DEBUG] Integration short-circuit for {symbol}: {health_reason}")
                return self._create_degraded_analysis_result(symbol, health_reason, time.time() - analysis_start_time)
            
            logger.info(f"Volume agents health check passed: {health_reason}")
            
            # ENHANCED: Validate input data before proceeding
            validation_error = self.orchestrator._validate_inputs(stock_data, symbol, indicators)
            if validation_error:
                logger.warning(f"Input validation failed for {symbol}: {validation_error}")
                return self._create_degraded_analysis_result(symbol, f"Input validation failed: {validation_error}", time.time() - analysis_start_time)
            
            # Run the orchestrator with enhanced error handling and timeout
            try:
                # Add timeout protection for the entire orchestrator operation
                import asyncio
                timeout_seconds = 180  # 180 seconds timeout for volume agents analysis (reduced from 300s)
                
                result = await asyncio.wait_for(
                    self.orchestrator.analyze_stock_volume_comprehensive(stock_data, symbol, indicators),
                    timeout=timeout_seconds
                )
                
            except asyncio.TimeoutError:
                processing_time = time.time() - analysis_start_time
                logger.error(f"Volume agents analysis timed out after {timeout_seconds} seconds for {symbol}")
                print(f"[VOLUME_AGENT_DEBUG] Integration timeout for {symbol} after {timeout_seconds}s")
                return self._create_degraded_analysis_result(
                    symbol, 
                    f"Analysis timed out after {timeout_seconds} seconds",
                    processing_time
                )
            except Exception as orchestrator_error:
                processing_time = time.time() - analysis_start_time
                logger.error(f"Orchestrator execution failed for {symbol}: {orchestrator_error}")
                import traceback
                logger.error(f"Orchestrator error traceback: {traceback.format_exc()}")
                print(f"[VOLUME_AGENT_DEBUG] Integration error for {symbol}: {orchestrator_error}")
                return self._create_degraded_analysis_result(
                    symbol, 
                    f"Orchestrator execution failed: {str(orchestrator_error)}",
                    processing_time
                )
            
            # ENHANCED: Check for minimum acceptable results
            if result.successful_agents == 0:
                logger.warning(f"All volume agents failed for {symbol}, providing fallback analysis")
                print(f"[VOLUME_AGENT_DEBUG] Integration result for {symbol}: all agents failed")
                return self._create_degraded_analysis_result(
                    symbol, 
                    "All volume agents failed during execution",
                    result.total_processing_time,
                    result
                )
            
            # ENHANCED: Warn if majority of agents failed
            total_agents = result.successful_agents + result.failed_agents
            failure_rate = result.failed_agents / total_agents if total_agents > 0 else 1.0
            
            if failure_rate > 0.6:  # More than 60% failed
                logger.warning(f"High failure rate ({failure_rate:.1%}) for {symbol} volume agents")
                print(f"[VOLUME_AGENT_DEBUG] High failure rate for {symbol}: {failure_rate:.1%}")
            
            # Extract and combine LLM responses for final decision agent
            combined_llm_analysis = self.extract_and_combine_llm_responses(result.individual_results)
            
            # Format result for main system compatibility
            formatted_result = {
                'success': result.successful_agents > 0,
                'processing_time': result.total_processing_time,
                'volume_analysis': result.unified_analysis,
                'combined_llm_analysis': combined_llm_analysis,  # For final decision agent
                'individual_agents': {
                    agent_name: {
                        'success': agent_result.success,
                        'confidence': agent_result.confidence_score or 0.0,
                        'key_data': agent_result.analysis_data or {},
                        'processing_time': agent_result.processing_time,
                        'error_message': agent_result.error_message if not agent_result.success else None,
                        'has_llm_response': bool(agent_result.llm_response) if agent_result.success else False
                    }
                    for agent_name, agent_result in result.individual_results.items()
                },
                'consensus_analysis': {
                    'overall_confidence': result.overall_confidence,
                    'consensus_signals': result.consensus_signals,
                    'conflicting_signals': result.conflicting_signals,
                    'successful_agents': result.successful_agents,
                    'failed_agents': result.failed_agents,
                    'failure_rate': failure_rate,
                    'quality_assessment': self._assess_result_quality(result),
                    'llm_responses_available': len([r for r in result.individual_results.values() if r.success and r.llm_response])
                }
            }
            
            # Debug: Integration completed summary
            try:
                ca = formatted_result.get('consensus_analysis', {})
                print(f"[VOLUME_AGENT_DEBUG] Integration complete for {symbol}: success={formatted_result.get('success')} successful_agents={ca.get('successful_agents')} failed_agents={ca.get('failed_agents')} overall_confidence={ca.get('overall_confidence')}")
            except Exception:
                print(f"[VOLUME_AGENT_DEBUG] Integration complete for {symbol}: summary unavailable")
            
            return formatted_result
            
        except Exception as e:
            processing_time = time.time() - analysis_start_time
            logger.error(f"Volume agent integration failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            print(f"[VOLUME_AGENT_DEBUG] Integration crashed for {symbol}: {e}")
            
            return self._create_degraded_analysis_result(
                symbol, 
                f"Integration system error: {str(e)}",
                processing_time
            )

    def get_agent_charts(self, result: AggregatedVolumeAnalysis) -> Dict[str, bytes]:
        """
        Extract chart images from volume agents results
        """
        charts = {}
        for agent_name, agent_result in result.individual_results.items():
            if agent_result.success and agent_result.chart_image:
                charts[f'{agent_name}_chart'] = agent_result.chart_image
        return charts

    def is_volume_agents_healthy(self) -> bool:
        """
        Check if the volume agents system is healthy and ready to use
        """
        try:
            # Check if all required components are available in the distributed architecture
            required_agent_names = [
                'volume_anomaly',
                'institutional_activity', 
                'volume_confirmation',
                'support_resistance',
                'volume_momentum'
            ]
            
            # Check if agents exist and are functional
            for agent_name in required_agent_names:
                agent = self.orchestrator.llm_agents.get(agent_name)
                if agent is None:
                    return False
                    
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def _create_degraded_analysis_result(self, symbol: str, error_message: str, processing_time: float, 
                                       partial_result: AggregatedVolumeAnalysis = None) -> Dict[str, Any]:
        """
        Create a degraded analysis result when the system partially or completely fails
        """
        base_result = {
            'success': False,
            'processing_time': processing_time,
            'error': error_message,
            'degraded_mode': True,
            'fallback_message': 'Volume agents analysis failed, using degraded analysis'
        }
        
        if partial_result and partial_result.successful_agents > 0:
            # Some agents succeeded, provide partial results
            base_result.update({
                'success': True,  # Partial success
                'volume_analysis': partial_result.unified_analysis,
                'individual_agents': {
                    agent_name: {
                        'success': agent_result.success,
                        'confidence': agent_result.confidence_score or 0.0,
                        'key_data': agent_result.analysis_data or {},
                        'processing_time': agent_result.processing_time,
                        'error_message': agent_result.error_message if not agent_result.success else None
                    }
                    for agent_name, agent_result in partial_result.individual_results.items()
                },
                'consensus_analysis': {
                    'overall_confidence': partial_result.overall_confidence,
                    'consensus_signals': partial_result.consensus_signals,
                    'conflicting_signals': partial_result.conflicting_signals,
                    'successful_agents': partial_result.successful_agents,
                    'failed_agents': partial_result.failed_agents,
                    'partial_analysis_warning': 'Results based on partial agent success'
                }
            })
        else:
            # Complete failure, provide minimal fallback
            base_result.update({
                'volume_analysis': {
                    'symbol': symbol,
                    'analysis_timestamp': datetime.now().isoformat(),
                    'error': error_message,
                    'volume_summary': {'status': 'analysis_failed'},
                    'key_findings': ['Volume analysis system unavailable'],
                    'risk_assessment': {
                        'overall_risk': 'unknown',
                        'risk_factors': [{'type': 'system_failure', 'description': error_message}]
                    }
                },
                'individual_agents': {},
                'consensus_analysis': {
                    'overall_confidence': 0.0,
                    'successful_agents': 0,
                    'failed_agents': 5,
                    'system_status': 'degraded'
                }
            })
        
        return base_result
    
    def _assess_result_quality(self, result: AggregatedVolumeAnalysis) -> str:
        """
        Assess the quality of the volume agents analysis result
        """
        total_agents = result.successful_agents + result.failed_agents
        
        if total_agents == 0:
            return 'no_data'
        
        success_rate = result.successful_agents / total_agents
        
        if success_rate >= 0.8:
            return 'excellent'
        elif success_rate >= 0.6:
            return 'good'
        elif success_rate >= 0.4:
            return 'fair'
        elif success_rate > 0:
            return 'poor'
        else:
            return 'failed'

    def get_agent_health_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get health status for all volume agents with detailed diagnostics
        """
        health_status = {}
        
        try:
            # Check each agent individually - use the correct orchestrator structure
            agents = {
                'volume_anomaly': self.orchestrator.llm_agents.get('volume_anomaly'),
                'institutional_activity': self.orchestrator.llm_agents.get('institutional_activity'),
                'volume_confirmation': self.orchestrator.llm_agents.get('volume_confirmation'),
                'support_resistance': self.orchestrator.llm_agents.get('support_resistance'),
                'volume_momentum': self.orchestrator.llm_agents.get('volume_momentum')
            }
            
            for agent_name, agent in agents.items():
                try:
                    is_healthy = agent is not None
                    
                    # For distributed agents architecture, check if agent has required methods
                    has_llm_capability = False
                    if is_healthy:
                        try:
                            # Check if agent has the expected interface
                            if hasattr(agent, 'analyze_complete') or hasattr(agent, 'analyze') or hasattr(agent, '__call__'):
                                has_llm_capability = True
                        except Exception:
                            has_llm_capability = False
                    
                    diagnostics = {
                        'initialized': is_healthy,
                        'llm_capability': has_llm_capability,
                        'agent_type': type(agent).__name__ if agent else 'None',
                        'using_distributed_architecture': True,
                        'last_check': datetime.now().isoformat()
                    }
                    
                    health_status[agent_name] = {
                        'healthy': is_healthy and has_llm_capability,
                        'diagnostics': diagnostics,
                        'status': 'operational' if is_healthy and has_llm_capability else 'degraded'
                    }
                    
                except Exception as agent_check_error:
                    health_status[agent_name] = {
                        'healthy': False,
                        'diagnostics': {
                            'initialized': False,
                            'error': str(agent_check_error),
                            'last_check': datetime.now().isoformat()
                        },
                        'status': 'error'
                    }
                    
        except Exception as overall_error:
            logger.error(f"Error checking agent health: {overall_error}")
            # Return default unhealthy status for all agents
            for agent_name in ['volume_anomaly', 'institutional_activity', 'volume_confirmation', 'support_resistance', 'volume_momentum']:
                health_status[agent_name] = {
                    'healthy': False,
                    'diagnostics': {
                        'initialized': False,
                        'error': str(overall_error),
                        'last_check': datetime.now().isoformat()
                    },
                    'status': 'system_error'
                }
        
        return health_status

    def get_system_health_summary(self) -> Dict[str, Any]:
        """
        Get overall system health summary with recommendations
        """
        health_status = self.get_agent_health_status()
        
        healthy_agents = sum(1 for agent in health_status.values() if agent['healthy'])
        total_agents = len(health_status)
        health_percentage = (healthy_agents / total_agents) * 100 if total_agents > 0 else 0
        
        # Determine system status
        if health_percentage >= 80:
            system_status = 'healthy'
            recommendation = 'System operating normally'
        elif health_percentage >= 60:
            system_status = 'degraded'
            recommendation = 'Some agents unavailable - partial functionality expected'
        elif health_percentage >= 20:
            system_status = 'impaired'
            recommendation = 'Majority of agents unavailable - limited functionality'
        else:
            system_status = 'critical'
            recommendation = 'Volume analysis system largely non-functional - use fallback methods'
        
        return {
            'system_status': system_status,
            'health_percentage': health_percentage,
            'healthy_agents': healthy_agents,
            'total_agents': total_agents,
            'recommendation': recommendation,
            'agent_status': health_status,
            'last_updated': datetime.now().isoformat()
        }

    def should_use_volume_agents(self) -> Tuple[bool, str]:
        """
        Determine whether to use volume agents or fall back to traditional analysis
        
        Returns:
            Tuple[bool, str]: (should_use, reason)
        """
        try:
            health_summary = self.get_system_health_summary()
            
            # Use volume agents if at least 40% are healthy
            should_use = health_summary['health_percentage'] >= 40
            
            if should_use:
                reason = f"Volume agents available ({health_summary['healthy_agents']}/{health_summary['total_agents']} healthy)"
            else:
                reason = f"Volume agents system too degraded ({health_summary['health_percentage']:.1f}% healthy) - using fallback"
            
            return should_use, reason
            
        except Exception as e:
            logger.error(f"Error checking volume agents availability: {e}")
            return False, f"Health check failed: {str(e)}"

    def get_agent_performance_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get performance metrics for all volume agents with historical tracking
        """
        try:
            # Initialize metrics tracking if not exists
            if not hasattr(self, '_agent_metrics'):
                self._agent_metrics = {
                    'volume_anomaly': {'total_calls': 0, 'successful_calls': 0, 'total_processing_time': 0.0, 'avg_confidence': 0.0, 'last_failure_time': None},
                    'institutional_activity': {'total_calls': 0, 'successful_calls': 0, 'total_processing_time': 0.0, 'avg_confidence': 0.0, 'last_failure_time': None},
                    'volume_confirmation': {'total_calls': 0, 'successful_calls': 0, 'total_processing_time': 0.0, 'avg_confidence': 0.0, 'last_failure_time': None},
                    'support_resistance': {'total_calls': 0, 'successful_calls': 0, 'total_processing_time': 0.0, 'avg_confidence': 0.0, 'last_failure_time': None},
                    'volume_momentum': {'total_calls': 0, 'successful_calls': 0, 'total_processing_time': 0.0, 'avg_confidence': 0.0, 'last_failure_time': None}
                }
            
            performance_metrics = {}
            current_time = datetime.now()
            
            for agent_name, metrics in self._agent_metrics.items():
                total_calls = metrics['total_calls']
                successful_calls = metrics['successful_calls']
                
                success_rate = (successful_calls / total_calls) if total_calls > 0 else 0.0
                avg_processing_time = (metrics['total_processing_time'] / successful_calls) if successful_calls > 0 else 0.0
                
                # Calculate agent reliability score
                reliability_score = self._calculate_agent_reliability(
                    success_rate, avg_processing_time, metrics['avg_confidence'], 
                    metrics['last_failure_time'], current_time
                )
                
                performance_metrics[agent_name] = {
                    'success_rate': success_rate,
                    'total_calls': total_calls,
                    'successful_calls': successful_calls,
                    'failed_calls': total_calls - successful_calls,
                    'avg_processing_time': avg_processing_time,
                    'avg_confidence': metrics['avg_confidence'],
                    'reliability_score': reliability_score,
                    'status': self._determine_agent_status(reliability_score, success_rate),
                    'last_failure_time': metrics['last_failure_time'],
                    'recommendations': self._get_agent_recommendations(reliability_score, success_rate)
                }
            
            return performance_metrics
            
        except Exception as e:
            logger.error(f"Error getting agent performance metrics: {e}")
            return {}
    
    def _calculate_agent_reliability(self, success_rate: float, avg_processing_time: float, 
                                   avg_confidence: float, last_failure_time: datetime = None, 
                                   current_time: datetime = None) -> float:
        """
        Calculate agent reliability score (0.0 to 1.0)
        """
        try:
            # Base score from success rate (40% weight)
            success_component = success_rate * 0.4
            
            # Performance component (30% weight) - faster is better
            max_acceptable_time = 60.0  # 60 seconds
            performance_component = max(0.0, (max_acceptable_time - avg_processing_time) / max_acceptable_time) * 0.3
            
            # Confidence component (20% weight)
            confidence_component = avg_confidence * 0.2
            
            # Recency component (10% weight) - penalize recent failures
            recency_component = 0.1
            if last_failure_time and current_time:
                time_since_failure = (current_time - last_failure_time).total_seconds()
                # Reduce score if failure was recent (within last hour)
                if time_since_failure < 3600:  # 1 hour
                    recency_penalty = (3600 - time_since_failure) / 3600 * 0.1
                    recency_component = max(0.0, recency_component - recency_penalty)
            
            reliability_score = success_component + performance_component + confidence_component + recency_component
            return min(1.0, max(0.0, reliability_score))
            
        except Exception:
            return 0.5  # Default moderate reliability
    
    def _determine_agent_status(self, reliability_score: float, success_rate: float) -> str:
        """
        Determine agent operational status based on metrics
        """
        if reliability_score >= 0.8 and success_rate >= 0.8:
            return 'excellent'
        elif reliability_score >= 0.6 and success_rate >= 0.6:
            return 'good'
        elif reliability_score >= 0.4 and success_rate >= 0.4:
            return 'fair'
        elif reliability_score >= 0.2 or success_rate >= 0.2:
            return 'poor'
        else:
            return 'critical'
    
    def _get_agent_recommendations(self, reliability_score: float, success_rate: float) -> List[str]:
        """
        Get recommendations for agent improvement
        """
        recommendations = []
        
        if success_rate < 0.5:
            recommendations.append('Consider disabling agent temporarily for maintenance')
        elif success_rate < 0.7:
            recommendations.append('Monitor agent closely - investigate failure causes')
        
        if reliability_score < 0.3:
            recommendations.append('Agent requires immediate attention')
        elif reliability_score < 0.6:
            recommendations.append('Review agent configuration and performance')
        
        if not recommendations:
            recommendations.append('Agent performing within acceptable parameters')
        
        return recommendations
    
    def update_agent_metrics(self, agent_name: str, success: bool, processing_time: float, confidence: float = None):
        """
        Update performance metrics for an agent after execution
        """
        try:
            if not hasattr(self, '_agent_metrics'):
                self.get_agent_performance_metrics()  # Initialize metrics
            
            if agent_name not in self._agent_metrics:
                return
            
            metrics = self._agent_metrics[agent_name]
            
            # Update call counts
            metrics['total_calls'] += 1
            if success:
                metrics['successful_calls'] += 1
                metrics['total_processing_time'] += processing_time
                
                # Update running average confidence
                if confidence is not None:
                    current_avg = metrics['avg_confidence']
                    successful_calls = metrics['successful_calls']
                    metrics['avg_confidence'] = ((current_avg * (successful_calls - 1)) + confidence) / successful_calls
            else:
                metrics['last_failure_time'] = datetime.now()
                
        except Exception as e:
            logger.error(f"Error updating agent metrics for {agent_name}: {e}")
    
    def should_disable_agent(self, agent_name: str) -> Tuple[bool, str]:
        """
        Determine if an agent should be temporarily disabled based on performance
        """
        try:
            performance_metrics = self.get_agent_performance_metrics()
            
            if agent_name not in performance_metrics:
                return False, "No performance data available"
            
            metrics = performance_metrics[agent_name]
            
            # Disable criteria
            if metrics['success_rate'] < 0.1 and metrics['total_calls'] >= 5:
                return True, f"Very low success rate: {metrics['success_rate']:.1%}"
            
            if metrics['reliability_score'] < 0.2 and metrics['total_calls'] >= 3:
                return True, f"Poor reliability score: {metrics['reliability_score']:.2f}"
            
            if metrics['status'] == 'critical':
                return True, "Agent status is critical"
            
            # Check for consecutive recent failures
            if metrics['failed_calls'] >= 3 and metrics['success_rate'] < 0.3:
                return True, f"Multiple recent failures with low success rate"
            
            return False, "Agent performance within acceptable limits"
            
        except Exception as e:
            logger.error(f"Error checking disable criteria for {agent_name}: {e}")
            return False, f"Error checking agent: {str(e)}"

    def extract_and_combine_llm_responses(self, individual_results: Dict[str, VolumeAgentResult]) -> str:
        """
        Extract and combine LLM responses from successful volume agents into a comprehensive text summary
        for use by the final decision agent.
        """
        try:
            successful_results = {k: v for k, v in individual_results.items() if v.success and v.llm_response}
            
            if not successful_results:
                return ""
            
            combined_analysis = f"# Volume Analysis Summary from {len(successful_results)} Volume Agents\n\n"
            
            # Sort agents by their configured weights (most important first)
            agent_weights = {name: config.get('weight', 0.2) for name, config in self.orchestrator.agent_config.items()}
            sorted_agents = sorted(successful_results.keys(), key=lambda x: agent_weights.get(x, 0.2), reverse=True)
            
            for agent_name in sorted_agents:
                result = successful_results[agent_name]
                agent_display_name = agent_name.replace('_', ' ').title()
                weight = agent_weights.get(agent_name, 0.2)
                confidence = result.confidence_score or 0.5
                
                combined_analysis += f"## {agent_display_name} Agent Analysis (Weight: {weight:.0%}, Confidence: {confidence:.0%})\n\n"
                
                # Clean up the LLM response
                llm_response = result.llm_response.strip()
                
                # Remove any JSON formatting or technical artifacts
                if llm_response.startswith('{') and llm_response.endswith('}'):
                    try:
                        import json
                        parsed = json.loads(llm_response)
                        if isinstance(parsed, dict) and 'analysis' in parsed:
                            llm_response = parsed['analysis']
                        elif isinstance(parsed, dict) and 'result' in parsed:
                            llm_response = parsed['result']
                    except json.JSONDecodeError:
                        pass  # Keep original response
                
                combined_analysis += f"{llm_response}\n\n"
                
                # Add key metrics if available
                if result.analysis_data:
                    key_metrics = self._extract_key_metrics_for_summary(agent_name, result.analysis_data)
                    if key_metrics:
                        combined_analysis += f"**Key Metrics:** {key_metrics}\n\n"
            
            # Add overall summary
            combined_analysis += f"## Volume Analysis Summary\n\n"
            combined_analysis += f"Based on analysis from {len(successful_results)} volume agents: {', '.join([name.replace('_', ' ').title() for name in sorted_agents])}\n\n"
            
            # Calculate overall confidence
            weighted_confidence = sum(result.confidence_score * agent_weights.get(agent_name, 0.2) 
                                    for agent_name, result in successful_results.items() 
                                    if result.confidence_score) / len(successful_results)
            
            combined_analysis += f"**Overall Volume Analysis Confidence:** {weighted_confidence:.0%}\n\n"
            
            return combined_analysis
            
        except Exception as e:
            logger.error(f"Error combining LLM responses: {e}")
            return f"Volume analysis available from {len(individual_results)} agents, but synthesis failed: {str(e)}"
    
    def _extract_key_metrics_for_summary(self, agent_name: str, analysis_data: Dict[str, Any]) -> str:
        """
        Extract key metrics from agent analysis data for summary
        """
        try:
            metrics = []
            
            if agent_name == 'volume_anomaly':
                anomalies = analysis_data.get('anomalies', [])
                if anomalies:
                    high_sig = len([a for a in anomalies if a.get('significance') == 'high'])
                    metrics.append(f"{len(anomalies)} anomalies detected ({high_sig} high significance)")
            
            elif agent_name == 'institutional_activity':
                activity_level = analysis_data.get('activity_level', 'unknown')
                activity_pattern = analysis_data.get('activity_pattern', 'unknown')
                if activity_level != 'unknown':
                    metrics.append(f"Activity: {activity_level}")
                if activity_pattern != 'unknown':
                    metrics.append(f"Pattern: {activity_pattern}")
            
            elif agent_name == 'volume_confirmation':
                confirmation = analysis_data.get('price_volume_confirmation', {})
                status = confirmation.get('status', 'unknown')
                if status != 'unknown':
                    metrics.append(f"Price-Volume: {status}")
            
            elif agent_name == 'support_resistance':
                levels = analysis_data.get('key_levels', [])
                if levels:
                    strong_levels = len([l for l in levels if l.get('strength', 0) > 0.7])
                    metrics.append(f"{len(levels)} S/R levels ({strong_levels} strong)")
            
            elif agent_name == 'volume_momentum':
                momentum = analysis_data.get('volume_momentum', {})
                direction = momentum.get('direction', 'unknown')
                strength = momentum.get('strength', 0)
                if direction != 'unknown':
                    metrics.append(f"Momentum: {direction} ({strength:.0%} strength)")
            
            return " | ".join(metrics) if metrics else ""
            
        except Exception as e:
            logger.warning(f"Error extracting key metrics for {agent_name}: {e}")
            return ""

    def _create_fallback_volume_analysis(self, stock_data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Create basic volume analysis when all agents fail
        """
        try:
            current_volume = stock_data['volume'].iloc[-1]
            volume_ma_20 = stock_data['volume'].rolling(window=20).mean().iloc[-1]
            
            # Basic volume metrics
            volume_ratio = current_volume / volume_ma_20 if volume_ma_20 > 0 else 1.0
            volume_percentile = self._calculate_volume_percentile(stock_data)
            volume_trend = self._determine_volume_trend(stock_data)
            
            return {
                'analysis_method': 'fallback',
                'current_volume': int(current_volume),
                'volume_ma_20': int(volume_ma_20),
                'volume_ratio': round(volume_ratio, 2),
                'volume_percentile': round(volume_percentile, 1),
                'volume_trend': volume_trend,
                'basic_signals': {
                    'high_volume': volume_ratio > 2.0,
                    'above_average': volume_ratio > 1.5,
                    'low_volume': volume_ratio < 0.5
                },
                'confidence': 0.3,  # Low confidence for fallback analysis
                'limitations': [
                    'Basic volume metrics only',
                    'No advanced pattern recognition',
                    'No institutional analysis',
                    'Limited signal reliability'
                ]
            }
        except Exception as e:
            logger.error(f"Fallback volume analysis failed: {e}")
            return {
                'analysis_method': 'fallback',
                'error': str(e),
                'confidence': 0.0
            }
