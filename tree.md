# Backend Directory Tree

```
backend/
├── .venv/ [Python 3.10 virtual environment - complete Python ecosystem]
├── __pycache__/ [Python cache directory]
├── agents/
│   ├── __init__.py
│   ├── final_decision/
│   │   ├── __init__.py
│   │   ├── cache/ [5 CSV cache files + metadata]
│   │   ├── MIGRATION_SUMMARY.md
│   │   ├── optimized_final_decision.txt
│   │   ├── processor.py
│   │   ├── prompt_processor.py
│   │   └── verification_results/
│   ├── indicators/
│   │   ├── __init__.py
│   │   ├── context_engineer.py
│   │   ├── indicator_summary_prompt.txt
│   │   ├── indicators_agents.py
│   │   ├── integration_manager.py
│   │   ├── llm_integration.py
│   │   ├── MIGRATION_COMPLETE.md
│   │   ├── momentum/
│   │   │   ├── __init__.py
│   │   │   └── processor.py
│   │   ├── prompt_manager.py
│   │   ├── prompt_testing/
│   │   │   └── indicator_summary/
│   │   │       └── cache/
│   │   ├── test_new_system.py
│   │   ├── tests/
│   │   │   └── test_processors_schema.py
│   │   └── trend/
│   │       ├── __init__.py
│   │       └── processor.py
│   ├── mtf_analysis/
│   │   ├── __init__.py
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   └── processor.py
│   │   ├── integration_manager.py
│   │   ├── intraday/
│   │   │   ├── __init__.py
│   │   │   └── processor.py
│   │   ├── MTF_LLM_AGENT_INTEGRATION.md
│   │   ├── mtf_agents.py
│   │   ├── mtf_llm_agent.py
│   │   ├── optimized_mtf_comparison.txt
│   │   ├── orchestrator.py
│   │   ├── position/
│   │   │   ├── __init__.py
│   │   │   └── processor.py
│   │   ├── prompt_testing/
│   │   │   └── mtf_comprehensive/
│   │   │       └── cache/
│   │   ├── swing/
│   │   │   ├── __init__.py
│   │   │   └── processor.py
│   │   └── test_mtf_llm_agent_only.py
│   ├── patterns/ [ENHANCED PATTERN RECOGNITION]
│   │   ├── __init__.py
│   │   ├── pattern_agents.py
│   │   ├── utils.py
│   │   ├── cross_validation_agent/
│   │   │   ├── __init__.py
│   │   │   ├── cross_validation_agent.md
│   │   │   ├── multi_stock_test.py
│   │   │   ├── processor.py
│   │   │   └── pattern_detection/
│   │   │       ├── __init__.py
│   │   │       ├── detector.py
│   │   │       └── processor.py
│   │   └── market_structure_agent/
│   │       ├── __init__.py
│   │       ├── agent.py
│   │       ├── charts.py
│   │       ├── integration.py
│   │       ├── llm_agent.py
│   │       ├── market_structure_analysis.txt
│   │       ├── multi_stock_test.py
│   │       └── processor.py
│   ├── risk_analysis/
│   │   ├── __init__.py
│   │   ├── multi_stock_test.py
│   │   ├── quantitative_risk/
│   │   │   └── processor.py
│   │   ├── risk_analysis_prompt.txt
│   │   └── risk_llm_agent.py
│   ├── sector/
│   │   ├── __init__.py
│   │   ├── benchmarking.py
│   │   ├── cache/
│   │   │   ├── [30+ sector-specific cache directories]
│   │   │   │   └── [Sector-specific CSV files and metadata]
│   │   │   ├── cache_metadata.json
│   │   │   └── sector_cache_manifest.json
│   │   ├── cache_config.json
│   │   ├── cache_manager.py
│   │   ├── classifier.py
│   │   ├── direct_test.py
│   │   ├── enhanced_classifier.py
│   │   ├── MIGRATION_SUMMARY.md
│   │   ├── processor.py
│   │   ├── prompt_response_test.py
│   │   ├── sector_synthesis_template.txt
│   │   ├── sector_synthesis_test_results/
│   │   │   ├── sector_prompt_HDFCBANK_20251002_195551.txt
│   │   │   └── sector_response_HDFCBANK_20251002_195600.txt
│   │   ├── simple_test.py
│   │   └── structure_test.py
│   └── volume/
│       ├── __init__.py
│       ├── institutional_activity/
│       │   ├── __init__.py
│       │   ├── agent.py
│       │   ├── cache/
│       │   ├── charts.py
│       │   ├── institutional_activity_analysis.txt
│       │   ├── integration.py
│       │   ├── llm_agent.py
│       │   ├── multi_stock_test.py
│       │   └── processor.py
│       ├── support_resistance/
│       │   ├── __init__.py
│       │   ├── agent.py
│       │   ├── cache/
│       │   ├── charts.py
│       │   ├── integration.py
│       │   ├── llm_agent.py
│       │   ├── multi_stock_test.py
│       │   ├── processor.py
│       │   └── volume_support_resistance.txt
│       ├── test_institutional_integration.py
│       ├── test_support_resistance_migration.py
│       ├── volume_agents.py
│       ├── volume_anomaly/
│       │   ├── __init__.py
│       │   ├── agent.py
│       │   ├── cache/
│       │   ├── charts.py
│       │   ├── integration.py
│       │   ├── llm_agent.py
│       │   ├── multi_stock_test.py
│       │   ├── processor.py
│       │   └── volume_anomaly_detection.txt
│       ├── volume_confirmation/
│       │   ├── __init__.py
│       │   ├── cache/
│       │   ├── charts.py
│       │   ├── context.py
│       │   ├── llm_agent.py
│       │   ├── multi_stock_test.py
│       │   ├── processor.py
│       │   └── volume_confirmation_analysis.txt
│       └── volume_momentum/
│           ├── __init__.py
│           ├── agent.py
│           ├── cache/
│           ├── charts.py
│           ├── llm_agent.py
│           ├── multi_stock_test.py
│           ├── processor.py
│           └── volume_trend_momentum.txt
├── analysis/
│   ├── advanced_analysis.py
│   ├── datasets.py
│   ├── heavy_load.py
│   ├── inference.py
│   └── risk_scoring.py
├── analysis_datasets/ [Directory for analysis-specific datasets]
├── api/
│   ├── __init__.py
│   └── responses.py
├── cache/
│   ├── [47+ CSV files - cached stock data]
│   └── cache_metadata.json
├── config/
│   ├── __init__.py
│   ├── constraints.txt
│   ├── deployment_config.py
│   ├── requirements_data_database_service.txt
│   ├── requirements-dev.txt
│   ├── requirements.txt
│   ├── runtime.txt
│   └── storage_config.py
├── core/
│   ├── __init__.py
│   ├── chart_manager.py
│   ├── config.py
│   ├── instrument_filter.py
│   ├── orchestrator.py
│   ├── path_utils.py
│   ├── redis_unified_cache_manager.py
│   ├── sector_manager.py
│   ├── supabase_client.py
│   └── utils.py
├── data/
│   ├── __init__.py
│   ├── analysis_datasets/
│   │   ├── analysis_datasets.json
│   │   ├── portfolio_dataset.json
│   │   ├── sector_dataset_backup.json
│   │   ├── sector_dataset.json
│   │   └── trading_dataset.json
│   ├── enhanced_sector_data/
│   │   ├── filtered_equity_stocks.csv
│   │   ├── instrument_breakdown.json
│   │   ├── major_stocks.json
│   │   └── sector_performance.json
│   ├── sector_category/
│   │   ├── __init__.py
│   │   ├── auto.json
│   │   ├── banking.json
│   │   ├── chemicals.json
│   │   ├── consumer_durables.json
│   │   ├── financial_services_25_50.json
│   │   ├── financial_services_ex_bank.json
│   │   ├── financial_services.json
│   │   ├── fmcg.json
│   │   ├── healthcare.json
│   │   ├── it.json
│   │   ├── media.json
│   │   ├── metal.json
│   │   ├── midsmall_financial_services.json
│   │   ├── midsmall_healthcare.json
│   │   ├── midsmall_it_telecom.json
│   │   ├── nifty500_healthcare.json
│   │   ├── oil_and_gas.json
│   │   ├── pharma.json
│   │   ├── private_bank.json
│   │   ├── psu_bank.json
│   │   └── realty.json
│   ├── signals/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── regimes.py
│   │   ├── schema.py
│   │   ├── scoring.py
│   │   └── weights_config.json
│   ├── updated_sector_dataset.json
│   └── zerodha_instruments.csv
├── demo_clean_logs.py [Demo script for log cleaning]
├── demo_table_format.py [Demo script for table formatting]
├── docs/
│   ├── agent_migration_guide.md
│   └── decision_flow.md
├── enhanced_sector_data/ [Link/directory for enhanced sector data]
├── ENV_CONFIG_GUIDE.md [Environment configuration documentation]
├── llm/ [New unified LLM system]
│   ├── __init__.py
│   ├── client.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.py
│   ├── examples/
│   │   └── simple_usage.py
│   ├── key_manager.py
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── gemini.py
│   ├── tests/
│   │   ├── test_basic_functionality.py
│   │   ├── test_gemini_live.py
│   │   ├── test_gemini_requests.py
│   │   └── test_key_rotation.py
│   ├── token_counter.py
│   ├── USAGE_GUIDE.md
│   └── utils.py
├── LOG_FORMAT_SUMMARY.md [Log format documentation]
├── logs/ [Directory for service logs]
├── ml/
│   ├── __init__.py
│   ├── analysis/
│   │   └── market_regime.py
│   ├── architecture.md
│   ├── bayesian_scorer.py
│   ├── indicators/
│   │   ├── technical_indicators.py
│   │   └── volume_profile.py
│   ├── inference.py
│   └── quant_system/
│       ├── __init__.py
│       ├── advanced/
│       │   └── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── base_models.py
│       │   ├── config.py
│       │   ├── registry.py
│       │   └── utils.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── dataset_builder.py
│       │   ├── enhanced_pipeline.py
│       │   ├── market_data_integration.py
│       │   └── pipeline.py
│       ├── engines/
│       │   ├── __init__.py
│       │   ├── hybrid_ml.py
│       │   ├── pattern_ml.py
│       │   ├── raw_data_ml.py
│       │   └── unified_manager.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── model_comparison.py
│       │   ├── price_analysis.py
│       │   └── robust_evaluation.py
│       ├── features/
│       │   ├── __init__.py
│       │   ├── enhanced_features.py
│       │   ├── feature_engineer.py
│       │   └── technical_indicators.py
│       ├── REORGANIZATION_SUMMARY.md
│       ├── scripts/
│       │   ├── __init__.py
│       │   ├── run_tests.py
│       │   └── train_models.py
│       ├── trading/
│       │   └── __init__.py
│       └── utils/
│           └── __init__.py
├── output/
│   └── charts/ [Generated chart outputs]
├── patterns/
│   ├── confirmation.py
│   ├── database.py
│   ├── optimized_continuation_levels.txt
│   ├── recognition.py
│   └── visualization.py
├── README.md [Main backend documentation]
├── sector_category -> ./data/sector_category [Symbolic link]
├── services/
│   ├── __init__.py
│   ├── analysis_service.py
│   ├── data_service.py
│   ├── database_service.py
│   └── enhanced_data_service.py
├── test_log_format.py [Log format testing]
├── test_token_counter_standalone.py [Token counter testing]
├── test_token_integration.py [Token integration testing]
├── TOKEN_TRACKING_GUIDE.md [Token tracking documentation]
├── tree.md [This file - directory structure documentation]
├── utils/
│   ├── __init__.py
│   ├── llm_response_extractor.py
│   ├── README.md
│   └── test_extractor.py
├── zerodha/
│   ├── __init__.py
│   ├── cache/
│   │   ├── b69a63e156b3e23d8f3129be5d8028f1.csv
│   │   └── cache_metadata.json
│   ├── client.py
│   ├── token_updater.py
│   └── ws_client.py
└── zerodha_instruments.csv -> ./data/zerodha_instruments.csv [Symbolic link]
```

## Summary

The backend directory contains a comprehensive stock analysis and trading system with the following main components:

### Core Modules
- **agents/**: Advanced agent-based architecture with specialized processors:
  - `final_decision/`: Final trading decision processing with dedicated cache
  - `indicators/`: Technical indicator agents with momentum/trend processors and testing framework
  - `mtf_analysis/`: Multi-timeframe analysis with core/intraday/position/swing processors
  - `patterns/`: **ENHANCED** pattern recognition with new specialized agents:
    - `cross_validation_agent/`: Cross-validation pattern detection with pattern_detection module
    - `market_structure_agent/`: Market structure analysis with comprehensive testing
  - `risk_analysis/`: Risk assessment with quantitative risk processing
  - `sector/`: Advanced sector analysis with expanded caching (30+ sector directories)
  - `volume/`: Comprehensive volume analysis with 5 specialized sub-agents:
    - institutional_activity, support_resistance, volume_anomaly, volume_confirmation, volume_momentum
- **analysis/**: Advanced analysis modules including heavy load processing and risk scoring
- **api/**: FastAPI endpoints and responses
- **cache/**: Cached stock data storage (47+ CSV files)
- **config/**: Configuration files, requirements, and deployment settings
- **core/**: Core utilities, cache management, orchestrators, and client connections
- **data/**: Structured data storage including:
  - Analysis datasets (portfolio, sector, trading)
  - Enhanced sector data with performance metrics
  - 19+ sector category JSON files
  - Signal processing and weights configuration
- **docs/**: Documentation for agent migration and decision flow
- **llm/**: Production-ready unified LLM system with:
  - Provider-agnostic architecture
  - Key rotation and management
  - Token tracking and usage monitoring
  - Comprehensive test suite
- **ml/**: Enhanced machine learning components with:
  - Market regime analysis and quantitative systems
  - Complete quant_system with core/data/engines/evaluation/features modules
- **patterns/**: Pattern recognition and confirmation modules
- **services/**: Backend services for data processing and analysis
- **utils/**: Utility functions and LLM response extraction tools
- **zerodha/**: Zerodha trading platform integration with WebSocket support

### Key Features
- **Enhanced Agent Architecture**: Advanced agent system with 25+ specialized processors
- **Next-Generation Pattern Recognition**: Revolutionary pattern analysis with:
  - Cross-validation agent with dedicated pattern detection module
  - Market structure agent with comprehensive analysis capabilities
- **Comprehensive Volume Analysis**: 5 dedicated volume analysis agents with individual caching systems
- **Multi-Timeframe Analysis**: 4 timeframe processors (core, intraday, position, swing)
- **Advanced Caching System**: Expanded intelligent caching with 100+ distributed cache files
- **Sector Analysis Excellence**: 30+ sector-specific cache directories for comprehensive market coverage
- **Unified LLM System**: Production-ready LLM integration with advanced token management
- **Extensive Testing Framework**: Multi-stock testing and cross-validation systems
- **Modern Architecture**: Continuous evolution with enhanced agent capabilities

### File Statistics
- **Total Python files**: 150+ core Python modules (significantly expanded with new agents)
- **Configuration files**: 8 configuration and requirements files
- **Data files**: 100+ CSV cache files (distributed across agents + 30+ sector directories), 19 sector JSON files, multiple datasets
- **Documentation files**: 10+ markdown documentation files including specialized agent guides
- **Test files**: 30+ test, migration, and verification files (expanded testing framework)
- **Virtual Environment**: Complete Python 3.10 environment (.venv/)
- **Cache directories**: 35+ specialized cache directories across agents and sectors
- **Agent processors**: 25+ specialized processing modules (enhanced with new pattern agents)

### Architecture Evolution
- **Revolutionary Pattern Recognition**: Added cross-validation and market structure agents with advanced detection capabilities
- **Enhanced Agent System**: Expanded from 22+ to 25+ specialized processors with new pattern analysis capabilities
- **Advanced Caching**: Evolved from 70+ to 100+ cache files with 30+ sector-specific directories
- **Comprehensive Testing**: Enhanced multi-stock testing framework with cross-validation capabilities
- **Modern LLM Integration**: Production-ready unified LLM system with advanced token management
- **Scalable Architecture**: Modular design supporting continuous agent expansion and enhancement
- **Market Coverage Excellence**: 30+ sector-specific analysis capabilities for comprehensive market insights
