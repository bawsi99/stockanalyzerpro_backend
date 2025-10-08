# Backend Directory Tree

```
backend/
├── .venv/ [Python 3.10 virtual environment - complete Python ecosystem]
├── __pycache__/ [Python cache directory]
├── agents/
│   ├── __init__.py
│   ├── README.md
│   ├── final_decision/
│   │   ├── __init__.py
│   │   ├── MIGRATION_SUMMARY.md
│   │   ├── optimized_final_decision.txt
│   │   ├── processor.py
│   │   ├── prompt_processor.py
│   │   └── test_migration.py
│   ├── indicators/
│   │   ├── __init__.py
│   │   ├── context_engineer.py
│   │   ├── indicator_summary_prompt.txt
│   │   ├── indicators_agents.py
│   │   ├── integration_manager.py
│   │   ├── llm_integration.py
│   │   ├── MIGRATION_COMPLETE.md
│   │   ├── prompt_manager.py
│   │   └── test_new_system.py
│   ├── mtf_analysis/
│   │   ├── __init__.py
│   │   ├── integration_manager.py
│   │   ├── MTF_LLM_AGENT_INTEGRATION.md
│   │   ├── mtf_agents.py
│   │   ├── mtf_llm_agent.py
│   │   ├── optimized_mtf_comparison.txt
│   │   ├── orchestrator.py
│   │   ├── test_migration.py
│   │   └── test_mtf_llm_agent_only.py
│   ├── patterns/
│   │   ├── __init__.py
│   │   ├── optimized_pattern_analysis.txt
│   │   ├── optimized_reversal_patterns.txt
│   │   ├── optimized_technical_overview.txt
│   │   └── patterns_agents.py
│   ├── risk_analysis/
│   │   ├── __init__.py
│   │   ├── multi_stock_test.py
│   │   ├── risk_analysis_prompt.txt
│   │   ├── risk_llm_agent.py
│   │   └── test_migration.py
│   ├── sector/
│   │   ├── __init__.py
│   │   ├── benchmarking.py
│   │   ├── cache_config.json
│   │   ├── cache_manager.py
│   │   ├── classifier.py
│   │   ├── direct_test.py
│   │   ├── enhanced_classifier.py
│   │   ├── MIGRATION_SUMMARY.md
│   │   ├── processor.py
│   │   ├── prompt_response_test.py
│   │   ├── sector_synthesis_template.txt
│   │   ├── simple_test.py
│   │   ├── structure_test.py
│   │   └── test_migration.py
│   └── volume/
│       ├── __init__.py
│       ├── test_institutional_integration.py
│       ├── test_support_resistance_migration.py
│       └── volume_agents.py
├── analysis/
│   ├── advanced_analysis.py
│   ├── datasets.py
│   ├── heavy_load.py
│   ├── inference.py
│   └── risk_scoring.py
├── analysis_datasets/ [Directory for analysis-specific datasets]
├── analysis_service.log [Service log file]
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
├── examples/
│   ├── __init__.py
│   └── new_mechanism_example.py
├── GEMINI_OPTIMIZATION_REPORT.md [Gemini optimization documentation]
├── gemini.archive/ [Archive of old Gemini implementation]
│   ├── __init__.py
│   ├── api_key_manager.py
│   ├── audit_llm_calls.py
│   ├── context_engineer.py
│   ├── debug_config.py
│   ├── debug_logger.py
│   ├── error_utils.py
│   ├── gemini_client.py
│   ├── gemini_core.py
│   ├── image_utils.py
│   ├── parallel_pattern_detection.py
│   ├── prompt_manager.py
│   ├── schema.py
│   └── token_tracker.py
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
│       └── REORGANIZATION_SUMMARY.md
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
- **agents/**: Modern agent-based architecture with specialized components:
  - `final_decision/`: Final trading decision processing
  - `indicators/`: Technical indicator analysis agents
  - `mtf_analysis/`: Multi-timeframe analysis agents
  - `patterns/`: Chart pattern recognition agents
  - `risk_analysis/`: Risk assessment and scoring agents
  - `sector/`: Sector analysis and classification agents
  - `volume/`: Volume analysis agents
- **analysis/**: Advanced analysis modules including heavy load processing
- **api/**: FastAPI endpoints and responses
- **cache/**: Cached stock data storage (47+ CSV files)
- **config/**: Configuration files, requirements, and deployment settings
- **core/**: Core utilities, cache management, orchestrators, and client connections
- **data/**: Structured data storage including:
  - Analysis datasets (portfolio, sector, trading)
  - Enhanced sector data with performance metrics
  - 20+ sector category JSON files
  - Signal processing and weights configuration
- **docs/**: Documentation for agent migration and decision flow
- **examples/**: Example implementations and usage patterns
- **gemini.archive/**: Archived old Gemini implementation
- **llm/**: New unified LLM system with:
  - Provider-agnostic architecture
  - Key rotation and management
  - Token tracking and usage monitoring
  - Comprehensive test suite
- **ml/**: Machine learning components including market regime analysis and quantitative systems
- **patterns/**: Pattern recognition and confirmation modules
- **services/**: Backend services for data processing and analysis
- **utils/**: Utility functions and LLM response extraction tools
- **zerodha/**: Zerodha trading platform integration with WebSocket support

### Key Features
- **Agent-Based Architecture**: Modular, specialized agents for different analysis tasks
- **Unified LLM System**: Provider-agnostic LLM integration with advanced token management
- **Advanced Data Management**: Comprehensive caching, sector classification, and signal processing
- **Real-time Integration**: WebSocket support for live data feeds
- **Pattern Recognition**: Advanced chart pattern detection and confirmation
- **Risk Management**: Sophisticated risk analysis and scoring systems
- **Extensible Design**: Well-documented migration paths and modular architecture

### File Statistics
- **Total Python files**: 80+ core Python modules
- **Configuration files**: 7 configuration and requirements files
- **Data files**: 47+ CSV cache files, 20+ sector JSON files, multiple datasets
- **Documentation files**: 10+ markdown documentation files
- **Test files**: 15+ test and migration files
- **Virtual Environment**: Complete Python 3.10 environment (.venv/)
- **Archive files**: 13 archived Gemini implementation files

### Architecture Evolution
- **Agent Migration**: Successfully migrated from monolithic to agent-based architecture
- **LLM Unification**: Replaced multiple LLM implementations with unified system
- **Enhanced Documentation**: Comprehensive guides for migration and usage
- **Improved Testing**: Extensive test coverage for new agent system
- **Better Organization**: Clear separation of concerns with specialized modules
- **Modern Python**: Updated to use Python 3.10 with .venv virtual environment
